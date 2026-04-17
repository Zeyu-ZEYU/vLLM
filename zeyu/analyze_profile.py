#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Post-processing script that merges iteration JSONL logs, nsys kernel
timeline, and (optional) ncu SM metrics into consolidated output files.

Usage:
    python zeyu/analyze_profile.py <profile_output_dir>

Produces:
    <profile_output_dir>/consolidated_iterations.jsonl
    <profile_output_dir>/consolidated_requests.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Load iteration JSONL
# ---------------------------------------------------------------------------
def load_iterations(path: Path) -> list[dict]:
    """Load iterations.jsonl produced by IterationLogger."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_requests(path: Path) -> list[dict]:
    """Load requests.jsonl produced by IterationLogger."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Parse nsys NVTX CSV — identify iteration NVTX ranges
# ---------------------------------------------------------------------------
def parse_nsys_nvtx_iterations(path: Path) -> list[dict]:
    """Parse nsys NVTX CSV and build per-iteration time windows.

    vLLM emits NVTX ranges like ``gpu_model_runner: preprocess``,
    ``gpu_model_runner: forward``, ``gpu_model_runner: sample`` for each
    iteration.  We use the ``preprocess`` ranges as iteration start
    markers and extend each iteration window to the end of the
    corresponding ``sample`` range.  If preprocess ranges are absent, we
    fall back to ``forward`` ranges alone.
    """
    if not path.exists():
        return []

    # Collect all relevant NVTX ranges.
    preprocess_ranges: list[dict] = []
    forward_ranges: list[dict] = []
    sample_ranges: list[dict] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", row.get("Range", ""))
            try:
                start_ns = int(row.get("Start (ns)", 0))
                # nvtx_pushpop_trace may have "End (ns)" instead of
                # "Duration (ns)"; handle both.
                end_raw = row.get("End (ns)")
                dur_raw = row.get("Duration (ns)")
                if end_raw is not None:
                    end_ns = int(end_raw)
                    duration_ns = end_ns - start_ns
                elif dur_raw is not None:
                    duration_ns = int(dur_raw)
                    end_ns = start_ns + duration_ns
                else:
                    continue
            except (ValueError, TypeError):
                continue
            if start_ns == 0 and duration_ns == 0:
                continue
            entry = {
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": duration_ns,
                "name": name,
            }
            if "gpu_model_runner: preprocess" in name:
                preprocess_ranges.append(entry)
            elif "gpu_model_runner: forward" in name:
                forward_ranges.append(entry)
            elif "gpu_model_runner: sample" in name:
                sample_ranges.append(entry)

    # Sort all by start time.
    preprocess_ranges.sort(key=lambda r: r["start_ns"])
    forward_ranges.sort(key=lambda r: r["start_ns"])
    sample_ranges.sort(key=lambda r: r["start_ns"])

    # Build iteration windows.
    # Prefer preprocess as start, sample as end (covers the full step).
    # Fall back to forward ranges if preprocess is missing.
    if preprocess_ranges:
        iterations = []
        for i, pp in enumerate(preprocess_ranges):
            # Find the sample range that ends this iteration.
            end_ns = pp["end_ns"]
            for sr in sample_ranges:
                if sr["start_ns"] >= pp["start_ns"]:
                    end_ns = sr["end_ns"]
                    break
            iterations.append(
                {
                    "start_ns": pp["start_ns"],
                    "end_ns": end_ns,
                    "duration_ns": end_ns - pp["start_ns"],
                    "name": pp["name"],
                }
            )
        return iterations

    # Fallback: use forward ranges as iteration boundaries.
    return forward_ranges


def parse_nsys_nvtx_by_name(path: Path, keyword: str) -> list[dict]:
    """Parse nsys NVTX CSV for sub-ranges whose name contains *keyword*."""
    if not path.exists():
        return []

    ranges = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", row.get("Range", ""))
            if keyword not in name:
                continue
            try:
                start_ns = int(row.get("Start (ns)", 0))
                end_raw = row.get("End (ns)")
                dur_raw = row.get("Duration (ns)")
                if end_raw is not None:
                    end_ns = int(end_raw)
                    duration_ns = end_ns - start_ns
                elif dur_raw is not None:
                    duration_ns = int(dur_raw)
                    end_ns = start_ns + duration_ns
                else:
                    continue
            except (ValueError, TypeError):
                continue
            if start_ns == 0 and duration_ns == 0:
                continue
            ranges.append(
                {
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ns": duration_ns,
                }
            )
    ranges.sort(key=lambda r: r["start_ns"])
    return ranges


# ---------------------------------------------------------------------------
# Parse nsys kernel CSV — GPU kernel timeline
# ---------------------------------------------------------------------------
def parse_nsys_kernels(path: Path) -> list[dict]:
    """Parse nsys cuda_gpu_trace CSV.

    Returns list of dicts with start_ns, duration_ns, name.
    """
    if not path.exists():
        return []

    kernels = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start_ns = int(row.get("Start (ns)", 0))
                duration_ns = int(row.get("Duration (ns)", 0))
            except (ValueError, TypeError):
                continue
            kernels.append(
                {
                    "start_ns": start_ns,
                    "duration_ns": duration_ns,
                    "end_ns": start_ns + duration_ns,
                    "name": row.get("Name", ""),
                }
            )

    kernels.sort(key=lambda k: k["start_ns"])
    return kernels


# ---------------------------------------------------------------------------
# Parse nsys gpu_metrics CSV (SM-level aggregate metrics)
# ---------------------------------------------------------------------------
def parse_nsys_gpu_metrics(path: Path) -> list[dict]:
    """Parse nsys gpu_metrics CSV.

    Columns vary by nsys version but typically include:
      * ``Timestamp``/``Timestamp (ns)``  — sample time in ns
      * ``SM Active %`` / ``SM Active``    — fraction of GPU SMs active
      * ``SM Warp Occupancy`` / similar
      * ``SMs busy`` / ``Active SMs``      — count of active SMs

    Returns a list of dicts: ``{ts_ns, sm_active_pct, sm_occupancy_pct,
    num_active_sms}``. Fields that aren't present in the CSV are
    omitted.
    """
    if not path.exists():
        return []

    samples = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        # Resolve column names heuristically.
        ts_col = None
        for c in cols:
            low = c.lower()
            if "timestamp" in low and "ns" in low:
                ts_col = c
                break
        if ts_col is None:
            # fallback: any "Start (ns)" or "Timestamp"
            for c in cols:
                low = c.lower()
                if "timestamp" in low or "time" in low:
                    ts_col = c
                    break
        sm_active_col = None
        sm_occ_col = None
        active_sms_col = None
        for c in cols:
            low = c.lower()
            if sm_active_col is None and (
                "sm active" in low
                or "smactive" in low
                or low.startswith("sms active")
            ):
                sm_active_col = c
            if sm_occ_col is None and (
                "warp occupancy" in low or "sm occupancy" in low
            ):
                sm_occ_col = c
            if active_sms_col is None and (
                "active sms" in low or "smsbusy" in low or "sms busy" in low
            ):
                active_sms_col = c

        for row in reader:
            try:
                if ts_col:
                    ts_ns = int(float(row.get(ts_col, 0)))
                else:
                    continue
            except (ValueError, TypeError):
                continue
            entry = {"ts_ns": ts_ns}
            if sm_active_col:
                try:
                    entry["sm_active_pct"] = float(row.get(sm_active_col, 0))
                except (ValueError, TypeError):
                    pass
            if sm_occ_col:
                try:
                    entry["sm_occupancy_pct"] = float(row.get(sm_occ_col, 0))
                except (ValueError, TypeError):
                    pass
            if active_sms_col:
                try:
                    entry["num_active_sms"] = float(row.get(active_sms_col, 0))
                except (ValueError, TypeError):
                    pass
            samples.append(entry)

    samples.sort(key=lambda s: s["ts_ns"])
    return samples


def aggregate_gpu_metrics_in_window(
    samples: list[dict], start_ns: int, end_ns: int
) -> dict:
    """Average SM metrics across samples that fall in [start_ns, end_ns)."""
    if not samples:
        return {}
    # Binary-search could be done; linear fine for small N.
    in_range = [s for s in samples if start_ns <= s["ts_ns"] < end_ns]
    if not in_range:
        return {}
    out: dict = {"num_gpu_metric_samples": len(in_range)}
    for field, outk in (
        ("sm_active_pct", "sm_active_pct_mean"),
        ("sm_occupancy_pct", "sm_occupancy_pct_mean"),
        ("num_active_sms", "num_active_sms_mean"),
    ):
        vals = [s[field] for s in in_range if field in s]
        if vals:
            out[outk] = round(sum(vals) / len(vals), 3)
            out[outk.replace("_mean", "_max")] = round(max(vals), 3)
    return out


def correlate_kernels_to_ranges(
    kernels: list[dict], ranges: list[dict]
) -> list[list[dict]]:
    """For each NVTX range, find all CUDA kernels that overlap with it.

    Returns a list parallel to ranges, each element is a list of kernels.
    Uses two-pointer sweep (both sorted by start_ns).
    """
    result: list[list[dict]] = [[] for _ in ranges]
    if not kernels or not ranges:
        return result

    ki = 0
    for ri, rng in enumerate(ranges):
        # Advance kernel pointer to first kernel that could overlap.
        while ki < len(kernels) and kernels[ki]["end_ns"] <= rng["start_ns"]:
            ki += 1
        # Collect all kernels that start within this range.
        j = ki
        while j < len(kernels) and kernels[j]["start_ns"] < rng["end_ns"]:
            result[ri].append(kernels[j])
            j += 1

    return result


def compute_gpu_util(
    kernels: list[dict],
    wall_time_ns: int,
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
) -> dict:
    """Compute GPU utilization from a set of kernels within a time window.

    Returns dict with gpu_util_pct, total_kernel_time_ns, num_kernels.
    Handles overlapping kernels by computing non-overlapping coverage.

    If *window_start_ns* and *window_end_ns* are provided, kernel
    intervals are clipped to that window so that kernels extending
    beyond the measurement range do not inflate utilization above 100%.
    """
    if wall_time_ns <= 0 or not kernels:
        return {
            "gpu_util_pct": 0.0,
            "total_kernel_time_ns": 0,
            "num_kernels": 0,
            "kernel_launch_gap_ns": 0,
            "kernel_launch_gap_pct": 0.0,
        }

    # Sort intervals by start, merge overlaps to get true busy time.
    intervals = sorted(
        [(k["start_ns"], k["end_ns"]) for k in kernels],
        key=lambda x: x[0],
    )

    # Clip intervals to the measurement window.
    if window_start_ns is not None and window_end_ns is not None:
        intervals = [
            (max(s, window_start_ns), min(e, window_end_ns))
            for s, e in intervals
        ]
        intervals = [(s, e) for s, e in intervals if s < e]

    if not intervals:
        return {
            "gpu_util_pct": 0.0,
            "total_kernel_time_ns": 0,
            "num_kernels": len(kernels),
            "kernel_launch_gap_ns": wall_time_ns,
            "kernel_launch_gap_pct": 100.0,
        }
    merged_time = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged_time += cur_end - cur_start
            cur_start, cur_end = start, end
    merged_time += cur_end - cur_start

    gpu_util_pct = round(merged_time / wall_time_ns * 100, 2)
    gap_ns = wall_time_ns - merged_time
    return {
        "gpu_util_pct": gpu_util_pct,
        "total_kernel_time_ns": merged_time,
        "num_kernels": len(kernels),
        "kernel_launch_gap_ns": gap_ns,
        "kernel_launch_gap_pct": round(100 - gpu_util_pct, 2),
    }


# ---------------------------------------------------------------------------
# Parse ncu CSV — SM-level metrics
# ---------------------------------------------------------------------------
def parse_ncu_csv(path: Path) -> dict[str, dict]:
    """Parse ncu --csv output.

    Returns dict mapping kernel_name -> averaged SM metrics.
    ncu CSV has many columns; we extract the SM-related ones.
    """
    if not path.exists():
        return {}

    # ncu CSV may have comment lines starting with "==".
    rows: list[dict] = []
    with open(path, newline="") as f:
        # Skip leading comment lines.
        lines = [
            line for line in f if not line.startswith("==") and line.strip()
        ]

    if not lines:
        return {}

    reader = csv.DictReader(lines)
    for row in reader:
        rows.append(row)

    if not rows:
        return {}

    # Find relevant metric columns.
    sm_throughput_col = None
    sm_warps_col = None
    kernel_name_col = None

    for col in rows[0]:
        col_lower = col.lower()
        if "kernel name" in col_lower or col_lower == "name":
            kernel_name_col = col
        if "sm__throughput" in col_lower and "pct" in col_lower:
            sm_throughput_col = col
        if "sm__warps_active" in col_lower and "pct" in col_lower:
            sm_warps_col = col

    if kernel_name_col is None:
        return {}

    # Aggregate by kernel name.
    agg: dict[str, dict] = defaultdict(
        lambda: {
            "sm_throughput_pct_sum": 0.0,
            "sm_warps_active_pct_sum": 0.0,
            "count": 0,
        }
    )

    for row in rows:
        name = row.get(kernel_name_col, "")
        entry = agg[name]
        entry["count"] += 1
        if sm_throughput_col and row.get(sm_throughput_col):
            try:
                entry["sm_throughput_pct_sum"] += float(
                    row[sm_throughput_col]
                )
            except ValueError:
                pass
        if sm_warps_col and row.get(sm_warps_col):
            try:
                entry["sm_warps_active_pct_sum"] += float(row[sm_warps_col])
            except ValueError:
                pass

    result = {}
    for name, entry in agg.items():
        n = entry["count"]
        result[name] = {
            "avg_sm_throughput_pct": round(
                entry["sm_throughput_pct_sum"] / n, 2
            )
            if n > 0
            else 0.0,
            "avg_sm_warp_occupancy_pct": round(
                entry["sm_warps_active_pct_sum"] / n, 2
            )
            if n > 0
            else 0.0,
            "num_launches": n,
        }

    return result


def enrich_with_ncu(
    kernels: list[dict], ncu_by_name: dict[str, dict]
) -> dict:
    """Compute average SM metrics for a set of kernels using ncu data.

    Matches kernels by name (substring match if exact fails).
    """
    if not ncu_by_name or not kernels:
        return {}

    throughput_sum = 0.0
    occupancy_sum = 0.0
    matched = 0

    for k in kernels:
        kname = k["name"]
        ncu_entry = ncu_by_name.get(kname)
        if ncu_entry is None:
            # Try substring match.
            for ncu_name, ncu_val in ncu_by_name.items():
                if ncu_name in kname or kname in ncu_name:
                    ncu_entry = ncu_val
                    break
        if ncu_entry is not None:
            throughput_sum += ncu_entry["avg_sm_throughput_pct"]
            occupancy_sum += ncu_entry["avg_sm_warp_occupancy_pct"]
            matched += 1

    if matched == 0:
        return {}

    return {
        "avg_sm_throughput_pct": round(throughput_sum / matched, 2),
        "avg_sm_warp_occupancy_pct": round(occupancy_sum / matched, 2),
        "ncu_matched_kernels": matched,
    }


# ---------------------------------------------------------------------------
# Process a single directory of iteration + nsys + ncu data
# ---------------------------------------------------------------------------
def process_single_dir(
    profile_dir: Path,
    *,
    label: str = "",
) -> tuple[list[dict], list[dict]]:
    """Process one set of iteration logs + nsys/ncu CSVs.

    Returns (consolidated_iters, consolidated_reqs).
    """
    iter_path = profile_dir / "iterations.jsonl"
    req_path = profile_dir / "requests.jsonl"

    if not iter_path.exists():
        print(f"  {label}No iterations.jsonl in {profile_dir}")
        return [], []

    iterations = load_iterations(iter_path)
    requests = load_requests(req_path) if req_path.exists() else []
    print(
        f"  {label}Loaded {len(iterations)} iterations, "
        f"{len(requests)} requests."
    )

    # --- Parse nsys data (look in profile_dir and parent) ---
    nsys_kernel_csv = None
    nsys_nvtx_csv = None
    search_dirs = [profile_dir, profile_dir.parent]

    for d in search_dirs:
        if nsys_kernel_csv is None:
            for c in [
                "nsys_kernels_cuda_gpu_trace.csv",
                "nsys_kernels.csv",
            ]:
                p = d / c
                if p.exists():
                    nsys_kernel_csv = p
                    break
        if nsys_nvtx_csv is None:
            for c in [
                "nsys_nvtx_pushpop_nvtx_pushpop_trace.csv",
                "nsys_nvtx_pushpop.csv",
                "nsys_nvtx_nvtx_gpu_proj_trace.csv",
                "nsys_nvtx.csv",
            ]:
                p = d / c
                if p.exists():
                    nsys_nvtx_csv = p
                    break

    nvtx_ranges = (
        parse_nsys_nvtx_iterations(nsys_nvtx_csv)
        if nsys_nvtx_csv
        else []
    )
    ve_ranges = (
        parse_nsys_nvtx_by_name(nsys_nvtx_csv, "vision_encoder")
        if nsys_nvtx_csv
        else []
    )
    fwd_ranges = (
        parse_nsys_nvtx_by_name(nsys_nvtx_csv, "gpu_model_runner: forward")
        if nsys_nvtx_csv
        else []
    )
    kernels = (
        parse_nsys_kernels(nsys_kernel_csv) if nsys_kernel_csv else []
    )
    print(
        f"  {label}Parsed nsys: {len(nvtx_ranges)} iter NVTX, "
        f"{len(ve_ranges)} VE, {len(fwd_ranges)} fwd, "
        f"{len(kernels)} kernels."
    )

    if nvtx_ranges and kernels:
        print(
            f"  {label}NVTX time: "
            f"{nvtx_ranges[0]['start_ns']}..{nvtx_ranges[-1]['end_ns']}"
        )
        print(
            f"  {label}Kernel time: "
            f"{kernels[0]['start_ns']}..{kernels[-1]['end_ns']}"
        )

    ncu_csv_path = None
    for d in search_dirs:
        p = d / "ncu_metrics.csv"
        if p.exists():
            ncu_csv_path = p
            break
    ncu_by_name = parse_ncu_csv(ncu_csv_path) if ncu_csv_path else {}

    # --- Parse nsys GPU metrics (SM active etc.) ---
    gpu_metrics_csv = None
    for d in search_dirs:
        for candidate in [
            "nsys_gpu_metrics_gpu_metrics.csv",
            "nsys_gpu_metrics.csv",
        ]:
            p = d / candidate
            if p.exists():
                gpu_metrics_csv = p
                break
        if gpu_metrics_csv is not None:
            break
    gpu_metric_samples = (
        parse_nsys_gpu_metrics(gpu_metrics_csv) if gpu_metrics_csv else []
    )
    if gpu_metric_samples:
        print(
            f"  {label}Parsed {len(gpu_metric_samples)} GPU metric samples "
            f"from {gpu_metrics_csv.name}"
        )

    # --- Correlate ---
    kernels_per_iter = correlate_kernels_to_ranges(kernels, nvtx_ranges)

    consolidated_iters = []
    for i, it in enumerate(iterations):
        record = dict(it)
        if label:
            record["gpu_role"] = label.strip(" []")

        if i < len(nvtx_ranges):
            rng = nvtx_ranges[i]
            iter_kernels = kernels_per_iter[i]
            gpu_metrics = compute_gpu_util(
                iter_kernels,
                rng["duration_ns"],
                window_start_ns=rng["start_ns"],
                window_end_ns=rng["end_ns"],
            )
            record.update(gpu_metrics)

            ve_kernel_time_ns = 0
            for ve_rng in ve_ranges:
                if (
                    ve_rng["start_ns"] >= rng["start_ns"]
                    and ve_rng["end_ns"] <= rng["end_ns"]
                ):
                    ve_kernels = [
                        k
                        for k in iter_kernels
                        if k["start_ns"] >= ve_rng["start_ns"]
                        and k["start_ns"] < ve_rng["end_ns"]
                    ]
                    ve_gpu = compute_gpu_util(
                        ve_kernels,
                        ve_rng["duration_ns"],
                        window_start_ns=ve_rng["start_ns"],
                        window_end_ns=ve_rng["end_ns"],
                    )
                    ve_kernel_time_ns = ve_gpu["total_kernel_time_ns"]
                    record["vision_encoder_gpu_util_pct"] = ve_gpu[
                        "gpu_util_pct"
                    ]
                    ve_gap_ns = ve_gpu["kernel_launch_gap_ns"]
                    record["vision_encoder_kernel_launch_gap_ns"] = ve_gap_ns
                    record["vision_encoder_kernel_launch_gap_pct"] = round(
                        ve_gap_ns / rng["duration_ns"] * 100, 2
                    ) if rng["duration_ns"] > 0 else 0.0
            record["vision_encoder_kernel_time_ns"] = ve_kernel_time_ns

            for fwd_rng in fwd_ranges:
                if (
                    fwd_rng["start_ns"] >= rng["start_ns"]
                    and fwd_rng["end_ns"] <= rng["end_ns"]
                ):
                    fwd_kernels = [
                        k
                        for k in iter_kernels
                        if k["start_ns"] >= fwd_rng["start_ns"]
                        and k["start_ns"] < fwd_rng["end_ns"]
                    ]
                    fwd_gpu = compute_gpu_util(
                        fwd_kernels,
                        fwd_rng["duration_ns"],
                        window_start_ns=fwd_rng["start_ns"],
                        window_end_ns=fwd_rng["end_ns"],
                    )
                    record["text_forward_gpu_util_pct"] = fwd_gpu[
                        "gpu_util_pct"
                    ]
                    record["text_forward_kernel_time_ns"] = fwd_gpu[
                        "total_kernel_time_ns"
                    ]
                    fwd_gap_ns = fwd_gpu["kernel_launch_gap_ns"]
                    record["text_forward_kernel_launch_gap_ns"] = fwd_gap_ns
                    record["text_forward_kernel_launch_gap_pct"] = round(
                        fwd_gap_ns / rng["duration_ns"] * 100, 2
                    ) if rng["duration_ns"] > 0 else 0.0
                    break  # one forward per iteration

            if ncu_by_name and iter_kernels:
                sm_metrics = enrich_with_ncu(iter_kernels, ncu_by_name)
                record.update(sm_metrics)

            # Add GPU metrics sampled by nsys during this iteration's NVTX
            # window (aggregate SM active %, active-SM count, etc.).
            if gpu_metric_samples:
                gpu_mx = aggregate_gpu_metrics_in_window(
                    gpu_metric_samples, rng["start_ns"], rng["end_ns"]
                )
                if gpu_mx:
                    record.update(gpu_mx)

        per_req_phases = []
        for req_id in it.get("prefill_req_ids", []):
            ext_id = req_id.rsplit("-", 1)[0]
            per_req_phases.append({"id": ext_id, "phase": "prefill"})
        for req_id in it.get("decode_req_ids", []):
            ext_id = req_id.rsplit("-", 1)[0]
            per_req_phases.append({"id": ext_id, "phase": "decode"})
        record["requests"] = per_req_phases

        consolidated_iters.append(record)

    # --- Build consolidated requests ---
    iter_gpu_util_map = {}
    iter_sm_map = {}
    for ci in consolidated_iters:
        idx = ci["iter"]
        iter_gpu_util_map[idx] = ci.get("gpu_util_pct", None)
        iter_sm_map[idx] = {
            "sm_throughput": ci.get("avg_sm_throughput_pct", None),
            "sm_occupancy": ci.get("avg_sm_warp_occupancy_pct", None),
        }

    consolidated_reqs = []
    for req in requests:
        record = dict(req)
        if label:
            record["gpu_role"] = label.strip(" []")

        for phase in ("encoder", "prefill", "decode"):
            phase_iters = req.get(f"{phase}_iters", [])
            utils = [
                iter_gpu_util_map[i]
                for i in phase_iters
                if i in iter_gpu_util_map
                and iter_gpu_util_map[i] is not None
            ]
            record[f"{phase}_avg_gpu_util_pct"] = (
                round(sum(utils) / len(utils), 2) if utils else None
            )
            sm_vals = [
                iter_sm_map[i]
                for i in phase_iters
                if i in iter_sm_map
                and iter_sm_map[i]["sm_throughput"] is not None
            ]
            if sm_vals:
                record[f"{phase}_avg_sm_throughput_pct"] = round(
                    sum(v["sm_throughput"] for v in sm_vals) / len(sm_vals),
                    2,
                )
                record[f"{phase}_avg_sm_occupancy_pct"] = round(
                    sum(v["sm_occupancy"] for v in sm_vals) / len(sm_vals),
                    2,
                )

        consolidated_reqs.append(record)

    consolidated_reqs.sort(key=lambda r: r.get("first_iter") or 0)
    return consolidated_iters, consolidated_reqs


def print_summary(consolidated_iters: list[dict], label: str = ""):
    """Print GPU utilization summary for a set of iterations."""
    if not consolidated_iters:
        return
    gpu_utils = [
        ci["gpu_util_pct"]
        for ci in consolidated_iters
        if "gpu_util_pct" in ci
    ]
    if gpu_utils:
        print(
            f"\n{label}GPU utilization across "
            f"{len(gpu_utils)} iterations:"
        )
        print(f"  avg: {sum(gpu_utils) / len(gpu_utils):.1f}%")
        print(f"  min: {min(gpu_utils):.1f}%")
        print(f"  max: {max(gpu_utils):.1f}%")

    encoder_iters = [
        ci for ci in consolidated_iters if ci.get("has_encoder")
    ]
    prefill_only = [
        ci
        for ci in consolidated_iters
        if ci.get("num_prefill_reqs", 0) > 0
        and ci.get("num_decode_reqs", 0) == 0
        and not ci.get("has_encoder")
    ]
    decode_only = [
        ci
        for ci in consolidated_iters
        if ci.get("num_decode_reqs", 0) > 0
        and ci.get("num_prefill_reqs", 0) == 0
    ]
    mixed = [
        ci
        for ci in consolidated_iters
        if ci.get("num_prefill_reqs", 0) > 0
        and ci.get("num_decode_reqs", 0) > 0
    ]
    print(f"\n{label}Iteration breakdown:")
    print(f"  encoder:     {len(encoder_iters)}")
    print(f"  prefill-only: {len(prefill_only)}")
    print(f"  decode-only:  {len(decode_only)}")
    print(f"  mixed (chunked prefill): {len(mixed)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Merge iteration logs + nsys + ncu into consolidated files"
    )
    parser.add_argument(
        "profile_dir",
        type=str,
        help="Path to the profile output directory",
    )
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    if not profile_dir.exists():
        print(f"Error: directory not found: {profile_dir}", file=sys.stderr)
        sys.exit(1)

    # Detect disaggregated mode: prefill/ and decode/ subdirectories.
    prefill_dir = profile_dir / "prefill"
    decode_dir = profile_dir / "decode"
    is_disagg = prefill_dir.exists() and decode_dir.exists()

    if is_disagg:
        print("Detected disaggregated mode (prefill/ + decode/ subdirs).")
        print("\n--- Prefill GPU ---")
        p_iters, p_reqs = process_single_dir(
            prefill_dir, label="[prefill] "
        )
        print("\n--- Decode GPU ---")
        d_iters, d_reqs = process_single_dir(
            decode_dir, label="[decode] "
        )

        # Merge: interleave by wall-clock timestamp, tag with gpu_role.
        all_iters = p_iters + d_iters
        all_iters.sort(key=lambda r: r.get("ts_wall", 0))

        # Merge requests: combine prefill-side and decode-side info per
        # external request ID.
        req_by_ext: dict[str, dict] = {}
        for req in p_reqs + d_reqs:
            ext_id = req.get("external_id", "")
            if ext_id not in req_by_ext:
                req_by_ext[ext_id] = dict(req)
            else:
                existing = req_by_ext[ext_id]
                # Merge iteration lists and GPU metrics from both sides.
                for key in (
                    "encoder_iters",
                    "prefill_iters",
                    "decode_iters",
                ):
                    existing.setdefault(key, []).extend(
                        req.get(key, [])
                    )
                # Prefer non-None values for GPU metrics.
                for key, val in req.items():
                    if key not in existing or existing[key] is None:
                        existing[key] = val

        all_reqs = list(req_by_ext.values())
        all_reqs.sort(key=lambda r: r.get("first_iter") or 0)

        consolidated_iters = all_iters
        consolidated_reqs = all_reqs

        print_summary(p_iters, label="[Prefill GPU] ")
        print_summary(d_iters, label="[Decode GPU] ")
    else:
        # Single-GPU mode: iterations.jsonl at top level.
        iter_path = profile_dir / "iterations.jsonl"
        if not iter_path.exists():
            print(
                f"Error: {iter_path} not found", file=sys.stderr
            )
            sys.exit(1)

        consolidated_iters, consolidated_reqs = process_single_dir(
            profile_dir
        )
        print_summary(consolidated_iters)

    # --- Write outputs ---
    out_iters = profile_dir / "consolidated_iterations.jsonl"
    with open(out_iters, "w") as f:
        for record in consolidated_iters:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_reqs = profile_dir / "consolidated_requests.jsonl"
    with open(out_reqs, "w") as f:
        for record in consolidated_reqs:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nConsolidated output written to:")
    print(f"  {out_iters}")
    print(f"  {out_reqs}")


if __name__ == "__main__":
    main()
