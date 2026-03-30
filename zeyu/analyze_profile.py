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
import re
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
ANNOTATE_PATTERN = re.compile(
    r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)"
)


def parse_nsys_nvtx(path: Path) -> list[dict]:
    """Parse nsys NVTX GPU projection trace CSV.

    Returns a list of dicts with start_ns, duration_ns, name for each
    NVTX range that matches the annotate_profile pattern.
    """
    if not path.exists():
        return []

    ranges = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", row.get("Range", ""))
            if not ANNOTATE_PATTERN.search(name):
                continue
            try:
                start_ns = int(row.get("Start (ns)", 0))
                duration_ns = int(row.get("Duration (ns)", 0))
            except (ValueError, TypeError):
                continue
            ranges.append(
                {
                    "start_ns": start_ns,
                    "end_ns": start_ns + duration_ns,
                    "duration_ns": duration_ns,
                    "name": name,
                }
            )

    # Sort by start time.
    ranges.sort(key=lambda r: r["start_ns"])
    return ranges


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
                duration_ns = int(row.get("Duration (ns)", 0))
            except (ValueError, TypeError):
                continue
            ranges.append(
                {
                    "start_ns": start_ns,
                    "end_ns": start_ns + duration_ns,
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
    kernels: list[dict], wall_time_ns: int
) -> dict:
    """Compute GPU utilization from a set of kernels within a time window.

    Returns dict with gpu_util_pct, total_kernel_time_ns, num_kernels.
    Handles overlapping kernels by computing non-overlapping coverage.
    """
    if wall_time_ns <= 0 or not kernels:
        return {
            "gpu_util_pct": 0.0,
            "total_kernel_time_ns": 0,
            "num_kernels": 0,
        }

    # Sort intervals by start, merge overlaps to get true busy time.
    intervals = sorted(
        [(k["start_ns"], k["end_ns"]) for k in kernels],
        key=lambda x: x[0],
    )
    merged_time = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged_time += cur_end - cur_start
            cur_start, cur_end = start, end
    merged_time += cur_end - cur_start

    return {
        "gpu_util_pct": round(merged_time / wall_time_ns * 100, 2),
        "total_kernel_time_ns": merged_time,
        "num_kernels": len(kernels),
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

    # --- Load data ---
    iter_path = profile_dir / "iterations.jsonl"
    req_path = profile_dir / "requests.jsonl"

    if not iter_path.exists():
        print(f"Error: {iter_path} not found", file=sys.stderr)
        sys.exit(1)

    iterations = load_iterations(iter_path)
    requests = load_requests(req_path) if req_path.exists() else []
    print(f"Loaded {len(iterations)} iterations, {len(requests)} requests.")

    # --- Parse nsys data ---
    # Try common nsys output naming conventions.
    nsys_kernel_csv = None
    nsys_nvtx_csv = None
    for candidate in [
        "nsys_kernels_cuda_gpu_trace.csv",
        "nsys_kernels.csv",
    ]:
        p = profile_dir / candidate
        if p.exists():
            nsys_kernel_csv = p
            break

    for candidate in [
        "nsys_nvtx_nvtx_gpu_proj_trace.csv",
        "nsys_nvtx.csv",
    ]:
        p = profile_dir / candidate
        if p.exists():
            nsys_nvtx_csv = p
            break

    nvtx_ranges = parse_nsys_nvtx(nsys_nvtx_csv) if nsys_nvtx_csv else []
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
    kernels = parse_nsys_kernels(nsys_kernel_csv) if nsys_kernel_csv else []
    print(
        f"Parsed nsys: {len(nvtx_ranges)} iteration NVTX ranges, "
        f"{len(ve_ranges)} vision_encoder ranges, "
        f"{len(fwd_ranges)} text_forward ranges, {len(kernels)} kernels."
    )

    # --- Parse ncu data ---
    ncu_csv = profile_dir / "ncu_metrics.csv"
    ncu_by_name = parse_ncu_csv(ncu_csv)
    if ncu_by_name:
        print(f"Parsed ncu: {len(ncu_by_name)} unique kernel names.")

    # --- Correlate kernels to iteration NVTX ranges ---
    kernels_per_iter = correlate_kernels_to_ranges(kernels, nvtx_ranges)

    # --- Build consolidated iterations ---
    consolidated_iters = []
    for i, it in enumerate(iterations):
        record = dict(it)  # copy all iteration fields

        # Match with NVTX range (by sequential order).
        if i < len(nvtx_ranges):
            rng = nvtx_ranges[i]
            iter_kernels = kernels_per_iter[i]
            gpu_metrics = compute_gpu_util(iter_kernels, rng["duration_ns"])
            record.update(gpu_metrics)

            # Check if this iteration overlaps a vision_encoder NVTX range.
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
                        ve_kernels, ve_rng["duration_ns"]
                    )
                    ve_kernel_time_ns = ve_gpu["total_kernel_time_ns"]
                    record["vision_encoder_gpu_util_pct"] = ve_gpu[
                        "gpu_util_pct"
                    ]
            record["vision_encoder_kernel_time_ns"] = ve_kernel_time_ns

            # Check if this iteration overlaps a text forward NVTX range.
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
                        fwd_kernels, fwd_rng["duration_ns"]
                    )
                    record["text_forward_gpu_util_pct"] = fwd_gpu[
                        "gpu_util_pct"
                    ]
                    record["text_forward_kernel_time_ns"] = fwd_gpu[
                        "total_kernel_time_ns"
                    ]
                    break  # one forward per iteration

            # Enrich with ncu SM metrics.
            if ncu_by_name and iter_kernels:
                sm_metrics = enrich_with_ncu(iter_kernels, ncu_by_name)
                record.update(sm_metrics)

        # Build per-request phase list for this iteration.
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
    # Map iteration GPU metrics back to requests by phase.
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

        # Average GPU util per phase.
        for phase in ("encoder", "prefill", "decode"):
            iters_key = f"{phase}_iters"
            phase_iters = req.get(iters_key, [])
            utils = [
                iter_gpu_util_map[i]
                for i in phase_iters
                if i in iter_gpu_util_map
                and iter_gpu_util_map[i] is not None
            ]
            record[f"{phase}_avg_gpu_util_pct"] = (
                round(sum(utils) / len(utils), 2) if utils else None
            )

            # SM metrics.
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

    # Sort requests by first_iter.
    consolidated_reqs.sort(
        key=lambda r: r.get("first_iter") or 0
    )

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

    # --- Print summary ---
    if consolidated_iters:
        gpu_utils = [
            ci["gpu_util_pct"]
            for ci in consolidated_iters
            if "gpu_util_pct" in ci
        ]
        if gpu_utils:
            print(f"\nGPU utilization across {len(gpu_utils)} iterations:")
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

        print(f"\nIteration breakdown:")
        print(f"  encoder:     {len(encoder_iters)}")
        print(f"  prefill-only: {len(prefill_only)}")
        print(f"  decode-only:  {len(decode_only)}")
        print(f"  mixed (chunked prefill): {len(mixed)}")


if __name__ == "__main__":
    main()
