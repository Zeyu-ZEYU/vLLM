#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Merge prefill-side and decode-side metrics from a cross-node disagg run
into a single ``disagg_summary.json``.

Input layout (produced by ``disagg_run.sh``):

    <out_dir>/
      prefill/
        latency.json                      # from prefill role
        iterations.jsonl                  # per-iteration log
        requests.jsonl
        consolidated_iterations.jsonl     # (when analyze_profile.py was run)
        consolidated_requests.jsonl
      decode/
        latency.json                      # from decode role
        iterations.jsonl
        requests.jsonl
        consolidated_iterations.jsonl     # (when analyze_profile.py was run)
        consolidated_requests.jsonl

Output: <out_dir>/disagg_summary.json

Produces per-request metrics aggregated across vision encoder /
prefill / decode phases, and a global summary.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _external_id(internal_id: str) -> str:
    """Strip random suffix (if any) and any decode/prefill address markers.

    Our disagg request IDs have the form:
        req<N>___prefill_addr_<ip>:<port>___decode_addr_<ip>:<port>_<uid>
    We return the leading ``req<N>`` (or external ID) prefix so that
    iteration-side req_ids can match latency.json's request_index.
    """
    prefix = internal_id.split("___", 1)[0]
    return prefix  # e.g. "req0"


def _req_index_from_id(internal_id: str) -> int | None:
    """Extract the integer index N from a request ID of form 'reqN___...'."""
    prefix = _external_id(internal_id)
    if prefix.startswith("req") and prefix[3:].isdigit():
        return int(prefix[3:])
    return None


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------
def _iter_contains_req(iter_record: dict, internal_id: str, phase: str) -> bool:
    """Return True if ``internal_id`` is in the given phase of this iteration."""
    if phase == "encoder":
        return internal_id in iter_record.get("encoder_req_ids", [])
    if phase == "prefill":
        return internal_id in iter_record.get("prefill_req_ids", [])
    if phase == "decode":
        return internal_id in iter_record.get("decode_req_ids", [])
    return False


def _aggregate_iter_metrics(iterations: list[dict]) -> dict:
    """Average GPU-util, memory, kernel-gap across a list of iterations.

    Only fields present in at least one iteration are emitted.
    """
    def _mean_of(key: str) -> float | None:
        vals = [r.get(key) for r in iterations if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None

    def _sum_of(key: str) -> float | None:
        vals = [r.get(key) for r in iterations if isinstance(r.get(key), (int, float))]
        return round(sum(vals), 3) if vals else None

    if not iterations:
        return {"num_iterations": 0}

    out: dict = {"num_iterations": len(iterations)}
    for k in (
        "gpu_util_pct",
        "kernel_launch_gap_pct",
        "vision_encoder_gpu_util_pct",
        "vision_encoder_kernel_launch_gap_pct",
        "text_forward_gpu_util_pct",
        "text_forward_kernel_launch_gap_pct",
        "gpu_mem_allocated_MiB",
        "gpu_mem_peak_MiB",
        "sm_active_pct_mean",
        "sm_occupancy_pct_mean",
        "num_active_sms_mean",
    ):
        v = _mean_of(k)
        if v is not None:
            out[f"avg_{k}"] = v
    for k in (
        "kernel_launch_gap_ns",
        "total_kernel_time_ns",
        "vision_encoder_kernel_launch_gap_ns",
        "vision_encoder_kernel_time_ns",
        "text_forward_kernel_launch_gap_ns",
        "text_forward_kernel_time_ns",
    ):
        v = _sum_of(k)
        if v is not None:
            out[f"sum_{k}"] = v
    return out


def _compute_per_token_tbt_ms(
    decode_iters_for_req: list[dict],
) -> list[float]:
    """Per-token TBT (in ms) for a single request.

    Each decode iteration that lists the request in ``decode_req_ids``
    produced exactly one output token for it. TBT[i] = ts_mono[i] -
    ts_mono[i-1], in milliseconds. The first iteration has no predecessor
    so yields no TBT value.
    """
    if len(decode_iters_for_req) < 2:
        return []
    sorted_iters = sorted(
        decode_iters_for_req, key=lambda r: r.get("ts_mono") or r.get("iter", 0)
    )
    tbts: list[float] = []
    prev = sorted_iters[0]
    for cur in sorted_iters[1:]:
        dt = (cur.get("ts_mono", 0) - prev.get("ts_mono", 0)) * 1000.0
        if dt > 0:
            tbts.append(round(dt, 3))
        prev = cur
    return tbts


def _pct(sorted_vals: list[float], q: float) -> float:
    """Simple percentile (q in [0, 100]) on a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(q / 100.0 * (len(sorted_vals) - 1)))))
    return round(sorted_vals[k], 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: merge_disagg.py <disagg_output_dir>", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    if not out_dir.exists():
        print(f"Error: {out_dir} not found", file=sys.stderr)
        sys.exit(1)

    # --- Load per-role latency + iterations (prefer consolidated) ---
    prefill_latency = load_json(out_dir / "prefill" / "latency.json")
    decode_latency = load_json(out_dir / "decode" / "latency.json")

    # Prefer consolidated iterations (from analyze_profile.py) when available,
    # because they carry GPU util / SM metrics.
    prefill_iters = load_jsonl(out_dir / "prefill" / "consolidated_iterations.jsonl")
    if not prefill_iters:
        prefill_iters = load_jsonl(out_dir / "prefill" / "iterations.jsonl")
    decode_iters = load_jsonl(out_dir / "decode" / "consolidated_iterations.jsonl")
    if not decode_iters:
        decode_iters = load_jsonl(out_dir / "decode" / "iterations.jsonl")

    prefill_reqs = prefill_latency.get("requests", [])
    decode_reqs = decode_latency.get("requests", [])

    # Index latency records by request_index.
    prefill_by_idx = {
        r["request_index"]: r for r in prefill_reqs if "request_index" in r
    }
    decode_by_idx = {
        r["request_index"]: r for r in decode_reqs if "request_index" in r
    }

    all_idx = sorted(set(prefill_by_idx.keys()) | set(decode_by_idx.keys()))

    # Index iterations by request index → list of iterations (per phase).
    # For each iteration we check every req_id list and bucket it.
    idx_to_encoder_iters: dict[int, list[dict]] = {}
    idx_to_prefill_iters: dict[int, list[dict]] = {}
    idx_to_decode_iters: dict[int, list[dict]] = {}

    def _bucket(iters: list[dict], target_bucket: dict, phase: str, key_list: str):
        for it in iters:
            for rid in it.get(key_list, []) or []:
                ri = _req_index_from_id(rid)
                if ri is not None:
                    target_bucket.setdefault(ri, []).append(it)

    _bucket(prefill_iters, idx_to_encoder_iters, "encoder", "encoder_req_ids")
    _bucket(prefill_iters, idx_to_prefill_iters, "prefill", "prefill_req_ids")
    _bucket(decode_iters, idx_to_decode_iters, "decode", "decode_req_ids")

    # --- Build per-request merged records ---
    merged_requests = []
    all_tbts_ms: list[float] = []
    for idx in all_idx:
        p = prefill_by_idx.get(idx, {})
        d = decode_by_idx.get(idx, {})

        ve_iters = idx_to_encoder_iters.get(idx, [])
        pf_iters = idx_to_prefill_iters.get(idx, [])
        # Exclude prefill iterations that are ALSO encoder iterations so we don't
        # double-count the same iteration under both VE and prefill.
        # (In practice an encoder iteration is also a prefill iteration, so we
        # report VE as a subset and keep "pure" prefill separately.)
        pf_only_iters = [it for it in pf_iters if it not in ve_iters]
        dc_iters = idx_to_decode_iters.get(idx, [])

        per_token_tbt_ms = _compute_per_token_tbt_ms(dc_iters)
        all_tbts_ms.extend(per_token_tbt_ms)
        sorted_tbt = sorted(per_token_tbt_ms) if per_token_tbt_ms else []
        tbt_summary = {
            "count": len(per_token_tbt_ms),
            "mean_ms": round(
                sum(per_token_tbt_ms) / len(per_token_tbt_ms), 3
            ) if per_token_tbt_ms else 0.0,
            "min_ms": sorted_tbt[0] if sorted_tbt else 0.0,
            "p50_ms": _pct(sorted_tbt, 50),
            "p95_ms": _pct(sorted_tbt, 95),
            "p99_ms": _pct(sorted_tbt, 99),
            "max_ms": sorted_tbt[-1] if sorted_tbt else 0.0,
        }

        rec = {
            "request_index": idx,
            "image_source": p.get("image_source") or d.get("image_source") or "",
            "question": p.get("question") or d.get("question") or "",
            "generated_text": d.get("generated_text", ""),

            # --- Phase durations (latency) ---
            "vision_encoder_time_ms": p.get("vision_encoder_time_ms", 0.0),
            "prefill_time_ms": p.get("prefill_time_ms", 0.0),
            "kv_transfer_time_ms": d.get("kv_transfer_time_ms", None),
            "decode_time_ms": d.get("decode_time_ms", 0.0),

            # --- Token counts ---
            "num_prompt_tokens": p.get("num_prompt_tokens", 0),
            "num_generation_tokens": d.get("num_generation_tokens", 0),
            "tpot_ms": d.get("tpot_ms", 0.0),

            # --- Per-token TBT (ms) ---
            "per_token_tbt_ms": per_token_tbt_ms,
            "tbt_stats": tbt_summary,

            # --- Per-phase aggregated iteration metrics ---
            "vision_encoder": _aggregate_iter_metrics(ve_iters),
            "prefill": _aggregate_iter_metrics(pf_only_iters),
            "decode": _aggregate_iter_metrics(dc_iters),

            # --- Raw timestamps (may have different clock bases) ---
            "prefill_arrival_ts": p.get("arrival_ts", 0.0),
            "prefill_scheduled_ts": p.get("scheduled_ts", 0.0),
            "prefill_first_token_ts": p.get("first_token_ts", 0.0),
            "decode_scheduled_ts": d.get("scheduled_ts", 0.0),
            "decode_first_token_ts": d.get("first_token_ts", 0.0),
            "decode_last_token_ts": d.get("last_token_ts", 0.0),
        }

        # JCT approximation: VE + prefill + KV transfer + decode.
        jct_ms = (
            rec["vision_encoder_time_ms"]
            + rec["prefill_time_ms"]
            + rec["decode_time_ms"]
        )
        if rec["kv_transfer_time_ms"] is not None and rec["kv_transfer_time_ms"] > 0:
            jct_ms += rec["kv_transfer_time_ms"]
        rec["jct_ms"] = round(jct_ms, 3)

        merged_requests.append(rec)

    # --- Aggregate summary ---
    n = len(merged_requests)
    summary: dict = {"num_requests": n}

    def _mean(key: str) -> float:
        vals = [r.get(key) for r in merged_requests if r.get(key) is not None]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def _mean_nonzero(key: str) -> float:
        vals = [
            r[key] for r in merged_requests
            if isinstance(r.get(key), (int, float)) and r[key] > 0
        ]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    if n > 0:
        summary.update({
            "avg_vision_encoder_time_ms": _mean_nonzero("vision_encoder_time_ms"),
            "avg_prefill_time_ms": _mean("prefill_time_ms"),
            "avg_decode_time_ms": _mean("decode_time_ms"),
            "avg_tpot_ms": _mean("tpot_ms"),
            "avg_kv_transfer_time_ms": _mean("kv_transfer_time_ms"),
            "avg_jct_ms": _mean("jct_ms"),
            "total_decode_tokens": sum(
                r.get("num_generation_tokens", 0) for r in merged_requests
            ),
        })

        if all_tbts_ms:
            sorted_all = sorted(all_tbts_ms)
            summary["tbt_stats"] = {
                "count": len(all_tbts_ms),
                "mean_ms": round(sum(all_tbts_ms) / len(all_tbts_ms), 3),
                "min_ms": round(sorted_all[0], 3),
                "p50_ms": _pct(sorted_all, 50),
                "p95_ms": _pct(sorted_all, 95),
                "p99_ms": _pct(sorted_all, 99),
                "max_ms": round(sorted_all[-1], 3),
            }

        # Aggregate phase-level iteration metrics across all requests.
        for phase in ("vision_encoder", "prefill", "decode"):
            def _phase_mean(field: str) -> float | None:
                vals = [
                    r[phase].get(field)
                    for r in merged_requests
                    if isinstance(r.get(phase), dict)
                    and isinstance(r[phase].get(field), (int, float))
                ]
                return round(sum(vals) / len(vals), 3) if vals else None

            phase_summary: dict = {}
            for fk in (
                "avg_gpu_util_pct",
                "avg_kernel_launch_gap_pct",
                "avg_gpu_mem_allocated_MiB",
                "avg_gpu_mem_peak_MiB",
                "avg_sm_active_pct_mean",
                "avg_sm_occupancy_pct_mean",
                "avg_num_active_sms_mean",
                "avg_vision_encoder_gpu_util_pct",
                "avg_vision_encoder_kernel_launch_gap_pct",
                "avg_text_forward_gpu_util_pct",
                "avg_text_forward_kernel_launch_gap_pct",
                "sum_kernel_launch_gap_ns",
                "sum_total_kernel_time_ns",
                "sum_vision_encoder_kernel_launch_gap_ns",
                "sum_vision_encoder_kernel_time_ns",
                "sum_text_forward_kernel_launch_gap_ns",
                "sum_text_forward_kernel_time_ns",
            ):
                v = _phase_mean(fk)
                if v is not None:
                    phase_summary[fk] = v
            if phase_summary:
                summary[phase] = phase_summary

        # RPS based on total wall times from each side.
        p_wall = prefill_latency.get("wall_time_s", 0.0)
        d_wall = decode_latency.get("wall_time_s", 0.0)
        end_to_end_wall = max(p_wall, 0.0) + max(d_wall, 0.0)
        if end_to_end_wall > 0:
            summary["rps_end_to_end"] = round(n / end_to_end_wall, 3)
        if d_wall > 0:
            summary["rps_decode_only"] = round(n / d_wall, 3)
        summary["prefill_wall_time_s"] = round(p_wall, 3)
        summary["decode_wall_time_s"] = round(d_wall, 3)

    out = {
        "mode": "disagg",
        "model": prefill_latency.get("model") or decode_latency.get("model") or "",
        "summary": summary,
        "requests": merged_requests,
    }

    summary_file = out_dir / "disagg_summary.json"
    with open(summary_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {summary_file}")

    # --- Console summary table ---
    print("\n" + "=" * 100)
    print(f"{'DISAGG SUMMARY':^100}")
    print(f"Model: {out['model']}")
    print("=" * 100)
    hdr = (
        f"{'#':>3} | {'Image Source':<24} | "
        f"{'VE(ms)':>8} | {'Pref(ms)':>9} | {'KV(ms)':>7} | "
        f"{'Dec(ms)':>8} | {'GenTok':>6} | {'TPOT':>6} | {'p50TBT':>7} | {'JCT(ms)':>9}"
    )
    print(hdr)
    print("-" * 100)
    for r in merged_requests:
        kv = r.get("kv_transfer_time_ms")
        kv_str = f"{kv:>7.2f}" if isinstance(kv, (int, float)) else f"{'N/A':>7}"
        p50 = r.get("tbt_stats", {}).get("p50_ms", 0)
        print(
            f"{r['request_index']:>3} | "
            f"{r['image_source'][:24]:<24} | "
            f"{r.get('vision_encoder_time_ms', 0):>8.2f} | "
            f"{r.get('prefill_time_ms', 0):>9.2f} | "
            f"{kv_str} | "
            f"{r.get('decode_time_ms', 0):>8.2f} | "
            f"{r.get('num_generation_tokens', 0):>6} | "
            f"{r.get('tpot_ms', 0):>6.2f} | "
            f"{p50:>7.2f} | "
            f"{r.get('jct_ms', 0):>9.2f}"
        )
    print("-" * 100)
    if n > 0:
        avg_p50 = summary.get("tbt_stats", {}).get("p50_ms", 0)
        print(
            f"{'AVG':>3} | {'':<24} | "
            f"{summary['avg_vision_encoder_time_ms']:>8.2f} | "
            f"{summary['avg_prefill_time_ms']:>9.2f} | "
            f"{summary['avg_kv_transfer_time_ms']:>7.2f} | "
            f"{summary['avg_decode_time_ms']:>8.2f} | "
            f"{summary['total_decode_tokens'] // n:>6} | "
            f"{summary['avg_tpot_ms']:>6.2f} | "
            f"{avg_p50:>7.2f} | "
            f"{summary['avg_jct_ms']:>9.2f}"
        )
        if "rps_end_to_end" in summary:
            print(
                f"  RPS (end-to-end) = {summary['rps_end_to_end']:.3f}  |  "
                f"RPS (decode-only) = {summary.get('rps_decode_only', 0):.3f}  |  "
                f"prefill_wall = {summary['prefill_wall_time_s']:.3f}s  |  "
                f"decode_wall = {summary['decode_wall_time_s']:.3f}s"
            )
        for phase in ("vision_encoder", "prefill", "decode"):
            if phase in summary:
                ps = summary[phase]
                line = f"  [{phase:14}] "
                if "avg_avg_gpu_util_pct" in ps:
                    line += f"gpu_util={ps['avg_avg_gpu_util_pct']:.1f}%  "
                if "avg_avg_kernel_launch_gap_pct" in ps:
                    line += f"kernel_gap={ps['avg_avg_kernel_launch_gap_pct']:.1f}%  "
                if "avg_avg_gpu_mem_allocated_MiB" in ps:
                    line += f"mem={ps['avg_avg_gpu_mem_allocated_MiB']:.0f}MiB  "
                if "avg_avg_num_active_sms_mean" in ps:
                    line += f"active_SMs={ps['avg_avg_num_active_sms_mean']:.1f}  "
                if "avg_avg_sm_active_pct_mean" in ps:
                    line += f"sm_active={ps['avg_avg_sm_active_pct_mean']:.1f}%  "
                print(line)
    print("=" * 100)


if __name__ == "__main__":
    main()
