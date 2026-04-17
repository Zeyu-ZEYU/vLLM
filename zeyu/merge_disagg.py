#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Merge prefill-side and decode-side metrics from a cross-node disagg run
into a single ``disagg_summary.json``.

Input layout (produced by ``disagg_run.sh``):

    <out_dir>/
      prefill/
        latency.json          # from prefill role
        iterations.jsonl      # per-iteration log (optional)
        requests.jsonl
      decode/
        latency.json          # from decode role
        iterations.jsonl
        requests.jsonl

Output:

    <out_dir>/disagg_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


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


def main():
    if len(sys.argv) != 2:
        print("Usage: merge_disagg.py <disagg_output_dir>", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    if not out_dir.exists():
        print(f"Error: {out_dir} not found", file=sys.stderr)
        sys.exit(1)

    prefill_latency = load_json(out_dir / "prefill" / "latency.json")
    decode_latency = load_json(out_dir / "decode" / "latency.json")

    prefill_reqs = prefill_latency.get("requests", [])
    decode_reqs = decode_latency.get("requests", [])

    # Index by request_index (set by run_prefill_role / run_decode_role).
    prefill_by_idx = {r["request_index"]: r for r in prefill_reqs if "request_index" in r}
    decode_by_idx = {r["request_index"]: r for r in decode_reqs if "request_index" in r}

    all_idx = sorted(set(prefill_by_idx.keys()) | set(decode_by_idx.keys()))

    merged_requests = []
    for idx in all_idx:
        p = prefill_by_idx.get(idx, {})
        d = decode_by_idx.get(idx, {})

        rec = {
            "request_index": idx,
            "image_source": p.get("image_source") or d.get("image_source") or "",
            "question": p.get("question") or d.get("question") or "",
            "generated_text": d.get("generated_text", ""),
            # --- Prefill-side metrics (Node 0) ---
            "vision_encoder_time_ms": p.get("vision_encoder_time_ms", 0.0),
            "prefill_time_ms": p.get("prefill_time_ms", 0.0),
            "num_prompt_tokens": p.get("num_prompt_tokens", 0),
            # --- Decode-side metrics (Node 1) ---
            "decode_time_ms": d.get("decode_time_ms", 0.0),
            "num_generation_tokens": d.get("num_generation_tokens", 0),
            "tpot_ms": d.get("tpot_ms", 0.0),
            # --- KV transfer (decode side estimate) ---
            "kv_transfer_time_ms": d.get("kv_transfer_time_ms", None),
            # --- Raw timestamps (may have different clock bases) ---
            "prefill_arrival_ts": p.get("arrival_ts", 0.0),
            "prefill_scheduled_ts": p.get("scheduled_ts", 0.0),
            "prefill_first_token_ts": p.get("first_token_ts", 0.0),
            "decode_scheduled_ts": d.get("scheduled_ts", 0.0),
            "decode_first_token_ts": d.get("first_token_ts", 0.0),
            "decode_last_token_ts": d.get("last_token_ts", 0.0),
        }

        # JCT approximation: sum of local durations + KV transfer estimate.
        jct_parts = [
            rec["vision_encoder_time_ms"],
            rec["prefill_time_ms"],
            rec["decode_time_ms"],
        ]
        jct_ms = sum(jct_parts)
        if rec["kv_transfer_time_ms"] is not None and rec["kv_transfer_time_ms"] > 0:
            jct_ms += rec["kv_transfer_time_ms"]
        rec["jct_ms"] = round(jct_ms, 3)

        merged_requests.append(rec)

    # --- Aggregate summary ---
    n = len(merged_requests)
    summary = {"num_requests": n}

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
        f"{'#':>3} | {'Image Source':<26} | "
        f"{'VE(ms)':>8} | {'Pref(ms)':>9} | {'KV(ms)':>8} | "
        f"{'Dec(ms)':>9} | {'GenTok':>6} | {'TPOT':>6} | {'JCT(ms)':>9}"
    )
    print(hdr)
    print("-" * 100)
    for r in merged_requests:
        kv = r.get("kv_transfer_time_ms")
        kv_str = f"{kv:>8.2f}" if isinstance(kv, (int, float)) else f"{'N/A':>8}"
        print(
            f"{r['request_index']:>3} | "
            f"{r['image_source'][:26]:<26} | "
            f"{r.get('vision_encoder_time_ms', 0):>8.2f} | "
            f"{r.get('prefill_time_ms', 0):>9.2f} | "
            f"{kv_str} | "
            f"{r.get('decode_time_ms', 0):>9.2f} | "
            f"{r.get('num_generation_tokens', 0):>6} | "
            f"{r.get('tpot_ms', 0):>6.2f} | "
            f"{r.get('jct_ms', 0):>9.2f}"
        )
    print("-" * 100)
    if n > 0:
        print(
            f"{'AVG':>3} | {'':<26} | "
            f"{summary['avg_vision_encoder_time_ms']:>8.2f} | "
            f"{summary['avg_prefill_time_ms']:>9.2f} | "
            f"{summary['avg_kv_transfer_time_ms']:>8.2f} | "
            f"{summary['avg_decode_time_ms']:>9.2f} | "
            f"{summary['total_decode_tokens'] // n:>6} | "
            f"{summary['avg_tpot_ms']:>6.2f} | "
            f"{summary['avg_jct_ms']:>9.2f}"
        )
        if "rps_end_to_end" in summary:
            print(
                f"  RPS (end-to-end) = {summary['rps_end_to_end']:.3f}  |  "
                f"RPS (decode-only) = {summary.get('rps_decode_only', 0):.3f}  |  "
                f"prefill_wall = {summary['prefill_wall_time_s']:.3f}s  |  "
                f"decode_wall = {summary['decode_wall_time_s']:.3f}s"
            )
    print("=" * 100)


if __name__ == "__main__":
    main()
