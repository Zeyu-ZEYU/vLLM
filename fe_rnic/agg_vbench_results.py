#!/usr/bin/env python3
"""
Aggregate one matrix's worth of results: vbench JSONs (client side) +
LMCache adapter JSONLs (server side).

For each (mode, conc, length) point this produces:
  - Client metrics (TTFT, TPOT, ITL, E2EL): mean / p25 / p50 / p75 / p99,
    pulled verbatim from the vbench JSON.
  - RPS / throughput / completed / failed (from vbench JSON).
  - Server-side d_* (computed per request by joining producer+consumer
    JSONLs on req_id, then aggregated as mean / p25 / p50 / p75 / p99):
      * d_prefill_start_load_kv (CPU bound of start_load_kv hook)
      * d_prefill (cudaEvent.elapsed_time of forward, GPU-anchored)
      * d_kv_mnck_in  (t_kv_mnck_in_end - t_kv_start, prefill-side)
      * d_kv_mnck     (t_kv_mnck_out_start - t_kv_mnck_in_end, cross)
      * d_kv_mnck_out (t_kv_end - t_kv_mnck_out_start, decode-side)
      * d_total_kv    (in + dwell + out)
      * d_total_kv_no_mnck (in + out, excl. dwell)

The producer JSONL (prefill side) carries `t_prefill_start_load_kv_*`,
`t_prefill_wait_for_save_start`, `t_kv_start`, `t_kv_mnck_in_end`,
`t_prefill_start`, `t_prefill_end`, `d_prefill_ms`. The consumer JSONL
(decode side) carries `t_kv_mnck_out_start`, `t_kv_end`. Join key is
`req_id` — the LMCache adapter normalizes prefill's vLLM internal id
to the two-part `cmpl-<hex>` form, and the proxy injects the same id
as the decode-side `correlation_id` so both sides write the same key.

Usage:
    python3 agg_vbench_results.py \
        --bench-dir /home/zeyu/exp_results/fe_rnic/bench_vllm \
        --output    /home/zeyu/exp_results/fe_rnic/bench_vllm/summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np

LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
CONCURRENCIES = [50, 100, 150, 200]
MODES = ["tail", "head"]

CLIENT_METRICS = ["ttft", "tpot", "itl", "e2el"]
SERVER_METRICS = [
    "d_prefill_start_load_kv_ms",
    "d_prefill_ms",
    "d_kv_mnck_in_ms",
    "d_kv_mnck_ms",
    "d_kv_mnck_out_ms",
    "d_total_kv_ms",
    "d_total_kv_no_mnck_ms",
]


def _read_jsonl(path: str) -> list[dict]:
    """Read a possibly-empty JSONL file. Skip malformed lines."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {
            "mean": 0.0, "p25": 0.0, "p50": 0.0,
            "p75": 0.0, "p99": 0.0, "n": 0,
        }
    arr = np.asarray(xs, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p25":  float(np.percentile(arr, 25)),
        "p50":  float(np.percentile(arr, 50)),
        "p75":  float(np.percentile(arr, 75)),
        "p99":  float(np.percentile(arr, 99)),
        "n":    int(arr.size),
    }


def _compute_per_request_d_star(
    prefill_records: dict[str, dict],
    decode_records: dict[str, dict],
) -> list[dict]:
    """For each req_id present in BOTH a prefill and a decode JSONL,
    compute the seven server-side d_* metrics in milliseconds."""
    out: list[dict] = []
    for rid, p in prefill_records.items():
        d = decode_records.get(rid)
        if d is None:
            continue
        # Prefill-side
        t_sl_s   = p.get("t_prefill_start_load_kv_start", 0.0) or 0.0
        t_sl_e   = p.get("t_prefill_start_load_kv_end",   0.0) or 0.0
        t_kv_s   = p.get("t_kv_start",        0.0) or 0.0
        t_kv_in  = p.get("t_kv_mnck_in_end",  0.0) or 0.0
        t_pf_s   = p.get("t_prefill_start",   0.0) or 0.0
        t_pf_e   = p.get("t_prefill_end",     0.0) or 0.0
        d_pf     = p.get("d_prefill_ms")  # may be None on older entries
        # Decode-side
        t_kv_out_s = d.get("t_kv_mnck_out_start", 0.0) or 0.0
        t_kv_end   = d.get("t_kv_end",            0.0) or 0.0

        # Skip if any timestamp we need is zero (joined record incomplete).
        if not (t_sl_s and t_sl_e and t_kv_s and t_kv_in
                and t_kv_out_s and t_kv_end):
            continue

        d_prefill_start_load_kv = (t_sl_e - t_sl_s) * 1000
        d_prefill_event = d_pf if d_pf is not None and d_pf > 0 else \
            ((t_pf_e - t_pf_s) * 1000 if (t_pf_s and t_pf_e) else 0.0)
        d_kv_mnck_in    = (t_kv_in - t_kv_s) * 1000
        d_kv_mnck       = (t_kv_out_s - t_kv_in) * 1000
        d_kv_mnck_out   = (t_kv_end - t_kv_out_s) * 1000
        d_total_kv      = d_kv_mnck_in + d_kv_mnck + d_kv_mnck_out
        d_total_kv_no_mnck = d_kv_mnck_in + d_kv_mnck_out

        out.append({
            "req_id": rid,
            "d_prefill_start_load_kv_ms": d_prefill_start_load_kv,
            "d_prefill_ms":               d_prefill_event,
            "d_kv_mnck_in_ms":            d_kv_mnck_in,
            "d_kv_mnck_ms":               d_kv_mnck,
            "d_kv_mnck_out_ms":           d_kv_mnck_out,
            "d_total_kv_ms":              d_total_kv,
            "d_total_kv_no_mnck_ms":      d_total_kv_no_mnck,
        })
    return out


def _client_stats_from_vbench(vb: dict) -> dict:
    """Pull TTFT/TPOT/ITL/E2EL means + percentiles from a vllm bench
    serve result JSON. vllm bench writes p25/median/p75/p99 explicitly
    (and `median_*_ms` is just p50)."""
    out = {}
    for m in CLIENT_METRICS:
        out[m] = {
            "mean": vb.get(f"mean_{m}_ms",   0.0),
            "p25":  vb.get(f"p25_{m}_ms",    0.0),
            "p50":  vb.get(f"median_{m}_ms",
                           vb.get(f"p50_{m}_ms", 0.0)),
            "p75":  vb.get(f"p75_{m}_ms",    0.0),
            "p99":  vb.get(f"p99_{m}_ms",    0.0),
            "std":  vb.get(f"std_{m}_ms",    0.0),
        }
    return out


def aggregate_one_run(
    bench_dir: Path, mode: str, conc: int, length: int,
) -> dict | None:
    vbench_json = bench_dir / f"vbench_{mode}_conc{conc}_L{length}.json"
    jsonl_dir = bench_dir / "jsonl" / mode / f"conc{conc}" / f"L{length}"
    if not vbench_json.exists():
        return None
    with open(vbench_json) as f:
        vb = json.load(f)

    # Prefill records on node 0 + lj1; consumer records on lj2 + lj3.
    prefill: dict[str, dict] = {}
    for fname in ("producer_node0.jsonl", "producer_lj1.jsonl"):
        for r in _read_jsonl(str(jsonl_dir / fname)):
            rid = r.get("req_id")
            if rid:
                prefill[rid] = r
    decode: dict[str, dict] = {}
    for fname in ("consumer_lj2.jsonl", "consumer_lj3.jsonl"):
        for r in _read_jsonl(str(jsonl_dir / fname)):
            rid = r.get("req_id")
            if rid:
                decode[rid] = r

    matched = _compute_per_request_d_star(prefill, decode)

    server_stats = {
        m: _stats([r[m] for r in matched if r[m] > 0])
        for m in SERVER_METRICS
    }

    return {
        "mode": mode,
        "conc": conc,
        "length": length,
        "rps": vb.get("request_throughput", 0.0),
        "output_throughput": vb.get("output_throughput", 0.0),
        "total_token_throughput": vb.get("total_token_throughput", 0.0),
        "completed": vb.get("completed", 0),
        "failed": vb.get("failed", 0),
        "duration": vb.get("duration", 0.0),
        "client": _client_stats_from_vbench(vb),
        "server": server_stats,
        "join_stats": {
            "n_producer": len(prefill),
            "n_consumer": len(decode),
            "n_matched":  len(matched),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", required=True,
                   help="Path containing vbench_*.json + jsonl/ tree.")
    p.add_argument("--output", required=True,
                   help="Where to write the aggregated JSON.")
    args = p.parse_args()

    bench_dir = Path(args.bench_dir)
    runs = []
    for mode in MODES:
        for conc in CONCURRENCIES:
            for L in LENGTHS:
                r = aggregate_one_run(bench_dir, mode, conc, L)
                if r is not None:
                    runs.append(r)

    out = {"runs": runs}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output} with {len(runs)} runs.")


if __name__ == "__main__":
    main()
