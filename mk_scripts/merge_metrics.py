#!/usr/bin/env python3
"""Merge the BL1 client (vllm bench serve) JSON dump and the server-side
sidecar JSONL into the project's spec metrics JSONL.

Output format (per ~/zeyu/mono_kernel/tasks/Motivation实验.md):
- Line 1: a header object with `rps`, `duration_s`, `num_completed`, etc.
- Lines 2..N+1: one object per input row (in input file order) with all
  per-request metrics defined in the spec.

Usage:
    python mk_scripts/merge_metrics.py \\
        --client  outputs/.bl1_client_<T>.json \\
        --server  outputs/.bl1_server_<T>.jsonl \\
        --inputs  inputs/requests/example.jsonl \\
        --label origin --workload example --args qwen3vl8b_n5_rps2 --time <T> \\
        --out outputs/origin_example_qwen3vl8b_n5_rps2_<T>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any


# vLLM v1's engine input processor wraps every externally-supplied request_id
# as "<server_prefix>-<external_id>-<8hex>" where:
#   server_prefix = "chatcmpl-" for /v1/chat/completions, "cmpl-" for
#                   /v1/completions, "" for low-level /v1/responses, etc.
#   <8hex>        = first 8 chars of a random UUID (input_processor.py:232)
# The merge needs to recover <external_id> to key by the input JSONL's `id`.
_RANDOMIZED_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")
_KNOWN_SERVER_PREFIXES = ("chatcmpl-", "cmpl-", "embd-", "rerank-")


def _recover_external_req_id(internal: str) -> str:
    """Best-effort recovery of the user-supplied request_id from vLLM's
    internal-form id."""
    s = _RANDOMIZED_SUFFIX_RE.sub("", internal)
    for prefix in _KNOWN_SERVER_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _safe_index(arr, i):
    try:
        return arr[i]
    except (IndexError, TypeError):
        return None


def _safe_sum(itl):
    if itl is None:
        return 0.0
    try:
        return float(sum(itl))
    except TypeError:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True, help="vllm bench serve JSON dump")
    ap.add_argument("--server", required=True,
                    help="BL1 server-side sidecar JSONL")
    ap.add_argument("--inputs", required=True,
                    help="Original requests JSONL (one row per line)")
    ap.add_argument("--label", required=True,
                    help="origin | disaggregation | pipeline")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--args", dest="args_tag", required=True)
    ap.add_argument("--time", dest="time_tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.client, "r", encoding="utf-8") as f:
        client = json.load(f)

    inputs = _load_jsonl(a.inputs)
    server_records = _load_jsonl(a.server)
    # Key server records by recovered external id so the input JSONL's `id`
    # field matches.
    server_by_id: dict[str, dict[str, Any]] = {}
    for r in server_records:
        internal = str(r.get("vllm_req_id", ""))
        ext = _recover_external_req_id(internal)
        server_by_id[ext] = r

    # Bench-serve --save-detailed parallel arrays (positional, in input order).
    output_lens = client.get("output_lens") or []
    ttfts = client.get("ttfts") or []
    itls = client.get("itls") or []
    start_times = client.get("start_times") or []
    generated_texts = client.get("generated_texts") or []
    errors = client.get("errors") or []

    rps = client.get("request_throughput")
    duration = client.get("duration")
    completed = client.get("completed")
    num_failed = client.get("failed", 0)
    model_id = client.get("model_id", "")

    header = {
        "rps": rps,
        "duration_s": duration,
        "num_completed": completed,
        "num_failed": num_failed,
        "label": a.label,
        "workload": a.workload,
        "args": a.args_tag,
        "time": a.time_tag,
        "model": model_id,
    }

    # Bench-serve only ran the first N input rows (--num-prompts N). The
    # parallel client arrays have length N; the remaining input rows have
    # no client metrics. Truncate to those N to avoid emitting placeholder
    # rows full of None.
    n_actual = min(len(inputs), len(output_lens) or len(inputs))
    inputs_truncated = inputs[:n_actual]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as out:
        out.write(json.dumps(header, ensure_ascii=False) + "\n")
        for i, row in enumerate(inputs_truncated):
            req_id = str(row.get("id", i))
            ttft = _safe_index(ttfts, i)
            itl = _safe_index(itls, i)
            num_otokens = _safe_index(output_lens, i) or 0
            start_time = _safe_index(start_times, i)
            err = _safe_index(errors, i) or ""

            # e2el = ttft + sum(itl) when streaming; otherwise approximated.
            e2el: Any
            if ttft is not None and itl is not None:
                e2el = ttft + _safe_sum(itl)
            elif ttft is not None:
                e2el = ttft
            else:
                e2el = None
            end_time = (start_time + e2el) if (
                start_time is not None and e2el is not None
            ) else None

            tpot = None
            if e2el is not None and ttft is not None and num_otokens and num_otokens > 1:
                tpot = (e2el - ttft) / max(num_otokens - 1, 1)

            srec = server_by_id.get(req_id)
            def _g(name):
                return srec.get(name) if srec else None

            out_row = {
                "id": row.get("id", i),
                "vllm_id": req_id,
                "start_time": start_time,
                "end_time": end_time,
                "output": _safe_index(generated_texts, i),
                "d_vision": _g("d_vision"),
                "d_prefill": _g("d_prefill"),
                "d_decode": _g("d_decode"),
                "num_otokens": num_otokens,
                "tpot": tpot,
                "ttft": ttft,
                "jct": e2el,
                "gu_vision": _g("gu_vision"),
                "gu_prefill": _g("gu_prefill"),
                "gu_decode": _g("gu_decode"),
                "gmu_vision": _g("gmu_vision"),
                "gmu_prefill": _g("gmu_prefill"),
                "gmu_decode": _g("gmu_decode"),
                "ko_vision": _g("ko_vision"),
                "ko_prefill": _g("ko_prefill"),
                "ko_decode": _g("ko_decode"),
                # SM-level metrics not collected in BL1 default run.
                "nsm_vision": None,
                "nsm_prefill": None,
                "nsm_decode": None,
                "smu_vision": None,
                "smu_prefill": None,
                "smu_decode": None,
                "error": err,
            }
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"Wrote merged metrics to {a.out}")


if __name__ == "__main__":
    main()
