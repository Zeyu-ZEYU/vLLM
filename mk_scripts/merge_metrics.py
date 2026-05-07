#!/usr/bin/env python3
"""Merge per-experiment sidecars into the project's spec metrics JSONL.

Two modes, picked by which flags are present:

* **BL1 (origin)**: single-instance run.
  Required: --client + --server. Optional: --server-sm.

* **BL2 (disaggregation)**: vision instance + text PD instance, two nodes.
  Required: --client + --server-vis + --server-text. Optional:
  --server-vis-sm, --server-text-sm, --server-vis-vemb, --server-text-vemb.

Output format (per ~/zeyu/mono_kernel/tasks/Motivation实验.md):
- Line 1: a header object with `rps`, `duration_s`, `num_completed`, etc.
- Lines 2..N+1: one object per input row (in input file order) with all
  per-request metrics defined in the spec. BL2 adds three vemb metrics:
  d_vemb_total, d_vemb_wait, d_vemb_pull (see _build_vemb_index docstring).

Per-request metric origin in BL2:
- d_vision, gu_vision, gmu_vision     ← vision-instance NVML sidecar
- d_prefill, d_decode, gu_prefill,    ← text-instance NVML sidecar
  gu_decode, gmu_prefill, gmu_decode
- ko/smu/nsm/sm_occ vision            ← vision-instance SM sidecar
- ko/smu/nsm/sm_occ prefill, decode   ← text-instance SM sidecar
- d_vemb_transfer                     ← max over (mm_hash) of
                                        (consumer.t_event - producer.t_event)
- ttft, jct, tpot, num_otokens, output, error ← client JSON (vllm bench)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
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
    internal-form id.

    Also unwinds the proxy-fanout id shape introduced by
    `examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py`,
    where each per-MM-item child request gets `<parent>:<idx>:<rand6>`
    appended before vLLM's own randomization. After stripping the
    `-<8hex>` and the `chatcmpl-` prefix, we additionally split on `:`
    and keep the leading parent id, which matches the input JSONL's
    `id` field.
    """
    s = _RANDOMIZED_SUFFIX_RE.sub("", internal)
    for prefix in _KNOWN_SERVER_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if ":" in s:
        s = s.split(":", 1)[0]
    return s


def _load_jsonl(path: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _index_by_external_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in records:
        internal = str(r.get("vllm_req_id", ""))
        out[_recover_external_req_id(internal)] = r
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


def _build_vemb_index(
    producer_records: list[dict[str, Any]],
    consumer_records: list[dict[str, Any]],
) -> tuple[
    dict[str, float], dict[str, float], dict[str, float]
]:
    """Pair producer/consumer events by mm_hash → three per-mm_hash metrics.

    Returns a tuple of three dicts, all keyed by mm_hash:

      total_by_hash[mh] = max(0, consumer.t_event - producer.t_event)
        cross-node, NTP-bound. Reflects user-perceived end-to-end latency
        for this image (from PUSH-sent to cache-populated).

      wait_by_hash[mh]  = consumer.d_wait
        single-clock perf_counter on consumer; time blocked in wait_for_notif
        for this mm_hash's PUSH to arrive.

      pull_by_hash[mh]  = consumer.d_pull
        single-clock perf_counter on consumer; time spent in consumer_pull
        (NIXL READ + scratch copy-out).

    For mm_hashes that appear in multiple events (rare; same hash consumed
    by multiple requests on this consumer instance), we use:
      - producer.t_event:   min  (earliest send finish)
      - consumer.t_event:   max  (latest receive finish)
      - consumer.d_wait:    max  (worst-case wait among events)
      - consumer.d_pull:    max  (worst-case pull among events)
    so total_by_hash captures the worst-case bound.
    """
    p_min: dict[str, float] = {}
    c_max: dict[str, float] = {}
    wait_max: dict[str, float] = {}
    pull_max: dict[str, float] = {}
    for r in producer_records:
        mh = str(r.get("mm_hash", ""))
        if not mh:
            continue
        t = float(r.get("t_event", 0.0))
        if mh not in p_min or t < p_min[mh]:
            p_min[mh] = t
    for r in consumer_records:
        mh = str(r.get("mm_hash", ""))
        if not mh:
            continue
        t = float(r.get("t_event", 0.0))
        if mh not in c_max or t > c_max[mh]:
            c_max[mh] = t
        d_wait = r.get("d_wait")
        if d_wait is not None:
            d_wait = float(d_wait)
            if mh not in wait_max or d_wait > wait_max[mh]:
                wait_max[mh] = d_wait
        d_pull = r.get("d_pull")
        if d_pull is not None:
            d_pull = float(d_pull)
            if mh not in pull_max or d_pull > pull_max[mh]:
                pull_max[mh] = d_pull

    total_by_hash: dict[str, float] = {}
    for mh, t_recv in c_max.items():
        if mh in p_min:
            total_by_hash[mh] = max(0.0, t_recv - p_min[mh])
    return total_by_hash, wait_max, pull_max


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True, help="vllm bench serve JSON dump")
    # BL1 (origin) mode
    ap.add_argument("--server", default=None,
                    help="BL1 single-server NVML sidecar JSONL")
    ap.add_argument("--server-sm", default=None,
                    help="BL1 single-server SM-pass sidecar JSONL (DCGM)")
    # BL2 (disaggregation) mode
    ap.add_argument("--server-vis", default=None,
                    help="BL2 vision-instance NVML sidecar JSONL")
    ap.add_argument("--server-vis-sm", default=None,
                    help="BL2 vision-instance SM-pass sidecar JSONL")
    ap.add_argument("--server-vis-vemb", default=None,
                    help="BL2 vision-side BL2 vemb sidecar (producer events)")
    ap.add_argument("--server-text", default=None,
                    help="BL2 text-instance NVML sidecar JSONL")
    ap.add_argument("--server-text-sm", default=None,
                    help="BL2 text-instance SM-pass sidecar JSONL")
    ap.add_argument("--server-text-vemb", default=None,
                    help="BL2 text-side BL2 vemb sidecar (consumer events)")
    ap.add_argument("--inputs", required=True,
                    help="Original requests JSONL (one row per line)")
    ap.add_argument("--label", required=True,
                    help="origin | disaggregation | pipeline")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--args", dest="args_tag", required=True)
    ap.add_argument("--time", dest="time_tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    is_bl2 = bool(a.server_vis or a.server_text)
    if is_bl2:
        if not a.server_vis or not a.server_text:
            ap.error("BL2 mode requires BOTH --server-vis and --server-text")
    else:
        if not a.server:
            ap.error("BL1 (origin) mode requires --server")

    with open(a.client, "r", encoding="utf-8") as f:
        client = json.load(f)

    inputs = _load_jsonl(a.inputs)

    # Index NVML sidecars by recovered external request id.
    if is_bl2:
        vis_by_id = _index_by_external_id(_load_jsonl(a.server_vis))
        text_by_id = _index_by_external_id(_load_jsonl(a.server_text))
        vis_sm_by_id = _index_by_external_id(_load_jsonl(a.server_vis_sm))
        text_sm_by_id = _index_by_external_id(_load_jsonl(a.server_text_sm))
        # vemb sidecars are keyed by mm_hash → time.
        text_vemb_records = _load_jsonl(a.server_text_vemb)
        vemb_total_by_hash, vemb_wait_by_hash, vemb_pull_by_hash = (
            _build_vemb_index(
                _load_jsonl(a.server_vis_vemb), text_vemb_records,
            )
        )
        # Build a req_id → list[mm_hash] map from the text-side consumer
        # vemb events (these include the request that triggered the load,
        # which the producer side cannot know on its own).
        req_to_mm_hashes: dict[str, list[str]] = defaultdict(list)
        for rec in text_vemb_records:
            mh = str(rec.get("mm_hash", ""))
            for rid in rec.get("req_ids", []) or []:
                ext = _recover_external_req_id(str(rid))
                req_to_mm_hashes[ext].append(mh)
    else:
        server_by_id = _index_by_external_id(_load_jsonl(a.server))
        server_sm_by_id = _index_by_external_id(_load_jsonl(a.server_sm))

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

    header: dict[str, Any] = {
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
    if is_bl2:
        header["bl2_n_pairs"] = len(vemb_total_by_hash)

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

            if is_bl2:
                vis_rec = vis_by_id.get(req_id)
                text_rec = text_by_id.get(req_id)
                vis_sm_rec = vis_sm_by_id.get(req_id)
                text_sm_rec = text_sm_by_id.get(req_id)

                def _vis(name):
                    return vis_rec.get(name) if vis_rec else None

                def _text(name):
                    return text_rec.get(name) if text_rec else None

                def _vis_sm(name):
                    return vis_sm_rec.get(name) if vis_sm_rec else None

                def _text_sm(name):
                    return text_sm_rec.get(name) if text_sm_rec else None

                # Source of truth for "which mm_hashes this request used"
                # is the text-side vemb sidecar (consumer events), which
                # records req_id alongside mm_hash. The vision NVML
                # sidecar's mm_hashes can also be used as a fallback,
                # though the proxy fanout gives that side a different
                # req_id shape.
                mm_hashes = req_to_mm_hashes.get(req_id) or []
                if not mm_hashes:
                    mm_hashes = (vis_rec or {}).get("mm_hashes") or \
                                (text_rec or {}).get("mm_hashes") or []
                # Per-request vemb metrics:
                # - d_vemb_total: max over mm_hashes of cross-node end-to-end
                #   (PUSH-sent on producer to cache-populated on consumer).
                #   The slowest image bounds when prefill can start.
                # - d_vemb_wait: sum over mm_hashes of consumer.d_wait. The
                #   consumer iterates mm_hashes serially in start_load_caches,
                #   so summing gives total wall-clock blocked on PUSH-arrival.
                # - d_vemb_pull: sum over mm_hashes of consumer.d_pull. Same
                #   serial reasoning, gives total wall-clock for NIXL READs.
                d_vemb_total = None
                d_vemb_wait_sum = 0.0
                d_vemb_pull_sum = 0.0
                wait_count = 0
                pull_count = 0
                for mh in mm_hashes:
                    mh_s = str(mh)
                    t = vemb_total_by_hash.get(mh_s)
                    if t is not None:
                        d_vemb_total = (
                            t if d_vemb_total is None else max(d_vemb_total, t)
                        )
                    w = vemb_wait_by_hash.get(mh_s)
                    if w is not None:
                        d_vemb_wait_sum += w
                        wait_count += 1
                    p = vemb_pull_by_hash.get(mh_s)
                    if p is not None:
                        d_vemb_pull_sum += p
                        pull_count += 1
                d_vemb_wait = d_vemb_wait_sum if wait_count > 0 else None
                d_vemb_pull = d_vemb_pull_sum if pull_count > 0 else None

                out_row = {
                    "id": row.get("id", i),
                    "vllm_id": req_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "output": _safe_index(generated_texts, i),
                    # Per-phase durations: d_phase = sum of execution
                    # forwards' elapsed_time; d_phase_span = first execution
                    # start to last execution end. Both seconds.
                    "d_vision": _vis("d_vision"),
                    "d_prefill": _text("d_prefill"),
                    "d_decode": _text("d_decode"),
                    "d_vision_span": _vis("d_vision_span"),
                    "d_prefill_span": _text("d_prefill_span"),
                    "d_decode_span": _text("d_decode_span"),
                    "n_executions_vision": _vis("n_executions_vision"),
                    "n_executions_prefill": _text("n_executions_prefill"),
                    "n_executions_decode": _text("n_executions_decode"),
                    "d_vemb_total": d_vemb_total,
                    "d_vemb_wait": d_vemb_wait,
                    "d_vemb_pull": d_vemb_pull,
                    "num_otokens": num_otokens,
                    "tpot": tpot,
                    "ttft": ttft,
                    "jct": e2el,
                    # gu/gmu/smu/ko/sm_occ are fractions in [0, 1] from the
                    # recorder; merger passes through unchanged.
                    "gu_vision": _vis("gu_vision"),
                    "gu_prefill": _text("gu_prefill"),
                    "gu_decode": _text("gu_decode"),
                    "gmu_vision": _vis("gmu_vision"),
                    "gmu_prefill": _text("gmu_prefill"),
                    "gmu_decode": _text("gmu_decode"),
                    "ko_vision": _vis_sm("ko_vision"),
                    "ko_prefill": _text_sm("ko_prefill"),
                    "ko_decode": _text_sm("ko_decode"),
                    "nsm_vision": _vis_sm("nsm_vision"),
                    "nsm_prefill": _text_sm("nsm_prefill"),
                    "nsm_decode": _text_sm("nsm_decode"),
                    "smu_vision": _vis_sm("smu_vision"),
                    "smu_prefill": _text_sm("smu_prefill"),
                    "smu_decode": _text_sm("smu_decode"),
                    "sm_occ_vision": _vis_sm("sm_occ_vision"),
                    "sm_occ_prefill": _text_sm("sm_occ_prefill"),
                    "sm_occ_decode": _text_sm("sm_occ_decode"),
                    "mm_hashes": mm_hashes,
                    "error": err,
                }
            else:
                srec = server_by_id.get(req_id)
                sm_rec = server_sm_by_id.get(req_id)

                def _g(name):
                    return srec.get(name) if srec else None

                def _sm(name):
                    return sm_rec.get(name) if sm_rec else None

                out_row = {
                    "id": row.get("id", i),
                    "vllm_id": req_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "output": _safe_index(generated_texts, i),
                    "d_vision": _g("d_vision"),
                    "d_prefill": _g("d_prefill"),
                    "d_decode": _g("d_decode"),
                    "d_vision_span": _g("d_vision_span"),
                    "d_prefill_span": _g("d_prefill_span"),
                    "d_decode_span": _g("d_decode_span"),
                    "n_executions_vision": _g("n_executions_vision"),
                    "n_executions_prefill": _g("n_executions_prefill"),
                    "n_executions_decode": _g("n_executions_decode"),
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
                    "ko_vision": _sm("ko_vision"),
                    "ko_prefill": _sm("ko_prefill"),
                    "ko_decode": _sm("ko_decode"),
                    "nsm_vision": _sm("nsm_vision"),
                    "nsm_prefill": _sm("nsm_prefill"),
                    "nsm_decode": _sm("nsm_decode"),
                    "smu_vision": _sm("smu_vision"),
                    "smu_prefill": _sm("smu_prefill"),
                    "smu_decode": _sm("smu_decode"),
                    "sm_occ_vision": _sm("sm_occ_vision"),
                    "sm_occ_prefill": _sm("sm_occ_prefill"),
                    "sm_occ_decode": _sm("sm_occ_decode"),
                    "error": err,
                }
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"Wrote merged metrics to {a.out}")


if __name__ == "__main__":
    main()
