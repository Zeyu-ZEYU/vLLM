#!/usr/bin/env python3
"""
PD Disaggregated Serving Benchmark

Sends requests to the proxy endpoint and collects:
  - RPS (throughput)
  - TTFT (time to first token)
  - TBT (time between tokens, a.k.a. ITL)
  - JCT (job completion time, end-to-end latency)
  - Decode count (output tokens per request)

Usage:
  python benchmark.py --url http://192.168.0.42:9090 --num-requests 10
  python benchmark.py --url http://192.168.0.42:9090 --synthetic --input-len 1024
  python benchmark.py --url http://192.168.0.42:9090 --synthetic --input-len 1024 --qps 2.0
"""

import argparse
import asyncio
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    req_id: int
    prompt_len: int
    output_tokens: int = 0
    ttft: float = 0.0    # seconds
    itl: list[float] = field(default_factory=list)  # inter-token latencies
    jct: float = 0.0     # job completion time (seconds)
    server_id: str = ""   # completion id returned by proxy (cmpl-...)
    # Server-side wall-clock timestamps (epoch seconds). Producer side
    # fills t_prefill_*, t_kv_start, t_kv_mnck_in_end; consumer side
    # fills t_kv_mnck_out_start, t_kv_end. All six are absolute so we
    # can join across prefill and decode nodes.
    t_prefill_start: float = 0.0
    t_prefill_end: float = 0.0
    t_kv_start: float = 0.0
    t_kv_mnck_in_end: float = 0.0
    t_kv_mnck_out_start: float = 0.0
    t_kv_end: float = 0.0
    kv_mode: str = ""     # "layerwise" or "non_layerwise"
    success: bool = True
    error: str = ""

    # Derived intervals (ms); computed after joining producer+consumer
    # JSONLs to the client result.
    @property
    def prefill_ms(self) -> float:
        return max(0.0, (self.t_prefill_end - self.t_prefill_start) * 1000)

    @property
    def kv_save_ms(self) -> float:
        """Prefill side: t_kv_start → t_kv_mnck_in_end.
        Includes GPU→CPU offload + Mooncake Put (local segment or RDMA
        WRITE to remote).
        """
        return max(0.0, (self.t_kv_mnck_in_end - self.t_kv_start) * 1000)

    @property
    def kv_gap_ms(self) -> float:
        """Between prefill's Put-CQE and decode's start-load-kv.
        Contains prefill response serialization + proxy dispatch +
        decode scheduler + decode forward entry.
        """
        return max(0.0, (self.t_kv_mnck_out_start - self.t_kv_mnck_in_end) * 1000)

    @property
    def kv_load_ms(self) -> float:
        """Decode side: t_kv_mnck_out_start → t_kv_end.
        Includes Mooncake Get (RDMA READ cross-node or local) +
        CPU→GPU memcpy. Measured up to cuda-sync after last load.
        """
        return max(0.0, (self.t_kv_end - self.t_kv_mnck_out_start) * 1000)

    @property
    def kv_total_ms(self) -> float:
        """End-to-end KV: t_kv_start on prefill → t_kv_end on decode."""
        return max(0.0, (self.t_kv_end - self.t_kv_start) * 1000)


PCTL_KEYS = ("p25", "p50", "p75", "p90", "p95", "p99")
PCTL_VALS = (25, 50, 75, 90, 95, 99)


def _pcts(data: list[float]) -> dict:
    """Compute P25/P50/P75/P90/P95/P99 for a list of values."""
    if not data:
        return {k: 0.0 for k in PCTL_KEYS}
    return {
        k: float(np.percentile(data, v))
        for k, v in zip(PCTL_KEYS, PCTL_VALS)
    }


@dataclass
class BenchmarkResult:
    num_requests: int
    num_completed: int
    num_failed: int
    total_time: float            # seconds
    rps: float                   # requests per second

    # TTFT — Time To First Token (ms)
    ttft: dict  # {p25, p50, p75, p90, p95, p99}

    # ITL — Inter-Token Latency / TBT (ms)
    itl: dict   # {p25, p50, p75, p90, p95, p99}

    # E2EL — End-to-End Latency / JCT (ms)
    e2el: dict  # {p25, p50, p75, p90, p95, p99}

    # Server-side breakdown (ms), percentiles
    prefill_time: dict   # {p25..p99}
    kv_save_time: dict   # {p25..p99} t_kv_start → t_kv_mnck_in_end (prefill)
    kv_gap_time: dict    # {p25..p99} t_kv_mnck_in_end → t_kv_mnck_out_start
    kv_load_time: dict   # {p25..p99} t_kv_mnck_out_start → t_kv_end (decode)
    kv_total_time: dict  # {p25..p99} t_kv_start → t_kv_end end-to-end

    # Decode
    avg_output_tokens: float
    avg_input_tokens: float

    # Means (in ms) — handy for sanity checking small runs.
    avg_ttft_ms: float = 0.0
    avg_jct_ms: float = 0.0
    avg_itl_ms: float = 0.0
    avg_prefill_ms: float = 0.0
    avg_kv_save_ms: float = 0.0
    avg_kv_gap_ms: float = 0.0
    avg_kv_load_ms: float = 0.0
    avg_kv_total_ms: float = 0.0
    # Decode-side: the first ITL is special — it includes decode startup
    # (proxy→decode HTTP, Mooncake Get of KV, first decode forward,
    # sampler). `decode_steady_ms` is sum of ITLs excluding the first.
    # `decode_total_ms` is sum of all ITLs = first→last token.
    avg_first_itl_ms: float = 0.0
    avg_decode_total_ms: float = 0.0
    avg_decode_steady_ms: float = 0.0

    # Raw per-request data
    requests: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Request sender
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    model: str,
    req_id: int,
) -> RequestResult:
    """Send a streaming completions request and collect timing."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "model": model,
        "stream": True,
    }

    result = RequestResult(req_id=req_id, prompt_len=len(prompt.split()))
    token_times: list[float] = []

    try:
        t_start = time.perf_counter()
        async with session.post(
            f"{url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            resp.raise_for_status()
            first_chunk = True
            async for line in resp.content:
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                token_times.append(time.perf_counter())
                # Capture server completion id from first chunk
                if first_chunk:
                    first_chunk = False
                    try:
                        chunk_data = json.loads(data_str)
                        result.server_id = chunk_data.get("id", "")
                    except (json.JSONDecodeError, KeyError):
                        pass

        t_end = time.perf_counter()

        if not token_times:
            result.success = False
            result.error = "no tokens received"
            return result

        result.ttft = token_times[0] - t_start
        result.jct = t_end - t_start
        result.output_tokens = len(token_times)
        result.itl = [
            token_times[i] - token_times[i - 1]
            for i in range(1, len(token_times))
        ]

    except Exception as e:
        result.success = False
        result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# Synthetic prompt generation
# ---------------------------------------------------------------------------

def generate_synthetic_prompt(input_len: int) -> str:
    """Generate a synthetic prompt of approximately input_len tokens."""
    # Use repeating words to approximate token count (1 word ≈ 1-1.5 tokens)
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "and", "cat", "sat", "on", "mat", "in", "big", "red", "house",
    ]
    # Approximate: 1 word ≈ 1.3 tokens, so generate more words
    num_words = int(input_len * 0.85)
    prompt_words = [random.choice(words) for _ in range(num_words)]
    return " ".join(prompt_words)


# ---------------------------------------------------------------------------
# Server-side metrics collection (from prefill worker JSONL files)
# ---------------------------------------------------------------------------

def _fetch_metrics_source(source: str) -> str:
    """Fetch JSONL content from one source. Returns "" on failure.

    source formats:
      "local:/path/to/file"
      "ssh:user@host:/path/to/file"
    """
    try:
        if source.startswith("local:"):
            path = source[len("local:"):]
            if not os.path.exists(path):
                return ""
            with open(path) as f:
                return f.read()
        if source.startswith("ssh:"):
            rest = source[len("ssh:"):]
            # rest = "user@host:/path"
            host, _, path = rest.partition(":")
            cmd = ["ssh", "-o", "BatchMode=yes", "-o",
                   "StrictHostKeyChecking=no", host, f"cat {path}"]
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=30)
            return out.stdout
    except Exception as e:
        print(f"  [warn] failed to fetch {source}: {e}")
    return ""


def _normalize_metric_req_id(rid: str) -> str:
    """Normalize to vLLM's two-segment completion id form
    ``cmpl-<hex>``. vLLM's internal Request.req_id is actually
    ``cmpl-<hex>-<rank>-<uuid>`` (four dash-separated parts; e.g.
    ``cmpl-b2dae2d7d6089fe7-0-beea237f``). The client's first streaming
    chunk ``id`` is the two-part prefix. Consumer JSONL (after the
    proxy correlation_id fix) writes the two-part form directly;
    producer JSONL writes the full four-part form. Collapsing both to
    two parts lets ``collect_server_metrics`` merge them under a single
    key.
    """
    if rid is None or not rid.startswith("cmpl-"):
        return rid
    parts = rid.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return rid


def collect_server_metrics(sources: list[str]) -> dict[str, dict]:
    """Fetch + parse JSONL from all sources; merge by normalized req_id.

    Producer and consumer JSONLs carry disjoint field sets (producer:
    t_prefill_*, t_kv_start, t_kv_mnck_in_end; consumer:
    t_kv_mnck_out_start, t_kv_end). Merging by ``dict.update`` keeps
    both sides' fields on the same req_id entry.

    The normalization step ensures that old producer JSONLs (with the
    long 'cmpl-<hex>-dp<N>-<uuid>' form) still join with new consumer
    JSONLs (short 'cmpl-<hex>' form from proxy correlation_id).
    """
    entries: dict[str, dict] = {}
    for src in sources:
        content = _fetch_metrics_source(src)
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_rid = rec.get("req_id")
            if not raw_rid:
                continue
            rid = _normalize_metric_req_id(raw_rid)
            existing = entries.setdefault(rid, {"req_id": rid})
            existing.update(rec)
            # Overwrite the stored req_id with the normalized form
            # (rec.update above may have re-written it from the file).
            existing["req_id"] = rid
    return entries


def attach_server_metrics(
    results: list[RequestResult], metrics: dict[str, dict],
) -> tuple[int, int]:
    """Populate each result with server-side timestamps.

    Matching strategy: adapter req_id looks like
    "cmpl-<hex>-<dp_rank>-<uuid>"; client sees server_id "cmpl-<hex>"
    (or the full form). Match by prefix in either direction.

    Returns ``(matched_producer, matched_consumer)``.
    """
    matched_prod = 0
    matched_cons = 0
    for r in results:
        if not r.success or not r.server_id:
            continue
        hit = None
        if r.server_id in metrics:
            hit = metrics[r.server_id]
        else:
            for rid, rec in metrics.items():
                if rid.startswith(r.server_id) or r.server_id.startswith(rid):
                    hit = rec
                    break
        if hit is None:
            continue
        has_producer = "t_prefill_start" in hit
        has_consumer = "t_kv_end" in hit
        if has_producer:
            r.t_prefill_start = float(hit["t_prefill_start"])
            r.t_prefill_end = float(hit["t_prefill_end"])
            r.t_kv_start = float(hit["t_kv_start"])
            r.t_kv_mnck_in_end = float(hit["t_kv_mnck_in_end"])
            if "mode" in hit:
                r.kv_mode = hit["mode"]
            matched_prod += 1
        if has_consumer:
            r.t_kv_mnck_out_start = float(hit["t_kv_mnck_out_start"])
            r.t_kv_end = float(hit["t_kv_end"])
            if "mode" in hit and not r.kv_mode:
                r.kv_mode = hit["mode"]
            matched_cons += 1
    return matched_prod, matched_cons


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def _send_warmup(
    session: aiohttp.ClientSession, url: str, model: str,
    input_len: int, max_tokens: int, count: int,
) -> None:
    """Fire ``count`` warmup requests and await completion.

    Results are discarded. Used to get past cold-start autotuning and
    first-request compilation so the measured run sees steady-state
    timing.
    """
    if count <= 0:
        return
    print(f"Warmup: {count} requests (input~{input_len}, out={max_tokens}) ...")
    tasks = [
        asyncio.create_task(
            send_request(
                session, url,
                generate_synthetic_prompt(input_len),
                max_tokens, model, -(i + 1),
            )
        )
        for i in range(count)
    ]
    warm_results = await asyncio.gather(*tasks)
    ok = sum(1 for r in warm_results if r.success)
    print(f"  Warmup done: {ok}/{count} succeeded")


def _truncate_metrics_sources(sources: list[str]) -> None:
    """Best-effort truncate of every metrics JSONL source.

    Called after warmup so that the main run's metrics don't mix with
    warmup entries.
    """
    for src in sources:
        try:
            if src.startswith("local:"):
                path = src[len("local:"):]
                if os.path.exists(path):
                    open(path, "w").close()
            elif src.startswith("ssh:"):
                rest = src[len("ssh:"):]
                host, _, path = rest.partition(":")
                cmd = [
                    "ssh", "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=no",
                    host, f": > {path}",
                ]
                subprocess.run(cmd, capture_output=True, timeout=15)
        except Exception as e:
            print(f"  [warn] truncate {src} failed: {e}")


async def run_benchmark(args) -> BenchmarkResult:
    """Run the benchmark and return results."""

    # Build prompts
    prompts = []
    max_tokens_list = []
    if args.dataset:
        import jsonlines  # noqa: F811
        with open(args.dataset, "r") as f:
            for line in f:
                req = json.loads(line.strip())
                prompts.append(req["prompt"])
                max_tokens_list.append(req.get("max_tokens", args.max_tokens))
        args.num_requests = len(prompts)
    elif args.synthetic:
        for _ in range(args.num_requests):
            prompts.append(generate_synthetic_prompt(args.input_len))
    elif args.prompt:
        prompts = [args.prompt] * args.num_requests
    else:
        prompts = ["Hello, my name is"] * args.num_requests

    if not max_tokens_list:
        max_tokens_list = [args.max_tokens] * len(prompts)

    # Poisson arrival if qps > 0
    arrival_times: list[float] = []
    if args.qps and args.qps > 0:
        for i in range(args.num_requests):
            if i == 0:
                arrival_times.append(0.0)
            else:
                interval = np.random.exponential(1.0 / args.qps)
                arrival_times.append(arrival_times[-1] + interval)
    else:
        arrival_times = [0.0] * args.num_requests  # all at once

    print(f"Benchmark: {args.num_requests} requests, "
          f"max_tokens={args.max_tokens}, model={args.model}")
    if args.synthetic:
        print(f"  Synthetic input: ~{args.input_len} tokens")
    if args.qps:
        print(f"  Poisson arrival: QPS={args.qps}")

    connector = aiohttp.TCPConnector(limit=max(args.concurrency, args.warmup))
    async with aiohttp.ClientSession(connector=connector) as session:
        if args.warmup > 0:
            warm_input_len = args.input_len if args.synthetic else 512
            warm_max_tokens = args.max_tokens
            await _send_warmup(
                session, args.url, args.model,
                warm_input_len, warm_max_tokens, args.warmup,
            )
            if args.metrics_sources:
                srcs = [s.strip() for s in args.metrics_sources.split(",")
                        if s.strip()]
                _truncate_metrics_sources(srcs)
            # Let any tail callbacks finish before measuring.
            await asyncio.sleep(0.5)

        tasks = []
        t_bench_start = time.perf_counter()

        for i in range(args.num_requests):
            # Wait for arrival time
            wait = arrival_times[i] - (time.perf_counter() - t_bench_start)
            if wait > 0:
                await asyncio.sleep(wait)

            task = asyncio.create_task(
                send_request(
                    session, args.url, prompts[i],
                    max_tokens_list[i], args.model, i,
                )
            )
            tasks.append(task)

        results: list[RequestResult] = await asyncio.gather(*tasks)
        t_bench_end = time.perf_counter()

    total_time = t_bench_end - t_bench_start

    # Collect per-request server-side metrics from prefill worker JSONLs
    if args.metrics_sources:
        sources = [s.strip() for s in args.metrics_sources.split(",")
                   if s.strip()]
        metrics_map = collect_server_metrics(sources)
        m_prod, m_cons = attach_server_metrics(results, metrics_map)
        print(f"  Server metrics: producer={m_prod}/{len(results)} "
              f"consumer={m_cons}/{len(results)} "
              f"(from {len(metrics_map)} merged entries)")

    # Aggregate metrics
    completed = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not completed:
        print("ERROR: All requests failed!")
        empty = {k: 0.0 for k in PCTL_KEYS}
        return BenchmarkResult(
            num_requests=args.num_requests,
            num_completed=0, num_failed=len(failed),
            total_time=total_time, rps=0,
            ttft=empty, itl=empty, e2el=empty,
            prefill_time=empty, kv_save_time=empty, kv_gap_time=empty,
            kv_load_time=empty, kv_total_time=empty,
            avg_output_tokens=0, avg_input_tokens=0,
        )

    ttfts = [r.ttft * 1000 for r in completed]  # ms
    e2els = [r.jct * 1000 for r in completed]    # ms
    all_itls = []
    for r in completed:
        all_itls.extend([x * 1000 for x in r.itl])  # ms

    # Only include requests where BOTH producer and consumer records
    # arrived — otherwise derived intervals are meaningless.
    def _has_full(r: RequestResult) -> bool:
        return r.t_prefill_start > 0 and r.t_kv_end > 0

    ok = [r for r in completed if _has_full(r)]
    prefill_vals = [r.prefill_ms for r in ok]
    kv_save_vals = [r.kv_save_ms for r in ok]
    kv_gap_vals = [r.kv_gap_ms for r in ok]
    kv_load_vals = [r.kv_load_ms for r in ok]
    kv_total_vals = [r.kv_total_ms for r in ok]

    # Decode-side ITL breakdown per request (ms)
    first_itl_vals = [r.itl[0] * 1000 for r in completed if r.itl]
    decode_total_vals = [sum(r.itl) * 1000 for r in completed if r.itl]
    decode_steady_vals = [
        sum(r.itl[1:]) * 1000 for r in completed if len(r.itl) > 1
    ]

    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    result = BenchmarkResult(
        num_requests=args.num_requests,
        num_completed=len(completed),
        num_failed=len(failed),
        total_time=total_time,
        rps=len(completed) / total_time if total_time > 0 else 0,

        ttft=_pcts(ttfts),
        itl=_pcts(all_itls),
        e2el=_pcts(e2els),

        prefill_time=_pcts(prefill_vals),
        kv_save_time=_pcts(kv_save_vals),
        kv_gap_time=_pcts(kv_gap_vals),
        kv_load_time=_pcts(kv_load_vals),
        kv_total_time=_pcts(kv_total_vals),

        avg_output_tokens=float(np.mean([r.output_tokens for r in completed])),
        avg_input_tokens=float(np.mean([r.prompt_len for r in completed])),

        avg_ttft_ms=_mean(ttfts),
        avg_jct_ms=_mean(e2els),
        avg_itl_ms=_mean(all_itls),
        avg_prefill_ms=_mean(prefill_vals),
        avg_kv_save_ms=_mean(kv_save_vals),
        avg_kv_gap_ms=_mean(kv_gap_vals),
        avg_kv_load_ms=_mean(kv_load_vals),
        avg_kv_total_ms=_mean(kv_total_vals),
        avg_first_itl_ms=_mean(first_itl_vals),
        avg_decode_total_ms=_mean(decode_total_vals),
        avg_decode_steady_ms=_mean(decode_steady_vals),

        requests=[asdict(r) for r in results],
    )

    return result


def print_results(result: BenchmarkResult):
    """Pretty-print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Completed:    {result.num_completed}/{result.num_requests} "
          f"({result.num_failed} failed)")
    print(f"  Total time:   {result.total_time:.2f}s")
    print(f"  RPS:          {result.rps:.2f} req/s")
    print(f"  Avg input:    {result.avg_input_tokens:.0f} tokens")
    print(f"  Avg output:   {result.avg_output_tokens:.0f} tokens")
    print()
    def _fmt(d: dict) -> str:
        return "  ".join(f"{k.upper()}={d[k]:.1f}" for k in PCTL_KEYS)

    print(f"  TTFT (ms):        {_fmt(result.ttft)}")
    print(f"  ITL  (ms):        {_fmt(result.itl)}")
    print(f"  E2EL (ms):        {_fmt(result.e2el)}")
    has_server = any(v > 0 for v in result.kv_total_time.values())
    if has_server:
        print(f"  Prefill  (ms):    {_fmt(result.prefill_time)}")
        print(f"  KV save  (ms):    {_fmt(result.kv_save_time)}")
        print(f"  KV gap   (ms):    {_fmt(result.kv_gap_time)}")
        print(f"  KV load  (ms):    {_fmt(result.kv_load_time)}")
        print(f"  KV total (ms):    {_fmt(result.kv_total_time)}")
    else:
        print("  (Server-side KV metrics: no joined producer+consumer "
              "records — check metrics_sources)")
    print()
    print("  Averages (ms):")
    print(f"    TTFT              {result.avg_ttft_ms:8.2f}")
    print(f"    JCT (E2EL)        {result.avg_jct_ms:8.2f}")
    print(f"    ITL (mean)        {result.avg_itl_ms:8.2f}")
    print(f"    first ITL         {result.avg_first_itl_ms:8.2f}  "
          "(incl. decode startup)")
    print(f"    decode total      {result.avg_decode_total_ms:8.2f}  "
          "(first → last token)")
    print(f"    decode steady     {result.avg_decode_steady_ms:8.2f}  "
          "(sum of non-first ITLs)")
    if has_server:
        print(f"    Prefill           {result.avg_prefill_ms:8.2f}  "
              "(t_prefill_end − t_prefill_start, cuda-synced)")
        print(f"    KV save           {result.avg_kv_save_ms:8.2f}  "
              "(prefill: t_kv_start → t_kv_mnck_in_end)")
        print(f"    KV gap            {result.avg_kv_gap_ms:8.2f}  "
              "(t_kv_mnck_in_end → t_kv_mnck_out_start)")
        print(f"    KV load           {result.avg_kv_load_ms:8.2f}  "
              "(decode: t_kv_mnck_out_start → t_kv_end, cuda-synced)")
        print(f"    KV total          {result.avg_kv_total_ms:8.2f}  "
              "(t_kv_start → t_kv_end, end-to-end)")
    print("=" * 60)


def save_results(result: BenchmarkResult, output_dir: str, tag: str = ""):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"bench_{tag}_{ts}.json" if tag else f"bench_{ts}.json"
    path = os.path.join(output_dir, fname)

    # Save without raw per-request data for summary
    summary = {k: v for k, v in asdict(result).items() if k != "requests"}
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {path}")

    # Also save full data with per-request details
    full_path = path.replace(".json", "_full.json")
    with open(full_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"Full data saved to: {full_path}")

    return path


def main():
    parser = argparse.ArgumentParser(description="PD Disagg Serving Benchmark")
    parser.add_argument("--url", type=str, default="http://localhost:9090",
                        help="Proxy URL")
    parser.add_argument("--model", type=str, default="Qwen3-235B",
                        help="Model name")
    parser.add_argument("--num-requests", type=int, default=10,
                        help="Number of requests")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Max output tokens per request")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Max concurrent requests")
    parser.add_argument("--qps", type=float, default=0,
                        help="Target QPS (Poisson arrival). 0 = send all at once")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Fire N warmup requests first (not counted in "
                             "results). Metrics JSONL is truncated after "
                             "warmup so only main-run entries are aggregated.")

    # Synthetic data
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic prompts")
    parser.add_argument("--input-len", type=int, default=1024,
                        help="Approximate input length in tokens (with --synthetic)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Fixed prompt to use for all requests")
    parser.add_argument("--dataset", type=str, default=None,
                        help="JSONL file from gen_synthetic_data.py")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results JSON")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag for result filename")

    # Server-side metrics: comma-separated list of JSONL sources.
    # Each source is "local:/path" or "ssh:user@host:/path". Producer
    # JSONL is written by prefill workers; consumer JSONL by decode
    # workers. Both are merged by adapter req_id and matched to the
    # client's server_id by prefix.
    parser.add_argument(
        "--metrics-sources", type=str,
        default=(
            # producer (prefill) — this node + node-1
            "local:/home/zeyu/lmcache_metrics_producer.jsonl,"
            "ssh:zeyu@lj1.zeyu.tw:/home/zeyu/lmcache_metrics_producer.jsonl,"
            # consumer (decode) — node-2 + node-3
            "ssh:zeyu@lj2.zeyu.tw:/home/zeyu/lmcache_metrics_consumer.jsonl,"
            "ssh:zeyu@lj3.zeyu.tw:/home/zeyu/lmcache_metrics_consumer.jsonl"
        ),
        help="Comma-separated list of JSONL metric sources "
             "(local:/path or ssh:user@host:/path). Empty to disable.",
    )

    args = parser.parse_args()

    result = asyncio.run(run_benchmark(args))
    print_results(result)

    if args.output_dir:
        save_results(result, args.output_dir, args.tag)


if __name__ == "__main__":
    main()
