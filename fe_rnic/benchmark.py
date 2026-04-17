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
    # Server-side metrics (from prefill worker JSONL, ms)
    prefill_time_ms: float = 0.0
    kv_total_time_ms: float = 0.0
    kv_exposed_time_ms: float = 0.0
    kv_mode: str = ""     # "layerwise" or "non_layerwise"
    success: bool = True
    error: str = ""


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

    # Server-side metrics (ms)
    prefill_time: dict   # {p25..p99} prefill compute only
    kv_total_time: dict  # {p25..p99} first layer start → last layer end
    kv_exposed_time: dict  # {p25..p99} prefill end → last layer end

    # Decode
    avg_output_tokens: float
    avg_input_tokens: float

    # Means (in ms) — handy for sanity checking small runs.
    avg_ttft_ms: float = 0.0
    avg_jct_ms: float = 0.0
    avg_itl_ms: float = 0.0
    avg_prefill_ms: float = 0.0
    avg_kv_total_ms: float = 0.0
    avg_kv_exposed_ms: float = 0.0

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


def collect_server_metrics(sources: list[str]) -> dict[str, dict]:
    """Fetch + parse JSONL from all sources. Key = adapter req_id."""
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
            rid = rec.get("req_id")
            if not rid:
                continue
            # Last write wins if req_id repeats (chunked prefill).
            if rid in entries and entries[rid].get("ts", 0) > rec.get("ts", 0):
                continue
            entries[rid] = rec
    return entries


def attach_server_metrics(
    results: list[RequestResult], metrics: dict[str, dict],
) -> int:
    """Populate each result with matched server metrics.

    Matching strategy: adapter's req_id looks like "cmpl-<hex>-<dp_rank>-<uuid>",
    while the client sees server_id = "cmpl-<hex>" (or the full form). Match by
    prefix in either direction.

    Returns the count of matched results.
    """
    matched = 0
    for r in results:
        if not r.success or not r.server_id:
            continue
        hit = None
        # Try exact match first
        if r.server_id in metrics:
            hit = metrics[r.server_id]
        else:
            # Prefix match: adapter req_id starts with client server_id
            for rid, rec in metrics.items():
                if rid.startswith(r.server_id) or r.server_id.startswith(rid):
                    hit = rec
                    break
        if hit:
            r.prefill_time_ms = float(hit.get("prefill_ms", 0))
            r.kv_total_time_ms = float(hit.get("kv_total_ms", 0))
            r.kv_exposed_time_ms = float(hit.get("kv_exposed_ms", 0))
            r.kv_mode = hit.get("mode", "")
            matched += 1
    return matched


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
        matched = attach_server_metrics(results, metrics_map)
        print(f"  Server metrics: {matched}/{len(results)} matched "
              f"from {len(metrics_map)} JSONL entries")

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
            prefill_time=empty, kv_total_time=empty,
            kv_exposed_time=empty,
            avg_output_tokens=0, avg_input_tokens=0,
        )

    ttfts = [r.ttft * 1000 for r in completed]  # ms
    e2els = [r.jct * 1000 for r in completed]    # ms
    all_itls = []
    for r in completed:
        all_itls.extend([x * 1000 for x in r.itl])  # ms

    prefill_vals = [r.prefill_time_ms for r in completed]
    kv_total_vals = [r.kv_total_time_ms for r in completed]
    kv_exposed_vals = [r.kv_exposed_time_ms for r in completed]

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
        kv_total_time=_pcts(kv_total_vals),
        kv_exposed_time=_pcts(kv_exposed_vals),

        avg_output_tokens=float(np.mean([r.output_tokens for r in completed])),
        avg_input_tokens=float(np.mean([r.prompt_len for r in completed])),

        avg_ttft_ms=_mean(ttfts),
        avg_jct_ms=_mean(e2els),
        avg_itl_ms=_mean(all_itls),
        avg_prefill_ms=_mean(prefill_vals),
        avg_kv_total_ms=_mean(kv_total_vals),
        avg_kv_exposed_ms=_mean(kv_exposed_vals),

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
    # Server-side metrics (from prefill worker logs)
    has_server = any(v > 0 for v in result.kv_total_time.values())
    if has_server:
        print(f"  Prefill (ms):     {_fmt(result.prefill_time)}")
        print(f"  KV total (ms):    {_fmt(result.kv_total_time)}")
        print(f"  KV exposed (ms):  {_fmt(result.kv_exposed_time)}")
    else:
        print("  (Server-side KV metrics: see prefill log [METRICS])")
    print()
    print("  Averages (ms):")
    print(f"    TTFT        {result.avg_ttft_ms:8.2f}")
    print(f"    JCT (E2EL)  {result.avg_jct_ms:8.2f}")
    print(f"    ITL         {result.avg_itl_ms:8.2f}")
    if has_server:
        print(f"    Prefill     {result.avg_prefill_ms:8.2f}")
        print(f"    KV total    {result.avg_kv_total_ms:8.2f}")
        print(f"    KV exposed  {result.avg_kv_exposed_ms:8.2f}")
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
    # Each source is "local:/path" or "ssh:user@host:/path". Content is
    # merged by adapter req_id and matched to client results by prefix.
    parser.add_argument(
        "--metrics-sources", type=str,
        default="local:/home/zeyu/lmcache_metrics.jsonl,"
                "ssh:zeyu@lj1.zeyu.tw:/home/zeyu/lmcache_metrics.jsonl",
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
