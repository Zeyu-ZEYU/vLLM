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
    success: bool = True
    error: str = ""


def _pcts(data: list[float]) -> dict:
    """Compute P25/P50/P75/P95 for a list of values."""
    if not data:
        return {"p25": 0, "p50": 0, "p75": 0, "p95": 0}
    return {
        "p25": float(np.percentile(data, 25)),
        "p50": float(np.percentile(data, 50)),
        "p75": float(np.percentile(data, 75)),
        "p95": float(np.percentile(data, 95)),
    }


@dataclass
class BenchmarkResult:
    num_requests: int
    num_completed: int
    num_failed: int
    total_time: float            # seconds
    rps: float                   # requests per second

    # ITL — Inter-Token Latency (ms)
    itl_p25: float
    itl_p50: float
    itl_p75: float
    itl_p95: float

    # E2EL — End-to-End Latency (ms)
    e2el_p25: float
    e2el_p50: float
    e2el_p75: float
    e2el_p95: float

    # TTFT (ms)
    ttft_mean: float
    ttft_median: float
    ttft_p99: float

    # Decode
    avg_output_tokens: float
    avg_input_tokens: float

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
            async for line in resp.content:
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                token_times.append(time.perf_counter())

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
# Main benchmark
# ---------------------------------------------------------------------------

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

    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
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

    # Aggregate metrics
    completed = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not completed:
        print("ERROR: All requests failed!")
        return BenchmarkResult(
            num_requests=args.num_requests,
            num_completed=0, num_failed=len(failed),
            total_time=total_time, rps=0,
            itl_p25=0, itl_p50=0, itl_p75=0, itl_p95=0,
            e2el_p25=0, e2el_p50=0, e2el_p75=0, e2el_p95=0,
            ttft_mean=0, ttft_median=0, ttft_p99=0,
            avg_output_tokens=0, avg_input_tokens=0,
        )

    ttfts = [r.ttft * 1000 for r in completed]  # ms
    e2els = [r.jct * 1000 for r in completed]    # ms (E2EL = JCT)
    all_itls = []
    for r in completed:
        all_itls.extend([x * 1000 for x in r.itl])  # ms

    itl_pcts = _pcts(all_itls)
    e2el_pcts = _pcts(e2els)

    result = BenchmarkResult(
        num_requests=args.num_requests,
        num_completed=len(completed),
        num_failed=len(failed),
        total_time=total_time,
        rps=len(completed) / total_time if total_time > 0 else 0,

        itl_p25=itl_pcts["p25"],
        itl_p50=itl_pcts["p50"],
        itl_p75=itl_pcts["p75"],
        itl_p95=itl_pcts["p95"],

        e2el_p25=e2el_pcts["p25"],
        e2el_p50=e2el_pcts["p50"],
        e2el_p75=e2el_pcts["p75"],
        e2el_p95=e2el_pcts["p95"],

        ttft_mean=float(np.mean(ttfts)),
        ttft_median=float(np.median(ttfts)),
        ttft_p99=float(np.percentile(ttfts, 99)) if len(ttfts) > 1 else ttfts[0],

        avg_output_tokens=float(np.mean([r.output_tokens for r in completed])),
        avg_input_tokens=float(np.mean([r.prompt_len for r in completed])),

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
    print(f"  TTFT (ms):    mean={result.ttft_mean:.1f}  "
          f"median={result.ttft_median:.1f}  p99={result.ttft_p99:.1f}")
    print(f"  ITL  (ms):    P25={result.itl_p25:.1f}  P50={result.itl_p50:.1f}  "
          f"P75={result.itl_p75:.1f}  P95={result.itl_p95:.1f}")
    print(f"  E2EL (ms):    P25={result.e2el_p25:.1f}  P50={result.e2el_p50:.1f}  "
          f"P75={result.e2el_p75:.1f}  P95={result.e2el_p95:.1f}")
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

    args = parser.parse_args()

    result = asyncio.run(run_benchmark(args))
    print_results(result)

    if args.output_dir:
        save_results(result, args.output_dir, args.tag)


if __name__ == "__main__":
    main()
