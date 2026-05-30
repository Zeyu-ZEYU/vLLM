#!/usr/bin/env python3
"""
Trace-replay load generator for the fe_rnic backend-bandwidth measurement.

Replays qwen_traceB requests against the disagg proxy's /v1/completions,
reproducing the trace's PREFIX SHARING so that reused-prefix KV is pulled
(inbound) from the decode-node Mooncake segment, and freshly-computed KV is
pushed (outbound, P->D) to it. With the prefill node holding no Mooncake
segment (PREFILL_SEGMENT_SIZE=0), both flows cross the backend network and
show up in bw_sampler.py.

The trace has only 16-token-block `hash_ids` (no raw tokens), so we synthesize
a DETERMINISTIC 16-token block per hash_id: requests sharing leading hash_ids
share leading token ids -> identical leading LMCache chunks -> prefix reuse.
Prompts are sent as raw token-id lists; the proxy forwards a list `prompt`
straight to vLLM, skipping /tokenize (see disagg_proxy_server.py).

Pacing modes:
  --mode timestamp : honor the trace's arrival timestamps (open loop, real load)
  --mode poisson   : Poisson arrivals at --rps (open loop, fixed mean rate)
  --mode asap      : dispatch as fast as --max-concurrency allows (saturate)

Run inside the fe_rnic container (venv has httpx):
  python3 trace_replay.py --trace ~/traces/qwen_traceB_blksz_16.jsonl.xz \
      --proxy 192.168.0.42:9090 --num-requests 1000 --mode poisson --rps 30
"""
import argparse
import asyncio
import json
import lzma
import random
import sys
import time

import httpx

BLOCK = 16


def block_tokens(hash_id, lo, hi):
    """Deterministic 16 token ids for one 16-token block hash."""
    rng = random.Random((hash_id * 2654435761) & 0xFFFFFFFFFFFF)
    span = hi - lo
    return [lo + rng.randrange(span) for _ in range(BLOCK)]


def build_prompt(hash_ids, input_length, cache, lo, hi, max_input_len):
    """Concatenate per-hash blocks into a token-id list of length min(input_length, cap)."""
    L = min(int(input_length), max_input_len)
    toks = []
    for h in hash_ids:
        if len(toks) >= L:
            break
        b = cache.get(h)
        if b is None:
            b = block_tokens(h, lo, hi)
            cache[h] = b
        toks.extend(b)
    if len(toks) < L:  # rare: hash_ids*16 < input_length -> deterministic pad
        pad = cache.setdefault(-1, block_tokens(-1, lo, hi))
        i = 0
        while len(toks) < L:
            toks.append(pad[i % BLOCK])
            i += 1
    return toks[:L]


def load_trace(path, n, start=0):
    op = lzma.open if path.endswith(".xz") else open
    out = []
    with op(path, "rt") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            out.append(json.loads(line))
            if n and len(out) >= n:
                break
    return out


def pct(arr, p):
    if not arr:
        return 0.0
    a = sorted(arr)
    return a[min(len(a) - 1, int(len(a) * p / 100))]


async def one_request(client, url, model, prompt, max_tokens, stats, idx, logf):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True}
    t0 = time.time()
    ttft = None
    try:
        async with client.stream("POST", url, json=body) as resp:
            resp.raise_for_status()
            async for _chunk in resp.aiter_bytes():
                if ttft is None:
                    ttft = time.time() - t0
        lat = time.time() - t0
        stats["ok"] += 1
        stats["ttft"].append(ttft if ttft is not None else lat)
        stats["lat"].append(lat)
        if logf:
            logf.write(json.dumps({"idx": idx, "n_tok": len(prompt), "max_tokens": max_tokens,
                                   "ttft": ttft, "lat": lat, "ok": True}) + "\n")
    except Exception as e:  # noqa: BLE001
        stats["errors"] += 1
        if stats["errors"] <= 8:
            print(f"[req {idx}] ERROR {type(e).__name__}: {e}", file=sys.stderr)
        if logf:
            logf.write(json.dumps({"idx": idx, "n_tok": len(prompt), "ok": False,
                                   "err": f"{type(e).__name__}: {e}"}) + "\n")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--proxy", default="localhost:9090")
    ap.add_argument("--model", default="Qwen3-235B")
    ap.add_argument("--num-requests", type=int, default=1000, help="replay N requests from --start-index (0 = all)")
    ap.add_argument("--start-index", type=int, default=0, help="skip the first N trace requests (warm-up vs measure split)")
    ap.add_argument("--mode", choices=["timestamp", "poisson", "asap"], default="poisson")
    ap.add_argument("--rps", type=float, default=30.0, help="poisson mean arrival rate")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="timestamp mode: scale inter-arrivals (>1 = slower replay)")
    ap.add_argument("--max-concurrency", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=8,
                    help="decode output tokens; 0 = use trace output_length (min 2)")
    ap.add_argument("--max-input-len", type=int, default=16000, help="cap prompt length (<= max-model-len)")
    ap.add_argument("--vocab-lo", type=int, default=1000)
    ap.add_argument("--vocab-hi", type=int, default=150000)
    ap.add_argument("--out", default="", help="optional per-request JSONL log")
    args = ap.parse_args()

    reqs = load_trace(args.trace, args.num_requests, args.start_index)
    print(f"loaded {len(reqs)} requests from {args.trace} (start_index={args.start_index})")

    cache = {}
    prompts = []
    for r in reqs:
        p = build_prompt(r["hash_ids"], r["input_length"], cache,
                         args.vocab_lo, args.vocab_hi, args.max_input_len)
        mt = args.max_tokens if args.max_tokens > 0 else max(2, int(r.get("output_length", 8)))
        prompts.append((p, mt))
    print(f"built {len(prompts)} prompts; {len(cache)} unique 16-token blocks; "
          f"avg_len={sum(len(p) for p, _ in prompts) // max(1, len(prompts))}")

    # arrival schedule (seconds from start)
    if args.mode == "timestamp":
        base = reqs[0].get("timestamp", 0.0)
        sched = [(r.get("timestamp", 0.0) - base) * args.time_scale for r in reqs]
    elif args.mode == "poisson":
        rng = random.Random(1234)
        sched, t = [], 0.0
        for _ in reqs:
            sched.append(t)
            t += rng.expovariate(args.rps)
    else:  # asap
        sched = [0.0] * len(reqs)

    url = f"http://{args.proxy}/v1/completions"
    sem = asyncio.Semaphore(args.max_concurrency)
    stats = {"ok": 0, "errors": 0, "ttft": [], "lat": []}
    inflight = {"n": 0, "max": 0}
    logf = open(args.out, "w") if args.out else None
    limits = httpx.Limits(max_connections=args.max_concurrency + 32,
                          max_keepalive_connections=args.max_concurrency + 32)
    timeout = httpx.Timeout(connect=30.0, read=900.0, write=900.0, pool=900.0)

    print(f"mode={args.mode} rps={args.rps} max_conc={args.max_concurrency} "
          f"max_tokens={args.max_tokens} -> {url}")

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        start = time.time()

        async def fire(i):
            async with sem:
                inflight["n"] += 1
                inflight["max"] = max(inflight["max"], inflight["n"])
                try:
                    await one_request(client, url, args.model, prompts[i][0],
                                      prompts[i][1], stats, i, logf)
                finally:
                    inflight["n"] -= 1

        tasks = []
        for i in range(len(reqs)):
            wait = sched[i] - (time.time() - start)
            if wait > 0:
                await asyncio.sleep(wait)
            tasks.append(asyncio.create_task(fire(i)))
            if (i + 1) % 100 == 0:
                print(f"  dispatched {i+1}/{len(reqs)} inflight={inflight['n']} "
                      f"ok={stats['ok']} err={stats['errors']} "
                      f"elapsed={time.time()-start:.1f}s")
        await asyncio.gather(*tasks)
        wall = time.time() - start

    if logf:
        logf.close()
    print("=" * 56)
    print(f"done: {stats['ok']} ok, {stats['errors']} err, wall={wall:.1f}s, "
          f"achieved_rps={stats['ok']/wall:.1f}, peak_inflight={inflight['max']}")
    if stats["ttft"]:
        print(f"TTFT ms: p50={pct(stats['ttft'],50)*1000:.0f} "
              f"p90={pct(stats['ttft'],90)*1000:.0f} p99={pct(stats['ttft'],99)*1000:.0f}")
        print(f"E2E  ms: p50={pct(stats['lat'],50)*1000:.0f} "
              f"p90={pct(stats['lat'],90)*1000:.0f} p99={pct(stats['lat'],99)*1000:.0f}")


if __name__ == "__main__":
    asyncio.run(main())
