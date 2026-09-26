"""Replay a Qwen serving trace against the proxy and log every request.

Prompts are built as token ids so that prefix reuse in the trace becomes real
prefix reuse in the engines: every 16-token block of a prompt is generated from
its hash id, so requests that share leading hash ids share leading tokens. Each
request asks for exactly its trace output length (``ignore_eos``).

Two load protocols:

- ``--concurrency C``: closed loop, C requests in flight at all times.
- ``--rate R``: open loop, Poisson arrivals at R requests/s. Timestamps are
  kept per request, so the time to first token can be measured from the
  scheduled arrival (queueing included).

Output: one JSON line per request with its trace index, scheduled, send,
first-token and last-token times, the requested output tokens
(``max_tokens``, generated exactly because of ``ignore_eos``), and the number
of streamed chunks.

Example::

    python -m shunt.harness.replay --trace traces/qwen_traceB_blksz_16.jsonl \\
        --url http://proxy:8000 --concurrency 512 --num-requests 20000 \\
        --out results/run/requests.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import json
import random
import time

from ..trace import BLOCK, TraceRecord, load_trace


@functools.lru_cache(maxsize=1 << 18)
def _block_tokens(h, lo: int, hi: int) -> tuple[int, ...]:
    rng = random.Random(f"blk-{h}")
    return tuple(rng.randrange(lo, hi) for _ in range(BLOCK))


def build_prompt(rec: TraceRecord, lo: int, hi: int, reuse: bool = True) -> list[int]:
    """Token ids of a request; identical hash ids give identical blocks.

    With ``reuse=False`` every prompt is unique (no prefix is shared).
    """
    toks: list[int] = []
    for h in rec.hash_ids:
        toks.extend(_block_tokens(h if reuse else f"{rec.index}-{h}", lo, hi))
        if len(toks) >= rec.input_length:
            break
    if len(toks) < rec.input_length:
        rng = random.Random(f"tail-{rec.index}")
        toks.extend(rng.randrange(lo, hi) for _ in range(rec.input_length - len(toks)))
    return toks[:rec.input_length]


async def _one(session, url: str, model: str, rec: TraceRecord, prompt: list[int],
               t_sched: float, max_out: int | None) -> dict:
    out_len = max(1, rec.output_length if max_out is None else min(rec.output_length,
                                                                    max_out))
    body = {"model": model, "prompt": prompt, "max_tokens": out_len, "stream": True,
            "temperature": 0.0, "ignore_eos": True}
    headers = {"X-Shunt-Request-Id": str(rec.index),
               "X-Shunt-Prefix-Tokens": str(rec.prefix_tokens),
               "X-Shunt-Fresh-Tokens": str(rec.fresh_tokens)}
    row = {"idx": rec.index, "input_len": rec.input_length, "prefix": rec.prefix_tokens,
           "max_tokens": out_len, "t_sched": t_sched, "t_send": time.time()}
    n = 0
    try:
        async with session.post(url + "/v1/completions", json=body,
                                headers=headers) as r:
            r.raise_for_status()
            async for line in r.content:
                if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                    continue
                now = time.time()
                if n == 0:
                    row["t_first"] = now
                row["t_last"] = now
                n += 1
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"
    row["n_chunks"] = n
    return row


async def run(args) -> None:
    import aiohttp

    recs = load_trace(args.trace, limit=args.num_requests, start=args.start)
    lo, hi = args.token_range
    prompts = [build_prompt(r, lo, hi, reuse=not args.no_reuse) for r in recs]
    out = open(args.out, "w", buffering=1)
    done = 0
    t0 = time.time()
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=args.timeout)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        async def finish(row):
            nonlocal done
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
            done += 1
            if args.progress and done % args.progress == 0:
                print(f"{done}/{len(recs)} done, {time.time() - t0:.0f}s", flush=True)

        if args.concurrency:
            it = iter(range(len(recs)))

            async def worker():
                for i in it:
                    if args.duration and time.time() - t0 > args.duration:
                        return
                    await finish(await _one(session, args.url, args.model, recs[i],
                                            prompts[i], time.time(), args.max_output))

            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        else:
            rng = random.Random(args.seed)
            tasks = []
            t_next = time.time()
            for i in range(len(recs)):
                if args.duration and t_next - t0 > args.duration:
                    break
                delay = t_next - time.time()
                if delay > 0:
                    await asyncio.sleep(delay)

                async def go(i=i, t_sched=t_next):
                    await finish(await _one(session, args.url, args.model, recs[i],
                                            prompts[i], t_sched, args.max_output))

                tasks.append(asyncio.create_task(go()))
                t_next += rng.expovariate(args.rate)
            await asyncio.gather(*tasks)
    out.close()
    meta = {"trace": args.trace, "start": args.start, "num_requests": len(recs),
            "concurrency": args.concurrency, "rate": args.rate, "seed": args.seed,
            "reuse": not args.no_reuse,
            "t_begin": t0, "t_end": time.time()}
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a Qwen trace against the proxy")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=0,
                    help="closed loop with this many requests in flight")
    ap.add_argument("--rate", type=float, default=0.0,
                    help="open loop: Poisson arrivals per second")
    ap.add_argument("--start", type=int, default=0, help="first trace record")
    ap.add_argument("--num-requests", type=int, default=None)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop issuing new requests after this many seconds")
    ap.add_argument("--max-output", type=int, default=None,
                    help="cap on output tokens per request")
    ap.add_argument("--token-range", type=int, nargs=2, default=(1000, 100000),
                    help="token ids are drawn from [lo, hi)")
    ap.add_argument("--no-reuse", action="store_true",
                    help="make every prompt unique (no prefix reuse)")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", type=int, default=1000)
    args = ap.parse_args()
    if not args.concurrency and args.rate <= 0:
        ap.error("set --concurrency (closed loop) or --rate (open loop)")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
