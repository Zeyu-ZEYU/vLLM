"""Replay the Qwen production trace against the Shunt proxy and record TTFT.

Reconstructs each request as a token-id prompt so the trace's prefix reuse
actually hits the engine's prefix cache: chunk hashes seed deterministic tokens,
so two requests that share a 16-token chunk hash send identical leading tokens
(§2.3). Each request also carries its reused-prefix and fresh-token counts in
``X-Shunt-*`` headers so the proxy's scheduler can price it (§3.2).

Two load models:

- ``--rate R``   open loop -- Poisson arrivals at R req/s (sub-saturated; the
  load sweeps of Figs 16b/17b vary R);
- ``--concurrency C``  closed loop -- C in-flight requests (saturating; the TTFT
  distributions of Figs 16a/17a).

Writes one JSON line per request with send/first-token timestamps and TTFT.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time

from ..trace import TraceRecord, load_trace

BLOCK = 16


def _stable_seed(h) -> int:
    """Deterministic across processes (unlike builtin hash on strings)."""
    if isinstance(h, int):
        return h & 0xFFFFFFFF
    return int(hashlib.md5(str(h).encode()).hexdigest()[:8], 16)


def build_token_prompt(record: TraceRecord, hash_ids: list, vocab: int = 150000
                       ) -> list[int]:
    """Deterministic token ids for a request, reuse-preserving.

    Chunk ``i`` is filled from ``hash_ids[i]`` so identical chunk hashes produce
    identical tokens (and thus a prefix-cache hit); the tail is padded past the
    last hash. Length equals the record's input length.
    """
    toks: list[int] = []
    for h in hash_ids:
        rng = random.Random(_stable_seed(h))
        toks.extend(rng.randrange(vocab) for _ in range(BLOCK))
    if len(toks) < record.input_length:
        rng = random.Random(record.req_id)
        toks.extend(rng.randrange(vocab)
                    for _ in range(record.input_length - len(toks)))
    return toks[:record.input_length]


def _payload(record: TraceRecord, tokens: list[int]) -> dict:
    out_len = max(1, record.output_length)
    return {
        "model": "shunt",
        "prompt": tokens,
        "max_tokens": out_len,
        "stream": True,
        "request_id": record.req_id,
        "temperature": 0.0,
    }


async def _one(session, url, record, hash_ids, results, sem=None):
    import aiohttp  # testbed dependency

    if sem:
        await sem.acquire()
    tokens = build_token_prompt(record, hash_ids)
    headers = {
        "X-Shunt-Prefix-Tokens": str(record.prefix_tokens),
        "X-Shunt-Fresh-Tokens": str(record.fresh_tokens),
    }
    t_send = time.time()
    t_first = None
    try:
        async with session.post(url + "/v1/completions", json=_payload(record, tokens),
                                headers=headers) as resp:
            async for _chunk in resp.content.iter_any():
                if t_first is None:
                    t_first = time.time()
                    break  # TTFT only; drain in background not needed here
    except Exception as e:  # noqa: BLE001
        results.append({"req_id": record.req_id, "error": str(e)})
        if sem:
            sem.release()
        return
    ttft = (t_first - t_send) if t_first else None
    results.append({"req_id": record.req_id, "t_send": t_send,
                    "t_first": t_first, "ttft": ttft,
                    "input_length": record.input_length,
                    "prefix_tokens": record.prefix_tokens})
    if sem:
        sem.release()


async def run(args) -> None:
    import aiohttp

    records = load_trace(args.trace, limit=args.limit)
    if args.start_index:
        records = records[args.start_index:]
    raw_hashes = _load_hashes(args.trace, len(records) + (args.start_index or 0))
    raw_hashes = raw_hashes[args.start_index or 0:]
    results: list[dict] = []
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if args.concurrency:
            sem = asyncio.Semaphore(args.concurrency)
            tasks = [asyncio.create_task(
                _one(session, args.url, r, h, results, sem))
                for r, h in zip(records, raw_hashes)]
            await asyncio.gather(*tasks)
        else:
            tasks = []
            rng = random.Random(0)
            for r, h in zip(records, raw_hashes):
                tasks.append(asyncio.create_task(
                    _one(session, args.url, r, h, results)))
                await asyncio.sleep(rng.expovariate(args.rate))  # Poisson gaps
            await asyncio.gather(*tasks)

    with open(args.out, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    oks = [r["ttft"] for r in results if r.get("ttft")]
    if oks:
        oks.sort()
        p = lambda q: oks[min(len(oks) - 1, int(q * len(oks)))]  # noqa: E731
        print(f"{len(oks)} ok; TTFT s  p50={p(.5):.3f} p90={p(.9):.3f} "
              f"p99={p(.99):.3f}")


def _load_hashes(path: str, n: int) -> list[list]:
    import lzma
    op = lzma.open if path.endswith(".xz") else open
    out = []
    with op(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            out.append(json.loads(line).get("hash_ids", []) or [])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay the trace and record TTFT")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", type=float, default=8.0, help="open-loop req/s")
    ap.add_argument("--concurrency", type=int, default=0,
                    help="closed-loop in-flight count (overrides --rate)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
