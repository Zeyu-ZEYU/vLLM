"""Load the open Qwen production trace and reconstruct per-request (P, N).

The trace (``qwen_traceB``, the business-serving subset of the public Qwen
serving cluster trace) gives each request an arrival timestamp, input/output
lengths, and a sequence of 16-token chunk hashes that mark prefix-KV reuse. We
replay a global prefix cache over the whole trace to split each request's input
into reused-prefix tokens P and freshly prefilled tokens N (§2.3, §3.2).
"""
from __future__ import annotations

import json
import lzma
from dataclasses import dataclass

from .types import Request

BLOCK = 16  # trace chunk-hash granularity (tokens)


@dataclass
class TraceRecord:
    req_id: str
    arrival: float
    input_length: int
    output_length: int
    prefix_tokens: int
    fresh_tokens: int


def _open(path: str):
    return lzma.open(path, "rt") if path.endswith(".xz") else open(path, "rt")


def load_trace(path: str, limit: int | None = None) -> list[TraceRecord]:
    """Parse the trace and reconstruct (P, N) with a global prefix-reuse sim.

    A request's leading chunk hashes that have been *seen before* count as reused
    prefix; the first unseen hash ends the reuse run. All of the request's hashes
    are then added to the seen set, so later requests can reuse them.
    """
    seen: set = set()
    out: list[TraceRecord] = []
    with _open(path) as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            r = json.loads(line)
            L = int(r["input_length"])
            hashes = r.get("hash_ids", []) or []
            matched = 0
            for h in hashes:
                if h in seen:
                    matched += 1
                else:
                    break
            seen.update(hashes)
            p = min(matched * BLOCK, L)
            out.append(TraceRecord(
                req_id=str(r.get("req_id", idx)),
                arrival=float(r.get("timestamp", r.get("arrival", idx))),
                input_length=L,
                output_length=int(r.get("output_length", 0)),
                prefix_tokens=p,
                fresh_tokens=L - p,
            ))
    return out


def to_requests(records: list[TraceRecord]) -> list[Request]:
    return [Request(req_id=r.req_id, prefix_tokens=r.prefix_tokens,
                    fresh_tokens=r.fresh_tokens) for r in records]


def iter_windows(records: list[TraceRecord], window: int):
    """Yield consecutive windows of ``window`` records (one prefill iteration's
    worth of requests for the offline imbalance analysis, §2.3)."""
    for i in range(0, len(records), window):
        chunk = records[i:i + window]
        if chunk:
            yield chunk
