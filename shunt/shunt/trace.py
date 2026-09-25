"""Load the Qwen serving traces (``qwen_traceB``, ``qwen_coder``, ...).

Each JSON line has ``chat_id``, ``timestamp`` (seconds), ``input_length``,
``output_length``, and ``hash_ids``: one hash per 16-token block of the prompt.
Identical leading hashes mean a shared prefix. :func:`load_trace` also splits
each prompt into reused-prefix and fresh tokens by replaying an unbounded
prefix cache over the trace in arrival order; the proxy and the oracle use this
split to price requests.
"""
from __future__ import annotations

import json
import lzma
from dataclasses import dataclass, field

BLOCK = 16


@dataclass
class TraceRecord:
    index: int
    req_id: str
    arrival: float
    input_length: int
    output_length: int
    hash_ids: list = field(default_factory=list)
    prefix_tokens: int = 0
    fresh_tokens: int = 0


def open_text(path: str):
    return lzma.open(path, "rt") if path.endswith(".xz") else open(path, "rt")


def load_trace(path: str, limit: int | None = None, start: int = 0
               ) -> list[TraceRecord]:
    """Parse the trace and split every prompt into prefix and fresh tokens.

    The prefix of a request is its run of leading blocks already seen in an
    earlier request; at least one token is always fresh. ``start`` skips that
    many leading records after the reuse replay has seen them.
    """
    seen: set = set()
    out: list[TraceRecord] = []
    with open_text(path) as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= start + limit:
                break
            r = json.loads(line)
            L = int(r["input_length"])
            hashes = r.get("hash_ids") or []
            matched = 0
            for h in hashes:
                if h in seen:
                    matched += 1
                else:
                    break
            seen.update(hashes)
            if idx < start:
                continue
            fresh = max(1, L - min(matched * BLOCK, L))
            out.append(TraceRecord(
                index=idx,
                req_id=str(r.get("chat_id", idx)),
                arrival=float(r.get("timestamp", 0.0)),
                input_length=L,
                output_length=int(r.get("output_length", 0)),
                hash_ids=list(hashes),
                prefix_tokens=L - fresh,
                fresh_tokens=fresh,
            ))
    return out


def iter_windows(records: list[TraceRecord], window: int):
    """Consecutive windows of ``window`` records."""
    for i in range(0, len(records), window):
        chunk = records[i:i + window]
        if chunk:
            yield chunk
