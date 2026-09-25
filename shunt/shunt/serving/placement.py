"""Placement of requests onto prefill DP ranks.

- ``rr``: round-robin, vLLM's default dispatch (Baseline).
- ``lpt``: Shunt's placement (RS). Requests that arrive within one tick are
  placed together with the LPT rule on their estimated worker-compute time.
- ``oracle``: a precomputed assignment per request (the ORS baseline), from
  ``python -m shunt.tools.oracle``; requests without one fall back to ``rr``.
- ``kva``: the rank holding the longest cached prefix; no cached prefix
  falls back to ``rr``.
- ``kva_lb``: like ``kva``, but when that rank's estimated in-flight load
  exceeds ``lb_factor`` times the mean, the least-loaded rank.
"""
from __future__ import annotations

import asyncio
import itertools
import time


class Placement:
    MODES = ("rr", "lpt", "oracle", "kva", "kva_lb")

    def __init__(self, mode: str, num_ranks: int, lpt_fn, tick_s: float = 0.002,
                 max_batch: int = 4096, oracle: dict | None = None, kv_index=None,
                 lb_factor: float = 1.5):
        if mode not in self.MODES:
            raise ValueError(f"placement must be one of {self.MODES}")
        if mode in ("kva", "kva_lb") and kv_index is None:
            raise ValueError(f"placement {mode} needs KV-event endpoints")
        self.mode = mode
        self.n = num_ranks
        self.lpt_fn = lpt_fn
        self.tick_s = tick_s
        self.max_batch = max_batch
        self.oracle = oracle or {}
        self.kv_index = kv_index
        self.lb_factor = lb_factor
        self._rr = itertools.cycle(range(num_ranks))
        self._queue: asyncio.Queue | None = None
        self._task = None
        self.load = [0.0] * num_ranks        # estimated in-flight prefill work
        self.decision_us: list[float] = []   # per LPT batch

    def start(self) -> None:
        if self.mode == "lpt":
            self._queue = asyncio.Queue()
            self._task = asyncio.get_event_loop().create_task(self._lpt_loop())

    async def place(self, req_key: str, est: float, tokens: list[int] | None) -> int:
        if self.mode == "rr":
            return next(self._rr)
        if self.mode == "oracle":
            r = self.oracle.get(req_key)
            return int(r) if r is not None else next(self._rr)
        if self.mode in ("kva", "kva_lb"):
            rank, hit = self.kv_index.longest_prefix(tokens or [])
            if rank < 0:
                return next(self._rr)
            if self.mode == "kva_lb":
                mean = sum(self.load) / self.n
                if mean > 0 and self.load[rank] > self.lb_factor * mean:
                    rank = min(range(self.n), key=lambda r: (self.load[r], r))
            return rank
        fut = asyncio.get_event_loop().create_future()
        await self._queue.put((est, fut))
        return await fut

    async def _lpt_loop(self) -> None:
        while True:
            first = await self._queue.get()
            await asyncio.sleep(self.tick_s)
            batch = [first]
            while not self._queue.empty() and len(batch) < self.max_batch:
                batch.append(self._queue.get_nowait())
            t0 = time.perf_counter()
            ranks = self.lpt_fn([e for e, _ in batch], self.n)
            self.decision_us.append((time.perf_counter() - t0) * 1e6)
            for (_, fut), r in zip(batch, ranks):
                if not fut.done():
                    fut.set_result(int(r))

    # in-flight load, used by kva_lb
    def started(self, rank: int, est: float) -> None:
        self.load[rank] += est

    def finished(self, rank: int, est: float) -> None:
        self.load[rank] = max(0.0, self.load[rank] - est)
