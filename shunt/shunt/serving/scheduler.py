"""The proxy-side request scheduler (RS, §3.2).

For each scheduling tick the proxy hands the batch of newly admitted requests to
:meth:`RequestScheduler.assign_batch`, which returns a target prefill DP worker
for each. Three modes back the evaluation:

- ``shunt``    -- the online LPT of Alg. 1 (the paper's RS);
- ``baseline`` -- vLLM's default round-robin dispatch;
- ``ors``      -- a precomputed offline-optimal assignment (the ORS baseline of
  §4.2), read in trace order from a rank file.

A request's compute time comes from the offline-profiled model (§3.1); the proxy
learns each request's reused-prefix and fresh-token counts from the driver (the
trace carries them) and falls back to the prompt length otherwise.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ..algorithms import lpt_schedule as py_lpt
from ..compute_model import ComputeModel

try:
    from .. import _native
    _HAVE_NATIVE = _native.HAVE_NATIVE
except Exception:  # pragma: no cover
    _HAVE_NATIVE = False


@dataclass
class SchedItem:
    req_id: str
    prefix_tokens: int
    fresh_tokens: int


class RequestScheduler:
    """Assigns admitted requests to prefill DP workers (§3.2, §4.1).

    Stateless across batches in ``shunt``/``baseline`` (each iteration's LPT
    starts fresh, matching Alg. 1 and the per-window oracle of §4.2); ``ors``
    walks the precomputed rank file in arrival order. ``last_decision_us`` holds
    the most recent batch's decision cost for the overhead report (§4.4).
    """

    def __init__(self, num_workers: int, model: ComputeModel,
                 mode: str = "shunt", ors_ranks: list[int] | None = None,
                 use_native: bool = True):
        self.W = num_workers
        self.model = model
        self.mode = mode
        self.ors_ranks = ors_ranks or []
        self.use_native = use_native and _HAVE_NATIVE
        self._rr = 0          # round-robin cursor (baseline)
        self._seen = 0        # global arrival index (ors)
        self.last_decision_us = 0.0

    def _compute_times(self, items: list[SchedItem]) -> list[float]:
        return [self.model.worker_compute_time(it.prefix_tokens, it.fresh_tokens)
                for it in items]

    def assign_batch(self, items: list[SchedItem]) -> dict[str, int]:
        t0 = time.perf_counter()
        if self.mode == "baseline":
            out = {}
            for it in items:
                out[it.req_id] = self._rr % self.W
                self._rr += 1
        elif self.mode == "ors":
            out = {}
            for it in items:
                idx = self._seen
                out[it.req_id] = (self.ors_ranks[idx] if idx < len(self.ors_ranks)
                                  else idx % self.W)
                self._seen += 1
        elif self.mode == "shunt":
            c = self._compute_times(items)
            if self.use_native:
                worker_of = _native.lpt_schedule(c, self.W)
            else:
                worker_of = py_lpt(c, self.W)
            out = {it.req_id: worker_of[i] for i, it in enumerate(items)}
        else:
            raise ValueError(f"unknown scheduling mode {self.mode!r}")
        self.last_decision_us = (time.perf_counter() - t0) * 1e6
        return out
