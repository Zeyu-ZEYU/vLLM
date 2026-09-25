"""Per-layer phase timing of the prefill forward, written once per iteration.

With ``SHUNT_TIMING=1`` every rank records CUDA events around the phases of
each layer: ``attn`` (the attention block, including elastic-attention
transfers), ``gate`` (the router), ``dispatch`` and ``combine`` (the A2A), and
``experts`` (the expert FFNs). Host-side waits for inbound KV (``kv_wait``) are
recorded per layer too. ``SHUNT_TIMING=2`` additionally keeps every phase's
start and end offset from the start of the iteration. Records are read back
without blocking once the GPU has passed them and are appended to
``step.rank<r>.jsonl``.
"""
from __future__ import annotations

import atexit
import threading
import time
from collections import deque
from contextlib import contextmanager

import torch

from . import state as _state


class Timing:
    def __init__(self, rt: _state.StepRuntime, level: int):
        self.rt = rt
        self.level = level
        self._tl = threading.local()
        self._lock = threading.Lock()
        self._cur: dict | None = None
        self._pending: deque = deque()
        rt.add_listener(self._on_plan)
        atexit.register(self.flush)

    # --- step boundaries --------------------------------------------------------

    def _on_plan(self, plan, seq: int) -> None:
        with self._lock:
            if self._cur is not None:
                self._pending.append(self._cur)
            ref = torch.cuda.Event(enable_timing=True)
            ref.record()
            self._cur = {"seq": seq, "t": time.time(), "ref": ref,
                         "nreq": len(self.rt.step_reqs),
                         "fresh": sum(r[2] for r in self.rt.step_reqs),
                         "events": [], "host": []}
        self._harvest(block=False)

    # --- recording ---------------------------------------------------------------

    def set_layer(self, layer: int) -> None:
        self._tl.layer = layer

    def _ubatch(self) -> int:
        try:
            from vllm.v1.worker.ubatching import dbo_current_ubatch_id
            return dbo_current_ubatch_id()
        except Exception:
            return 0

    @contextmanager
    def phase(self, name: str):
        cur = self._cur
        if cur is None:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            cur["events"].append((getattr(self._tl, "layer", -1), self._ubatch(),
                                  name, start, end))

    def add_host(self, name: str, seconds: float) -> None:
        cur = self._cur
        if cur is not None:
            cur["host"].append((getattr(self._tl, "layer", -1), name, seconds))

    # --- read back ---------------------------------------------------------------

    def _harvest(self, block: bool) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    return
                rec = self._pending[0]
                if not block and not all(ev[4].query() for ev in rec["events"]):
                    return
                self._pending.popleft()
            self._write(rec, block)

    def _write(self, rec: dict, block: bool) -> None:
        if not self.rt.logs.enabled():
            return
        num_layers = self.rt.model.num_layers
        phases: dict[str, list[float]] = {}
        timeline = []
        ref = rec["ref"]
        for layer, ub, name, s, e in rec["events"]:
            if block:
                e.synchronize()
            dur = s.elapsed_time(e)
            arr = phases.setdefault(name, [0.0] * num_layers)
            if 0 <= layer < num_layers:
                arr[layer] += dur
            if self.level >= 2:
                t0 = ref.elapsed_time(s)
                timeline.append([layer, ub, name, round(t0, 4), round(t0 + dur, 4)])
        host: dict[str, list[float]] = {}
        for layer, name, sec in rec["host"]:
            arr = host.setdefault(name, [0.0] * num_layers)
            if 0 <= layer < num_layers:
                arr[layer] += sec * 1e3
        out = {"seq": rec["seq"], "t": rec["t"], "nreq": rec["nreq"],
               "fresh": rec["fresh"], "phases_ms": phases, "host_ms": host}
        if timeline:
            out["timeline_ms"] = timeline
        self.rt.logs.write("step", out)

    def flush(self) -> None:
        with self._lock:
            if self._cur is not None:
                self._pending.append(self._cur)
                self._cur = None
        try:
            self._harvest(block=True)
        except Exception:
            pass


_TIMING: Timing | None = None
_DISABLED = False
_LOCK = threading.Lock()


def get() -> Timing | None:
    """The process's timer, or ``None`` when timing is off."""
    global _TIMING, _DISABLED
    if _TIMING is None and not _DISABLED:
        rt = _state.get()
        if rt is None or rt.settings.timing <= 0:
            _DISABLED = True
            return None
        with _LOCK:
            if _TIMING is None:
                _TIMING = Timing(rt, rt.settings.timing)
    return _TIMING


@contextmanager
def phase(name: str):
    t = get()
    if t is None:
        yield
    else:
        with t.phase(name):
            yield
