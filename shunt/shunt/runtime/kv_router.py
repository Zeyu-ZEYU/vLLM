"""Chunk-to-port routing of KV transfers from the KVLB plan (used by LMCache).

The plan gives each worker, per direction, the bytes per layer that go on its
own backend port, on each borrowed port of its node, and on the frontend. The
router turns that into a per-chunk decision: the first time it sees a chunk in
an iteration, it takes the first port in plan order (own, borrowed nearest
first, frontend) that still has room for the chunk, and every layer of that
chunk then uses the same port. Chunks beyond the plan's total go to the link
the plan uses that would finish earliest.
"""
from __future__ import annotations

import threading

from . import state as _state

FRONTEND = -1
IN, OUT = "in", "out"


class KVRouter:
    def __init__(self, rt: _state.StepRuntime):
        self.rt = rt
        self.bw_port = rt.deploy.bw_port
        self.bw_fe = rt.deploy.bw_fe
        self.seq = -1
        self._lock = threading.Lock()
        self._ledger: dict[str, list[list]] = {IN: [], OUT: []}
        self._decided: dict[str, dict] = {IN: {}, OUT: {}}
        self._routed: dict[str, dict[int, float]] = {IN: {}, OUT: {}}
        self._active: dict[str, bool] = {IN: False, OUT: False}
        rt.add_listener(self._on_plan)

    @property
    def own_port(self) -> int:
        return self.rt.rank % self.rt.deploy.workers_per_node

    def _on_plan(self, plan, seq: int) -> None:
        with self._lock:
            self._flush()
            self.seq = seq
            for d in (IN, OUT):
                self._decided[d] = {}
                self._routed[d] = {}
                if plan is None:
                    self._ledger[d] = []
                    self._active[d] = False
                    continue
                a = (plan.inbound if d == IN else plan.outbound)[self.rt.rank]
                ledger = [[self.own_port, a.own]]
                ledger += [[int(p), b] for p, b in a.borrow.items()]
                if a.frontend > 0:
                    ledger.append([FRONTEND, a.frontend])
                self._ledger[d] = ledger
                self._active[d] = True

    def route(self, direction: str, chunk_key, nbytes: float) -> int:
        """Port for this chunk: a node-local port index or :data:`FRONTEND`."""
        with self._lock:
            dec = self._decided[direction]
            port = dec.get(chunk_key)
            if port is not None:
                return port
            if not self._active[direction]:
                port = self.own_port
            else:
                ledger = self._ledger[direction]
                for entry in ledger:
                    if entry[1] >= 0.5 * nbytes:
                        entry[1] -= nbytes
                        port = entry[0]
                        break
                if port is None:
                    routed = self._routed[direction]
                    port = min(
                        (e[0] for e in ledger),
                        key=lambda p: (routed.get(p, 0.0) + nbytes)
                        / (self.bw_fe if p == FRONTEND else self.bw_port))
            dec[chunk_key] = port
            r = self._routed[direction]
            r[port] = r.get(port, 0.0) + nbytes
            return port

    def _flush(self) -> None:
        if self.seq < 0 or not self.rt.logs.enabled():
            return
        if not (self._routed[IN] or self._routed[OUT]):
            return
        self.rt.logs.write("route", {
            "seq": self.seq,
            "in": {str(k): v for k, v in self._routed[IN].items()},
            "out": {str(k): v for k, v in self._routed[OUT].items()},
        })


_ROUTER: KVRouter | None = None
_LOCK = threading.Lock()


def get() -> KVRouter | None:
    """The process's router, or ``None`` when the Shunt runtime is inactive."""
    global _ROUTER
    if _ROUTER is None:
        rt = _state.get()
        if rt is None:
            return None
        with _LOCK:
            if _ROUTER is None:
                _ROUTER = KVRouter(rt)
    return _ROUTER
