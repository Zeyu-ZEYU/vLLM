"""Per-iteration planning inside the prefill engine.

Once per iteration, before the forward pass, vLLM synchronizes all DP ranks
with one small all-reduce (``vllm/v1/worker/dp_utils.py``). Each rank writes
its own column. Shunt appends :data:`NUM_ROWS` rows to that tensor: the rank's
estimated worker-compute and attention times, fresh tokens, and inbound and
outbound KV bytes. After the all-reduce every rank holds the inputs of the
whole EP group and computes the same :class:`shunt.planner.GroupPlan` from
them. Elastic attention and the LMCache KV router then read that plan.
"""
from __future__ import annotations

import threading
import time

from .. import algorithms, native
from ..compute_model import ComputeModel
from ..config import ShuntConfig
from ..planner import GroupInputs, GroupPlan, plan_group
from . import settings as _settings
from .logs import LogSet

NUM_ROWS = 6  # tau_ns, attn_ns, fresh_tokens, kv_in_bytes, kv_out_bytes, num_reqs


class StepRuntime:
    """Process-wide planning state of one prefill DP rank."""

    def __init__(self, s: _settings.Settings):
        self.settings = s
        self.cfg = ShuntConfig.from_json(s.config_path)
        self.model = self.cfg.model
        self.deploy = self.cfg.deploy
        self.opts = self.cfg.options
        self.cm = ComputeModel.from_profile(self.cfg.compute_profile, self.model,
                                            self.deploy)
        if s.planner == "python":
            self.impl = algorithms
        else:
            native.require()
            self.impl = native
        self.rank = 0
        self.seq = 0                 # plan sequence number, identical on all ranks
        self.plan: GroupPlan | None = None
        self._pending: list[int] | None = None
        self._pending_reqs: list[tuple] = []
        self.step_reqs: list[tuple] = []
        self.step_t0 = 0.0
        self.logs = LogSet(None, 0)
        self._listeners: list = []
        self._lock = threading.Lock()

    # --- setup ------------------------------------------------------------------

    def set_rank(self, rank: int, group_size: int) -> None:
        if group_size != self.deploy.ep_group_workers:
            raise ValueError(
                f"Shunt config says ep_group_workers={self.deploy.ep_group_workers} "
                f"but the engine runs {group_size} DP ranks")
        self.rank = rank
        self.logs = LogSet(self.settings.log_dir, rank)

    def add_listener(self, fn) -> None:
        """``fn(plan, seq)`` is called whenever a new plan is installed."""
        self._listeners.append(fn)

    # --- per-iteration inputs ---------------------------------------------------

    def begin_step(self, reqs: list[tuple[str, int, int, int, int]]) -> None:
        """Record this rank's scheduled requests for the coming iteration.

        Each entry is ``(req_id, prefix_tokens, fresh_tokens, kv_in_tokens,
        kv_out_tokens)``: the context already computed before this iteration,
        the tokens computed now, and the KV tokens moved in and out.
        """
        cm = self.cm
        tau = attn = 0.0
        fresh = kin = kout = 0
        for _, p, n, ti, to in reqs:
            tau += cm.worker_time(p, n)
            attn += cm.attention_time(p, n)
            fresh += n
            kin += ti
            kout += to
        self._pending = [int(tau * 1e9), int(attn * 1e9), fresh,
                         int(cm.kv_bytes(kin)), int(cm.kv_bytes(kout)), len(reqs)]
        self._pending_reqs = reqs

    def dp_column(self) -> list[int]:
        """This rank's contribution to the appended all-reduce rows."""
        return self._pending if self._pending is not None else [0] * NUM_ROWS

    def on_dp_rows(self, rows: list[list[int]]) -> None:
        """Install the plan derived from the all-reduced rows (NUM_ROWS x G)."""
        self.seq += 1
        G = self.deploy.ep_group_workers
        if len(rows[0]) != G:
            raise ValueError(f"expected {G} DP ranks, got {len(rows[0])}")
        inp = GroupInputs(
            tau_wk=[x * 1e-9 for x in rows[0]],
            attn=[x * 1e-9 for x in rows[1]],
            fresh=[int(x) for x in rows[2]],
            v_in=[float(x) for x in rows[3]],
            v_out=[float(x) for x in rows[4]],
        )
        t0 = time.perf_counter()
        plan = plan_group(inp, self.cm, self.deploy, self.opts, self.impl) \
            if sum(inp.fresh) > 0 else None
        plan_us = (time.perf_counter() - t0) * 1e6
        with self._lock:
            self.plan = plan
            self.step_reqs = self._pending_reqs
            self._pending = None
            self._pending_reqs = []
            self.step_t0 = time.time()
        for fn in self._listeners:
            fn(plan, self.seq)
        if plan is not None and self.logs.enabled():
            self._log_plan(inp, plan, rows[5], plan_us)

    # --- logging ----------------------------------------------------------------

    def _log_plan(self, inp: GroupInputs, plan: GroupPlan, nreqs, plan_us: float
                  ) -> None:
        if self.rank == 0:
            self.logs.write("plan", {
                "seq": self.seq, "t": self.step_t0,
                "tau_wk": inp.tau_wk, "attn": inp.attn, "fresh": inp.fresh,
                "v_in": inp.v_in, "v_out": inp.v_out, "num_reqs": list(nreqs),
                "moves": plan.moves, "tau_post": plan.tau_post,
                "group_mean": plan.group_mean, "tau_ex": plan.tau_ex,
                "t_cmp": plan.t_cmp, "t_a2a": plan.t_a2a,
                "b_be": plan.b_be, "b_fe": plan.b_fe,
                "inbound": [_alloc(a) for a in plan.inbound],
                "outbound": [_alloc(a) for a in plan.outbound],
                "plan_us": plan_us,
            })
        self.logs.write("reqs", {
            "seq": self.seq, "t": self.step_t0,
            "reqs": [list(r) for r in self.step_reqs],
        })


def _alloc(a) -> dict:
    return {"own": a.own, "borrow": {str(k): v for k, v in a.borrow.items()},
            "frontend": a.frontend}


_RUNTIME: StepRuntime | None = None
_INIT_LOCK = threading.Lock()


def get() -> StepRuntime | None:
    """The process's runtime, or ``None`` when Shunt is inactive here."""
    global _RUNTIME
    if _RUNTIME is None:
        s = _settings.load()
        if not s.active:
            return None
        with _INIT_LOCK:
            if _RUNTIME is None:
                _RUNTIME = StepRuntime(s)
    return _RUNTIME
