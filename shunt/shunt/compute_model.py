"""The offline-profiled compute-time model (§3.1).

Shunt plans each iteration before any layer runs. To do that it needs, for every
request, the *worker-compute time* it adds (the head-splittable attention work
plus the small non-splittable gate), and one shared *expert-compute time* for the
EP group. The paper obtains these from "two functions profiled offline on the
target GPUs."

We express both as analytical FLOP counts (exact in shape) scaled by measured
coefficients (exact in magnitude). With the default coefficients the magnitude is
a nominal H20-3e rate; load a real profile with :meth:`ComputeModel.from_profile`
to get wall-clock-accurate times. Imbalance ratios (max/mean), which drive the
motivation figures and the elastic-attention trigger, are invariant to the
overall scale, so the defaults already reproduce them.

Symbols follow §3.1: ``P`` = reused-prefix tokens, ``N`` = freshly prefilled
tokens of a request.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .config import DEFAULT_EFFECTIVE_FLOPS, MODEL, TESTBED, ModelConfig, TestbedConfig


def _proj_flops_per_token(m: ModelConfig) -> float:
    """QKV projection + output projection, per token (2x for multiply-add)."""
    qkv = 2.0 * m.hidden * (m.q_dim + 2 * m.kv_dim)
    out = 2.0 * m.q_dim * m.hidden
    return qkv + out


def _attn_flops_per_pair(m: ModelConfig) -> float:
    """Attention score + value aggregation, per (query token, context token)."""
    return 4.0 * m.q_dim


def _gate_flops_per_token(m: ModelConfig) -> float:
    """Router gate projection hidden -> num_experts, per token."""
    return 2.0 * m.hidden * m.num_experts


def _expert_flops_per_token(m: ModelConfig) -> float:
    """One token through one expert FFN (SwiGLU: gate, up, down)."""
    return 6.0 * m.hidden * m.moe_intermediate


def _attn_pairs(prefix: float, fresh: float) -> float:
    """Number of (query, context) pairs a request attends over.

    Each of its ``fresh`` query tokens attends to the ``prefix`` reused context,
    plus the causal triangle among the fresh tokens themselves.
    """
    return fresh * prefix + fresh * (fresh + 1.0) / 2.0


@dataclass
class ComputeModel:
    """Maps request shapes to compute times (§3.1).

    Coefficients are seconds-per-FLOP for each term. The split between the
    head-splittable part (projections + attention) and the non-splittable gate
    matters for elastic attention (§3.3), so the gate is tracked separately.
    """

    model: ModelConfig
    testbed: TestbedConfig
    c_proj: float    # s per token,        QKV+O projections (splittable)
    c_attn: float    # s per (q,ctx) pair,  attention itself  (splittable)
    c_gate: float    # s per token,         router gate       (NOT splittable)
    c_expert: float  # s per token-expert,  one expert FFN

    @classmethod
    def default(
        cls,
        model: ModelConfig = MODEL,
        testbed: TestbedConfig = TESTBED,
        effective_flops: float = DEFAULT_EFFECTIVE_FLOPS,
    ) -> "ComputeModel":
        s = 1.0 / effective_flops
        return cls(
            model=model,
            testbed=testbed,
            c_proj=_proj_flops_per_token(model) * s,
            c_attn=_attn_flops_per_pair(model) * s,
            c_gate=_gate_flops_per_token(model) * s,
            c_expert=_expert_flops_per_token(model) * s,
        )

    @classmethod
    def from_profile(cls, path: str, model: ModelConfig = MODEL,
                     testbed: TestbedConfig = TESTBED) -> "ComputeModel":
        """Load measured coefficients written by ``shunt.profiling`` on the GPU.

        The JSON holds the four per-FLOP coefficients (any omitted falls back to
        the nominal default), so a profile run only needs to fit what it measured.
        """
        base = cls.default(model, testbed)
        with open(path) as f:
            prof = json.load(f)
        for k in ("c_proj", "c_attn", "c_gate", "c_expert"):
            if k in prof:
                setattr(base, k, float(prof[k]))
        return base

    # --- per-request worker compute (§3.1, §3.2) -------------------------------

    def worker_compute_time(self, prefix: float, fresh: float) -> float:
        """Worker-compute time a request adds: attention work plus its gate."""
        pairs = _attn_pairs(prefix, fresh)
        return (self.c_proj + self.c_gate) * fresh + self.c_attn * pairs

    def attention_time(self, prefix: float, fresh: float) -> float:
        """The head-splittable part of the worker compute (projections + attn)."""
        pairs = _attn_pairs(prefix, fresh)
        return self.c_proj * fresh + self.c_attn * pairs

    # --- shared expert term and A2A estimate (§3.1, §3.4) ----------------------

    def expert_time(self, total_fresh: float, ep_width: int | None = None) -> float:
        """Shared expert-compute time under an even token spread (§3.1).

        Spreading the EP group's ``total_fresh`` tokens evenly over the experts
        gives each GPU ``total_fresh * top_k / ep_width`` token-experts, the
        shortest the expert phase can run, so this never overstates the window.
        """
        E = ep_width if ep_width is not None else self.testbed.ep_group_workers
        per_gpu_token_experts = total_fresh * self.model.top_k / E
        return per_gpu_token_experts * self.c_expert

    def a2a_time(self, total_fresh: float, ep_width: int | None = None) -> float:
        """Conservative per-port cross-node A2A time, for the frontend budget (§3.4).

        Even spread fixes the per-port cross-node A2A volume; dividing by the port
        bandwidth gives the estimate. Only expert destinations on another node
        cross the backend RNIC (same-node routing rides NVLink).
        """
        E = ep_width if ep_width is not None else self.testbed.ep_group_workers
        wpn = self.testbed.workers_per_node
        cross_frac = max(0.0, (E - wpn) / E)
        # dispatch + combine, both directions
        total_cross_bytes = (
            2.0 * total_fresh * self.model.top_k * cross_frac
            * self.model.hidden * self.model.dtype_bytes
        )
        per_port_bytes = total_cross_bytes / E
        return per_port_bytes / (self.testbed.bw_port_GBps * 1e9)

    # --- KV volumes (§3.1) -----------------------------------------------------

    def kv_bytes_inbound(self, prefix_tokens: float) -> float:
        """Inbound prefix-KV bytes for one worker, one layer."""
        return prefix_tokens * self.model.kv_bytes_per_token_per_layer

    def kv_bytes_outbound(self, fresh_tokens: float) -> float:
        """Outbound new-KV bytes for one worker, one layer."""
        return fresh_tokens * self.model.kv_bytes_per_token_per_layer
