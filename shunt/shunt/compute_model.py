"""The two compute-time functions the planner uses, plus the KV and A2A volumes.

The worker-compute function maps a request's reused-prefix tokens ``P`` and
fresh tokens ``N`` to the per-layer time it adds on its DP worker: the QKV and
output projections, the attention itself, and the router gate. The first two
are the head-splittable attention time. The expert-compute function spreads the
EP group's total fresh tokens evenly over the experts and returns one per-layer
expert time shared by all workers.

Both are linear in shape features whose coefficients come from a profile
measured on the target GPU (``python -m shunt.profiling.compute``). Without a
profile, nominal coefficients derived from FLOP counts are used, which is only
suitable for testing the plumbing.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .config import DeploySpec, ModelSpec

# Nominal dense BF16 rate used only for the FLOP-derived default coefficients.
NOMINAL_FLOPS = 1.0e14


def attention_pairs(prefix: float, fresh: float) -> float:
    """(query, key) pairs a request attends over: every fresh token sees the
    prefix plus the causal triangle of the fresh tokens."""
    return fresh * prefix + fresh * (fresh + 1.0) / 2.0


@dataclass
class ComputeCoefficients:
    """Per-layer seconds per unit of each shape feature."""

    c_tok: float     # per fresh token: QKV and output projections (splittable)
    c_pair: float    # per (query, key) pair: attention kernel (splittable)
    c_gate: float    # per fresh token: router gate (not splittable)
    c_expert: float  # per token-expert assignment on one GPU
    c_expert0: float = 0.0  # fixed per-layer expert-phase time

    @classmethod
    def nominal(cls, m: ModelSpec) -> "ComputeCoefficients":
        s = 1.0 / NOMINAL_FLOPS
        proj = 2.0 * m.hidden * (m.q_dim + 2 * m.kv_dim) + 2.0 * m.q_dim * m.hidden
        return cls(
            c_tok=proj * s,
            c_pair=4.0 * m.q_dim * s,
            c_gate=2.0 * m.hidden * m.num_experts * s,
            c_expert=6.0 * m.hidden * m.moe_intermediate * s,
        )


class ComputeModel:
    """Per-layer compute, KV, and A2A estimates for one model and deployment."""

    def __init__(self, model: ModelSpec, deploy: DeploySpec,
                 coef: ComputeCoefficients | None = None):
        self.model = model
        self.deploy = deploy
        self.coef = coef or ComputeCoefficients.nominal(model)

    @classmethod
    def from_profile(cls, path: str | None, model: ModelSpec,
                     deploy: DeploySpec) -> "ComputeModel":
        """Load coefficients written by ``shunt.profiling.compute``."""
        if not path:
            return cls(model, deploy)
        with open(path) as f:
            prof = json.load(f)
        base = asdict(ComputeCoefficients.nominal(model))
        base.update({k: float(v) for k, v in prof.get("coefficients", prof).items()
                     if k in base})
        return cls(model, deploy, ComputeCoefficients(**base))

    # --- worker compute (per request, per layer) --------------------------------

    def attention_time(self, prefix: float, fresh: float) -> float:
        """Head-splittable part: projections plus attention."""
        c = self.coef
        return c.c_tok * fresh + c.c_pair * attention_pairs(prefix, fresh)

    def worker_time(self, prefix: float, fresh: float) -> float:
        """Worker-compute time a request adds: attention time plus its gate."""
        return self.attention_time(prefix, fresh) + self.coef.c_gate * fresh

    # --- shared expert term and A2A lower bound (per layer) ---------------------

    def expert_time(self, total_fresh: float) -> float:
        """Expert time under an even spread of the group's fresh tokens."""
        m, E = self.model, self.deploy.ep_group_workers
        return self.coef.c_expert0 + self.coef.c_expert * total_fresh * m.top_k / E

    def a2a_bytes_per_port(self, total_fresh: float) -> float:
        """Cross-node dispatch plus combine bytes one port sends per direction
        under an even spread of token-expert assignments."""
        m, d = self.model, self.deploy
        E = d.ep_group_workers
        cross = max(0.0, (E - d.workers_per_node) / E)
        per_port_assign = total_fresh * m.top_k / E
        return 2.0 * per_port_assign * cross * m.hidden * m.dtype_bytes

    def a2a_time(self, total_fresh: float) -> float:
        """Shortest possible A2A: the per-port cross-node volume at port speed."""
        return self.a2a_bytes_per_port(total_fresh) / self.deploy.bw_port

    # --- KV volumes (per layer) ------------------------------------------------

    def kv_bytes(self, tokens: float) -> float:
        return tokens * self.model.kv_bytes_per_token
