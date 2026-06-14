"""Shunt: balancing compute and KV traffic without A2A contention.

The planner and analyses that drive the paper's experiments. Submodules:

- :mod:`shunt.config`        model + testbed constants (§2.2, §2.3)
- :mod:`shunt.compute_model` the offline-profiled compute-time model (§3.1)
- :mod:`shunt.algorithms`    Alg. 1 (RS), Alg. 2 (EAP), Alg. 3 (KVLB) + oracle
- :mod:`shunt.planner`       per-iteration plan: RS -> EAP -> KVLB (§3.1)
- :mod:`shunt.trace`         load the Qwen production trace, reconstruct (P, N)
- :mod:`shunt.analysis`      offline measurement analyses (Figs 5-8)
"""
from __future__ import annotations

from .algorithms import allocate_offload, balance_heads, lpt_schedule, optimal_oracle
from .compute_model import ComputeModel
from .config import MODEL, TESTBED, ModelConfig, TestbedConfig
from .planner import plan_from_assignment, plan_iteration
from .trace import load_trace, to_requests
from .types import DirectionPlan, IterationPlan, Request

__all__ = [
    "ComputeModel", "ModelConfig", "TestbedConfig", "MODEL", "TESTBED",
    "lpt_schedule", "optimal_oracle", "balance_heads", "allocate_offload",
    "plan_iteration", "plan_from_assignment", "load_trace", "to_requests",
    "Request", "IterationPlan", "DirectionPlan",
]
