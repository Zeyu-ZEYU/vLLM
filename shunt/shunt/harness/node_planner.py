"""Publish the node-local plan each iteration (EAP + KVLB).

The proxy chooses each request's worker (RS / ORS / round-robin); this turns that
assignment into the two node-local plans the rest of the stack consumes:

- KVLB: one ``kvlb_plan_w<rank>.json`` per worker, holding the per-NIC weights for
  inbound and outbound KV in the routing backend's child order (own NIC first,
  borrowable NICs next, frontend last). The LMCache routing backend reads it.
- EAP: ``eap_heads.json`` holding each worker's post-split query-head count, which
  the model's elastic-attention path reads (§3.3).

In the deployment the proxy calls :func:`publish` once per scheduling tick over
that tick's batch, so the plans the engines apply match the requests in flight.
"""
from __future__ import annotations

import json
import os
import tempfile

from ..compute_model import ComputeModel
from ..config import TESTBED, TestbedConfig
from ..planner import plan_from_assignment
from ..types import DirectionPlan, Request


def _child_weights(dp: DirectionPlan, w: int, wpn: int) -> list[float]:
    """Per-NIC bytes for worker ``w`` in the routing backend's child order.

    The routing backend rotates the node's backend NICs so this worker's own NIC
    (rank ``w % wpn``) is first, then the rest in order, then the frontend. We
    emit the byte budgets in exactly that order.
    """
    own = w % wpn
    node_off = w - own
    weights = [dp.backend.get(node_off + (own + j) % wpn, 0.0) for j in range(wpn)]
    weights.append(dp.frontend)
    return weights


def _atomic_write(path: str, obj) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)   # atomic, so the reader never sees a half-written plan


def publish(requests: list[Request], worker_of: list[int], model: ComputeModel,
            plan_dir: str = "/tmp/shunt", testbed: TestbedConfig = TESTBED,
            theta: float = 1.5, enable_eap: bool = True, enable_kvlb: bool = True
            ) -> None:
    """Compute EAP+KVLB for this iteration's assignment and write the plan files."""
    plan = plan_from_assignment(requests, worker_of, model, testbed, theta,
                                enable_eap, enable_kvlb)
    wpn = testbed.workers_per_node
    for w in range(testbed.ep_group_workers):
        _atomic_write(os.path.join(plan_dir, f"kvlb_plan_w{w}.json"), {
            "inbound": _child_weights(plan.inbound[w], w, wpn),
            "outbound": _child_weights(plan.outbound[w], w, wpn),
        })
    _atomic_write(os.path.join(plan_dir, "eap_heads.json"),
                  {"head_counts": plan.head_counts, "t_cmp": plan.t_cmp})
