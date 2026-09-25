"""The per-iteration plan for one EP group: elastic attention, then KVLB.

Every prefill rank computes the same plan from the same group-wide inputs
(estimated worker-compute and attention times, fresh tokens, and KV volumes of
all DP workers), so the ranks agree on it without further coordination. The
proxy's placement (RS) happens earlier and only shapes these inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import algorithms as ref
from .compute_model import ComputeModel
from .config import DeploySpec, PlanOptions


@dataclass
class GroupInputs:
    """Per-worker planner inputs for one iteration (lists of length G).

    Times are per layer in seconds; KV volumes are per layer in bytes.
    """

    tau_wk: list[float]
    attn: list[float]
    fresh: list[int]
    v_in: list[float]
    v_out: list[float]

    @classmethod
    def empty(cls, G: int) -> "GroupInputs":
        return cls([0.0] * G, [0.0] * G, [0] * G, [0.0] * G, [0.0] * G)


@dataclass
class KVAlloc:
    """One worker's KV placement in one direction (bytes per layer).

    ``borrow`` maps a node-local port index to the bytes this worker sends
    through that port.
    """

    own: float = 0.0
    borrow: dict[int, float] = field(default_factory=dict)
    frontend: float = 0.0

    @property
    def total(self) -> float:
        return self.own + sum(self.borrow.values()) + self.frontend


@dataclass
class GroupPlan:
    """The plan every rank derives for one iteration.

    ``moves`` lists the head moves ``(donor, recipient)`` in global worker
    ranks, one per query head, in the order Algorithm S1 made them.
    """

    moves: list[tuple[int, int]]
    tau_post: list[float]
    t_cmp: float
    t_a2a: float
    b_be: float
    b_fe: float
    inbound: list[KVAlloc]
    outbound: list[KVAlloc]
    group_mean: float = 0.0
    tau_ex: float = 0.0

    def eap_active(self) -> bool:
        return bool(self.moves)


def plan_group(inp: GroupInputs, cm: ComputeModel, deploy: DeploySpec,
               opts: PlanOptions, impl=None) -> GroupPlan:
    """Compute the group plan with the reference (``impl=None``) or native core."""
    impl = impl or ref
    G = deploy.ep_group_workers
    wpn = deploy.workers_per_node
    H = cm.model.num_q_heads
    tau = list(inp.tau_wk)
    group_mean = sum(tau) / G if G else 0.0

    moves: list[tuple[int, int]] = []
    if opts.eap:
        for n in range(deploy.num_nodes):
            lo = n * wpn
            node_moves, post = impl.balance_heads(
                tau[lo:lo + wpn], inp.attn[lo:lo + wpn], group_mean, H, opts.theta)
            moves.extend((lo + s, lo + d) for s, d in node_moves)
            tau[lo:lo + wpn] = post

    total_fresh = sum(inp.fresh)
    tau_ex = cm.expert_time(total_fresh) if total_fresh else 0.0
    t_cmp = (max(tau) if tau else 0.0) + tau_ex
    t_a2a = cm.a2a_time(total_fresh)

    inbound = [KVAlloc(own=v) for v in inp.v_in]
    outbound = [KVAlloc(own=v) for v in inp.v_out]
    b_be = b_fe = 0.0
    if opts.kvlb_budget and total_fresh:
        window = max(0.0, t_cmp - t_a2a) if opts.dbo else t_cmp
        b_be = deploy.bw_port * window
        b_fe = deploy.bw_fe * (t_cmp + t_a2a) if opts.kvlb_frontend else 0.0
        for n in range(deploy.num_nodes):
            lo = n * wpn
            for vols, out in ((inp.v_in, inbound), (inp.v_out, outbound)):
                node = impl.allocate_offload(
                    vols[lo:lo + wpn], b_be, b_fe, deploy.bw_port, deploy.bw_fe,
                    deploy.pcie_distance, opts.kvlb_borrow, opts.kvlb_frontend)
                for j, a in enumerate(node):
                    out[lo + j] = KVAlloc(own=a["own"], borrow=dict(a["borrow"]),
                                          frontend=a["frontend"])
    return GroupPlan(moves=moves, tau_post=tau, t_cmp=t_cmp, t_a2a=t_a2a,
                     b_be=b_be, b_fe=b_fe, inbound=inbound, outbound=outbound,
                     group_mean=group_mean, tau_ex=tau_ex)
