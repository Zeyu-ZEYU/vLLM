"""Per-iteration planner: RS -> EAP -> KVLB into one :class:`IterationPlan` (§3.1).

This is the brain Shunt runs on the CPU before any layer executes. In the live
system the global RS step runs at the proxy and the two node-local steps (EAP,
KVLB) run in each prefill node's planner; :func:`plan_iteration` composes all
three so tests, the offline analyses, and the node planner share one code path.
"""
from __future__ import annotations

from .algorithms import allocate_offload, balance_heads, lpt_schedule, optimal_oracle
from .compute_model import ComputeModel
from .config import TESTBED, TestbedConfig
from .types import DirectionPlan, IterationPlan, Request


def plan_iteration(requests: list[Request], model: ComputeModel,
                   testbed: TestbedConfig = TESTBED, theta: float = 1.5,
                   use_oracle: bool = False,
                   enable_eap: bool = True, enable_kvlb: bool = True
                   ) -> IterationPlan:
    """Compute the full plan for one prefill iteration.

    ``enable_eap`` / ``enable_kvlb`` and ``use_oracle`` select the ablation arms
    of §4.3 (/EAP, /KVLB, and the ORS / Sh-ORS scheduling variants).
    """
    G = testbed.ep_group_workers
    wpn = testbed.workers_per_node
    nodes = testbed.num_prefill_nodes
    H = model.model.num_q_heads

    # --- RS: place requests on workers by the compute they add (Alg. 1) --------
    for r in requests:
        r.compute_time = model.worker_compute_time(r.prefix_tokens, r.fresh_tokens)
    ctimes = [r.compute_time for r in requests]
    worker_of = (optimal_oracle(ctimes, G) if use_oracle else lpt_schedule(ctimes, G))
    assignment = {r.req_id: worker_of[i] for i, r in enumerate(requests)}

    # --- per-worker aggregates -------------------------------------------------
    tau_wk = [0.0] * G
    attn = [0.0] * G
    v_in = [0.0] * G
    v_out = [0.0] * G
    for i, r in enumerate(requests):
        w = worker_of[i]
        tau_wk[w] += r.compute_time
        attn[w] += model.attention_time(r.prefix_tokens, r.fresh_tokens)
        v_in[w] += model.kv_bytes_inbound(r.prefix_tokens)
        v_out[w] += model.kv_bytes_outbound(r.fresh_tokens)

    total_fresh = sum(r.fresh_tokens for r in requests)
    tau_ex = model.expert_time(total_fresh)
    group_mean = sum(tau_wk) / G if G else 0.0

    # --- EAP: shrink the residual straggler within each node (Alg. 2) ----------
    head_counts = [H] * G
    if enable_eap:
        for n in range(nodes):
            lo, hi = n * wpn, (n + 1) * wpn
            h, t_post = balance_heads(tau_wk[lo:hi], attn[lo:hi], group_mean, H, theta)
            head_counts[lo:hi] = h
            tau_wk[lo:hi] = t_post

    # --- compute window after the split (§3.1) ---------------------------------
    t_cmp = (max(tau_wk) if tau_wk else 0.0) + tau_ex
    t_a2a = model.a2a_time(total_fresh)

    # --- KVLB: fit KV into the window, offload the rest (Alg. 3) ----------------
    inbound = [DirectionPlan(owner=w) for w in range(G)]
    outbound = [DirectionPlan(owner=w) for w in range(G)]
    if enable_kvlb:
        b_be = testbed.bw_port_GBps * 1e9 * t_cmp
        b_fe = testbed.bw_fe_GBps * 1e9 * (t_cmp + t_a2a)
        bw_port = testbed.bw_port_GBps * 1e9
        bw_fe = testbed.bw_fe_GBps * 1e9
        for n in range(nodes):
            lo, hi = n * wpn, (n + 1) * wpn
            for vol, out in ((v_in[lo:hi], inbound), (v_out[lo:hi], outbound)):
                local = allocate_offload(vol, b_be, b_fe, bw_port, bw_fe)
                for j, p in enumerate(local):
                    out[lo + j] = _shift_ports(p, lo)
    else:
        # /KVLB arm: all KV stays on each worker's own backend port (§4.3)
        for w in range(G):
            inbound[w].backend[w] = v_in[w]
            outbound[w].backend[w] = v_out[w]

    return IterationPlan(
        assignment=assignment, head_counts=head_counts, worker_compute=tau_wk,
        attention_time=attn, t_cmp=t_cmp, t_a2a=t_a2a,
        inbound=inbound, outbound=outbound,
    )


def _shift_ports(p: DirectionPlan, offset: int) -> DirectionPlan:
    """Remap a node-local DirectionPlan to global worker indices."""
    return DirectionPlan(
        owner=p.owner + offset,
        backend={port + offset: b for port, b in p.backend.items()},
        frontend=p.frontend,
    )
