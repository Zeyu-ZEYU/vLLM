"""End-to-end checks on the planner and the three algorithms.

Runnable directly (``python tests/test_planner.py``) or under pytest. Validates
the invariants the paper relies on: LPT beats round-robin on makespan, EAP
conserves heads and only fires above threshold, and KVLB conserves bytes and
respects budgets.
"""
from __future__ import annotations

import random

from shunt import (MODEL, TESTBED, ComputeModel, Request, allocate_offload,
                   balance_heads, lpt_schedule, plan_iteration)


def _synth_requests(n: int, seed: int = 0) -> list[Request]:
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        # long inputs, short outputs; occasional very long request (the straggler)
        L = rng.choice([512, 1024, 2048] * 5 + [16000])
        p = rng.randint(0, L // 2)
        reqs.append(Request(req_id=str(i), prefix_tokens=p, fresh_tokens=L - p))
    return reqs


def test_lpt_beats_roundrobin() -> None:
    model = ComputeModel.default()
    reqs = _synth_requests(512, seed=1)
    c = [model.worker_compute_time(r.prefix_tokens, r.fresh_tokens) for r in reqs]
    W = TESTBED.ep_group_workers

    def makespan(assign):
        load = [0.0] * W
        for i, w in enumerate(assign):
            load[w] += c[i]
        return max(load)

    rr = [i % W for i in range(len(reqs))]
    lpt = lpt_schedule(c, W)
    assert makespan(lpt) <= makespan(rr), "LPT must not be worse than round-robin"


def test_eap_conserves_heads_and_threshold() -> None:
    H = MODEL.num_q_heads
    # balanced node -> no split
    bal = [10.0] * 8
    heads, _ = balance_heads(bal, [8.0] * 8, group_mean=10.0, num_q_heads=H)
    assert heads == [H] * 8

    # skewed node above threshold -> split, heads conserved
    skew = [40.0] + [5.0] * 7
    attn = [38.0] + [4.0] * 7
    heads, post = balance_heads(skew, attn, group_mean=10.0, num_q_heads=H)
    assert sum(heads) == 8 * H, "total heads must be conserved on the node"
    assert max(post) < max(skew), "split must reduce the straggler's compute"
    assert heads[0] < H, "straggler should shed heads"


def test_kvlb_conserves_bytes_and_budget() -> None:
    W = 8
    vols = [100.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]   # worker 0 is the hotspot
    B_be, B_fe = 30.0, 50.0
    plans = allocate_offload(vols, B_be, B_fe, bw_port=1.0, bw_fe=1.0)
    # every byte placed somewhere
    for w in range(W):
        assert abs(plans[w].total - vols[w]) < 1e-6, "KVLB must place all bytes"
    # own port never exceeds its budget
    for w in range(W):
        assert plans[w].own <= B_be + 1e-9
    # hotspot must borrow and/or use the frontend
    assert plans[0].borrowed or plans[0].frontend > 0


def test_plan_iteration_end_to_end() -> None:
    model = ComputeModel.default()
    reqs = _synth_requests(512, seed=2)
    plan = plan_iteration(reqs, model, TESTBED, theta=1.5)
    G = TESTBED.ep_group_workers
    assert len(plan.assignment) == len(reqs)
    assert all(0 <= w < G for w in plan.assignment.values())
    assert len(plan.head_counts) == G
    # heads conserved per node
    wpn = TESTBED.workers_per_node
    for n in range(TESTBED.num_prefill_nodes):
        assert sum(plan.head_counts[n * wpn:(n + 1) * wpn]) == wpn * MODEL.num_q_heads
    assert plan.t_cmp > 0
    # KVLB conserves bytes against per-worker volumes
    for w in range(G):
        v_in = plan.inbound[w].total
        v_out = plan.outbound[w].total
        assert v_in >= 0 and v_out >= 0
    # inbound + outbound bytes summed over the group equal the trace's KV totals
    tot_in = sum(p.total for p in plan.inbound)
    tot_out = sum(p.total for p in plan.outbound)
    exp_in = sum(model.kv_bytes_inbound(r.prefix_tokens) for r in reqs)
    exp_out = sum(model.kv_bytes_outbound(r.fresh_tokens) for r in reqs)
    assert abs(tot_in - exp_in) < 1.0
    assert abs(tot_out - exp_out) < 1.0


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    main()
