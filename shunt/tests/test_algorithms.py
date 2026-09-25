"""Behaviour of the reference algorithms."""
import random

from shunt import algorithms as A


def test_lpt_places_heaviest_first_on_least_loaded():
    assert A.lpt_schedule([5, 1, 4, 2], 2) == [0, 0, 1, 1]


def test_oracle_meets_lower_bound_on_easy_instance():
    c = [7, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1]
    W = 3
    worker_of = A.optimal_oracle(c, W)
    loads = [0.0] * W
    for i, w in enumerate(worker_of):
        loads[w] += c[i]
    assert max(loads) == A.makespan_lower_bound(c, W)


def test_balance_heads_respects_threshold():
    moves, post = A.balance_heads([1.2, 1.0, 1.0, 1.0], [1, 1, 1, 1], 1.05, 64, 1.5)
    assert moves == [] and post == [1.2, 1.0, 1.0, 1.0]


def test_balance_heads_moves_only_own_heads_and_stops():
    t = [40.0, 6.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    a = [0.95 * x for x in t]
    moves, post = A.balance_heads(t, a, sum(t) / 16, 64, 1.5)
    assert moves, "a large straggler must shed heads"
    shed = {}
    for s, d in moves:
        assert s != d
        shed[s] = shed.get(s, 0) + 1
    assert all(n <= 64 for n in shed.values())
    # stop rule: no further move lowers the node's slowest time
    s = max(range(8), key=lambda w: (post[w], -w))
    d = min(range(8), key=lambda w: (post[w], w))
    e = a[s] / 64
    assert shed.get(s, 0) == 64 or post[d] + e >= post[s]
    assert max(post) < max(t)
    assert abs(sum(post) - sum(t)) < 1e-9


def test_waterfill_balances_finish_times():
    added = A.waterfill([10.0, 0.0, 5.0], [1.0, 1.0, 2.0], 9.0)
    fin = [(l + x) / b for l, x, b in zip([10.0, 0.0, 5.0], added, [1.0, 1.0, 2.0])]
    assert abs(sum(added) - 9.0) < 1e-9
    assert abs(fin[1] - fin[2]) < 1e-9 and fin[0] >= fin[1] - 1e-9


def _check_conservation(vols, allocs):
    for v, a in zip(vols, allocs):
        assert abs(a["own"] + sum(a["borrow"].values()) + a["frontend"] - v) < 1e-6


def test_offload_fits_budgets_when_possible():
    vols = [300.0, 50.0, 50.0, 50.0]
    allocs = A.allocate_offload(vols, 100.0, 100.0, 1.0, 1.0)
    _check_conservation(vols, allocs)
    assert allocs[0]["own"] == 100.0
    assert sum(allocs[0]["borrow"].values()) == 150.0  # 50 spare on each other port
    assert allocs[0]["frontend"] == 50.0
    assert list(allocs[0]["borrow"]) == [1, 2, 3]       # nearest ports first


def test_offload_spreads_beyond_budgets():
    vols = [1000.0, 0.0]
    allocs = A.allocate_offload(vols, 100.0, 100.0, 1.0, 1.0)
    _check_conservation(vols, allocs)
    port_load = [allocs[0]["own"], allocs[0]["borrow"].get(1, 0.0)]
    fe = allocs[0]["frontend"]
    assert abs(port_load[0] - port_load[1]) < 1e-6 and abs(port_load[0] - fe) < 1e-6


def test_offload_arms():
    vols = [500.0, 10.0, 10.0]
    no_borrow = A.allocate_offload(vols, 100.0, 100.0, 1.0, 1.0, allow_borrow=False)
    _check_conservation(vols, no_borrow)
    assert no_borrow[0]["borrow"] == {}
    no_fe = A.allocate_offload(vols, 100.0, 100.0, 1.0, 1.0, allow_frontend=False)
    _check_conservation(vols, no_fe)
    assert all(a["frontend"] == 0.0 for a in no_fe)


def test_offload_random_conservation():
    rng = random.Random(0)
    for _ in range(200):
        W = rng.randint(1, 8)
        vols = [rng.random() * 1000 for _ in range(W)]
        allocs = A.allocate_offload(vols, rng.random() * 300, rng.random() * 300,
                                    1.0, rng.choice([0.5, 1.0]),
                                    allow_borrow=rng.random() < 0.7,
                                    allow_frontend=rng.random() < 0.7)
        _check_conservation(vols, allocs)
