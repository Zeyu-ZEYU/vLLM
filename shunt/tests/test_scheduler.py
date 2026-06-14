"""Tests for the proxy-side request scheduler (RS)."""
from __future__ import annotations

import random

from shunt import ComputeModel
from shunt.serving.scheduler import RequestScheduler, SchedItem


def _items(n, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        L = rng.choice([512, 1024, 2048] * 5 + [16000])
        p = rng.randint(0, L // 2)
        out.append(SchedItem(str(i), p, L - p))
    return out


def _makespan(assign, items, model, W):
    load = [0.0] * W
    for it in items:
        load[assign[it.req_id]] += model.worker_compute_time(
            it.prefix_tokens, it.fresh_tokens)
    return max(load)


def test_baseline_round_robin() -> None:
    s = RequestScheduler(16, ComputeModel.default(), mode="baseline")
    items = _items(48)
    a = s.assign_batch(items)
    assert [a[it.req_id] for it in items[:18]] == [i % 16 for i in range(18)]


def test_ors_follows_file() -> None:
    ranks = [3, 1, 4, 1, 5, 9, 2, 6]
    s = RequestScheduler(16, ComputeModel.default(), mode="ors", ors_ranks=ranks)
    items = _items(8)
    a = s.assign_batch(items)
    assert [a[it.req_id] for it in items] == ranks


def test_shunt_beats_baseline_makespan() -> None:
    model = ComputeModel.default()
    items = _items(512, seed=3)
    sh = RequestScheduler(16, model, mode="shunt")
    bl = RequestScheduler(16, model, mode="baseline")
    a_sh = sh.assign_batch(items)
    a_bl = bl.assign_batch(items)
    assert _makespan(a_sh, items, model, 16) <= _makespan(a_bl, items, model, 16)
    assert sh.last_decision_us > 0


def main() -> None:
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print(f"PASS {k}")


if __name__ == "__main__":
    main()
