"""Tests for the elastic-attention head assignment (§3.3)."""
from __future__ import annotations

from shunt.serving.elastic import head_assignment


def test_no_straggler_does_not_fire() -> None:
    plan = head_assignment([64] * 8, 64)
    assert not plan.fires
    assert plan.kept == [64] * 8


def test_singleton_straggler_conserves() -> None:
    hc = [21, 68, 68, 71, 71, 71, 71, 71]   # one straggler shedding heads
    plan = head_assignment(hc, 64)
    shed = sum(max(0, 64 - h) for h in hc)
    absorbed = sum(c for lst in plan.assign.values() for _, c in lst)
    assert shed == absorbed == 43
    assert plan.kept[0] == 21
    assert all(plan.kept[r] == 64 for r in range(1, 8))
    assert plan.fires
    # every shed head is attributed to the straggler that shed it
    for helper, lst in plan.assign.items():
        for straggler, _ in lst:
            assert helper != straggler


def test_multiple_stragglers() -> None:
    # two stragglers, four helpers; head counts conserve (sum = 8*64 = 512)
    hc = [40, 40, 80, 80, 72, 72, 64, 64]
    assert sum(hc) == 8 * 64
    plan = head_assignment(hc, 64)
    shed = sum(max(0, 64 - h) for h in hc)
    absorbed = sum(c for lst in plan.assign.values() for _, c in lst)
    assert shed == absorbed == 48


def main() -> None:
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print(f"PASS {k}")


if __name__ == "__main__":
    main()
