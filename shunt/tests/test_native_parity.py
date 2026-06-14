"""Parity: the C++ cores must agree with the Python reference (§3.2-§3.4).

Skips cleanly if ``csrc/libshunt.so`` is not built. Run after ``make`` in csrc/.
"""
from __future__ import annotations

import random

from shunt import algorithms as py
from shunt import _native as nat


def _need_native() -> bool:
    if not nat.HAVE_NATIVE:
        print("SKIP: libshunt.so not built (run `make -C csrc`)")
        return False
    return True


def test_lpt_parity() -> None:
    if not _need_native():
        return
    rng = random.Random(7)
    for W in (16, 64, 256):
        c = [rng.uniform(0.1, 5.0) for _ in range(W * 32)]
        a = py.lpt_schedule(c, W)
        b = nat.lpt_schedule(c, W)
        # identical assignment (same lowest-index tie-break)
        assert a == b, f"LPT mismatch at W={W}"


def test_balance_heads_parity() -> None:
    if not _need_native():
        return
    rng = random.Random(8)
    H = 64
    for _ in range(50):
        t = [rng.uniform(1, 40) for _ in range(8)]
        att = [x * rng.uniform(0.8, 0.99) for x in t]
        gm = sum(t) / len(t) * rng.uniform(0.4, 1.0)
        ha, pa = py.balance_heads(t, att, gm, H)
        hb, pb = nat.balance_heads(t, att, gm, H)
        assert ha == hb, "head counts mismatch"
        assert all(abs(x - y) < 1e-9 for x, y in zip(pa, pb)), "post-split mismatch"


def test_offload_parity() -> None:
    if not _need_native():
        return
    rng = random.Random(9)
    for _ in range(50):
        W = 8
        vol = [rng.uniform(0, 1e9) for _ in range(W)]
        B_be = rng.uniform(1e7, 5e8)
        B_fe = rng.uniform(1e7, 5e8)
        plans = py.allocate_offload(vol, B_be, B_fe, 25e9, 25e9)
        own, fe, borrow = nat.allocate_offload(vol, B_be, B_fe, 25e9, 25e9)
        for w in range(W):
            assert abs(plans[w].own - own[w]) < 1e-3, f"own[{w}] mismatch"
            assert abs(plans[w].frontend - fe[w]) < 1e-3, f"fe[{w}] mismatch"
            for lender, b in plans[w].borrowed.items():
                assert abs(b - borrow[w][lender]) < 1e-3, "borrow mismatch"


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")


if __name__ == "__main__":
    main()
