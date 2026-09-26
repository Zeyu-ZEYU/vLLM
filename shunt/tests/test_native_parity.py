"""The C++ cores must reproduce the reference algorithms exactly."""
import random

import pytest

from shunt import algorithms as A
from shunt import native as N

pytestmark = pytest.mark.skipif(not N.available(), reason="libshunt.so not built")


def test_lpt_parity():
    rng = random.Random(1)
    for _ in range(100):
        W = rng.randint(1, 20)
        c = [rng.choice([rng.random(), 1.0]) for _ in range(rng.randint(0, 200))]
        assert N.lpt_schedule(c, W) == A.lpt_schedule(c, W)
        init = [rng.random() * 3 for _ in range(W)]
        assert N.lpt_schedule(c, W, init) == A.lpt_schedule(c, W, init)


def test_balance_heads_parity():
    rng = random.Random(2)
    for _ in range(300):
        W = rng.randint(1, 8)
        t = [rng.random() * rng.choice([1, 1, 10]) for _ in range(W)]
        a = [x * rng.uniform(0.5, 1.0) for x in t]
        gm = sum(t) / W * rng.uniform(0.3, 1.2)
        H = rng.choice([8, 16, 64])
        m1, p1 = A.balance_heads(t, a, gm, H, 1.5)
        m2, p2 = N.balance_heads(t, a, gm, H, 1.5)
        assert m1 == m2
        assert p1 == pytest.approx(p2, rel=1e-12, abs=1e-15)


def test_allocate_offload_parity():
    rng = random.Random(3)
    for _ in range(300):
        W = rng.randint(1, 8)
        vols = [rng.random() * 1000 * rng.choice([1, 5]) for _ in range(W)]
        pcie = None
        if rng.random() < 0.5:
            pcie = [[abs(i - j) + (i // 2 != j // 2) for j in range(W)] for i in range(W)]
        args = (vols, rng.random() * 300, rng.random() * 300, 1.0,
                rng.choice([0.5, 1.0]), pcie, rng.random() < 0.7, rng.random() < 0.7)
        r1 = A.allocate_offload(*args)
        r2 = N.allocate_offload(*args)
        for a, b in zip(r1, r2):
            assert a["own"] == pytest.approx(b["own"], rel=1e-9, abs=1e-9)
            assert a["frontend"] == pytest.approx(b["frontend"], rel=1e-9, abs=1e-9)
            keys = set(a["borrow"]) | set(b["borrow"])
            for k in keys:
                assert a["borrow"].get(k, 0.0) == pytest.approx(
                    b["borrow"].get(k, 0.0), rel=1e-9, abs=1e-9)
