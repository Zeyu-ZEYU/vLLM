"""ctypes bindings to the C++ planner cores (``csrc/libshunt.so``).

The C++ is the production / measured path (the microsecond decision costs of
Figs 18-19 come from it). Import this only when the library is built; callers
should fall back to the pure-Python reference in :mod:`shunt.algorithms`
otherwise. ``HAVE_NATIVE`` says whether the library loaded.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_DEF = Path(__file__).resolve().parent.parent / "csrc" / "libshunt.so"
_PATH = os.environ.get("SHUNT_NATIVE_LIB", str(_DEF))

HAVE_NATIVE = False
_lib = None
try:
    _lib = ctypes.CDLL(_PATH)
    HAVE_NATIVE = True
except OSError:
    HAVE_NATIVE = False

_d = ctypes.POINTER(ctypes.c_double)
_i = ctypes.POINTER(ctypes.c_int32)

if HAVE_NATIVE:
    _lib.shunt_lpt_schedule.argtypes = [_d, ctypes.c_int, ctypes.c_int, _i]
    _lib.shunt_balance_heads.argtypes = [
        _d, _d, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double, _i, _d]
    _lib.shunt_allocate_offload.argtypes = [
        _d, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, _d, _d, _d, _d]


def _dp(a: np.ndarray):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return a, a.ctypes.data_as(_d)


def lpt_schedule(compute_times, num_workers: int) -> list[int]:
    c, cp = _dp(np.asarray(compute_times))
    n = c.size
    out = np.zeros(n, dtype=np.int32)
    _lib.shunt_lpt_schedule(cp, n, num_workers, out.ctypes.data_as(_i))
    return out.tolist()


def balance_heads(worker_compute, attention_time, group_mean, num_q_heads,
                  theta=1.5):
    t, tp = _dp(np.asarray(worker_compute))
    a, ap = _dp(np.asarray(attention_time))
    W = t.size
    hc = np.zeros(W, dtype=np.int32)
    post = np.zeros(W, dtype=np.float64)
    _lib.shunt_balance_heads(tp, ap, W, float(group_mean), int(num_q_heads),
                             float(theta), hc.ctypes.data_as(_i),
                             post.ctypes.data_as(_d))
    return hc.tolist(), post.tolist()


def allocate_offload(volumes, budget_be, budget_fe, bw_port=1.0, bw_fe=1.0,
                     pcie_distance=None):
    v, vp = _dp(np.asarray(volumes))
    W = v.size
    pcie_p = None
    if pcie_distance is not None:
        _pc, pcie_p = _dp(np.asarray(pcie_distance).reshape(-1))
    own = np.zeros(W, dtype=np.float64)
    fe = np.zeros(W, dtype=np.float64)
    borrow = np.zeros(W * W, dtype=np.float64)
    _lib.shunt_allocate_offload(vp, W, float(budget_be), float(budget_fe),
                                float(bw_port), float(bw_fe),
                                pcie_p if pcie_p else ctypes.cast(0, _d),
                                own.ctypes.data_as(_d), fe.ctypes.data_as(_d),
                                borrow.ctypes.data_as(_d))
    return own.tolist(), fe.tolist(), borrow.reshape(W, W).tolist()
