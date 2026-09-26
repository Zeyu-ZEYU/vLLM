"""ctypes bindings to the C++ planner cores (``csrc/libshunt.so``).

The functions mirror :mod:`shunt.algorithms` and can be passed as ``impl`` to
:func:`shunt.planner.plan_group`. Build the library with ``make -C csrc``; set
``SHUNT_NATIVE_LIB`` to load it from another path.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_DEFAULT = Path(__file__).resolve().parent.parent / "csrc" / "libshunt.so"
LIB_PATH = os.environ.get("SHUNT_NATIVE_LIB", str(_DEFAULT))

_lib = None
_err: str | None = None
try:
    _lib = ctypes.CDLL(LIB_PATH)
except OSError as e:  # library not built
    _err = str(e)

_d = ctypes.POINTER(ctypes.c_double)
_i = ctypes.POINTER(ctypes.c_int32)

if _lib is not None:
    _lib.shunt_lpt_schedule.argtypes = [_d, ctypes.c_int, ctypes.c_int, _i, _d]
    _lib.shunt_lpt_schedule.restype = None
    _lib.shunt_balance_heads.argtypes = [
        _d, _d, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double,
        _i, _i]
    _lib.shunt_balance_heads.restype = ctypes.c_int
    _lib.shunt_allocate_offload.argtypes = [
        _d, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, _d, ctypes.c_int, ctypes.c_int, _d, _d, _d]
    _lib.shunt_allocate_offload.restype = None


def available() -> bool:
    return _lib is not None


def require() -> None:
    if _lib is None:
        raise RuntimeError(
            f"Shunt native library not found at {LIB_PATH} ({_err}); "
            "build it with `make -C csrc` in the shunt package directory.")


def _f64(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def lpt_schedule(compute_times, num_workers: int, init_load=None) -> list[int]:
    require()
    c = _f64(compute_times)
    out = np.zeros(c.size, dtype=np.int32)
    init = ctypes.cast(0, _d)
    if init_load is not None:
        load = _f64(init_load)
        init = load.ctypes.data_as(_d)
    _lib.shunt_lpt_schedule(c.ctypes.data_as(_d), c.size, int(num_workers),
                            out.ctypes.data_as(_i), init)
    return out.tolist()


def balance_heads(worker_compute, attention_time, group_mean: float,
                  num_q_heads: int, theta: float):
    require()
    t = _f64(worker_compute).copy()
    a = _f64(attention_time)
    W = t.size
    cap = max(1, W * int(num_q_heads))
    donors = np.zeros(cap, dtype=np.int32)
    recips = np.zeros(cap, dtype=np.int32)
    n = _lib.shunt_balance_heads(t.ctypes.data_as(_d), a.ctypes.data_as(_d), W,
                                 float(group_mean), int(num_q_heads), float(theta),
                                 donors.ctypes.data_as(_i), recips.ctypes.data_as(_i))
    moves = list(zip(donors[:n].tolist(), recips[:n].tolist()))
    return moves, t.tolist()


def allocate_offload(volumes, budget_be: float, budget_fe: float, bw_port: float,
                     bw_fe: float, pcie_distance=None, allow_borrow: bool = True,
                     allow_frontend: bool = True) -> list[dict]:
    require()
    v = _f64(volumes)
    W = v.size
    pcie = None
    pcie_p = ctypes.cast(0, _d)
    if pcie_distance is not None:
        pcie = _f64(pcie_distance).reshape(-1)
        pcie_p = pcie.ctypes.data_as(_d)
    own = np.zeros(W)
    fe = np.zeros(W)
    borrow = np.zeros(W * W)
    _lib.shunt_allocate_offload(v.ctypes.data_as(_d), W, float(budget_be),
                                float(budget_fe), float(bw_port), float(bw_fe),
                                pcie_p, int(bool(allow_borrow)),
                                int(bool(allow_frontend)), own.ctypes.data_as(_d),
                                fe.ctypes.data_as(_d), borrow.ctypes.data_as(_d))
    b = borrow.reshape(W, W)
    return [{"own": float(own[w]),
             "borrow": {p: float(b[w, p]) for p in range(W) if b[w, p] > 0.0},
             "frontend": float(fe[w])} for w in range(W)]
