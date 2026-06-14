"""Per-iteration imbalance of compute and KV traffic (Figs 6-8).

An "iteration" is a window of ``ep_group_workers * batch`` consecutive trace
requests run in lockstep across the DP workers (§2.3). We report, per iteration,
the max/mean over workers of (a) worker-compute time, (b) inbound prefix-KV
volume, (c) outbound new-KV volume, under two assignments:

- ``roundrobin`` -- vLLM's default dispatch (worker = request index mod W),
  giving the natural imbalance of Figs 6 and 7;
- ``oracle`` -- the offline min-makespan compute balance of Fig 8, which evens
  compute and outbound-KV but cannot remove the straggler and even worsens the
  inbound prefix-KV.

KV volumes are in tokens here (the per-token/per-layer constant cancels in
max/mean); compute is the model's worker-compute time.
"""
from __future__ import annotations

import argparse

import numpy as np

from ..algorithms import optimal_oracle
from ..compute_model import ComputeModel
from ..config import TESTBED
from ..trace import TraceRecord, load_trace


def _maxmean(values: np.ndarray) -> float:
    m = values.mean()
    return float(values.max() / m) if m > 0 else 1.0


def imbalance_series(records: list[TraceRecord], model: ComputeModel,
                     mode: str = "roundrobin", batch: int = 32,
                     max_windows: int | None = None
                     ) -> dict[str, np.ndarray]:
    """Return per-iteration max/mean series for compute, inbound, outbound."""
    W = TESTBED.ep_group_workers
    per = W * batch
    P = np.array([r.prefix_tokens for r in records], dtype=np.float64)
    N = np.array([r.fresh_tokens for r in records], dtype=np.float64)
    C = np.array([model.worker_compute_time(r.prefix_tokens, r.fresh_tokens)
                  for r in records], dtype=np.float64)

    T = len(records) // per
    if max_windows is not None:
        T = min(T, max_windows)
    comp = np.empty(T); inb = np.empty(T); outb = np.empty(T)
    for t in range(T):
        sl = slice(t * per, (t + 1) * per)
        Cw, Pw, Nw = C[sl], P[sl], N[sl]
        if mode == "roundrobin":
            cs = np.array([Cw[w::W].sum() for w in range(W)])
            ps = np.array([Pw[w::W].sum() for w in range(W)])
            ns = np.array([Nw[w::W].sum() for w in range(W)])
        elif mode == "oracle":
            assign = optimal_oracle(list(Cw), W)
            a = np.array(assign)
            cs = np.array([Cw[a == w].sum() for w in range(W)])
            ps = np.array([Pw[a == w].sum() for w in range(W)])
            ns = np.array([Nw[a == w].sum() for w in range(W)])
        else:
            raise ValueError(mode)
        comp[t] = _maxmean(cs); inb[t] = _maxmean(ps); outb[t] = _maxmean(ns)
    return {"compute": comp, "inbound": inb, "outbound": outb}


def _summ(name: str, s: np.ndarray) -> str:
    p10, p50, p90, mx = np.percentile(s, [10, 50, 90, 100])
    frac3 = float((s > 3.0).mean())
    return (f"  {name:9s} p10={p10:.2f} p50={p50:.2f} p90={p90:.2f} "
            f"max={mx:.2f}  frac>3x={frac3:.2%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Trace imbalance analysis (Figs 6-8)")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap requests read from the trace (for a quick check)")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap iterations analyzed (the oracle is slow)")
    ap.add_argument("--profile", default=None, help="compute-model profile json")
    a = ap.parse_args()

    model = (ComputeModel.from_profile(a.profile) if a.profile
             else ComputeModel.default())
    records = load_trace(a.trace, limit=a.limit)
    print(f"loaded {len(records)} requests")

    for mode in ("roundrobin", "oracle"):
        s = imbalance_series(records, model, mode=mode, max_windows=a.max_windows)
        print(f"[{mode}]  ({len(s['compute'])} iterations)")
        for k in ("compute", "inbound", "outbound"):
            print(_summ(k, s[k]))


if __name__ == "__main__":
    main()
