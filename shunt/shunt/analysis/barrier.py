"""Idle time at the A2A barrier and A2A occupancy of the compute window
(Sec. 4.2, "Optimized A2A and DBO").

- Idle share: in every iteration, the share of the straggler's worker-compute
  time the other ranks spend idle, 1 - mean/max of the ranks' measured worker
  compute (median and P90 over iterations).
- A2A occupancy (runs with ``SHUNT_TIMING=2``): per layer of the straggler, the
  share of its compute phases (attention, gate, routing, experts) during which
  a dispatch or combine of the other micro-batch is in flight, averaged over
  the layers (median over iterations).

Example::

    python -m shunt.analysis.barrier Baseline=results/dbo/baseline+dbo \\
        Shunt=results/dbo/shunt+dbo --out tables/dbo_barrier
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from . import style
from .load import COMPUTE_PHASES, Run, phase_sum

A2A = ("dispatch", "combine", "mk_prepare", "mk_finalize")
COMPUTE = COMPUTE_PHASES + ("mk_experts",)


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    tot = 0.0
    for s0, e0 in a:
        for s1, e1 in b:
            tot += max(0.0, min(e0, e1) - max(s0, s1))
    return tot


def analyze(run: Run) -> dict:
    idle, occ = [], []
    G = run.num_ranks
    for seq in run.busy_steps():
        d = run.steps[seq]
        if len(d) < G:
            continue
        wc = np.array([phase_sum(d[r], COMPUTE_PHASES) for r in range(G)])
        if wc.max() <= 0:
            continue
        idle.append(1 - wc.mean() / wc.max())
        s = d[int(wc.argmax())]
        tl = s.get("timeline_ms")
        if not tl:
            continue
        by_layer: dict[int, dict[str, list]] = defaultdict(lambda: {"c": [], "a": []})
        for layer, _ub, name, t0, t1 in tl:
            if name in COMPUTE:
                by_layer[layer]["c"].append((t0, t1))
            elif name in A2A:
                by_layer[layer]["a"].append((t0, t1))
        fr = []
        for layer, x in by_layer.items():
            c = sum(t1 - t0 for t0, t1 in x["c"])
            if c > 0:
                fr.append(_overlap(x["c"], x["a"]) / c)
        if fr:
            occ.append(float(np.mean(fr)))
    return {"iterations": len(idle),
            "idle median": np.median(idle) if idle else np.nan,
            "idle P90": np.percentile(idle, 90) if idle else np.nan,
            "A2A occupancy of the window (median)": np.median(occ) if occ else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    cols = [(label, analyze(Run(p))) for label, p in style.labeled(a.runs)]
    rows = [[k] + [c[k] if k == "iterations" else
                   (f"{100 * c[k]:.1f}%" if np.isfinite(c[k]) else "--")
                   for _, c in cols] for k in cols[0][1]]
    style.write_table(rows, ["metric"] + [l for l, _ in cols], a.out,
                      title="idle time at the barrier")


if __name__ == "__main__":
    main()
