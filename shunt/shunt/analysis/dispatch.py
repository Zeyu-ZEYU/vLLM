"""Dispatch policies side by side (Table S5): where prompt tokens come from,
the worker-compute imbalance before any split, and the TTFT.

The imbalance column uses the planner's per-iteration estimates of the ranks'
worker compute before elastic attention (``tau_wk`` in the plans), so it is
comparable between runs with and without elastic attention.

Example::

    python -m shunt.analysis.dispatch Baseline=results/kva/baseline \\
        "LPT alone=results/kva/lpt" "KVA alone=results/kva/kva" \\
        "Shunt (KVA)=results/kva/shunt-kva" --out tables/tab_s5_dispatch
"""
from __future__ import annotations

import argparse

import numpy as np

from . import style
from .load import Run, imbalance, pct
from .trace_stats import split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for label, path in style.labeled(a.runs):
        r = Run(path)
        local, inbound, _ = split(r)
        imb = [imbalance(p["tau_wk"]) for p in r.plans if sum(p["tau_wk"]) > 0]
        t = r.ttft()
        rows.append([label, f"{100 * local:.1f}%", f"{100 * inbound:.1f}%",
                     float(np.median(imb)) if imb else float("nan"),
                     pct(t, 50), pct(t, 99)])
    style.write_table(rows, ["dispatch", "local", "inbound", "compute max/mean",
                             "P50 (s)", "P99 (s)"], a.out, title="dispatch policies")


if __name__ == "__main__":
    main()
