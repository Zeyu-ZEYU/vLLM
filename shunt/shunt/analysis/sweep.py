"""Mean TTFT versus offered load from open-loop runs (Figs. 13b, 14b, 15b;
Tables 2a, S2, S3).

Each series is a set of open-loop runs of one system at different rates; the
rate is read from the run's ``meta.json``. TTFT counts from each request's
scheduled arrival. Runs whose mean TTFT exceeds ``--cap`` are marked as such in
the table and left out of the plot.

Example::

    python -m shunt.analysis.sweep --series "Baseline=results/sweep/baseline-r*" \\
        --series "Shunt=results/sweep/shunt-r*" --out figs/fig13b_load
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

from . import style
from .load import Run


def collect(pattern: str) -> dict[float, float]:
    out = {}
    for path in sorted(glob.glob(pattern)):
        r = Run(path)
        load = r.meta.get("load", "")
        if not load.startswith("open:"):
            continue
        t = r.ttft(from_schedule=True)
        if t.size:
            out[float(load.split(":")[1])] = float(t.mean())
    return dict(sorted(out.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--series", action="append", required=True,
                    help="label=glob of run directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=float, default=100.0,
                    help="mean TTFT (s) above which a run counts as saturated")
    ap.add_argument("--dashed", default="", help="comma-separated labels drawn dashed")
    a = ap.parse_args()
    style.setup()
    import matplotlib.pyplot as plt

    dashed = set(filter(None, a.dashed.split(",")))
    series = [(label, collect(pat)) for label, pat in style.labeled(a.series)]
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    for i, (label, pts) in enumerate(series):
        xs = [x for x, y in pts.items() if y <= a.cap]
        ys = [pts[x] for x in xs]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.1, color=style.PALETTE[i % 8],
                ls="--" if label in dashed else "-", label=label)
    ax.set_xlabel("Offered load (requests/s)")
    ax.set_ylabel("Mean TTFT (s)")
    ax.legend(framealpha=0.8)
    style.save(fig, f"{a.out}.pdf")
    rates = sorted({x for _, pts in series for x in pts})
    rows = []
    for label, pts in series:
        row = [label]
        for x in rates:
            y = pts.get(x)
            row.append("--" if y is None else (f">{a.cap:g}" if y > a.cap else y))
        rows.append(row)
    style.write_table(rows, ["system"] + [f"{x:g}" for x in rates], a.out,
                      title="mean TTFT (s) vs offered load (requests/s)")


if __name__ == "__main__":
    main()
