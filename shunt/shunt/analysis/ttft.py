"""TTFT distributions of closed-loop runs (Figs. 9, 13a, 14a, 15a, S5; Tables 1,
2b, S4, and the TTFT columns of Table S5).

Boxes span P25-P75 with the median marked; whiskers end at P1 and P99. The
table adds the mean TTFT, the completion rate (requests/s), and the median
decode time per output token.

Examples::

    python -m shunt.analysis.ttft box Baseline=results/main/baseline \\
        ORS=results/main/ors Combo=results/main/combo Shunt=results/main/shunt \\
        Sh-ORS=results/main/sh-ors --out figs/fig13a_ttft
    python -m shunt.analysis.ttft table NCCL/Baseline=results/a2a/baseline \\
        NCCL/Shunt=results/a2a/shunt DBO/Baseline=results/a2a/baseline+dbo \\
        --out tables/tab1_a2a
"""
from __future__ import annotations

import argparse

import numpy as np

from . import style
from .load import Run, pct


def stats(run: Run) -> list:
    t = run.ttft()
    tp = run.tpot()
    return [len(t), pct(t, 1), pct(t, 25), pct(t, 50), pct(t, 75), pct(t, 99),
            float(t.mean()) if t.size else float("nan"), run.completion_rate(),
            float(np.median(tp)) * 1e3 if tp.size else float("nan"), run.errors()]


HEADER = ["system", "requests", "P1 (s)", "P25 (s)", "P50 (s)", "P75 (s)", "P99 (s)",
          "mean (s)", "req/s", "TPOT (ms)", "errors"]


def box(runs: list[tuple[str, str]], out: str, xlabel: str = "TTFT (s)",
        warmup_s: float = 0.0) -> None:
    style.setup()
    import matplotlib.pyplot as plt

    data = [(label, Run(p, warmup_s=warmup_s)) for label, p in runs]
    fig, ax = plt.subplots(figsize=(3.4, 0.35 * len(data) + 0.6))
    for i, (label, r) in enumerate(data):
        t = r.ttft()
        if not t.size:
            continue
        p1, p25, p50, p75, p99 = (pct(t, q) for q in (1, 25, 50, 75, 99))
        y = len(data) - 1 - i
        ax.add_patch(plt.Rectangle((p25, y - 0.3), p75 - p25, 0.6, fill=True,
                                   facecolor=style.PALETTE[i % 8], alpha=0.35,
                                   edgecolor=style.PALETTE[i % 8]))
        ax.plot([p50, p50], [y - 0.3, y + 0.3], color="black", lw=1.2)
        ax.plot([p1, p25], [y, y], color="black", lw=0.8)
        ax.plot([p75, p99], [y, y], color="black", lw=0.8)
        for x in (p1, p99):
            ax.plot([x, x], [y - 0.15, y + 0.15], color="black", lw=0.8)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([label for label, _ in data][::-1])
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_xlim(left=0)
    style.save(fig, f"{out}.pdf")
    table(runs, out, warmup_s)


def table(runs: list[tuple[str, str]], out: str | None, warmup_s: float = 0.0) -> None:
    rows = [[label] + stats(Run(p, warmup_s=warmup_s)) for label, p in runs]
    style.write_table(rows, HEADER, out, title="TTFT (closed loop)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("box", "table"):
        p = sub.add_parser(name)
        p.add_argument("runs", nargs="+", help="label=run_dir")
        p.add_argument("--out", required=name == "box")
        p.add_argument("--warmup-s", type=float, default=0.0,
                       help="drop requests sent in the first seconds")
    a = ap.parse_args()
    runs = style.labeled(a.runs)
    if a.cmd == "box":
        box(runs, a.out, warmup_s=a.warmup_s)
    else:
        table(runs, a.out, a.warmup_s)


if __name__ == "__main__":
    main()
