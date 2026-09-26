"""Per-iteration imbalance across DP ranks (Figs. 5-7, Tables S1 and S7).

Every iteration's imbalance is the max/mean over the DP ranks of the measured
worker-compute time (attention, gate, and routing phases summed over the
layers, from runs with ``SHUNT_TIMING>=1``) and of the inbound and outbound KV
tokens (from the ranks' request logs).

Examples::

    # Fig. 5 (compute) and Fig. 6 (KV) from a Baseline run
    python -m shunt.analysis.imbalance plot --run results/motivation/baseline \\
        --kind compute --out figs/fig5_straggler
    python -m shunt.analysis.imbalance plot --run results/motivation/baseline \\
        --kind kv --out figs/fig6_kv
    # Fig. 7 from the ORS run (compute, inbound, and outbound on one plot)
    python -m shunt.analysis.imbalance plot --run results/motivation/ors \\
        --kind all --out figs/fig7_oracle
    # Tables S1 and S7: one column per run
    python -m shunt.analysis.imbalance table traceB=results/.../baseline \\
        coder=results/.../baseline-coder --out tables/tab_s1_imbalance
"""
from __future__ import annotations

import argparse

import numpy as np

from . import style
from .load import Run, kv_imbalance, pct, worker_compute_imbalance


def _series(run: Run, kind: str) -> dict[str, np.ndarray]:
    out = {}
    if kind in ("compute", "all"):
        out["Worker compute"] = worker_compute_imbalance(run)
    if kind in ("kv", "all"):
        kin, kout = kv_imbalance(run)
        out["Inbound prefix-KV"] = kin
        out["Outbound new-KV"] = kout
    return out


def plot(run: Run, kind: str, out: str) -> None:
    style.setup()
    import matplotlib.pyplot as plt

    series = _series(run, kind)
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    for i, (label, v) in enumerate(series.items()):
        ax.plot(np.arange(len(v)), v, lw=0.8, color=style.PALETTE[i], label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Imbalance\n(max/mean)")
    ax.legend(loc="upper right", framealpha=0.8)
    style.save(fig, f"{out}_series.pdf")

    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    for i, (label, v) in enumerate(series.items()):
        x = np.sort(v)
        ax.plot(x, np.arange(1, len(x) + 1) / max(1, len(x)), lw=1.2,
                color=style.PALETTE[i], label=label)
    ax.set_xlabel("Imbalance (max/mean)")
    ax.set_ylabel("CDF")
    ax.legend(loc="lower right", framealpha=0.8)
    style.save(fig, f"{out}_cdf.pdf")
    rows = [[label, len(v), pct(v, 50), pct(v, 90), pct(v, 99),
             float(v.max()) if v.size else float("nan"),
             float((v > 3).mean()) if v.size else float("nan")]
            for label, v in series.items()]
    style.write_table(rows, ["series", "iterations", "median", "P90", "P99", "max",
                             "share>3"], f"{out}_summary", title=run.name)


def table(runs: list[tuple[str, str]], out: str | None) -> None:
    header = ["metric"] + [label for label, _ in runs]
    cols = []
    for _, path in runs:
        r = Run(path)
        wc = worker_compute_imbalance(r)
        kin, kout = kv_imbalance(r)
        cols.append([pct(wc, 50), pct(wc, 90), pct(wc, 99),
                     float(wc.max()) if wc.size else float("nan"),
                     pct(kin, 50), pct(kin, 99), float(kin.max()) if kin.size else float("nan"),
                     pct(kout, 50), pct(kout, 99),
                     float(kout.max()) if kout.size else float("nan")])
    names = ["compute median", "compute P90", "compute P99", "compute worst",
             "inbound median", "inbound P99", "inbound worst",
             "outbound median", "outbound P99", "outbound worst"]
    rows = [[n] + [c[i] for c in cols] for i, n in enumerate(names)]
    style.write_table(rows, header, out, title="per-iteration imbalance (max/mean)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plot")
    p.add_argument("--run", required=True)
    p.add_argument("--kind", choices=["compute", "kv", "all"], default="compute")
    p.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("runs", nargs="+", help="label=run_dir")
    t.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "plot":
        plot(Run(a.run), a.kind, a.out)
    else:
        table(style.labeled(a.runs), a.out)


if __name__ == "__main__":
    main()
