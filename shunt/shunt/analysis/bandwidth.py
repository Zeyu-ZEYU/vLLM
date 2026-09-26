"""RNIC utilization over time on a prefill and a decode node (Figs. 8 and S2).

Reads the samples of ``shunt.harness.bw_sampler`` (``bw/<host>.jsonl`` in a run
directory). For every sample time it averages the utilization over the chosen
devices of a node, separately for outbound (transmit) and inbound (receive)
traffic, and plots a window of the run.

Examples::

    # Fig. 8: backend bonds of one prefill and one decode node
    python -m shunt.analysis.bandwidth --prefill results/bw/baseline/bw/p0.jsonl \\
        --decode results/bw/baseline/bw/d0.jsonl \\
        --devices mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \\
        --window 45 --out figs/fig8_backend_bw
    # Fig. S2: the frontend interface
    python -m shunt.analysis.bandwidth --prefill .../p0.jsonl --decode .../d0.jsonl \\
        --devices eth0 --window 50 --percent-max 0.1 --out figs/figS2_frontend_bw
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from . import style
from .load import read_jsonl
from pathlib import Path


def series(path: str, devices: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_t: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for r in read_jsonl(Path(path)):
        if r["dev"] in devices:
            by_t[round(r["t"], 4)].append((r["tx_util"], r["rx_util"]))
    ts = np.array(sorted(by_t))
    tx = np.array([np.mean([u for u, _ in by_t[t]]) for t in ts])
    rx = np.array([np.mean([u for _, u in by_t[t]]) for t in ts])
    return ts, np.clip(tx, 0, 1), np.clip(rx, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--decode", required=True)
    ap.add_argument("--devices", required=True, help="comma-separated device names")
    ap.add_argument("--start", type=float, default=None,
                    help="seconds after the first sample (default: middle of the run)")
    ap.add_argument("--window", type=float, default=45.0)
    ap.add_argument("--percent-max", type=float, default=100.0, help="y-axis limit (%)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    devs = set(a.devices.split(","))
    style.setup()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    rows = []
    lines = [("Prefill out", a.prefill, 1, "#90cdf4"), ("Prefill in", a.prefill, 2, "#feb2b2"),
             ("Decode out", a.decode, 1, "#2b6cb0"), ("Decode in", a.decode, 2, "#c53030")]
    for label, path, idx, color in lines:
        s = series(path, devs)
        ts, u = s[0], s[idx]
        if not ts.size:
            continue
        t0 = ts[0] + (a.start if a.start is not None
                      else max(0.0, (ts[-1] - ts[0] - a.window) / 2))
        m = (ts >= t0) & (ts <= t0 + a.window)
        ax.plot(ts[m] - t0, 100 * u[m], lw=0.7, color=color, label=label)
        rows.append([label, int(m.sum()), 100 * float(u[m].mean()),
                     100 * float(np.percentile(u[m], 99)), 100 * float(u[m].max())])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Utilization (%)")
    ax.set_ylim(0, a.percent_max)
    ax.legend(ncol=2, framealpha=0.8)
    style.save(fig, f"{a.out}.pdf")
    style.write_table(rows, ["line", "samples", "mean %", "P99 %", "max %"], a.out,
                      title="RNIC utilization")


if __name__ == "__main__":
    main()
