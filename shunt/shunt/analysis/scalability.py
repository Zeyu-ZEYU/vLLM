"""Fig 19: per-iteration decision cost of RS, EAP, KVLB vs DP worker count.

Reads the ``scalability_results.txt`` written by ``csrc/scalability_bench`` and
plots the three costs on a log y axis. RS grows ~linearly with the worker count;
EAP and KVLB are node-local and stay flat (§4.5).
"""
from __future__ import annotations

import argparse

import numpy as np

from ..plots import BLUE, COL_H, COL_W, GREEN, RED, apply_style


def load(path: str):
    W, rs, eap, kv = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            a = line.split()
            W.append(int(a[0])); rs.append(float(a[1]))
            eap.append(float(a[2])); kv.append(float(a[3]))
    return np.array(W), np.array(rs), np.array(eap), np.array(kv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig 19: scalability of RS/EAP/KVLB")
    ap.add_argument("--results", required=True, help="scalability_results.txt")
    ap.add_argument("--out", default="eval_scalability.pdf")
    a = ap.parse_args()

    W, rs, eap, kv = load(a.results)
    import matplotlib.pyplot as plt
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    ax.plot(W, rs, "-o", color=GREEN, lw=1.2, ms=3, label="RS")
    ax.plot(W, eap, "-s", color=RED, lw=1.2, ms=3, label="EAP")
    ax.plot(W, kv, "-^", color=BLUE, lw=1.2, ms=3, label="KVLB")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("DP workers")
    ax.set_ylabel(r"Decision cost ($\mu$s)")
    ax.grid(True, lw=0.4, alpha=0.4, which="both")
    ax.tick_params(direction="in", length=2.5, which="both")
    ax.legend(loc="upper left", frameon=False, handlelength=1.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {a.out}  (RS {rs[0]:.1f}->{rs[-1]:.0f}us, "
          f"EAP {eap[0]:.2f}us, KVLB {kv[0]:.2f}us)")


if __name__ == "__main__":
    main()
