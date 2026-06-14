"""Fig 18: elastic-attention overhead breakdown (broken-axis stacked bars).

Reads the JSONL from ``shunt.bench.elastic_attn_bench``. Each bar is one
(severity, helpers) case, stacked into exposed pre-transfer, attention compute,
and exposed post-transfer (fractions of the phase). The y axis is broken so the
small transfer band near the bottom is legible next to the 94-100% compute band.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..plots import BLUE, GREEN, RED, apply_style


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig 18: elastic overhead breakdown")
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="eval_overhead_elastic.pdf")
    a = ap.parse_args()
    rows = load(a.results)
    rows.sort(key=lambda r: (r["helpers"], r["severity"]))

    import matplotlib.pyplot as plt
    apply_style()
    pre = np.array([r["pre_frac"] * 100 for r in rows])
    attn = np.array([r["attn_frac"] * 100 for r in rows])
    post = np.array([r["post_frac"] * 100 for r in rows])
    labels = [f"{r['severity']}\n{['i','iii','v','vii'][[1,3,5,7].index(r['helpers'])]}"
              if r["helpers"] in (1, 3, 5, 7) else f"{r['severity']}" for r in rows]
    x = np.arange(len(rows))

    # broken axis: bottom band shows the transfers, top band the compute
    fig, (top, bot) = plt.subplots(2, 1, sharex=True, figsize=(3.3, 2.0),
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})
    for ax in (top, bot):
        ax.bar(x, attn, bottom=pre, color=GREEN, width=0.7, label="Attention")
        ax.bar(x, pre, color=RED, width=0.7, label="Pre-transfer")
        ax.bar(x, post, bottom=pre + attn, color=BLUE, width=0.7, label="Post-transfer")
    top.set_ylim(92, 100.5)
    bot.set_ylim(0, max(8, (pre + post).max() * 1.3))
    top.spines["bottom"].set_visible(False)
    bot.spines["top"].set_visible(False)
    top.tick_params(bottom=False)
    # zig-zag break marks
    d = 0.012
    for ax, ys in ((top, [0]), (bot, [1])):
        kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=0.6)
        ax.plot((-d, d), (ys[0] - d, ys[0] + d), **kw)
    bot.set_xticks(x[::max(1, len(x) // 10)])
    bot.set_xticklabels([labels[i] for i in range(0, len(x), max(1, len(x) // 10))])
    bot.set_xlabel("Severity (top) / helpers (bottom)")
    top.set_ylabel("% of phase")
    top.legend(loc="lower left", frameon=False, ncol=3, handlelength=1.0,
               columnspacing=0.8, fontsize=5.5)
    for ax in (top, bot):
        ax.tick_params(direction="in", length=2.5)
        ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {a.out}  (max exposed transfer {(pre + post).max():.1f}% of phase)")


if __name__ == "__main__":
    main()
