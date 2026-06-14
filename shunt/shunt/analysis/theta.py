"""Fig 20: TTFT sensitivity to the compute-straggler threshold theta (§4.6).

Sweep theta (run the serving experiment at each value via the proxy's ``theta``
config) and pass the per-theta TTFT JSONL here. Draws one box per theta (P25/50/75
+ P1/99 whiskers); the default theta=1.5 is highlighted. Below 1.5 every
percentile is flat (splitting mildly-skewed iterations does not pay), above 1.5
only the tail stretches.

    python -m shunt.analysis.theta --default 1.5 --out f.pdf \
        1.0=t10.jsonl 1.5=t15.jsonl 2.0=t20.jsonl 2.5=t25.jsonl 3.0=t30.jsonl
"""
from __future__ import annotations

import argparse

from ..plots import BLUE, RED, SUB_H, SUB_W, apply_style
from .ttft import _box_stats, load_ttfts


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig 20: theta sensitivity")
    ap.add_argument("pairs", nargs="+", help="theta=file.jsonl")
    ap.add_argument("--default", type=float, default=1.5)
    ap.add_argument("--out", default="elastic_theta_box.pdf")
    a = ap.parse_args()

    thetas = [float(p.split("=", 1)[0]) for p in a.pairs]
    files = [p.split("=", 1)[1] for p in a.pairs]
    stats = [_box_stats(load_ttfts(f)) for f in files]

    import matplotlib.pyplot as plt
    apply_style()
    fig, ax = plt.subplots(figsize=(SUB_W * 1.4, SUB_H))
    bp = ax.bxp(stats, positions=thetas, showfliers=False, patch_artist=True,
                widths=0.25)
    for th, box in zip(thetas, bp["boxes"]):
        box.set(facecolor=(RED if abs(th - a.default) < 1e-6 else BLUE),
                alpha=0.65, lw=0.6)
    for med in bp["medians"]:
        med.set(color="black", lw=1.0)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("TTFT (s)")
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    ax.tick_params(direction="in", length=2.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    for th, st in zip(thetas, stats):
        print(f"  theta={th}: p50={st['med']:.2f}s p75={st['q3']:.2f}s "
              f"p99={st['whishi']:.2f}s")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
