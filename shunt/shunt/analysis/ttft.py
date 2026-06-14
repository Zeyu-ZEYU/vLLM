"""Figs 16-17: TTFT distribution and mean TTFT vs offered load.

Consumes the per-request JSONL written by ``shunt.harness.trace_replay``. The box
mode draws one box per configuration (Baseline / ORS / Shunt / Sh-ORS, or the
ablation arms): box edges and median at P25/P50/P75, whiskers at P1/P99. The load
mode plots mean TTFT against the offered request rate.

Box::   python -m shunt.analysis.ttft box --out f.pdf \
            Baseline=base.jsonl ORS=ors.jsonl Shunt=shunt.jsonl Sh-ORS=shors.jsonl
Load::  python -m shunt.analysis.ttft load --out f.pdf \
            Shunt=rate8:s8.jsonl,rate12:s12.jsonl,rate16:s16.jsonl  Baseline=...
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..plots import BLUE, COL_H, COL_W, GREEN, GREY, RED, SUB_H, SUB_W, apply_style

_PALETTE = [GREY, GREEN, RED, BLUE, "#8e44ad"]


def load_ttfts(path: str) -> np.ndarray:
    vals = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("ttft"):
                vals.append(r["ttft"])
    return np.array(sorted(vals), dtype=np.float64)


def _box_stats(v: np.ndarray) -> dict:
    p = np.percentile(v, [1, 25, 50, 75, 99])
    return {"whislo": p[0], "q1": p[1], "med": p[2], "q3": p[3], "whishi": p[4],
            "fliers": []}


def cmd_box(pairs: list[tuple[str, str]], out: str) -> None:
    import matplotlib.pyplot as plt
    apply_style()
    labels = [p[0] for p in pairs]
    stats = [_box_stats(load_ttfts(p[1])) for p in pairs]
    fig, ax = plt.subplots(figsize=(SUB_W * 1.3, SUB_H))
    bp = ax.bxp(stats, showfliers=False, patch_artist=True, widths=0.6)
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=_PALETTE[i % len(_PALETTE)], alpha=0.65, lw=0.6)
    for med in bp["medians"]:
        med.set(color="black", lw=1.0)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("TTFT (s)")
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    ax.tick_params(direction="in", length=2.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    for lab, st in zip(labels, stats):
        print(f"  {lab:10s} p50={st['med']:.2f}s p99={st['whishi']:.2f}s")
    print(f"wrote {out}")


def cmd_load(series: list[tuple[str, list[tuple[float, str]]]], out: str) -> None:
    import matplotlib.pyplot as plt
    apply_style()
    fig, ax = plt.subplots(figsize=(SUB_W * 1.3, SUB_H))
    for i, (label, points) in enumerate(series):
        rates = [r for r, _ in points]
        means = [float(load_ttfts(f).mean()) for _, f in points]
        ax.plot(rates, means, "-o", color=_PALETTE[i % len(_PALETTE)], lw=1.2,
                ms=3, label=label)
    ax.set_xlabel("Offered load (req/s)")
    ax.set_ylabel("Mean TTFT (s)")
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.tick_params(direction="in", length=2.5)
    ax.legend(loc="upper left", frameon=False, handlelength=1.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Figs 16-17: TTFT box / load")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("box")
    b.add_argument("pairs", nargs="+", help="Label=file.jsonl")
    b.add_argument("--out", default="eval_ttft_box.pdf")
    ld = sub.add_parser("load")
    ld.add_argument("series", nargs="+",
                    help="Label=rate:file,rate:file (mean TTFT vs rate)")
    ld.add_argument("--out", default="eval_load_ttft.pdf")
    a = ap.parse_args()

    if a.cmd == "box":
        pairs = [tuple(p.split("=", 1)) for p in a.pairs]
        cmd_box(pairs, a.out)
    else:
        series = []
        for s in a.series:
            label, rest = s.split("=", 1)
            pts = []
            for tok in rest.split(","):
                rate, f = tok.split(":", 1)
                pts.append((float(rate.replace("rate", "")), f))
            series.append((label, pts))
        cmd_load(series, a.out)


if __name__ == "__main__":
    main()
