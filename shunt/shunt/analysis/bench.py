"""Figures and tables of the microbenchmarks (Figs. S3 and S4, Table S6).

Examples::

    python -m shunt.analysis.bench overhead bench/eap_overhead.json --out figs/figS3
    python -m shunt.analysis.bench ring bench/ring.json bench/eap_overhead.json \\
        --out tables/tab_s6_ring
    python -m shunt.analysis.bench scalability bench/scalability.csv --out figs/figS4
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from . import style


def overhead(path: str, out: str) -> None:
    style.setup()
    import matplotlib.pyplot as plt

    res = json.load(open(path))["results"]
    res.sort(key=lambda r: (r["imbalance"], r["helpers"]))
    labels = [f"{r['imbalance']:.0f}/{r['helpers']}" for r in res]
    pre = np.array([100 * r["pre_share"] for r in res])
    att = np.array([100 * r["attention_share"] for r in res])
    post = np.array([100 * r["post_share"] for r in res])
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    x = np.arange(len(res))
    ax.bar(x, pre, color=style.PALETTE[1], label="Pre-attention transfer")
    ax.bar(x, att, bottom=pre, color=style.PALETTE[0], label="Attention")
    ax.bar(x, post, bottom=pre + att, color=style.PALETTE[3],
           label="Post-attention transfer")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlabel("max/mean / helpers")
    ax.set_ylabel("Share of the phase (%)")
    ax.legend(framealpha=0.8, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    style.save(fig, f"{out}.pdf")
    rows = [[f"{r['imbalance']:.2f}", r["helpers"], r["heads_moved"],
             r["complete_s"] * 1e3, 100 * r["pre_share"], 100 * r["attention_share"],
             100 * r["post_share"]] for r in res]
    style.write_table(rows, ["max/mean", "helpers", "heads moved", "phase (ms)",
                             "pre %", "attention %", "post %"], out,
                      title="elastic attention phase")


def ring(path: str, eap_path: str | None, out: str | None) -> None:
    res = json.load(open(path))["results"]
    eap = {}
    if eap_path:
        for r in json.load(open(eap_path))["results"]:
            eap[(round(r["imbalance"], 2), r["helpers"])] = r["complete_s"]
    rows = []
    for r in sorted(res, key=lambda r: (r["imbalance"], r["helpers"])):
        e = eap.get((round(r["imbalance"], 2), r["helpers"]))
        rows.append([f"{r['imbalance']:.2f}", r["helpers"], r["ring_size"],
                     e * 1e3 if e else "--", r["makespan_s"] * 1e3,
                     f"{100 * (r['makespan_s'] / e - 1):+.0f}%" if e else "--",
                     100 * r["straggler_idle"],
                     f"{100 * r['helper_idle_mean']:.0f} / {100 * r['helper_idle_max']:.0f}",
                     r["mean_wait_ms"]])
    style.write_table(rows, ["max/mean", "helpers", "ring size", "EAP (ms)", "ring (ms)",
                             "ring vs EAP", "straggler idle %",
                             "helper idle mean/max %", "mean wait (ms)"], out,
                      title="ring attention vs elastic attention")


def scalability(path: str, out: str) -> None:
    style.setup()
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(path)))
    w = [int(r["workers"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    for i, (key, label) in enumerate((("rs_us", "RS (LPT)"), ("eap_us", "EAP"),
                                      ("kvlb_us", "KVLB"))):
        ax.plot(w, [float(r[key]) for r in rows], marker="o", ms=3,
                color=style.PALETTE[i], label=label)
    ax.set_yscale("log")
    ax.set_xlabel("DP workers")
    ax.set_ylabel("Decision cost (µs)")
    ax.legend(framealpha=0.8)
    style.save(fig, f"{out}.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overhead")
    o.add_argument("result")
    o.add_argument("--out", required=True)
    r = sub.add_parser("ring")
    r.add_argument("result")
    r.add_argument("eap", nargs="?")
    r.add_argument("--out")
    s = sub.add_parser("scalability")
    s.add_argument("result")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "overhead":
        overhead(a.result, a.out)
    elif a.cmd == "ring":
        ring(a.result, a.eap, a.out)
    else:
        scalability(a.result, a.out)


if __name__ == "__main__":
    main()
