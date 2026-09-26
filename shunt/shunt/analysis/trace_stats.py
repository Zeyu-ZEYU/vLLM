"""Trace statistics and the prompt-token split (Table S1, Fig. S1, Sec. 2.3).

- ``stats``: requests, input and output length percentiles, and the share of
  prompt tokens covered by reused prefixes, per trace file.
- ``lengths``: the input and output length distributions of one trace.
- ``split``: from a run, the share of prompt tokens served from the prefill
  instance's local prefix cache, fetched from the KV pool (inbound
  prefix-KV), and computed, over each request's first prefill step.

Examples::

    python -m shunt.analysis.trace_stats stats business=traces/qwen_traceB_blksz_16.jsonl \\
        coding=traces/qwen_coder_blksz_16.jsonl --out tables/tab_s1_traces
    python -m shunt.analysis.trace_stats lengths traces/qwen_traceB_blksz_16.jsonl \\
        --out figs/figS1_lengths
    python -m shunt.analysis.trace_stats split Baseline=results/main/baseline
"""
from __future__ import annotations

import argparse

import numpy as np

from . import style
from .load import Run
from ..trace import load_trace


def stats(traces: list[tuple[str, str]], out: str | None) -> None:
    cols = []
    for _, path in traces:
        recs = load_trace(path)
        L = np.array([r.input_length for r in recs])
        O = np.array([r.output_length for r in recs])
        P = np.array([r.prefix_tokens for r in recs])
        cols.append([len(recs), np.percentile(L, 50), np.percentile(L, 99), L.max(),
                     np.percentile(O, 50), f"{100 * P.sum() / L.sum():.1f}%"])
    names = ["requests", "input P50", "input P99", "input max", "output P50",
             "prefix share"]
    rows = [[n] + [c[i] for c in cols] for i, n in enumerate(names)]
    style.write_table(rows, ["metric"] + [l for l, _ in traces], out, title="traces")


def lengths(path: str, out: str, bins: int = 60) -> None:
    style.setup()
    import matplotlib.pyplot as plt

    recs = load_trace(path)
    L = np.array([r.input_length for r in recs])
    O = np.array([max(1, r.output_length) for r in recs])
    fig, ax = plt.subplots(figsize=(3.4, 1.4))
    edges = np.logspace(0, np.log10(max(L.max(), O.max()) + 1), bins)
    for i, (label, v) in enumerate((("Input", L), ("Output", O))):
        w = np.ones_like(v, dtype=float) / len(v)
        ax.hist(v, bins=edges, weights=w, alpha=0.6, color=style.PALETTE[i], label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Share of requests")
    ax.legend(framealpha=0.8)
    style.save(fig, f"{out}.pdf")
    rows = [["input", np.percentile(L, 50), np.percentile(L, 99), L.max()],
            ["output", np.percentile(O, 50), np.percentile(O, 99), O.max()]]
    style.write_table(rows, ["", "P50", "P99", "max"], out, title=path)


def split(run: Run) -> tuple[float, float, float]:
    seen: dict[str, list] = {}
    fresh_total: dict[str, int] = {}
    for seq in sorted(run.rank_reqs):
        for rows in run.rank_reqs[seq].values():
            for rid, p, n, kin, _ in rows:
                if rid not in seen:
                    seen[rid] = [p, kin]
                fresh_total[rid] = fresh_total.get(rid, 0) + n
    local = sum(max(0, p - kin) for p, kin in seen.values())
    inbound = sum(kin for _, kin in seen.values())
    computed = sum(fresh_total.values())
    tot = local + inbound + computed
    return (local / tot, inbound / tot, computed / tot) if tot else (np.nan,) * 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("traces", nargs="+", help="label=trace_file")
    s.add_argument("--out")
    l_ = sub.add_parser("lengths")
    l_.add_argument("trace")
    l_.add_argument("--out", required=True)
    p = sub.add_parser("split")
    p.add_argument("runs", nargs="+", help="label=run_dir")
    p.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "stats":
        stats(style.labeled(a.traces), a.out)
    elif a.cmd == "lengths":
        lengths(a.trace, a.out)
    else:
        rows = [[label] + [f"{100 * x:.1f}%" for x in split(Run(path))]
                for label, path in style.labeled(a.runs)]
        style.write_table(rows, ["run", "local cache", "inbound (pool)", "computed"],
                          a.out, title="prompt tokens")


if __name__ == "__main__":
    main()
