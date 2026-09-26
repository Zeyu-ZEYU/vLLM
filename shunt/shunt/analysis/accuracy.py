"""Prediction accuracy of the planner's estimates (Table 3).

Compares the plan of every iteration (``plan.rank000.jsonl``) with the phase
times measured in that iteration (``step.rank*.jsonl``, ``SHUNT_TIMING>=1``).
Per-layer measured times are medians over the iteration's layers:

- worker compute of rank w: attention, gate, and routing phases, against the
  plan's post-split estimate for w (ranks with requests only);
- expert term: the slowest rank's expert phase, against the shared estimate;
- A2A: the fastest rank's dispatch plus combine (the rank that waits least at
  the barrier), against the lower-bound estimate;
- budget window: the largest worker compute plus the slowest expert phase
  (minus the A2A with ``--dbo``), against the plan's window.

Relative error = predicted / measured - 1. The script also reports how often
the predicted straggler is the measured one, the share of worker-compute
estimates within 5%, how often the budget window is below the measured one,
and, for ranks whose inbound KV fit its budget, how often the host waited for
a layer's KV (``kv_wait``) and the wait relative to the layer's time.

Example::

    python -m shunt.analysis.accuracy NCCL=results/accuracy/shunt \\
        DBO=results/accuracy/shunt+dbo:dbo --out tables/tab3_accuracy
"""
from __future__ import annotations

import argparse

import numpy as np

from . import style
from .load import COMPUTE_PHASES, Run


def _layer_arrays(step: dict, names) -> np.ndarray:
    ph = step.get("phases_ms", {})
    arrs = [np.asarray(ph[n]) for n in names if n in ph]
    return np.sum(arrs, axis=0) if arrs else np.zeros(0)


def _a2a_layers(step: dict) -> np.ndarray:
    ph = step.get("phases_ms", {})
    if "mk_prepare" in ph:
        return np.asarray(ph["mk_prepare"]) + np.asarray(ph.get("mk_finalize", 0))
    return np.asarray(ph.get("dispatch", 0)) + np.asarray(ph.get("combine", 0))


def _expert_layers(step: dict) -> np.ndarray:
    ph = step.get("phases_ms", {})
    if "mk_experts" in ph:
        return np.asarray(ph["mk_experts"])
    return np.asarray(ph.get("moe", 0)) - np.asarray(ph.get("route", 0))


def analyze(run: Run, dbo: bool) -> dict:
    plans = {p["seq"]: p for p in run.plans}
    err = {"wk": [], "ex": [], "a2a": [], "win": []}
    same_straggler = total = 0
    late = fit = 0
    delays = []
    for seq, ranks in run.steps.items():
        p = plans.get(seq)
        if p is None or not ranks:
            continue
        G = len(p["tau_post"])
        if len(ranks) < G:
            continue
        wk = {r: np.median(_layer_arrays(ranks[r], COMPUTE_PHASES)) * 1e-3
              for r in range(G)}
        ex = np.median(np.max([_expert_layers(ranks[r]) for r in range(G)], axis=0)) * 1e-3
        a2a = np.median(np.min([_a2a_layers(ranks[r]) for r in range(G)], axis=0)) * 1e-3
        busy = [r for r in range(G) if ranks[r].get("nreq", 0) > 0]
        if not busy:
            continue
        for r in busy:
            if wk[r] > 0:
                err["wk"].append(p["tau_post"][r] / wk[r] - 1)
        if ex > 0:
            err["ex"].append(p["tau_ex"] / ex - 1)
        if a2a > 0 and p["t_a2a"] > 0:
            err["a2a"].append(p["t_a2a"] / a2a - 1)
        win_meas = max(wk.values()) + ex - (a2a if dbo else 0.0)
        win_pred = p["t_cmp"] - (p["t_a2a"] if dbo else 0.0)
        if win_meas > 0:
            err["win"].append(win_pred / win_meas - 1)
        total += 1
        same_straggler += int(np.argmax(p["tau_post"]) == max(wk, key=wk.get))
        # inbound KV that fit its budget: did any layer wait for it?
        for r in busy:
            a = p["inbound"][r]
            if p["v_in"][r] <= 0 or a["borrow"] or a["frontend"] or a["own"] > p["b_be"]:
                continue
            fit += 1
            waits = np.asarray(ranks[r].get("host_ms", {}).get("kv_wait", []))
            layer = _layer_arrays(ranks[r], ("attn",))
            if waits.size and waits.max() > 0.05:
                late += 1
                m = waits > 0.05
                delays.append(float(np.mean(waits[m] / np.maximum(layer[m], 1e-6))))
    out = {k: np.asarray(v) * 100 for k, v in err.items()}
    out["same_straggler"] = same_straggler / total if total else float("nan")
    out["wk_within_5"] = float(np.mean(np.abs(out["wk"]) <= 5)) if out["wk"].size else float("nan")
    out["win_below"] = float(np.mean(out["win"] < 0)) if out["win"].size else float("nan")
    out["late_share"] = late / fit if fit else float("nan")
    out["late_delay"] = float(np.mean(delays)) if delays else float("nan")
    out["iterations"] = total
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="label=run_dir[:dbo]")
    ap.add_argument("--out")
    a = ap.parse_args()
    cols = []
    for label, path in style.labeled(a.runs):
        dbo = path.endswith(":dbo")
        cols.append((label, analyze(Run(path.removesuffix(":dbo")), dbo)))

    def trio(v):
        return " / ".join(f"{np.percentile(v, q):+.1f}" for q in (5, 50, 95)) if v.size else "--"

    rows = [[name] + [trio(c[key]) for _, c in cols]
            for name, key in (("Worker compute", "wk"), ("Expert term", "ex"),
                              ("A2A time", "a2a"), ("Budget window", "win"))]
    rows += [[name] + [f"{c[key] * 100:.1f}%" if np.isfinite(c[key]) else "--"
                       for _, c in cols]
             for name, key in (("predicted straggler is measured one", "same_straggler"),
                               ("worker compute within 5%", "wk_within_5"),
                               ("budget window below measured", "win_below"),
                               ("in-budget inbound KV late", "late_share"),
                               ("delay of late layers", "late_delay"))]
    rows.append(["iterations"] + [c["iterations"] for _, c in cols])
    style.write_table(rows, ["estimate (P5 / P50 / P95, %)"] + [l for l, _ in cols],
                      a.out, title="relative error, predicted / measured - 1")


if __name__ == "__main__":
    main()
