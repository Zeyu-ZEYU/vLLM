"""How often KV outgrows its window, and where the overflow goes (Sec. 4.3,
supplementary Sec. S5, and the DBO overflow shares of Sec. 4.2).

From the plans (``plan.rank000.jsonl``):

- per port and iteration, how often inbound and outbound KV exceed the port's
  budget, and the distribution of volume / budget;
- how often the frontend is used, and the KV bytes beyond every budget;

from the requests per rank (``reqs``): the share of requests on a port whose
KV overflowed in that iteration; and from the KV router (``route``): the share
of KV bytes sent through borrowed ports and through the frontend.

Example::

    python -m shunt.analysis.overflow NCCL=results/overflow/shunt \\
        DBO=results/overflow/shunt+dbo --out tables/overflow
"""
from __future__ import annotations

import argparse

import numpy as np

from . import style
from .load import Run


def _beyond(p: dict, key: str, wpn: int) -> float:
    """Bytes (per layer) no budget absorbed, summed over the nodes."""
    allocs, b_be, b_fe = p[key], p["b_be"], p["b_fe"]
    G = len(allocs)
    total = 0.0
    for lo in range(0, G, wpn):
        port = [0.0] * wpn
        fe = 0.0
        for j in range(wpn):
            a = allocs[lo + j]
            port[j] += a["own"]
            for q, b in a["borrow"].items():
                port[int(q)] += b
            fe += a["frontend"]
        total += sum(max(0.0, x - b_be) for x in port) + max(0.0, fe - b_fe)
    return total


def analyze(run: Run, wpn: int) -> dict:
    over_in = over_out = over_any = ports = 0
    ratio_in, ratio_out = [], []
    fe_iters = iters = 0
    beyond = volume = 0.0
    overflowing: dict[int, set] = {}
    for p in run.plans:
        b = p["b_be"]
        if b <= 0:
            continue
        iters += 1
        G = len(p["v_in"])
        hot = set()
        for w in range(G):
            vi, vo = p["v_in"][w], p["v_out"][w]
            ports += 1
            oi, oo = vi > b, vo > b
            over_in += oi
            over_out += oo
            over_any += oi or oo
            if oi or oo:
                hot.add(w)
            if vi > 0:
                ratio_in.append(vi / b)
            if vo > 0:
                ratio_out.append(vo / b)
        overflowing[p["seq"]] = hot
        if any(a["frontend"] > 0 for a in p["inbound"] + p["outbound"]):
            fe_iters += 1
        beyond += _beyond(p, "inbound", wpn) + _beyond(p, "outbound", wpn)
        volume += sum(p["v_in"]) + sum(p["v_out"])
    touched = reqs = 0
    for seq, ranks in run.rank_reqs.items():
        hot = overflowing.get(seq)
        if hot is None:
            continue
        for r, rows in ranks.items():
            reqs += len(rows)
            touched += len(rows) if r in hot else 0
    own = borrowed = frontend = 0.0
    for seq, ranks in run.routes.items():
        for r, rec in ranks.items():
            me = str(r % wpn)
            for d in ("in", "out"):
                for port, v in rec[d].items():
                    if port == me:
                        own += v
                    elif port == "-1":
                        frontend += v
                    else:
                        borrowed += v
    moved = own + borrowed + frontend
    ri, ro = np.asarray(ratio_in), np.asarray(ratio_out)
    return {
        "port-iterations": ports,
        "inbound over budget": over_in / ports if ports else np.nan,
        "outbound over budget": over_out / ports if ports else np.nan,
        "either over budget": over_any / ports if ports else np.nan,
        "requests on overflowing ports": touched / reqs if reqs else np.nan,
        "KV bytes borrowed": borrowed / moved if moved else np.nan,
        "KV bytes on frontend": frontend / moved if moved else np.nan,
        "iterations using frontend": fe_iters / iters if iters else np.nan,
        "KV bytes beyond all budgets": beyond / volume if volume else np.nan,
        "inbound/budget P90": float(np.percentile(ri, 90)) if ri.size else np.nan,
        "inbound/budget P99": float(np.percentile(ri, 99)) if ri.size else np.nan,
        "inbound/budget max": float(ri.max()) if ri.size else np.nan,
        "outbound/budget P99": float(np.percentile(ro, 99)) if ro.size else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    ap.add_argument("--workers-per-node", type=int, default=None,
                    help="default: from the run's config/shunt.json")
    ap.add_argument("--out")
    a = ap.parse_args()
    cols = []
    for label, path in style.labeled(a.runs):
        run = Run(path)
        wpn = a.workers_per_node
        if wpn is None:
            import json
            cfg = json.loads((run.path / "config" / "shunt.json").read_text())
            wpn = int(cfg["deploy"]["workers_per_node"])
        cols.append((label, analyze(run, wpn)))
    keys = list(cols[0][1])
    rows = []
    for k in keys:
        row = [k]
        for _, c in cols:
            v = c[k]
            row.append(v if k in ("port-iterations",) or "/budget" in k
                       else (f"{100 * v:.1f}%" if np.isfinite(v) else "--"))
        rows.append(row)
    style.write_table(rows, ["metric"] + [l for l, _ in cols], a.out,
                      title="KV overflow")


if __name__ == "__main__":
    main()
