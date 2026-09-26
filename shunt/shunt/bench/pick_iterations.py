"""Pick iterations of a measured run by worker-compute imbalance.

For each target max/mean, takes the iteration of the run whose measured
imbalance (over all DP ranks) is closest, and records the requests (reused
prefix and fresh tokens) of every rank on the straggler's node, straggler
first, then the others from lightest to heaviest. The output feeds
``shunt.bench.elastic_overhead`` and ``shunt.bench.ring_attention``.

Example::

    python -m shunt.bench.pick_iterations --run results/motivation/baseline-timing \\
        --targets 3 5 7 9 11 --workers-per-node 8 --out bench/iterations.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..analysis.load import COMPUTE_PHASES, Run, imbalance, phase_sum


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--targets", type=float, nargs="+", default=[3, 5, 7, 9, 11])
    ap.add_argument("--workers-per-node", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Run(a.run)
    G = run.num_ranks
    cands = []
    for seq in run.busy_steps():
        d = run.steps[seq]
        reqs = run.rank_reqs.get(seq, {})
        if len(d) < G:
            continue
        wc = [phase_sum(d[r], COMPUTE_PHASES) for r in range(G)]
        cands.append((imbalance(wc), seq, wc, reqs))
    picked = []
    for t in a.targets:
        imb, seq, wc, reqs = min(cands, key=lambda c: abs(c[0] - t))
        s = int(np.argmax(wc))
        lo = s - s % a.workers_per_node
        node = list(range(lo, lo + a.workers_per_node))
        others = sorted((r for r in node if r != s), key=lambda r: wc[r])
        picked.append({
            "target": t, "imbalance": imb, "seq": seq,
            "ranks": [[[p, n] for _, p, n, _, _ in reqs.get(r, [])]
                      for r in [s] + others],
            "measured_compute_ms": [wc[r] for r in [s] + others],
        })
        print(f"target {t}: iteration {seq}, max/mean {imb:.2f}")
    with open(a.out, "w") as f:
        json.dump({"run": str(run.path), "iterations": picked}, f, indent=1)


if __name__ == "__main__":
    main()
