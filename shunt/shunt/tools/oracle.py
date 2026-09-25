"""Compute the ORS baseline's placement offline.

Splits the trace window into consecutive groups of ``--window`` requests (one
prefill iteration's worth), computes each request's worker-compute time with
the profiled compute model, and assigns each group to the DP ranks with the
min-makespan oracle. The output maps a request's trace index to its rank and
is read by the proxy (``placement: oracle``, ``oracle_file``).

Example::

    python -m shunt.tools.oracle --trace qwen_traceB_blksz_16.jsonl.xz \\
        --config shunt.json --start 0 --num-requests 20000 --window 512 \\
        --out ors_ranks.json
"""
from __future__ import annotations

import argparse
import json

from ..algorithms import makespan_lower_bound, optimal_oracle
from ..compute_model import ComputeModel
from ..config import ShuntConfig
from ..trace import iter_windows, load_trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--config", required=True, help="ShuntConfig JSON")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num-requests", type=int, default=None)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = ShuntConfig.from_json(a.config)
    cm = ComputeModel.from_profile(cfg.compute_profile, cfg.model, cfg.deploy)
    G = cfg.deploy.ep_group_workers
    recs = load_trace(a.trace, limit=a.num_requests, start=a.start)
    ranks: dict[str, int] = {}
    at_bound = total = 0
    for win in iter_windows(recs, a.window):
        c = [cm.worker_time(r.prefix_tokens, r.fresh_tokens) for r in win]
        assign = optimal_oracle(c, G)
        loads = [0.0] * G
        for r, w, x in zip(win, assign, c):
            ranks[str(r.index)] = w
            loads[w] += x
        total += 1
        at_bound += max(loads) <= makespan_lower_bound(c, G) * (1 + 1e-9)
    with open(a.out, "w") as f:
        json.dump(ranks, f)
    print(f"{len(ranks)} requests in {total} windows; "
          f"{at_bound} windows reach the makespan lower bound")


if __name__ == "__main__":
    main()
