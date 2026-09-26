"""Per-iteration decision cost of LPT placement, EAP, and KVLB (Fig. S4).

Builds (if needed) and runs the C++ benchmark ``csrc/scalability_bench``: LPT
places ``--req-per-worker`` requests per DP worker for every worker count;
EAP balances one node's heads and KVLB allocates both KV directions of one
node, whose cost does not depend on the worker count. Writes a CSV.

Example::

    python -m shunt.bench.scalability --cpu 11 --out bench/scalability.csv
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

CSRC = Path(__file__).resolve().parents[2] / "csrc"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cpu", type=int, default=-1, help="pin to this core")
    ap.add_argument("--req-per-worker", type=int, default=32)
    ap.add_argument("--workers", default="16,1000,2000,3000,4000,5000,6000,7000,8000")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    subprocess.run(["make", "-C", str(CSRC), "-s", "scalability_bench"], check=True)
    out = subprocess.run([str(CSRC / "scalability_bench"), "--cpu", str(a.cpu),
                          "--req-per-worker", str(a.req_per_worker),
                          "--workers", a.workers, "--trials", str(a.trials)],
                         check=True, capture_output=True, text=True).stdout
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(out)
    print(out, end="")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
