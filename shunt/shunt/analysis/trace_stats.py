"""Fig 5: input / output token-length distributions of the Qwen trace (§2.3).

Inputs are long and outputs short, the regime that most stresses prefill-side KV
transfer. Runs entirely from the trace; no testbed needed.
"""
from __future__ import annotations

import argparse

import numpy as np

from ..plots import COL_H, COL_W, RED, BLUE, apply_style
from ..trace import load_trace


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig 5: trace I/O length distribution")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default="trace_io_length_dist.pdf")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    records = load_trace(a.trace, limit=a.limit)
    inp = np.array([r.input_length for r in records], dtype=np.float64)
    out = np.array([max(1, r.output_length) for r in records], dtype=np.float64)
    print(f"{len(records)} reqs; input  p50={np.median(inp):.0f} p99={np.percentile(inp,99):.0f}")
    print(f"          ; output p50={np.median(out):.0f} p99={np.percentile(out,99):.0f}")

    import matplotlib.pyplot as plt
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    for arr, c, lab in [(inp, RED, "Input"), (out, BLUE, "Output")]:
        s = np.sort(arr)
        ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=c, lw=1.2, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Token length")
    ax.set_ylabel("CDF of requests")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.tick_params(direction="in", length=2.5)
    ax.legend(loc="upper left", frameon=False, handlelength=1.1)
    fig.tight_layout(pad=0.2)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
