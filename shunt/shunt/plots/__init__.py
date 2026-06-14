"""Shared matplotlib styling for the paper figures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# the paper's palette
RED = "#c0392b"     # inbound prefix-KV / straggler
BLUE = "#1f3a93"    # outbound new-KV
GREEN = "#1e8449"
GREY = "#555555"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 7.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# small two-panel column figure size used by the paper subfigures
SUB_W, SUB_H = 1.62, 1.25
COL_W, COL_H = 3.3, 1.9
