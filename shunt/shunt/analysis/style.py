"""Shared matplotlib style and output helpers for the figures and tables."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2b6cb0", "#c53030", "#2f855a", "#b7791f", "#6b46c1", "#319795",
           "#d53f8c", "#4a5568"]


def setup() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
        "grid.alpha": 0.3, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def write_table(rows: list[list], header: list[str], path: str | Path | None,
                title: str = "") -> None:
    """Print a table and write it as CSV, Markdown, and LaTeX (booktabs)."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)
    widths = [max(len(fmt(x)) for x in col) for col in zip(header, *rows)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(title)
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(fmt(v).ljust(w) for v, w in zip(r, widths)))
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[fmt(v) for v in r] for r in rows])
    with open(p.with_suffix(".md"), "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(v) for v in r) + " |\n")
    with open(p.with_suffix(".tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "r" * (len(header) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(fmt(v) for v in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {p.with_suffix('.csv')}, .md, .tex")


def labeled(specs: list[str]) -> list[tuple[str, str]]:
    """Parse ``label=path`` arguments (``path`` alone uses its directory name)."""
    out = []
    for s in specs:
        if "=" in s:
            label, path = s.split("=", 1)
        else:
            path = s
            label = Path(s.rstrip("/")).name
        out.append((label, path))
    return out
