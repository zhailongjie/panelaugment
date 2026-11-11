"""Plot the cortical MERFISH BP-11010522 / +10-gene example.

Install the optional plotting dependency: python -m pip install matplotlib
Run from the repository root:
    python examples/merfish/plot_results.py --out-dir example-figures

The CSV contains balanced accuracy in percent, one row per test donor.
This script redraws the recorded evaluation; it does not train a classifier.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read_scores(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4 or len({row["donor"] for row in rows}) != 4:
        raise ValueError("The example must contain all four distinct test donors")
    for row in rows:
        for key in ("base_BA_percent", "pa_BA_percent", "BA_change_pp"):
            row[key] = float(row[key])
        if not all(0 <= row[key] <= 100 for key in ("base_BA_percent", "pa_BA_percent")):
            raise ValueError("Balanced accuracy must be expressed in percent")
        if abs(row["pa_BA_percent"] - row["base_BA_percent"] - row["BA_change_pp"]) > 1e-10:
            raise ValueError("The stored paired change does not match its scores")
    return rows


def make_figure(rows: list[dict]):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    })
    base_color, pa_color = "#7A828C", "#D97732"
    ink, muted = "#233244", "#657180"
    fig = plt.figure(figsize=(183 / 25.4, 107 / 25.4), facecolor="white")
    ax = fig.add_axes([0.235, 0.17, 0.635, 0.575])
    fig.text(0.045, 0.93, "A 10-gene extension of a 500-gene panel",
             fontsize=11, fontweight="bold", color=ink)
    fig.text(0.045, 0.865, "Cortical MERFISH  ·  BP-11010522  ·  500 → 510 genes",
             fontsize=8.5, color=muted)

    handles = [
        Line2D([], [], color=base_color, marker="o", linestyle="none",
               markersize=5, label="Base (500 genes)"),
        Line2D([], [], color=pa_color, marker="D", linestyle="none",
               markersize=4.5, label="Base + PA (510 genes)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.225, 0.818),
               ncol=2, handletextpad=0.45, columnspacing=1.5, borderaxespad=0,
               fontsize=8)

    positions = [4, 3, 2, 1]
    base = [row["base_BA_percent"] for row in rows]
    pa = [row["pa_BA_percent"] for row in rows]
    for y, x0, x1 in zip(positions, base, pa):
        ax.plot([x0, x1], [y, y], color="#C8CDD3", linewidth=1.8, zorder=2)
    ax.scatter(base, positions, s=26, color=base_color, marker="o", zorder=3)
    ax.scatter(pa, positions, s=26, color=pa_color, marker="D", zorder=4)

    avg_base, avg_pa = mean(base), mean(pa)
    mean_y = -0.45
    ax.axhline(0.35, color="#DDE2E7", linewidth=0.6, zorder=1)
    ax.plot([avg_base, avg_pa], [mean_y, mean_y], color="#C8CDD3", linewidth=2.5, zorder=2)
    ax.scatter([avg_base], [mean_y], s=35, color=base_color, marker="o", zorder=3)
    ax.scatter([avg_pa], [mean_y], s=35, color=pa_color, marker="D", zorder=4)
    for x, color in ((avg_base, base_color), (avg_pa, pa_color)):
        ax.annotate(f"{x:.2f}%", xy=(x, mean_y), xytext=(0, -15),
                    textcoords="offset points", ha="center", va="top", color=color,
                    fontsize=8, fontweight="bold")
    ax.text(1.04, mean_y, f"+{avg_pa - avg_base:.2f} pp", transform=ax.get_yaxis_transform(),
            va="center", ha="left", color=pa_color, fontweight="bold", fontsize=8)

    ax.set_xlim(75, 100)
    ax.set_ylim(-1.7, 4.6)
    ax.set_xticks([75, 80, 85, 90, 95, 100])
    ax.set_xlabel("Balanced accuracy (%)", labelpad=7, color=ink)
    ax.set_yticks(positions + [mean_y], [row["donor"] for row in rows] + ["Donor mean"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=12, colors=ink)
    ax.tick_params(axis="x", length=3, color="#A5ADB6", labelcolor=muted)
    ax.spines["bottom"].set_color("#A5ADB6")
    # Guides stop above the mean row, leaving the direct value labels clear.
    ax.vlines([75, 80, 85, 90, 95, 100], ymin=0.35, ymax=4.6,
              color="#EDF0F3", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.text(0.045, 0.025, "Four test donors; mean gives each donor equal weight.  pp = percentage points.",
             fontsize=7, color=muted)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("example-figures"))
    args = parser.parse_args()
    rows = read_scores(Path(__file__).with_name("donor_scores.csv"))
    fig = make_figure(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "merfish-example.png", dpi=600)
    fig.savefig(args.out_dir / "merfish-example.svg")
    fig.savefig(args.out_dir / "merfish-example.pdf")
    plt.close(fig)
    print(f"Saved MERFISH example figures to {args.out_dir}")


if __name__ == "__main__":
    main()
