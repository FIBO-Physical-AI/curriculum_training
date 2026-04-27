from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from curriculum_rl.figures._util import (
    CONDITION_COLOR,
    CONDITION_LABEL,
    CONDITION_ORDER,
    infer_condition,
)


def plot_epte_violin(
    epte_csv: Path,
    out_path: Path,
    num_bins: int = 8,
) -> None:
    if not epte_csv.is_file():
        raise FileNotFoundError(f"{epte_csv} not found")
    rows = list(csv.DictReader(epte_csv.open()))
    if not rows:
        raise ValueError(f"{epte_csv} is empty")

    by_cond_bin: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        cond = r.get("condition") or infer_condition(r.get("experiment", ""))
        if cond is None:
            continue
        by_cond_bin[(cond, int(r["bin_idx"]))].append(float(r["epte_sp"]))

    conditions = [c for c in CONDITION_ORDER if any((c, b) in by_cond_bin for b in range(num_bins))]
    n_cond = len(conditions)
    width = 0.8 / max(n_cond, 1)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = np.arange(num_bins)
    for i, cond in enumerate(conditions):
        offset = (i - (n_cond - 1) / 2) * width
        positions = x + offset
        data = [by_cond_bin.get((cond, b), [0.0]) for b in range(num_bins)]
        parts = ax.violinplot(
            data, positions=positions, widths=width * 0.9,
            showmeans=False, showextrema=False, showmedians=False,
        )
        color = CONDITION_COLOR[cond]
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor("white")
            pc.set_alpha(0.55)
            pc.set_linewidth(0.6)
        means = [float(np.mean(d)) if len(d) else np.nan for d in data]
        mins = [float(np.min(d)) if len(d) else np.nan for d in data]
        maxs = [float(np.max(d)) if len(d) else np.nan for d in data]
        ax.scatter(positions, means, s=24, color=color, edgecolors="#111827", lw=0.6, zorder=5, label=CONDITION_LABEL[cond])
        for p, lo, hi in zip(positions, mins, maxs):
            if not np.isnan(lo):
                ax.plot([p, p], [lo, hi], color=color, lw=0.8, alpha=0.5, zorder=2)

    ax.axhline(1.0, color="#9ca3af", lw=0.7, ls="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i*0.5:.1f}-{(i+1)*0.5:.1f}" for i in range(num_bins)], fontsize=9)
    ax.set_xlabel("velocity bin (m/s)", fontsize=11)
    ax.set_ylabel("EPTE-SP", fontsize=11)
    ax.set_ylim(-0.05, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Per-rollout EPTE-SP distribution", fontsize=12, fontweight="bold", pad=28)
    ax.legend(
        frameon=False, fontsize=10, ncol=n_cond,
        loc="lower center", bbox_to_anchor=(0.5, 1.005),
    )
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epte-csv", type=Path, default=Path("src/results/epte_sp.csv"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/epte_violin.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    args = parser.parse_args(argv)
    plot_epte_violin(args.epte_csv, args.out, num_bins=args.num_bins)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
