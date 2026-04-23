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


def _load_epte_csv(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_epte_bars(
    epte_csv: Path,
    out_path: Path,
    num_bins: int = 8,
) -> None:
    if not epte_csv.is_file():
        raise FileNotFoundError(
            f"{epte_csv} not found. Run scripts/eval_epte.py on each checkpoint first to produce it."
        )
    rows = _load_epte_csv(epte_csv)
    if not rows:
        raise ValueError(f"{epte_csv} is empty")

    per_seed_all: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    per_seed_mean_table: dict[tuple[str, int, int], float] = {}
    for r in rows:
        cond = r.get("condition") or infer_condition(r.get("experiment", ""))
        if cond is None:
            continue
        seed = int(r["seed"])
        bin_idx = int(r["bin_idx"])
        epte = float(r["epte_sp"])
        per_seed_all[(cond, seed, bin_idx)].append(epte)
    for k, v in per_seed_all.items():
        per_seed_mean_table[k] = float(np.mean(v))

    by_cond_seeds: dict[str, dict[int, list[float]]] = {}
    for (cond, seed, bin_idx), val in per_seed_mean_table.items():
        by_cond_seeds.setdefault(cond, {}).setdefault(bin_idx, []).append(val)

    by_cond_all: dict[str, dict[int, list[float]]] = {}
    for (cond, seed, bin_idx), vals in per_seed_all.items():
        by_cond_all.setdefault(cond, {}).setdefault(bin_idx, []).extend(vals)

    conditions = [c for c in CONDITION_ORDER if c in by_cond_seeds]
    x = np.arange(num_bins)
    width = 0.8 / max(len(conditions), 1)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, cond in enumerate(conditions):
        means = np.full(num_bins, np.nan)
        lows = np.full(num_bins, np.nan)
        highs = np.full(num_bins, np.nan)
        for b in range(num_bins):
            vals = by_cond_seeds[cond].get(b, [])
            if vals:
                means[b] = float(np.mean(vals))
                lows[b] = float(np.min(vals))
                highs[b] = float(np.max(vals))
        err = np.stack([np.nan_to_num(means - lows), np.nan_to_num(highs - means)])
        offset = (i - (len(conditions) - 1) / 2) * width
        ax.bar(
            x + offset, means, width=width, yerr=err, capsize=3,
            label=CONDITION_LABEL[cond], color=CONDITION_COLOR[cond],
            edgecolor="white", lw=1.0,
            error_kw={"ecolor": "#374151", "lw": 1.0},
        )
        for b in range(num_bins):
            pts = by_cond_all[cond].get(b, [])
            if not pts:
                continue
            xs = np.full(len(pts), x[b] + offset) + np.random.uniform(-width*0.28, width*0.28, size=len(pts))
            ax.scatter(xs, pts, s=4, alpha=0.25, color=CONDITION_COLOR[cond], edgecolors="none", zorder=3)
        for b in range(num_bins):
            if not np.isnan(means[b]):
                ax.text(x[b] + offset, min(means[b] + 0.03, 1.08), f"{means[b]:.2f}",
                        ha="center", va="bottom", fontsize=7, color="#111827")
    ax.axhline(1.0, color="#9ca3af", lw=0.7, ls="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i * 0.5:.1f}-{(i + 1) * 0.5:.1f}" for i in range(num_bins)], rotation=0, fontsize=9)
    ax.set_xlabel("velocity bin (m/s)", fontsize=11)
    ax.set_ylabel("EPTE-SP  (lower is better)", fontsize=11)
    ax.set_ylim(-0.02, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Final-policy quality (bars = seed-mean ± min-max range, dots = per-rollout EPTE)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epte-csv", type=Path, default=Path("src/results/epte_sp.csv"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/epte_bars.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    args = parser.parse_args(argv)
    plot_epte_bars(args.epte_csv, args.out, num_bins=args.num_bins)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
