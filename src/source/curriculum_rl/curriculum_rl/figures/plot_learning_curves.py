from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from curriculum_rl.figures._util import (
    CONDITION_COLOR,
    CONDITION_LABEL,
    CONDITION_ORDER,
    aggregate_runs_by_condition,
    apply_style,
    find_runs,
)


def plot_learning_curves(
    logs_root: Path,
    out_path: Path,
    num_bins: int = 8,
    num_steps_per_env: int = 24,
) -> None:
    runs = find_runs(logs_root)
    if not runs:
        raise FileNotFoundError(f"no curriculum.csv files under {logs_root}")
    by_cond = aggregate_runs_by_condition(runs, num_bins=num_bins)
    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False

    ncols = 4
    nrows = int(np.ceil(num_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).flatten()

    for b in range(num_bins):
        ax = axes[b]
        for cond in CONDITION_ORDER:
            if cond not in by_cond:
                continue
            steps, _w, rewards = by_cond[cond]
            it = steps / num_steps_per_env
            ax.plot(it, rewards[:, b], label=CONDITION_LABEL[cond], color=CONDITION_COLOR[cond], lw=2.2, alpha=0.95)
        ax.set_title(f"{b * 0.5:.1f} - {(b + 1) * 0.5:.1f} m/s", fontsize=11, color="#111827", pad=6)
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(length=3)

    for b in range(num_bins, len(axes)):
        axes[b].axis("off")

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.18, wspace=0.15, hspace=0.32)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.01),
        fontsize=11,
        handlelength=2.2,
    )
    fig.text(0.5, 0.09, "PPO iteration", ha="center", va="center", fontsize=12)
    fig.text(0.01, 0.54, "per-bin mean return", ha="left", va="center", rotation="vertical", fontsize=12)
    fig.suptitle("Per-bin learning curves", fontsize=14, fontweight="bold", y=0.97)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, default=Path("unitree_rl_lab/logs/rsl_rl"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/learning_curves.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    args = parser.parse_args(argv)
    plot_learning_curves(args.logs_root, args.out, num_bins=args.num_bins)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
