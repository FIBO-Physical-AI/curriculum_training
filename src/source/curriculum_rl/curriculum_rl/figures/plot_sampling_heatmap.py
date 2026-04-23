from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from curriculum_rl.figures._util import (
    CONDITION_LABEL,
    CONDITION_ORDER,
    HEATMAP_CMAP,
    aggregate_runs_by_condition,
    apply_style,
    find_runs,
)


def plot_sampling_heatmap(
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

    conditions_present = [c for c in CONDITION_ORDER if c in by_cond]
    n = len(conditions_present)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, cond in zip(axes, conditions_present):
        steps, weights, _r = by_cond[cond]
        normalized = weights / np.nansum(weights, axis=1, keepdims=True).clip(min=1e-9)
        it = steps / num_steps_per_env
        xmin = float(it.min()) if len(it) else 0.0
        xmax = float(it.max()) if len(it) else 1.0
        im = ax.imshow(
            normalized.T,
            aspect="auto",
            origin="lower",
            cmap=HEATMAP_CMAP,
            extent=[xmin, xmax, -0.5, num_bins - 0.5],
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(CONDITION_LABEL[cond], fontsize=12, fontweight="bold", color="#111827", pad=6)
        ax.set_xlabel("PPO iteration", fontsize=11)
        ax.set_yticks(np.arange(num_bins))
        ax.set_yticklabels([f"{i * 0.5:.1f} - {(i + 1) * 0.5:.1f}" for i in range(num_bins)])
        ax.grid(False)
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("#9ca3af")
    axes[0].set_ylabel("velocity bin (m/s)", fontsize=11)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02, aspect=24)
    cbar.set_label("sampling probability", fontsize=11)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=3)
    fig.suptitle("Task-sampling distribution over training", fontsize=14, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, default=Path("unitree_rl_lab/logs/rsl_rl"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/sampling_heatmap.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    args = parser.parse_args(argv)
    plot_sampling_heatmap(args.logs_root, args.out, num_bins=args.num_bins)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
