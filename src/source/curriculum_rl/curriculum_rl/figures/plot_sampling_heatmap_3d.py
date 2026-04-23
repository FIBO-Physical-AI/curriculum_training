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


def plot_sampling_heatmap_3d(
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

    conditions_present = [c for c in CONDITION_ORDER if c in by_cond]
    n = len(conditions_present)
    fig = plt.figure(figsize=(5.6 * n + 1.6, 8.0))
    surf = None
    for idx, cond in enumerate(conditions_present, start=1):
        ax = fig.add_subplot(1, n, idx, projection="3d")
        steps, weights, _r = by_cond[cond]
        normalized = weights / np.nansum(weights, axis=1, keepdims=True).clip(min=1e-9)
        it = steps / num_steps_per_env
        bins = np.arange(num_bins)
        X, Y = np.meshgrid(it, bins)
        Z = normalized.T
        surf = ax.plot_surface(X, Y, Z, cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0, edgecolor="#ffffff", linewidth=0.15, antialiased=True, rstride=1, cstride=1, alpha=0.95)
        ax.set_title(CONDITION_LABEL[cond], fontsize=13, fontweight="bold", pad=14, color="#111827")
        ax.set_xlabel("PPO iteration", fontsize=10, labelpad=14)
        ax.set_ylabel("bin (m/s)", fontsize=10, labelpad=10)
        ax.set_zlabel("")
        ax.set_yticks(np.arange(num_bins))
        ax.set_yticklabels([f"{i * 0.5:.1f}" for i in range(num_bins)], fontsize=8)
        ax.tick_params(axis="x", labelsize=9, pad=2)
        ax.tick_params(axis="z", labelsize=9, pad=2)
        ax.set_zlim(0, max(1.0, float(np.nanmax(Z)) * 1.05))
        ax.view_init(elev=30, azim=-60)
        ax.xaxis.pane.set_edgecolor("#d1d5db"); ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_edgecolor("#d1d5db"); ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_edgecolor("#d1d5db"); ax.zaxis.pane.set_alpha(0.1)
        ax.grid(True, alpha=0.2)

    fig.subplots_adjust(left=0.02, right=0.86, wspace=0.02, bottom=0.06, top=0.90)
    cbar_ax = fig.add_axes([0.89, 0.25, 0.013, 0.55])
    cbar = fig.colorbar(surf, cax=cbar_ax)
    cbar.set_label("sampling probability", fontsize=11, labelpad=10)
    cbar.outline.set_visible(False)
    fig.suptitle("Task-sampling distribution over training (3D)", fontsize=14, fontweight="bold", y=0.96)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, default=Path("unitree_rl_lab/logs/rsl_rl"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/sampling_heatmap_3d.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    args = parser.parse_args(argv)
    plot_sampling_heatmap_3d(args.logs_root, args.out, num_bins=args.num_bins)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
