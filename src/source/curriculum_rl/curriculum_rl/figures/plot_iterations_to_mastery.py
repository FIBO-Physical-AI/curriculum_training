from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from curriculum_rl.eval.iterations_to_mastery import (
    MASTERY_THRESHOLD_DEFAULT,
    iterations_to_mastery_from_curves,
)
from curriculum_rl.figures._util import (
    CONDITION_COLOR,
    CONDITION_LABEL,
    CONDITION_ORDER,
    aggregate_runs_by_condition,
    apply_style,
    find_runs,
)


def plot_iterations_to_mastery(
    logs_root: Path,
    out_path: Path,
    num_bins: int = 8,
    num_steps_per_env: int = 24,
    mastery_threshold: float = MASTERY_THRESHOLD_DEFAULT,
) -> None:
    runs = find_runs(logs_root)
    if not runs:
        raise FileNotFoundError(f"no curriculum.csv files under {logs_root}")
    by_cond = aggregate_runs_by_condition(runs, num_bins=num_bins)
    apply_style()

    conditions = [c for c in CONDITION_ORDER if c in by_cond]
    itm: dict[str, np.ndarray] = {}
    for cond in conditions:
        steps, _w, rewards = by_cond[cond]
        iters = steps / num_steps_per_env
        itm[cond] = iterations_to_mastery_from_curves(rewards, iters, mastery_threshold=mastery_threshold)

    x = np.arange(num_bins)
    width = 0.8 / max(len(conditions), 1)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, cond in enumerate(conditions):
        vals = itm[cond]
        finite = np.isfinite(vals)
        xs = x[finite] + (i - (len(conditions) - 1) / 2) * width
        ax.bar(
            xs,
            vals[finite],
            width=width,
            label=CONDITION_LABEL[cond],
            color=CONDITION_COLOR[cond],
            edgecolor="white",
            lw=1.2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i * 0.5:.1f} - {(i + 1) * 0.5:.1f}" for i in range(num_bins)], rotation=25, ha="right", fontsize=10)
    ax.set_xlabel("velocity bin (m/s)", fontsize=12)
    ax.set_ylabel("first PPO iteration to mastery", fontsize=12)
    ax.set_title("Iterations to mastery", fontsize=14, fontweight="bold", color="#111827")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, axis="y", alpha=0.4, lw=0.7)
    ax.grid(False, axis="x")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, default=Path("unitree_rl_lab/logs/rsl_rl"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/iterations_to_mastery.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=MASTERY_THRESHOLD_DEFAULT)
    args = parser.parse_args(argv)
    plot_iterations_to_mastery(args.logs_root, args.out, num_bins=args.num_bins, mastery_threshold=args.threshold)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
