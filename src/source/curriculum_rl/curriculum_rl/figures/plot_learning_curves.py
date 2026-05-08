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
    aggregate_ren_by_condition,
    apply_style,
    find_runs,
)

_SIGNAL_YLABEL = {
    "mean_reward": "per-bin mean return",
    "r_lin": "per-bin r_lin (velocity tracking)",
    "r_en": "per-bin r_en (energy reward)",
}

_SIGNAL_TITLE = {
    "mean_reward": "Per-bin learning curves",
    "r_lin": "Per-bin r_lin learning curves",
    "r_en": "Per-bin r_en learning curves",
}


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0 or window <= 1:
        return y.copy()
    w = min(window, n)
    half = w // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = y[lo:hi]
        out[i] = float(np.nanmean(seg)) if seg.size else np.nan
    return out


def plot_learning_curves(
    logs_root: Path,
    out_path: Path,
    num_bins: int = 8,
    num_steps_per_env: int = 24,
    smooth_window: int = 20,
    signal: str = "mean_reward",
) -> None:
    runs = find_runs(logs_root)
    if not runs:
        raise FileNotFoundError(f"no curriculum.csv files under {logs_root}")

    if signal == "r_en":
        by_cond_ren = aggregate_ren_by_condition(runs, num_bins=num_bins)
        by_cond = {cond: (steps, None, r_en) for cond, (steps, r_en) in by_cond_ren.items()}
    else:
        by_cond = aggregate_runs_by_condition(runs, num_bins=num_bins, signal=signal)

    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False

    ncols = 4
    nrows = int(np.ceil(num_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).flatten()

    bin_width = 4.0 / num_bins
    for b in range(num_bins):
        ax = axes[b]
        for cond in CONDITION_ORDER:
            if cond not in by_cond:
                continue
            steps, _w, rewards = by_cond[cond]
            it = steps / num_steps_per_env
            raw = rewards[:, b]
            smoothed = _rolling_mean(raw, smooth_window)
            ax.plot(it, raw, color=CONDITION_COLOR[cond], lw=0.8, alpha=0.22)
            ax.plot(it, smoothed, label=CONDITION_LABEL[cond], color=CONDITION_COLOR[cond], lw=2.2, alpha=0.98)
        ax.set_title(f"{b * bin_width:.1f} - {(b + 1) * bin_width:.1f} m/s", fontsize=11, color="#111827", pad=6)
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(length=3)

    for b in range(num_bins, len(axes)):
        axes[b].axis("off")

    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.18, wspace=0.15, hspace=0.40)
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
    fig.text(0.01, 0.54, _SIGNAL_YLABEL.get(signal, signal), ha="left", va="center", rotation="vertical", fontsize=12)
    fig.suptitle(_SIGNAL_TITLE.get(signal, "Per-bin learning curves"), fontsize=14, fontweight="bold", y=0.96)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, default=Path("unitree_rl_lab/logs/rsl_rl"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/learning_curves.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument("--signal", default="mean_reward", choices=["mean_reward", "r_lin", "r_en"])
    args = parser.parse_args(argv)
    plot_learning_curves(
        args.logs_root, args.out,
        num_bins=args.num_bins,
        smooth_window=args.smooth_window,
        signal=args.signal,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
