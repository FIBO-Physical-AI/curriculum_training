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


def _terminal_slope(y: np.ndarray, it: np.ndarray, n_tail: int = 200) -> tuple[float, float, float]:
    if y.size < 2:
        return float("nan"), float("nan"), float("nan")
    n = min(n_tail, y.size)
    y_tail = y[-n:]
    it_tail = it[-n:]
    if np.all(np.isnan(y_tail)):
        return float("nan"), float("nan"), float("nan")
    mask = ~np.isnan(y_tail)
    if mask.sum() < 2:
        return float(np.nanmean(y_tail)), float(np.nanstd(y_tail)), float("nan")
    slope, _ = np.polyfit(it_tail[mask], y_tail[mask], 1)
    return float(np.nanmean(y_tail)), float(np.nanstd(y_tail)), float(slope)


def plot_convergence(
    logs_root: Path,
    out_path: Path,
    num_bins: int = 8,
    num_steps_per_env: int = 24,
    smooth_window: int = 20,
    n_tail: int = 200,
    report_path: Path | None = None,
) -> None:
    runs = find_runs(logs_root)
    if not runs:
        raise FileNotFoundError(f"no curriculum.csv files under {logs_root}")
    by_cond = aggregate_runs_by_condition(runs, num_bins=num_bins)

    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False

    ncols = 4
    nrows = int(np.ceil(num_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).flatten()

    report_lines: list[str] = []
    report_lines.append(
        f"{'cond':<15} {'bin':>4} {'reward_mean':>12} {'reward_std':>11} {'slope/iter':>12}  status"
    )
    report_lines.append("-" * 72)

    for b in range(num_bins):
        ax = axes[b]
        ax.axhline(1.0, color="#9ca3af", lw=0.7, ls="--", alpha=0.7)
        ax.axhline(0.7, color="#d1d5db", lw=0.6, ls=":", alpha=0.6)

        for cond in CONDITION_ORDER:
            if cond not in by_cond:
                continue
            steps, _w, rewards = by_cond[cond]
            it = steps / num_steps_per_env
            raw = rewards[:, b]
            smoothed = _rolling_mean(raw, smooth_window)
            ax.plot(it, raw, color=CONDITION_COLOR[cond], lw=0.8, alpha=0.22)
            ax.plot(it, smoothed, label=CONDITION_LABEL[cond],
                    color=CONDITION_COLOR[cond], lw=2.2, alpha=0.98)

            mean, std, slope = _terminal_slope(smoothed, it, n_tail=n_tail)
            if np.isnan(mean):
                status = "no data"
            elif np.isnan(slope):
                status = "insufficient tail"
            else:
                converged = abs(slope) < 1e-4 and std < 0.05
                if converged and mean >= 0.95:
                    status = "CONVERGED to optimal"
                elif converged and mean < 0.95:
                    status = "CONVERGED suboptimal"
                elif slope > 1e-4:
                    status = "still improving"
                else:
                    status = "degrading"
            report_lines.append(
                f"{cond:<15} {b:>4} {mean:>12.3f} {std:>11.3f} {slope:>12.5f}  {status}"
            )

        ax.set_title(f"{b * 0.5:.1f} - {(b + 1) * 0.5:.1f} m/s",
                     fontsize=11, color="#111827", pad=6)
        ax.set_ylim(-0.03, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.7, 1.0])
        ax.tick_params(length=3)

    for b in range(num_bins, len(axes)):
        axes[b].axis("off")

    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.18,
                        wspace=0.15, hspace=0.40)
    handles, labels = axes[0].get_legend_handles_labels()
    ref_handles = [
        plt.Line2D([0], [0], color="#9ca3af", lw=1.2, ls="--", label="y=1.0 (max reward)"),
        plt.Line2D([0], [0], color="#d1d5db", lw=1.2, ls=":", label="y=0.7 (gamma threshold)"),
    ]
    fig.legend(
        handles + ref_handles, labels + [h.get_label() for h in ref_handles],
        loc="lower center", ncol=len(handles) + len(ref_handles),
        bbox_to_anchor=(0.5, 0.01), fontsize=10, handlelength=2.2,
    )
    fig.text(0.5, 0.09, "PPO iteration", ha="center", va="center", fontsize=12)
    fig.text(0.01, 0.54, "per-bin mean tracking reward",
             ha="left", va="center", rotation="vertical", fontsize=12)
    fig.suptitle("Convergence: per-bin reward vs optimal",
                 fontsize=14, fontweight="bold", y=0.96)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    if report_path is None:
        report_path = out_path.with_suffix(".txt")
    report_path.write_text("\n".join(report_lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, default=Path("unitree_rl_lab/logs/rsl_rl"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/convergence.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument("--n-tail", type=int, default=200)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    plot_convergence(
        args.logs_root, args.out,
        num_bins=args.num_bins,
        smooth_window=args.smooth_window,
        n_tail=args.n_tail,
        report_path=args.report,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
