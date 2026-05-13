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
    infer_condition,
)


def _find_traces(traces_dir: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    if not traces_dir.is_dir():
        return out
    for path in sorted(traces_dir.glob("*.npz")):
        cond = infer_condition(path.stem)
        if cond is None:
            continue
        out.setdefault(cond, []).append(path)
    return out


def _alive_window_kernel(vx: np.ndarray, fall_step: np.ndarray, v_cmd: float) -> np.ndarray:
    T = vx.shape[1]
    mask = np.arange(T)[None, :] < fall_step[:, None]
    sign_v = np.sign(v_cmd) if v_cmd != 0.0 else 1.0
    aligned = np.clip(vx * sign_v, 0.0, abs(v_cmd))
    denom = max(abs(v_cmd), 0.1)
    per_step = aligned / denom
    per_step_masked = np.where(mask, per_step, np.nan)
    return np.nanmean(per_step_masked, axis=1)


def _eval_kernel_per_cond(traces_dir: Path, num_bins: int) -> dict[str, dict[int, np.ndarray]]:
    cond_to_files = _find_traces(traces_dir)
    bin_width = 4.0 / num_bins
    out: dict[str, dict[int, np.ndarray]] = {}
    for cond, files in cond_to_files.items():
        per_bin: dict[int, list[np.ndarray]] = {}
        for fpath in files:
            z = np.load(fpath)
            for b in range(num_bins):
                key_vx = f"vx_b{b}"
                key_fs = f"fall_step_b{b}"
                if key_vx not in z.files:
                    continue
                vx = z[key_vx]
                if key_fs in z.files:
                    fs = z[key_fs]
                else:
                    fs = np.full(vx.shape[0], vx.shape[1], dtype=np.int32)
                v_cmd = (b + 0.5) * bin_width
                k = _alive_window_kernel(vx, fs, v_cmd)
                per_bin.setdefault(b, []).append(k)
        out[cond] = {b: np.concatenate(v) for b, v in per_bin.items() if v}
    return out


def _train_kernel_final(
    logs_root: Path,
    num_bins: int,
    tail_fraction: float = 0.1,
) -> dict[str, np.ndarray]:
    runs = find_runs(logs_root)
    if not runs:
        return {}
    by_cond = aggregate_runs_by_condition(runs, num_bins=num_bins, signal="mean_reward")
    out: dict[str, np.ndarray] = {}
    for cond, (steps, _w, rewards) in by_cond.items():
        n = rewards.shape[0]
        tail_n = max(1, int(np.ceil(n * tail_fraction)))
        out[cond] = np.nanmean(rewards[-tail_n:], axis=0)
    return out


def plot_train_vs_eval_kernel(
    traces_dir: Path,
    logs_root: Path,
    out_path: Path,
    num_bins: int = 8,
    tail_fraction: float = 0.1,
) -> None:
    eval_per_cond = _eval_kernel_per_cond(traces_dir, num_bins)
    train_per_cond = _train_kernel_final(logs_root, num_bins, tail_fraction=tail_fraction)
    conditions = [c for c in CONDITION_ORDER if c in eval_per_cond or c in train_per_cond]
    if not conditions:
        raise FileNotFoundError(
            f"no data found in either {traces_dir} or {logs_root}"
        )

    bin_width = 4.0 / num_bins

    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False

    fig, axes = plt.subplots(1, len(conditions),
                             figsize=(4.0 * len(conditions), 4.0),
                             sharey=True)
    if len(conditions) == 1:
        axes = np.array([axes])

    x = np.arange(num_bins)
    width = 0.4
    for ci, cond in enumerate(conditions):
        ax = axes[ci]
        train = train_per_cond.get(cond, np.full(num_bins, np.nan))
        eval_means = np.array([
            float(np.nanmean(eval_per_cond.get(cond, {}).get(b, np.array([np.nan]))))
            for b in range(num_bins)
        ])
        eval_stds = np.array([
            float(np.nanstd(eval_per_cond.get(cond, {}).get(b, np.array([np.nan]))))
            for b in range(num_bins)
        ])
        c = CONDITION_COLOR[cond]
        ax.bar(x - width / 2, train, width=width,
               color=c, alpha=0.45, edgecolor=c, linewidth=0.8,
               label="training (last 10% iters)")
        ax.bar(x + width / 2, eval_means, width=width,
               yerr=eval_stds, capsize=2.0,
               color=c, alpha=0.95, edgecolor="white", linewidth=0.6,
               label="eval (alive window)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b * bin_width:.1f}" for b in range(num_bins)],
                           rotation=0, fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("velocity bin lower edge (m/s)", fontsize=10)
        ax.set_title(CONDITION_LABEL[cond], fontsize=12, fontweight="bold", color=c)
        ax.grid(True, axis="y", alpha=0.3, lw=0.6)
        if ci == 0:
            ax.set_ylabel(r"per-bin tracking kernel $r_{\mathrm{lin}}$", fontsize=11)
            ax.legend(loc="lower left", fontsize=9)

    fig.suptitle("Training-time vs eval-time per-bin tracking kernel",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.13, wspace=0.10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=Path("src/results/eval_traces"))
    parser.add_argument("--logs-root", type=Path, default=Path("unitree_rl_lab/logs/rsl_rl"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/train_vs_eval_kernel.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--tail-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)
    plot_train_vs_eval_kernel(
        args.traces_dir, args.logs_root, args.out,
        num_bins=args.num_bins, tail_fraction=args.tail_fraction,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
