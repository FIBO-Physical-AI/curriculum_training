from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from curriculum_rl.figures._util import (
    CONDITION_COLOR,
    CONDITION_LABEL,
    CONDITION_ORDER,
    apply_style,
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


def _load_joint_vel(files: list[Path], bin_idx: int) -> tuple[np.ndarray | None, list[str], float]:
    stacks: list[np.ndarray] = []
    joint_names: list[str] = []
    sim_dt = 0.02
    for fpath in files:
        z = np.load(fpath)
        key = f"joint_vel_b{bin_idx}"
        if key not in z.files:
            continue
        stacks.append(z[key])
        if "joint_names" in z.files and not joint_names:
            joint_names = [str(n) for n in z["joint_names"]]
        if "sim_dt" in z.files:
            sim_dt = float(z["sim_dt"])
    if not stacks:
        return None, joint_names, sim_dt
    return np.concatenate(stacks, axis=0), joint_names, sim_dt


def plot_joint_kinematics(
    traces_dir: Path,
    out_path: Path,
    bin_idx: int = 4,
    num_bins: int = 8,
) -> None:
    cond_to_files = _find_traces(traces_dir)
    if not cond_to_files:
        raise FileNotFoundError(f"no trace npz files found in {traces_dir}")

    cond_data: dict[str, np.ndarray] = {}
    joint_names: list[str] = []
    sim_dt = 0.02
    for cond in CONDITION_ORDER:
        files = cond_to_files.get(cond)
        if not files:
            continue
        jv, jn, dt = _load_joint_vel(files, bin_idx)
        if jv is None:
            continue
        cond_data[cond] = jv
        sim_dt = dt
        if jn:
            joint_names = jn

    if not cond_data:
        raise ValueError(
            f"no joint_vel_b{bin_idx} key in any trace; "
            "re-run eval_epte.py to log per-joint velocities"
        )

    n_joints = next(iter(cond_data.values())).shape[2]
    if not joint_names or len(joint_names) != n_joints:
        joint_names = [f"j{j}" for j in range(n_joints)]

    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False

    fig, axes = plt.subplots(
        n_joints, 3,
        figsize=(11, 1.3 * n_joints),
        sharex=True,
    )
    if n_joints == 1:
        axes = axes.reshape(1, -1)

    metric_titles = ["joint velocity (rad/s)", "joint acceleration (rad/s^2)", "joint jerk (rad/s^3)"]

    for j in range(n_joints):
        for ci, cond in enumerate(cond_data):
            jv = cond_data[cond][:, :, j]
            t = np.arange(jv.shape[1]) * sim_dt
            vel_mean = jv.mean(axis=0)
            vel_std = jv.std(axis=0)

            acc = np.diff(jv, axis=1) / sim_dt
            t_acc = t[1:]
            acc_mean = acc.mean(axis=0)
            acc_std = acc.std(axis=0)

            jerk = np.diff(acc, axis=1) / sim_dt
            t_jerk = t[2:]
            jerk_mean = jerk.mean(axis=0)
            jerk_std = jerk.std(axis=0)

            color = CONDITION_COLOR[cond]
            axes[j, 0].plot(t, vel_mean, color=color, lw=1.2, alpha=0.95)
            axes[j, 0].fill_between(t, vel_mean - vel_std, vel_mean + vel_std,
                                    color=color, alpha=0.18, linewidth=0)
            axes[j, 1].plot(t_acc, acc_mean, color=color, lw=1.2, alpha=0.95)
            axes[j, 1].fill_between(t_acc, acc_mean - acc_std, acc_mean + acc_std,
                                    color=color, alpha=0.18, linewidth=0)
            axes[j, 2].plot(t_jerk, jerk_mean, color=color, lw=1.2, alpha=0.95)
            axes[j, 2].fill_between(t_jerk, jerk_mean - jerk_std, jerk_mean + jerk_std,
                                    color=color, alpha=0.18, linewidth=0)

        axes[j, 0].set_ylabel(joint_names[j], fontsize=9)
        for c in range(3):
            axes[j, c].grid(True, alpha=0.25, lw=0.6)

    for c in range(3):
        axes[0, c].set_title(metric_titles[c], fontsize=11, fontweight="bold")
        axes[-1, c].set_xlabel("time (s)", fontsize=9)

    bin_width = 4.0 / num_bins
    v_cmd = (bin_idx + 0.5) * bin_width
    fig.suptitle(
        f"Per-joint kinematics — bin {bin_idx} (v_cmd={v_cmd:.2f} m/s)",
        fontsize=13, fontweight="bold", y=0.998,
    )

    legend_handles = [
        Line2D([0], [0], color=CONDITION_COLOR[c], lw=1.6, label=CONDITION_LABEL[c])
        for c in cond_data
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.002))

    fig.subplots_adjust(top=0.965, bottom=0.04, left=0.07, right=0.99,
                        hspace=0.35, wspace=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=Path("src/results/eval_traces"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/joint_kinematics.png"))
    parser.add_argument("--bin-idx", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=8)
    args = parser.parse_args(argv)
    plot_joint_kinematics(args.traces_dir, args.out, bin_idx=args.bin_idx, num_bins=args.num_bins)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
