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
    apply_style,
    infer_condition,
)

FOOT_LABELS = ["FL", "FR", "RL", "RR"]
FOOT_COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b"]


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


def plot_grf(
    traces_dir: Path,
    out_path: Path,
    num_bins: int = 8,
    t_window_s: float = 3.0,
) -> None:
    cond_to_files = _find_traces(traces_dir)
    if not cond_to_files:
        raise FileNotFoundError(f"no trace npz files in {traces_dir}")

    sim_dt = None
    for files in cond_to_files.values():
        z = np.load(files[0])
        if "force_b0" not in z.files:
            raise FileNotFoundError(
                f"{files[0]} has no force_b0 — re-run eval_epte.py to log foot forces"
            )
        sim_dt = float(z["sim_dt"]) if "sim_dt" in z.files else 0.02
        break

    conditions = [c for c in CONDITION_ORDER if c in cond_to_files]
    n_cols = len(conditions)
    n_rows = num_bins

    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 1.8 * n_rows),
        sharex=False,
        sharey=False,
    )
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for ci, cond in enumerate(conditions):
        files = cond_to_files[cond]

        for b in range(num_bins):
            ax = axes[b, ci]
            key = f"force_b{b}"
            key_fs = f"fall_step_b{b}"
            all_forces = []
            all_falls = []
            for fpath in files:
                z = np.load(fpath)
                if key not in z.files:
                    continue
                f_arr = z[key]
                all_forces.append(f_arr)
                n_jt = f_arr.shape[0]
                if key_fs in z.files:
                    all_falls.append(z[key_fs][:n_jt])
                else:
                    all_falls.append(np.full(n_jt, f_arr.shape[1], dtype=np.int32))

            if not all_forces:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", color="#9ca3af", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            forces = np.concatenate(all_forces, axis=0)
            falls = np.concatenate(all_falls, axis=0)
            n_show = min(int(t_window_s / sim_dt), forces.shape[1])
            t = np.arange(n_show) * sim_dt
            forces_win = forces[:, :n_show, :].astype(np.float32)
            mask = np.arange(n_show)[None, :] < falls[:, None]
            mask_b = np.broadcast_to(mask[:, :, None], forces_win.shape)
            forces_masked = np.where(mask_b, forces_win, np.nan)
            mean_f = np.nanmean(forces_masked, axis=0)

            for foot_i in range(mean_f.shape[1]):
                ax.plot(t, mean_f[:, foot_i],
                        color=FOOT_COLORS[foot_i],
                        lw=1.2,
                        label=FOOT_LABELS[foot_i] if b == 0 and ci == 0 else None)

            ax.set_xlim(0, n_show * sim_dt)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=7)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

            if ci == 0:
                v_cmd = (b + 0.5) * (4.0 / num_bins)
                ax.set_ylabel(f"bin {b}\n{v_cmd:.2f} m/s\nForce (N)", fontsize=7)
            if b == 0:
                ax.set_title(CONDITION_LABEL[cond], fontsize=11, fontweight="bold",
                             color=CONDITION_COLOR[cond])
            if b == n_rows - 1:
                ax.set_xlabel("time (s)", fontsize=8)

    foot_handles = [
        plt.Line2D([0], [0], color=FOOT_COLORS[i], lw=2, label=FOOT_LABELS[i])
        for i in range(4)
    ]
    fig.legend(
        handles=foot_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle("Ground Reaction Force per Foot", fontsize=13, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.93, bottom=0.075, left=0.08, right=0.99, hspace=0.55, wspace=0.20)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=Path("src/results/eval_traces"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/grf.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--t-window", type=float, default=3.0)
    args = parser.parse_args(argv)
    plot_grf(args.traces_dir, args.out, num_bins=args.num_bins, t_window_s=args.t_window)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
