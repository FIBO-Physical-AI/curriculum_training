from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from curriculum_rl.figures._util import (
    CONDITION_COLOR,
    CONDITION_LABEL,
    CONDITION_ORDER,
    apply_style,
    infer_condition,
)


FOOT_LABELS = ["FL", "FR", "RL", "RR"]


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


def _pick_rollout(contact: np.ndarray, vx: np.ndarray, v_cmd: float) -> int:
    n_roll, n_steps, _ = contact.shape
    survives = np.array([
        bool(contact[i].any(axis=1).all()) or (vx[i].shape[0] == n_steps and np.isfinite(vx[i, -1]))
        for i in range(n_roll)
    ])
    err = np.abs(vx.mean(axis=1) - v_cmd)
    if survives.any():
        idx = np.where(survives)[0]
        return int(idx[np.argmin(err[idx])])
    return int(np.argmin(err))


def _draw_gait_strip(ax, contact_one: np.ndarray, color: str, sim_dt: float, t_window_s: float) -> None:
    n_steps = contact_one.shape[0]
    n_show = min(int(t_window_s / sim_dt), n_steps)
    c = contact_one[:n_show]
    t = np.arange(n_show) * sim_dt

    for foot_i in range(4):
        y_low = 3 - foot_i - 0.4
        y_high = 3 - foot_i + 0.4
        contact_mask = c[:, foot_i].astype(bool)
        if contact_mask.any():
            edges = np.diff(np.concatenate(([False], contact_mask, [False])).astype(int))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            for s, e in zip(starts, ends):
                ax.fill_between(
                    [t[s], t[min(e, n_show - 1)]],
                    y_low, y_high,
                    color=color, lw=0,
                )

    ax.set_xlim(0, n_show * sim_dt)
    ax.set_ylim(-0.7, 3.7)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(FOOT_LABELS, fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_gait_diagram(
    traces_dir: Path,
    out_path: Path,
    num_bins: int = 8,
    t_window_s: float = 3.0,
) -> None:
    cond_to_files = _find_traces(traces_dir)
    if not cond_to_files:
        raise FileNotFoundError(f"no trace npz files found in {traces_dir}")

    sim_dt = None
    for files in cond_to_files.values():
        z = np.load(files[0])
        if "contact_b0" not in z.files:
            raise ValueError(
                f"{files[0]} has no contact_b0 — re-run eval_epte.py to log foot contacts"
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
        figsize=(4.0 * n_cols, 1.3 * n_rows),
        sharex=True,
    )
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for ci, cond in enumerate(conditions):
        files = cond_to_files[cond]
        z0 = np.load(files[0])
        for b in range(num_bins):
            ax = axes[b, ci]
            key_c = f"contact_b{b}"
            key_vx = f"vx_b{b}"
            key_vc = f"vcmd_b{b}"
            if key_c not in z0.files:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", color="#9ca3af", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            contact = z0[key_c]
            vx = z0[key_vx]
            v_cmd = float(z0[key_vc]) if key_vc in z0.files else 0.0

            r = _pick_rollout(contact, vx, v_cmd)
            _draw_gait_strip(ax, contact[r], CONDITION_COLOR[cond], sim_dt, t_window_s)

            if ci == 0:
                ax.set_ylabel(f"bin {b}\nv={v_cmd:.2f}", fontsize=8)
            if b == 0:
                ax.set_title(CONDITION_LABEL[cond], fontsize=11, fontweight="bold",
                             color=CONDITION_COLOR[cond])
            if b == n_rows - 1:
                ax.set_xlabel("time (s)", fontsize=9)

    fig.suptitle("Gait Diagram (Foot Contact Pattern)", fontsize=13, fontweight="bold", y=0.995)

    legend_handles = [
        Patch(facecolor="#6b7280", label="stance (foot in contact)"),
        Patch(facecolor="white", edgecolor="#6b7280", label="swing (foot in air)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.005))

    fig.subplots_adjust(top=0.93, bottom=0.075, left=0.06, right=0.99,
                        hspace=0.45, wspace=0.12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=Path("src/results/eval_traces"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/gait_diagram.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--t-window", type=float, default=3.0)
    args = parser.parse_args(argv)
    plot_gait_diagram(args.traces_dir, args.out, num_bins=args.num_bins, t_window_s=args.t_window)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
