from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from curriculum_rl.figures._util import apply_style, CONDITION_COLOR, CONDITION_LABEL, CONDITION_ORDER
from curriculum_rl.figures.plot_gait_classification import classify_gait, GAIT_COLORS

FOOT_LABELS = ["FL", "FR", "RL", "RR"]
FOOT_COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b"]

_RAMP_RE = re.compile(r"ramp_([a-z_]+)_seed(\d+)")


def _rolling_duty(contact: np.ndarray, window_steps: int) -> np.ndarray:
    n, n_feet = contact.shape
    out = np.full((n, n_feet), np.nan)
    for i in range(n):
        lo = max(0, i - window_steps // 2)
        hi = min(n, i + window_steps // 2 + 1)
        out[i] = contact[lo:hi].mean(axis=0)
    return out


def _classify_windows(
    contact: np.ndarray,
    sim_dt: float,
    window_s: float = 2.0,
    stride_s: float = 0.5,
) -> tuple[np.ndarray, list[str], list[float]]:
    window_steps = max(1, int(window_s / sim_dt))
    stride_steps = max(1, int(stride_s / sim_dt))
    n = len(contact)
    centers, gaits, confs = [], [], []
    pos = 0
    while pos + window_steps <= n:
        seg = contact[pos: pos + window_steps]
        gait, conf = classify_gait(seg)
        centers.append((pos + window_steps / 2) * sim_dt)
        gaits.append(gait)
        confs.append(conf)
        pos += stride_steps
    return np.array(centers), gaits, confs


def _find_ramps(ramps_dir: Path) -> dict[str, Path]:
    best: dict[str, Path] = {}
    for path in sorted(ramps_dir.glob("ramp_*.npz")):
        m = _RAMP_RE.match(path.stem)
        if m is None:
            continue
        cond = m.group(1)
        if cond not in best:
            best[cond] = path
    return best


def _draw_column(
    axes: list,
    ramp_path: Path,
    window_s: float,
    stride_s: float,
    show_ylabel: bool,
) -> None:
    z = np.load(ramp_path)
    t = z["t"]
    vcmd = z["vcmd"]
    vx = z["vx"]
    contact = z["contact"].astype(bool)
    sim_dt = float(z["sim_dt"])
    n_feet = contact.shape[1]

    window_steps = max(1, int(window_s / sim_dt))
    duty = _rolling_duty(contact, window_steps)
    centers, gaits, confs = _classify_windows(contact, sim_dt, window_s=window_s, stride_s=stride_s)

    ax1, ax2, ax3, ax4 = axes

    ax1.plot(t, vcmd, color="#9ca3af", lw=1.5, ls="--", label="v_cmd")
    ax1.plot(t, vx, color="#3b82f6", lw=1.8, label="v_x")
    if show_ylabel:
        ax1.set_ylabel("velocity (m/s)")
    ax1.legend(loc="upper left", frameon=False, fontsize=9)

    for foot_i in range(min(n_feet, 4)):
        y_low = (n_feet - 1 - foot_i) - 0.4
        y_high = (n_feet - 1 - foot_i) + 0.4
        c = contact[:, foot_i]
        edges = np.diff(np.concatenate(([False], c, [False])).astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            ax2.fill_between(
                [t[s], t[min(e, len(t) - 1)]],
                y_low, y_high,
                color=FOOT_COLORS[foot_i], lw=0, alpha=0.85,
            )
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(-0.7, n_feet - 0.3)
    ax2.set_yticks(range(n_feet))
    ax2.set_yticklabels(FOOT_LABELS[:n_feet][::-1], fontsize=9)
    if show_ylabel:
        ax2.set_ylabel("foot contact")

    mean_duty = duty.mean(axis=1)
    for foot_i in range(min(n_feet, 4)):
        ax3.plot(t, duty[:, foot_i], color=FOOT_COLORS[foot_i], lw=1.0, alpha=0.5,
                 label=FOOT_LABELS[foot_i])
    ax3.plot(t, mean_duty, color="#111827", lw=2.0, label="mean", zorder=5)
    ax3.axhline(0.5, color="#9ca3af", lw=0.8, ls=":")
    ax3.set_ylim(-0.05, 1.05)
    if show_ylabel:
        ax3.set_ylabel(f"duty factor\n({window_s:.1f}s window)")
    ax3.legend(loc="upper right", frameon=False, fontsize=8, ncol=5)

    stride_s_half = stride_s / 2.0
    for center, gait, conf in zip(centers, gaits, confs):
        color = GAIT_COLORS.get(gait, "#9ca3af")
        ax4.barh(0, stride_s, left=center - stride_s_half, height=0.6,
                 color=color, alpha=0.85, linewidth=0)
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_yticks([])
    if show_ylabel:
        ax4.set_ylabel("gait label")
    ax4.set_xlabel("time (s)")


def plot_gait_transition(
    ramps_dir: Path,
    out_path: Path,
    window_s: float = 2.0,
    stride_s: float = 0.5,
) -> None:
    cond_to_path = _find_ramps(ramps_dir)
    if not cond_to_path:
        raise FileNotFoundError(f"no ramp_*.npz files in {ramps_dir} — run eval_ramp.py first")

    conditions = [c for c in CONDITION_ORDER if c in cond_to_path]
    if not conditions:
        conditions = sorted(cond_to_path.keys())
    n_cols = len(conditions)

    apply_style()
    fig, all_axes = plt.subplots(
        4, n_cols,
        figsize=(7 * n_cols, 10),
        sharex="col",
    )
    if n_cols == 1:
        all_axes = all_axes.reshape(4, 1)

    for ci, cond in enumerate(conditions):
        col_axes = [all_axes[r, ci] for r in range(4)]
        _draw_column(col_axes, cond_to_path[cond], window_s, stride_s, show_ylabel=(ci == 0))
        label = CONDITION_LABEL.get(cond, cond)
        color = CONDITION_COLOR.get(cond, "#111827")
        all_axes[0, ci].set_title(label, fontsize=12, fontweight="bold", color=color)

    legend_gaits = sorted({
        g
        for path in cond_to_path.values()
        for _zz in [np.load(path)]
        for g in _classify_windows(
            _zz["contact"].astype(bool),
            float(_zz["sim_dt"]),
            window_s=window_s, stride_s=stride_s,
        )[1]
    })
    legend_patches = [
        mpatches.Patch(facecolor=GAIT_COLORS[g], label=g, alpha=0.85)
        for g in legend_gaits if g in GAIT_COLORS
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        frameon=False,
        fontsize=9,
        ncol=len(legend_patches),
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle("Gait Transition: 0 → 4 m/s ramp", fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.93, bottom=0.07, hspace=0.15, wspace=0.18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ramps-dir", type=Path, default=Path("src/results"),
                        help="directory containing ramp_*.npz files")
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/gait_transition.png"))
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--stride", type=float, default=0.5)
    args = parser.parse_args(argv)
    plot_gait_transition(args.ramps_dir, args.out, window_s=args.window, stride_s=args.stride)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
