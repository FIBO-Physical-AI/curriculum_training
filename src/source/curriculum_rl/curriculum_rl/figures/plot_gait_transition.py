from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from curriculum_rl.figures._util import apply_style, CONDITION_COLOR, CONDITION_LABEL, CONDITION_ORDER
from curriculum_rl.figures.plot_gait_classification import classify_gait, GAIT_COLORS, GAIT_NAMES

FOOT_LABELS = ["FL", "FR", "RL", "RR"]
FOOT_COLORS = ["#ef4444", "#bca700", "#3b82f6", "#10b981"]

_RAMP_RE = re.compile(r"ramp_([a-z_]+)_seed(\d+)")


def _classify_windows(
    contact: np.ndarray,
    sim_dt: float,
    window_s: float = 1.0,
    stride_s: float = 0.25,
) -> tuple[np.ndarray, list[str]]:
    window_steps = max(1, int(window_s / sim_dt))
    stride_steps = max(1, int(stride_s / sim_dt))
    n = len(contact)
    centers, gaits = [], []
    pos = 0
    while pos + window_steps <= n:
        seg = contact[pos: pos + window_steps]
        gait, _ = classify_gait(seg)
        centers.append(pos + window_steps // 2)
        gaits.append(gait)
        pos += stride_steps
    return np.array(centers, dtype=int), gaits


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
    vcmd = z["vcmd"]
    vx = z["vx"]
    contact = z["contact"].astype(bool)
    sim_dt = float(z["sim_dt"])
    n_feet = contact.shape[1]
    n_steps = len(vcmd)

    x = vcmd
    x_lo = float(x.min())
    x_hi = float(x.max())

    centers_idx, gaits = _classify_windows(contact, sim_dt, window_s=window_s, stride_s=stride_s)

    ax_v, ax_c, ax_g = axes

    ax_v.plot(x, vcmd, color="#9ca3af", lw=1.5, ls="--", label="v_cmd")
    ax_v.plot(x, vx, color="#3b82f6", lw=1.6, label="v_x")
    if show_ylabel:
        ax_v.set_ylabel("velocity (m/s)")
    ax_v.legend(loc="upper left", frameon=False, fontsize=9)

    for foot_i in range(min(n_feet, 4)):
        y_low = (n_feet - 1 - foot_i) - 0.4
        y_high = (n_feet - 1 - foot_i) + 0.4
        c = contact[:, foot_i]
        edges = np.diff(np.concatenate(([False], c, [False])).astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            e = min(e, n_steps - 1)
            xs = float(x[s])
            xe = float(x[e])
            if xe <= xs:
                continue
            ax_c.fill_between(
                [xs, xe], y_low, y_high,
                color=FOOT_COLORS[foot_i], lw=0, alpha=0.85,
            )
    ax_c.set_xlim(x_lo, x_hi)
    ax_c.set_ylim(-0.7, n_feet - 0.3)
    ax_c.set_yticks(range(n_feet))
    ax_c.set_yticklabels(FOOT_LABELS[:n_feet][::-1], fontsize=9)
    if show_ylabel:
        ax_c.set_ylabel("foot contact")

    if len(centers_idx):
        v_at_center = vcmd[centers_idx]
        v_per_step = (x_hi - x_lo) / max(n_steps - 1, 1)
        half = (stride_s / sim_dt) * v_per_step / 2.0
        for i in range(len(centers_idx)):
            v_c = float(v_at_center[i])
            color = GAIT_COLORS.get(gaits[i], "#9ca3af")
            ax_g.barh(0, 2 * half, left=v_c - half, height=0.7,
                      color=color, alpha=0.9, linewidth=0)
    ax_g.set_xlim(x_lo, x_hi)
    ax_g.set_ylim(-0.5, 0.5)
    ax_g.set_yticks([])
    if show_ylabel:
        ax_g.set_ylabel("gait")
    ax_g.set_xlabel("commanded velocity (m/s)")


def plot_gait_transition(
    ramps_dir: Path,
    out_path: Path,
    window_s: float = 1.0,
    stride_s: float = 0.25,
) -> None:
    cond_to_path = _find_ramps(ramps_dir)
    if not cond_to_path:
        raise FileNotFoundError(f"no ramp_*.npz files in {ramps_dir} — run eval_ramp.py first")

    conditions = [c for c in CONDITION_ORDER if c in cond_to_path]
    if not conditions:
        conditions = sorted(cond_to_path.keys())
    n_cols = len(conditions)
    n_rows = 3

    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, all_axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.0 * n_cols, 6.5),
        sharex="col",
        gridspec_kw={"height_ratios": [1.4, 1.4, 0.5]},
    )
    if n_cols == 1:
        all_axes = all_axes.reshape(n_rows, 1)

    seen_gaits: set[str] = set()
    for ci, cond in enumerate(conditions):
        col_axes = [all_axes[r, ci] for r in range(n_rows)]
        _draw_column(col_axes, cond_to_path[cond], window_s, stride_s, show_ylabel=(ci == 0))
        label = CONDITION_LABEL.get(cond, cond)
        color = CONDITION_COLOR.get(cond, "#111827")
        all_axes[0, ci].set_title(label, fontsize=12, fontweight="bold", color=color)

        z = np.load(cond_to_path[cond])
        _, gaits = _classify_windows(
            z["contact"].astype(bool),
            float(z["sim_dt"]),
            window_s=window_s, stride_s=stride_s,
        )
        seen_gaits.update(gaits)

    legend_patches = [
        mpatches.Patch(facecolor=GAIT_COLORS[g], label=g, alpha=0.9)
        for g in GAIT_NAMES if g in seen_gaits and g in GAIT_COLORS
    ]
    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            frameon=False,
            fontsize=10,
            ncol=len(legend_patches),
            bbox_to_anchor=(0.5, 0.0),
        )

    fig.suptitle("Gait Transition: 0 → 4 m/s ramp + 4 m/s hold",
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.93, bottom=0.10, left=0.07, right=0.99,
                        hspace=0.20, wspace=0.20)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ramps-dir", type=Path, default=Path("src/results"),
                        help="directory containing ramp_*.npz files")
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/gait_transition.png"))
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--stride", type=float, default=0.25)
    args = parser.parse_args(argv)
    plot_gait_transition(args.ramps_dir, args.out, window_s=args.window, stride_s=args.stride)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
