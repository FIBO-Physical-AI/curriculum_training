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


def _merge_runs(gaits: list[str]) -> list[tuple[int, int, str]]:
    if not gaits:
        return []
    runs: list[tuple[int, int, str]] = []
    start = 0
    for i in range(1, len(gaits)):
        if gaits[i] != gaits[start]:
            runs.append((start, i, gaits[start]))
            start = i
    runs.append((start, len(gaits), gaits[start]))
    return runs


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
) -> set[str]:
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
    seen = set(gaits)

    ax_v, ax_g, ax_c = axes

    ax_v.plot(x, vcmd, color="#9ca3af", lw=1.5, ls="--", label="$v_{cmd}$")
    ax_v.plot(x, vx, color="#1f2937", lw=1.8, label="$v_x$")
    ax_v.set_xlim(x_lo, x_hi)
    ax_v.set_ylim(min(0.0, float(vx.min()) - 0.3), float(max(vcmd.max(), vx.max())) + 0.3)
    if show_ylabel:
        ax_v.set_ylabel("velocity (m/s)", fontsize=10)
    ax_v.legend(loc="upper left", frameon=False, fontsize=9)
    ax_v.tick_params(labelbottom=False)
    ax_v.grid(True, alpha=0.25)

    runs = _merge_runs(gaits)
    if len(centers_idx):
        v_at_center = vcmd[centers_idx]
        for s_idx, e_idx, gait in runs:
            if e_idx <= s_idx:
                continue
            v_left = float(v_at_center[s_idx])
            v_right = float(v_at_center[min(e_idx, len(v_at_center) - 1)])
            if e_idx >= len(v_at_center):
                v_right = x_hi
            if s_idx == 0:
                v_left = x_lo
            color = GAIT_COLORS.get(gait, "#9ca3af")
            width = v_right - v_left
            if width <= 0:
                continue
            ax_g.add_patch(plt.Rectangle((v_left, 0.0), width, 1.0,
                                         facecolor=color, alpha=0.85, edgecolor="white", lw=0.5))
            if width > 0.35 * (x_hi - x_lo) * 0.20:
                ax_g.text(v_left + width / 2.0, 0.5, gait,
                          ha="center", va="center", fontsize=8.5,
                          fontweight="bold", color="#111827")
    ax_g.set_xlim(x_lo, x_hi)
    ax_g.set_ylim(0.0, 1.0)
    ax_g.set_yticks([])
    ax_g.set_xticks([])
    if show_ylabel:
        ax_g.set_ylabel("gait", fontsize=10, rotation=0, ha="right", va="center", labelpad=20)

    for foot_i in range(min(n_feet, 4)):
        y_low = (n_feet - 1 - foot_i) - 0.45
        y_high = (n_feet - 1 - foot_i) + 0.45
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
            ax_c.fill_between([xs, xe], y_low, y_high,
                              color="#111827", lw=0)
    ax_c.set_xlim(x_lo, x_hi)
    ax_c.set_ylim(-0.6, n_feet - 0.4)
    ax_c.set_yticks(range(n_feet))
    ax_c.set_yticklabels(FOOT_LABELS[:n_feet][::-1], fontsize=9)
    ax_c.set_xlabel("commanded velocity (m/s)", fontsize=10)
    ax_c.grid(True, axis="x", alpha=0.20)
    if show_ylabel:
        for sp in ("top", "right"):
            ax_c.spines[sp].set_visible(False)
    else:
        ax_c.set_yticklabels([])

    return seen


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
        figsize=(6.2 * n_cols, 5.2),
        sharex="col",
        gridspec_kw={"height_ratios": [1.6, 0.35, 1.2]},
    )
    if n_cols == 1:
        all_axes = all_axes.reshape(n_rows, 1)

    seen_gaits: set[str] = set()
    for ci, cond in enumerate(conditions):
        col_axes = [all_axes[r, ci] for r in range(n_rows)]
        seen = _draw_column(col_axes, cond_to_path[cond], window_s, stride_s,
                            show_ylabel=(ci == 0))
        seen_gaits.update(seen)
        label = CONDITION_LABEL.get(cond, cond)
        color = CONDITION_COLOR.get(cond, "#111827")
        all_axes[0, ci].set_title(label, fontsize=12, fontweight="bold", color=color)

    legend_patches = [
        mpatches.Patch(facecolor=GAIT_COLORS[g], label=g, alpha=0.85)
        for g in GAIT_NAMES if g in seen_gaits and g in GAIT_COLORS
    ]
    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            frameon=False,
            fontsize=9.5,
            ncol=len(legend_patches),
            bbox_to_anchor=(0.5, -0.01),
        )

    fig.suptitle("Gait emergence across 0 → 4 m/s ramp + 4 m/s hold",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.07, right=0.99,
                        hspace=0.10, wspace=0.18)

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
