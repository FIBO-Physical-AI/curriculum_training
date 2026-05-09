from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

from curriculum_rl.figures._util import apply_style, CONDITION_COLOR, CONDITION_LABEL, CONDITION_ORDER
from curriculum_rl.figures.plot_gait_classification import classify_gait, GAIT_COLORS, GAIT_NAMES

FOOT_LABELS = ["FL", "FR", "RL", "RR"]
FOOT_COLORS = ["#dc2626", "#ea580c", "#2563eb", "#059669"]

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


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) <= window:
        return x.astype(float)
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    padded = np.concatenate([np.full(pad, x[0]), x.astype(float), np.full(pad, x[-1])])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(x)]


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
    has_wz = "wz" in z.files
    wz = z["wz"] if has_wz else np.zeros_like(vx)
    n_feet = contact.shape[1]

    x = vcmd
    x_lo = float(x.min())
    x_hi = float(x.max())

    centers_idx, gaits = _classify_windows(contact, sim_dt, window_s=window_s, stride_s=stride_s)
    seen = set(gaits)

    ax_v, ax_g, ax_d = axes

    ax_v.plot(x, vcmd, color="#9ca3af", lw=1.6, ls="--", label=r"$v_{cmd}$")
    ax_v.plot(x, vx, color="#1f2937", lw=1.8, label=r"$v_x$")
    ax_v.set_xlim(x_lo, x_hi)
    v_top = float(max(vcmd.max(), vx.max())) + 0.3
    v_bot = min(0.0, float(vx.min()) - 0.3)
    ax_v.set_ylim(v_bot, v_top)
    if show_ylabel:
        ax_v.set_ylabel("velocity (m/s)", fontsize=10)
    ax_v.tick_params(labelbottom=False)
    ax_v.grid(True, alpha=0.25)

    if has_wz:
        ax_w = ax_v.twinx()
        ax_w.plot(x, wz, color="#dc2626", lw=1.0, alpha=0.7, label=r"$\omega_z$")
        wz_max = float(np.max(np.abs(wz))) if len(wz) else 1.0
        ax_w.set_ylim(-max(wz_max, 0.5) * 1.2, max(wz_max, 0.5) * 1.2)
        ax_w.tick_params(axis="y", colors="#dc2626", labelsize=8)
        if not show_ylabel:
            ax_w.set_ylabel(r"$\omega_z$ (rad/s)", color="#dc2626", fontsize=9)
        else:
            ax_w.set_yticklabels([])
        ax_w.axhline(0, color="#dc2626", lw=0.4, alpha=0.3)
        lines_v, labels_v = ax_v.get_legend_handles_labels()
        lines_w, labels_w = ax_w.get_legend_handles_labels()
        ax_v.legend(lines_v + lines_w, labels_v + labels_w,
                    loc="upper left", frameon=False, fontsize=9)
    else:
        ax_v.legend(loc="upper left", frameon=False, fontsize=9)

    runs = _merge_runs(gaits)
    if len(centers_idx):
        v_at_center = vcmd[centers_idx]
        for s_idx, e_idx, gait in runs:
            if e_idx <= s_idx:
                continue
            v_left = float(v_at_center[s_idx]) if s_idx > 0 else x_lo
            v_right = float(v_at_center[min(e_idx, len(v_at_center) - 1)]) \
                if e_idx < len(v_at_center) else x_hi
            color = GAIT_COLORS.get(gait, "#9ca3af")
            width = v_right - v_left
            if width <= 0:
                continue
            ax_g.add_patch(plt.Rectangle((v_left, 0.0), width, 1.0,
                                         facecolor=color, alpha=0.9,
                                         edgecolor="white", lw=0.6))
    ax_g.set_xlim(x_lo, x_hi)
    ax_g.set_ylim(0.0, 1.0)
    ax_g.set_yticks([])
    ax_g.set_xticks([])
    if show_ylabel:
        ax_g.set_ylabel("gait", fontsize=10, rotation=0,
                        ha="right", va="center", labelpad=12)

    duty_window = max(int(window_s / sim_dt), 1)
    for foot_i in range(min(n_feet, 4)):
        c = contact[:, foot_i].astype(float)
        duty = _rolling_mean(c, duty_window)
        ax_d.plot(x, duty, color=FOOT_COLORS[foot_i], lw=1.4,
                  label=FOOT_LABELS[foot_i], alpha=0.9)
    ax_d.set_xlim(x_lo, x_hi)
    ax_d.set_ylim(0.0, 1.0)
    ax_d.axhline(0.5, color="#9ca3af", lw=0.6, ls=":")
    ax_d.axhline(0.75, color="#9ca3af", lw=0.6, ls=":")
    if show_ylabel:
        ax_d.set_ylabel("duty factor", fontsize=10)
    ax_d.set_xlabel("commanded velocity (m/s)", fontsize=10)
    ax_d.grid(True, alpha=0.25)
    if show_ylabel:
        ax_d.legend(loc="lower left", frameon=False, fontsize=8.5,
                    ncol=4, handlelength=1.2, columnspacing=1.0)

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
        figsize=(6.5 * n_cols, 5.4),
        sharex="col",
        gridspec_kw={"height_ratios": [1.5, 0.30, 1.3]},
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
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle("Gait emergence across 0 → 4 m/s ramp + 4 m/s hold",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.07, right=0.99,
                        hspace=0.10, wspace=0.20)

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
