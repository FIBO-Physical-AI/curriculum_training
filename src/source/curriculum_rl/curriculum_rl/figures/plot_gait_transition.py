from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from curriculum_rl.figures._util import apply_style, CONDITION_COLOR, CONDITION_LABEL, CONDITION_ORDER

FOOT_LABELS = ["FL", "FR", "RL", "RR"]
FOOT_COLORS = ["#dc2626", "#ea580c", "#2563eb", "#059669"]

_RAMP_RE = re.compile(r"ramp_([a-z_]+)_seed(\d+)")


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
    show_ylabel: bool,
) -> None:
    z = np.load(ramp_path)
    vcmd = z["vcmd"]
    vx = z["vx"]
    contact = z["contact"].astype(bool)
    sim_dt = float(z["sim_dt"])
    has_wz = "wz" in z.files
    wz = z["wz"] if has_wz else None
    n_feet = contact.shape[1]

    x = vcmd
    x_lo = float(x.min())
    x_hi = float(x.max())

    ax_v, ax_d = axes

    ax_v.plot(x, vcmd, color="#9ca3af", lw=1.6, ls="--", label=r"$v_{cmd}$")
    ax_v.plot(x, vx, color="#1f2937", lw=1.8, label=r"$v_x$")
    ax_v.set_xlim(x_lo, x_hi)
    v_top = float(max(vcmd.max(), vx.max())) + 0.3
    v_bot = min(0.0, float(vx.min()) - 0.3)
    ax_v.set_ylim(v_bot, v_top)
    if show_ylabel:
        ax_v.set_ylabel("velocity (m/s)")
    ax_v.tick_params(labelbottom=False)
    ax_v.grid(True, alpha=0.25)

    if has_wz:
        ax_w = ax_v.twinx()
        ax_w.plot(x, wz, color="#dc2626", lw=1.0, alpha=0.7, label=r"$\omega_z$")
        wz_max = float(np.max(np.abs(wz))) if len(wz) else 1.0
        ax_w.set_ylim(-max(wz_max, 0.5) * 1.2, max(wz_max, 0.5) * 1.2)
        ax_w.tick_params(axis="y", colors="#dc2626")
        ax_w.set_ylabel(r"$\omega_z$ (rad/s)", color="#dc2626")
        lines_v, labels_v = ax_v.get_legend_handles_labels()
        lines_w, labels_w = ax_w.get_legend_handles_labels()
        ax_v.legend(lines_v + lines_w, labels_v + labels_w,
                    loc="upper left", frameon=False)
    else:
        ax_v.legend(loc="upper left", frameon=False)

    duty_window = max(int(window_s / sim_dt), 1)
    for foot_i in range(min(n_feet, 4)):
        c = contact[:, foot_i].astype(float)
        duty = _rolling_mean(c, duty_window)
        ax_d.plot(x, duty, color=FOOT_COLORS[foot_i], lw=1.4,
                  label=FOOT_LABELS[foot_i], alpha=0.9)
    ax_d.set_xlim(x_lo, x_hi)
    ax_d.set_ylim(0.0, 1.0)
    if show_ylabel:
        ax_d.set_ylabel("duty factor")
        ax_d.legend(loc="lower left", frameon=False, ncol=4,
                    handlelength=1.2, columnspacing=1.0)
    ax_d.set_xlabel("commanded velocity (m/s)")
    ax_d.grid(True, alpha=0.25)


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
    n_rows = 2

    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, all_axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.0 * n_cols, 5.0),
        sharex="col",
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    if n_cols == 1:
        all_axes = all_axes.reshape(n_rows, 1)

    for ci, cond in enumerate(conditions):
        col_axes = [all_axes[r, ci] for r in range(n_rows)]
        _draw_column(col_axes, cond_to_path[cond], window_s,
                     show_ylabel=(ci == 0))
        all_axes[0, ci].set_title(
            CONDITION_LABEL.get(cond, cond),
            color=CONDITION_COLOR.get(cond, "#111827"),
        )

    fig.subplots_adjust(top=0.93, bottom=0.10, left=0.07, right=0.96,
                        hspace=0.12, wspace=0.22)

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
    plot_gait_transition(args.ramps_dir, args.out, window_s=args.window)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
