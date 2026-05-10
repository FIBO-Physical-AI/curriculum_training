from __future__ import annotations

import argparse
import re
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
)
from curriculum_rl.figures.plot_gait_classification import classify_gait

_RAMP_RE = re.compile(r"ramp_([a-z_]+)_seed(\d+)")

GAIT_BAND_COLORS: dict[str, str] = {
    "Stand":     "#cbd5e1",
    "Walk":      "#60a5fa",
    "Trot":      "#fbbf24",
    "Pace":      "#a78bfa",
    "Bound":     "#ec4899",
    "Pronk":     "#fb923c",
    "Irregular": "#9ca3af",
    "n/a":       "#e5e7eb",
}


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


def _load_ramp(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    z = np.load(path)
    vcmd = np.asarray(z["vcmd"], dtype=float)
    vx = np.asarray(z["vx"], dtype=float)
    contact = np.asarray(z["contact"]).astype(bool)
    sim_dt = float(z["sim_dt"]) if "sim_dt" in z.files else 0.02
    if len(vcmd) > 1:
        diff = np.diff(vcmd)
        nonpos = np.where(diff <= 0)[0]
        ramp_end = int(nonpos[0]) + 1 if len(nonpos) else len(vcmd)
    else:
        ramp_end = len(vcmd)
    return vcmd[:ramp_end], vx[:ramp_end], contact[:ramp_end], sim_dt


def _classify_windows(
    contact: np.ndarray, vcmd: np.ndarray, sim_dt: float,
    window_s: float, stride_s: float,
) -> list[tuple[float, float, str]]:
    n = contact.shape[0]
    w = max(int(window_s / sim_dt), 20)
    s = max(int(stride_s / sim_dt), 1)
    raw: list[tuple[float, float, str]] = []
    for start in range(0, n - w + 1, s):
        end = start + w
        gait = classify_gait(contact[start:end])
        raw.append((float(vcmd[start]), float(vcmd[min(end - 1, n - 1)]), gait))
    if not raw:
        return raw
    merged: list[list] = [list(raw[0])]
    for v_lo, v_hi, gait in raw[1:]:
        if gait == merged[-1][2]:
            merged[-1][1] = v_hi
        else:
            merged.append([v_lo, v_hi, gait])
    return [(m[0], m[1], m[2]) for m in merged]


def plot_gait_transition(
    ramps_dir: Path,
    out_path: Path,
    window_s: float = 2.0,
    stride_s: float = 0.25,
) -> None:
    cond_to_path = _find_ramps(ramps_dir)
    if not cond_to_path:
        raise FileNotFoundError(f"no ramp_*.npz files in {ramps_dir}")

    conditions = [c for c in CONDITION_ORDER if c in cond_to_path]
    if not conditions:
        conditions = sorted(cond_to_path.keys())
    n_cond = len(conditions)

    ramp_data: dict[str, tuple] = {}
    band_data: dict[str, list] = {}
    for cond in conditions:
        vcmd, vx, contact, sim_dt = _load_ramp(cond_to_path[cond])
        ramp_data[cond] = (vcmd, vx, sim_dt)
        band_data[cond] = _classify_windows(contact, vcmd, sim_dt, window_s, stride_s)

    v_lo = float(min(d[0].min() for d in ramp_data.values()))
    v_hi = float(max(d[0].max() for d in ramp_data.values()))

    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False

    fig, axes = plt.subplots(
        n_cond, 1,
        figsize=(10.0, 2.4 * n_cond + 0.6),
        sharex=True,
    )
    if n_cond == 1:
        axes = np.array([axes])

    used_gaits: list[str] = []

    for ci, cond in enumerate(conditions):
        ax = axes[ci]
        vcmd, vx, _ = ramp_data[cond]

        for gv_lo, gv_hi, gait in band_data[cond]:
            color = GAIT_BAND_COLORS.get(gait, "#e5e7eb")
            ax.axvspan(gv_lo, gv_hi, color=color, alpha=0.45, lw=0)
            if gait not in used_gaits:
                used_gaits.append(gait)

        ax.plot(vcmd, vcmd, color="#9ca3af", lw=1.2, ls="--", label=r"$v_{cmd}$")
        ax.plot(vcmd, vx, color="#1f2937", lw=1.6, label=r"$v_x$")
        ax.set_xlim(v_lo, v_hi)
        v_top = max(float(vx.max()), v_hi) + 0.3
        v_bot = min(0.0, float(vx.min()) - 0.3)
        ax.set_ylim(v_bot, v_top)
        ax.set_ylabel("velocity (m/s)")
        ax.set_title(CONDITION_LABEL.get(cond, cond),
                     color=CONDITION_COLOR.get(cond, "#111827"),
                     fontweight="bold", loc="left")
        ax.legend(loc="upper left", frameon=False, ncol=2)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("commanded velocity (m/s)")

    handles = [Patch(facecolor=GAIT_BAND_COLORS.get(g, "#e5e7eb"),
                     alpha=0.55, label=g) for g in used_gaits]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 8), frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=10)

    fig.suptitle("Gait Transition over Velocity Ramp (0 → 4 m/s)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.92, bottom=0.13, left=0.07, right=0.98, hspace=0.35)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ramps-dir", type=Path, default=Path("src/results"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/gait_transition.png"))
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--stride", type=float, default=0.25)
    args = parser.parse_args(argv)
    plot_gait_transition(args.ramps_dir, args.out, window_s=args.window,
                         stride_s=args.stride)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
