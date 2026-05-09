from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from curriculum_rl.figures._util import (
    CONDITION_COLOR,
    CONDITION_LABEL,
    CONDITION_ORDER,
    apply_style,
    infer_condition,
)

FOOT_LABELS = ["FL", "FR", "RL", "RR"]

GAIT_NAMES = [
    "Stand/Walk",
    "Walk",
    "Amble",
    "Trot",
    "Pace",
    "Fast-trot",
    "Bound/Gallop",
    "Pronk",
    "Unknown",
]

GAIT_COLORS = {
    "Stand/Walk": "#93c5fd",
    "Walk": "#60a5fa",
    "Amble": "#34d399",
    "Trot": "#10b981",
    "Pace": "#f59e0b",
    "Fast-trot": "#f97316",
    "Bound/Gallop": "#ef4444",
    "Pronk": "#a855f7",
    "Unknown": "#9ca3af",
}


def _phase_offset(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n == 0:
        return 0.5
    corr = np.correlate(a.astype(float) - a.mean(), b.astype(float) - b.mean(), mode="full")
    if corr.max() == 0:
        return 0.5
    lag = int(np.argmax(corr)) - (n - 1)
    stride_period = _stride_period(a)
    if stride_period <= 0:
        return 0.5
    return abs(lag / stride_period) % 1.0


def _stride_period(contact: np.ndarray) -> float:
    edges = np.diff(np.concatenate(([0], contact.astype(int), [0])))
    starts = np.where(edges == 1)[0]
    if len(starts) < 2:
        return 0.0
    periods = np.diff(starts)
    return float(np.mean(periods))


def classify_gait(contact: np.ndarray) -> tuple[str, float]:
    n_steps, n_feet = contact.shape
    if n_steps == 0:
        return "Unknown", 0.0
    if n_feet < 4:
        return "Unknown", 0.0

    duty_per_foot = contact.mean(axis=0)
    df = float(duty_per_foot.mean())
    has_flight = bool(np.any(contact.sum(axis=1) == 0))

    if df > 0.90:
        return "Stand/Walk", df
    if df > 0.80:
        return "Walk", df

    fl, fr, rl, rr = contact[:, 0], contact[:, 1], contact[:, 2], contact[:, 3]

    diag1_phase = _phase_offset(fl, rr)
    diag2_phase = _phase_offset(fr, rl)
    diag_sync = (abs(diag1_phase) + abs(diag2_phase)) / 2.0

    lat1_phase = _phase_offset(fl, rl)
    lat2_phase = _phase_offset(fr, rr)
    lat_sync = (abs(lat1_phase) + abs(lat2_phase)) / 2.0

    all_phases = [_phase_offset(fl, fr), _phase_offset(fl, rl), _phase_offset(fl, rr)]
    all_sync = float(np.mean([abs(p) for p in all_phases]))

    if 0.65 < df <= 0.80:
        return "Amble", 1.0 - min(diag_sync, lat_sync)

    if df > 0.50:
        if diag_sync < lat_sync:
            return "Trot", float(np.clip(1.0 - diag_sync, 0.0, 1.0))
        return "Pace", float(np.clip(1.0 - lat_sync, 0.0, 1.0))

    if not has_flight:
        return "Fast-trot", float(1.0 - df)

    if all_sync < 0.15:
        return "Pronk", float(1.0 - all_sync)

    return "Bound/Gallop", float(1.0 - df)


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


def _draw_cell(ax, gaits: list[str], confs: list[float]) -> None:
    if not gaits:
        ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                ha="center", va="center", color="#9ca3af", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return

    counts = Counter(gaits)
    majority_gait, majority_n = counts.most_common(1)[0]
    pct = majority_n / len(gaits)
    mean_conf = float(np.mean([c for g, c in zip(gaits, confs) if g == majority_gait]))
    color = GAIT_COLORS.get(majority_gait, "#9ca3af")

    ax.set_facecolor(color + "33")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#9ca3af")

    ax.text(0.5, 0.62, majority_gait, transform=ax.transAxes,
            ha="center", va="center", fontsize=11, fontweight="bold", color="#111827")
    ax.text(0.5, 0.30, f"{pct:.0%}   conf={mean_conf:.2f}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#374151")


def plot_gait_classification(
    traces_dir: Path,
    out_path: Path,
    num_bins: int = 8,
) -> None:
    cond_to_files = _find_traces(traces_dir)
    if not cond_to_files:
        raise FileNotFoundError(f"no trace npz files in {traces_dir}")

    conditions = [c for c in CONDITION_ORDER if c in cond_to_files]
    n_cols = len(conditions)
    n_rows = num_bins

    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 1.3 * n_rows),
    )
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    bin_width = 4.0 / num_bins

    for ci, cond in enumerate(conditions):
        files = cond_to_files[cond]
        all_gaits: list[list[str]] = [[] for _ in range(num_bins)]
        all_conf: list[list[float]] = [[] for _ in range(num_bins)]

        for fpath in files:
            z = np.load(fpath)
            for b in range(num_bins):
                key = f"contact_b{b}"
                if key not in z.files:
                    continue
                contact = z[key]
                for r in range(contact.shape[0]):
                    gait, conf = classify_gait(contact[r])
                    all_gaits[b].append(gait)
                    all_conf[b].append(conf)

        for b in range(num_bins):
            ax = axes[b, ci]
            _draw_cell(ax, all_gaits[b], all_conf[b])

            if ci == 0:
                v_center = (b + 0.5) * bin_width
                ax.set_ylabel(f"bin {b}\nv={v_center:.2f}", fontsize=9)
            if b == 0:
                ax.set_title(CONDITION_LABEL[cond], fontsize=11, fontweight="bold",
                             color=CONDITION_COLOR[cond])

    fig.suptitle("Gait Classification per (condition, bin)",
                 fontsize=13, fontweight="bold", y=0.995)

    legend_patches = [
        mpatches.Patch(facecolor=GAIT_COLORS[g], label=g, alpha=0.85)
        for g in GAIT_NAMES if g != "Unknown"
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.005),
    )

    fig.subplots_adjust(top=0.96, bottom=0.07, left=0.06, right=0.99,
                        hspace=0.35, wspace=0.12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=Path("src/results/eval_traces"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/gait_classification.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    args = parser.parse_args(argv)
    plot_gait_classification(args.traces_dir, args.out, num_bins=args.num_bins)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
