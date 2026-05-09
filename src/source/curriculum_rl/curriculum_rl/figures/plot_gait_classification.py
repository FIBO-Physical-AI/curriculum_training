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
    "Four-Beat Walking",
    "Two-Beat Walking",
    "Trotting",
    "Fly Trotting",
    "Pacing",
    "Bounding",
    "Unknown",
]

GAIT_COLORS = {
    "Four-Beat Walking": "#60a5fa",
    "Two-Beat Walking": "#34d399",
    "Trotting": "#f59e0b",
    "Fly Trotting": "#ef4444",
    "Pacing": "#a78bfa",
    "Bounding": "#ec4899",
    "Unknown": "#9ca3af",
}


def _stride_period(contact: np.ndarray) -> float:
    edges = np.diff(np.concatenate(([0], contact.astype(int), [0])))
    starts = np.where(edges == 1)[0]
    if len(starts) < 2:
        return 0.0
    return float(np.mean(np.diff(starts)))


def _phase_offset(ref: np.ndarray, other: np.ndarray, period: float) -> float:
    n = len(ref)
    if n == 0 or period <= 0:
        return 0.0
    a = ref.astype(float) - ref.mean()
    b = other.astype(float) - other.mean()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    corr = np.correlate(b, a, mode="full")
    lag = int(np.argmax(corr)) - (n - 1)
    return (lag / period) % 1.0


def _cyclic_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs(a - b)
    return float(np.minimum(d, 1.0 - d).mean())


_TROT_TARGET = np.array([0.0, 0.5, 0.5, 0.0])
_PACE_TARGET = np.array([0.0, 0.5, 0.0, 0.5])
_BOUND_TARGET = np.array([0.0, 0.0, 0.5, 0.5])
_WALK_TARGETS = [
    np.array([0.0, 0.5, 0.25, 0.75]),
    np.array([0.0, 0.5, 0.75, 0.25]),
    np.array([0.0, 0.25, 0.5, 0.75]),
    np.array([0.0, 0.75, 0.5, 0.25]),
]


def classify_gait(contact: np.ndarray) -> tuple[str, float]:
    n_steps, n_feet = contact.shape
    if n_steps < 20 or n_feet < 4:
        return "Unknown", 0.0

    fl, fr, rl, rr = contact[:, 0], contact[:, 1], contact[:, 2], contact[:, 3]

    duty = float(contact.mean())
    n_in_contact = contact.sum(axis=1)
    flight_frac = float(np.mean(n_in_contact == 0))
    full_support_frac = float(np.mean(n_in_contact == 4))

    if duty > 0.97:
        return "Unknown", 0.4

    period = _stride_period(fl)
    if period < 4.0:
        if flight_frac > 0.04:
            return "Fly Trotting", float(np.clip(flight_frac * 5, 0.2, 1.0))
        if duty > 0.75:
            return "Four-Beat Walking", 0.3
        return "Trotting", 0.3

    phases = np.array([
        0.0,
        _phase_offset(fl, fr, period),
        _phase_offset(fl, rl, period),
        _phase_offset(fl, rr, period),
    ])

    score_trot = _cyclic_dist(phases, _TROT_TARGET)
    score_pace = _cyclic_dist(phases, _PACE_TARGET)
    score_bound = _cyclic_dist(phases, _BOUND_TARGET)
    score_walk = min(_cyclic_dist(phases, t) for t in _WALK_TARGETS)

    pattern_name, pattern_score = min(
        [("trot", score_trot), ("pace", score_pace),
         ("bound", score_bound), ("walk", score_walk)],
        key=lambda x: x[1],
    )
    conf = float(np.clip(1.0 - 4.0 * pattern_score, 0.0, 1.0))

    if flight_frac > 0.04:
        if pattern_name == "bound":
            return "Bounding", conf
        return "Fly Trotting", float(max(conf, np.clip(flight_frac * 5, 0.3, 1.0)))

    if pattern_name == "walk" and duty > 0.62:
        return "Four-Beat Walking", conf

    if pattern_name == "pace":
        return "Pacing", conf

    if pattern_name == "bound":
        return "Bounding", conf

    if full_support_frac > 0.12 and duty > 0.55:
        return "Two-Beat Walking", conf

    return "Trotting", conf


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
    seen_gaits: set[str] = set()

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
            seen_gaits.update(all_gaits[b])

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
        for g in GAIT_NAMES if g in seen_gaits
    ]
    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=len(legend_patches),
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
