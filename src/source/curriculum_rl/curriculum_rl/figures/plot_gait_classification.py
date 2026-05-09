from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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


def hildebrand_metrics(contact: np.ndarray) -> tuple[float, float] | None:
    """Return (duty_factor_pct, lateral_lag_pct) per Hildebrand convention.

    duty_factor: mean fraction of stride in stance, ×100.
    lateral_lag: phase offset of FL relative to RL (left-side ipsilateral pair),
                 in % of stride period.
    """
    n_steps, n_feet = contact.shape
    if n_steps < 20 or n_feet < 4:
        return None
    fl, rl = contact[:, 0], contact[:, 2]
    period = _stride_period(rl)
    if period < 4.0:
        return None
    duty = float(contact.mean()) * 100.0
    lag = _phase_offset(rl, fl, period) * 100.0
    if not (0.0 <= duty <= 100.0):
        return None
    return duty, lag


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


def _draw_hildebrand_background(ax) -> None:
    regions = [
        ((25, 75, 25, 50), "#fde68a", "Lateral-sequence\nwalk"),
        ((25, 75, 50, 75), "#bfdbfe", "Diagonal-sequence\nwalk"),
        ((75, 100, 0, 100), "#d1d5db", "Stand /\nstatic"),
        ((25, 50, 40, 60), "#fed7aa", "Trot"),
        ((25, 50, 0, 15), "#ddd6fe", "Pace"),
        ((25, 50, 85, 100), "#ddd6fe", "Pace"),
        ((0, 25, 25, 75), "#fecaca", "Run /\nfly trot"),
    ]
    for (x0, x1, y0, y1, color, label) in (
        (r[0][0], r[0][1], r[0][2], r[0][3], r[1], r[2]) for r in regions
    ):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    facecolor=color, edgecolor="none", alpha=0.45))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label,
                ha="center", va="center", fontsize=8, color="#374151", alpha=0.9)


def _draw_canonical_targets(ax) -> None:
    targets = [
        ("Trot",  35, 50, "x"),
        ("Pace",  50,  0, "x"),
        ("Walk",  75, 25, "x"),
        ("Bound", 35, 50, "+"),
    ]
    for name, x, y, marker in targets:
        ax.plot(x, y, marker=marker, color="#111827", ms=8, mew=1.5, zorder=4)


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

    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 5.0), sharey=True)
    if n_cols == 1:
        axes = np.array([axes])

    bin_width = 4.0 / num_bins
    cmap = plt.get_cmap("viridis")

    for ci, cond in enumerate(conditions):
        ax = axes[ci]
        _draw_hildebrand_background(ax)

        files = cond_to_files[cond]
        all_pts: list[tuple[float, float, int]] = []
        for fpath in files:
            z = np.load(fpath)
            for b in range(num_bins):
                key = f"contact_b{b}"
                if key not in z.files:
                    continue
                contact = z[key]
                for r in range(contact.shape[0]):
                    m = hildebrand_metrics(contact[r])
                    if m is None:
                        continue
                    all_pts.append((m[0], m[1], b))

        if all_pts:
            xs = np.array([p[0] for p in all_pts])
            ys = np.array([p[1] for p in all_pts])
            cs = np.array([p[2] for p in all_pts]) / max(num_bins - 1, 1)
            ax.scatter(xs, ys, c=cs, cmap=cmap, vmin=0, vmax=1,
                       s=24, alpha=0.55, edgecolors="white", linewidths=0.4, zorder=3)

        _draw_canonical_targets(ax)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("duty factor (% stride in stance)", fontsize=10)
        if ci == 0:
            ax.set_ylabel("lateral lag (FL − RL, % stride)", fontsize=10)
        ax.set_title(CONDITION_LABEL.get(cond, cond),
                     fontsize=12, fontweight="bold",
                     color=CONDITION_COLOR.get(cond, "#111827"))
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.set_aspect("equal")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=4.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), location="right",
                        shrink=0.7, pad=0.02, aspect=22)
    cbar.set_label("commanded velocity bin centre (m/s)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    target_legend = [
        mlines.Line2D([], [], color="#111827", marker="x", lw=0, ms=8, mew=1.5,
                      label="canonical Trot/Pace/Walk"),
        mlines.Line2D([], [], color="#111827", marker="+", lw=0, ms=8, mew=1.5,
                      label="canonical Bound"),
    ]
    fig.legend(handles=target_legend, loc="lower center", frameon=False,
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Hildebrand symmetric-gait plot — duty factor vs ipsilateral phase lag",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.06, right=0.92, wspace=0.10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
