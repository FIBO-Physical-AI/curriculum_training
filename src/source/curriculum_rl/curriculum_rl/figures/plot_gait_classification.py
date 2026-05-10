from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from curriculum_rl.figures._util import (
    CONDITION_COLOR,
    CONDITION_LABEL,
    CONDITION_ORDER,
    apply_style,
    infer_condition,
)

FOOT_LABELS = ["FL", "FR", "RL", "RR"]
FOOT_COLORS = ["#dc2626", "#ea580c", "#2563eb", "#059669"]


GAIT_TEMPLATES: dict[str, dict] = {
    "Walk":  {"phases": [0.0, 0.5, 0.75, 0.25], "duty": 0.65},
    "Trot":  {"phases": [0.0, 0.5, 0.5,  0.0],  "duty": 0.50},
    "Pace":  {"phases": [0.0, 0.5, 0.0,  0.5],  "duty": 0.50},
    "Bound": {"phases": [0.0, 0.0, 0.5,  0.5],  "duty": 0.40},
    "Pronk": {"phases": [0.0, 0.0, 0.0,  0.0],  "duty": 0.30},
}

PHASE_TOLERANCE = 0.10
STAND_DUTY = 0.95
MIN_STRIDE_STARTS = 2
MAX_STRIDE_PERIOD = 50


def _stance_starts(mask: np.ndarray) -> np.ndarray:
    edges = np.diff(np.concatenate(([False], mask.astype(bool), [False])).astype(int))
    return np.where(edges == 1)[0]


def _stride_period(contact_one_foot: np.ndarray) -> float:
    starts = _stance_starts(contact_one_foot)
    if len(starts) < 2:
        return 0.0
    return float(np.mean(np.diff(starts)))


def _first_stance(mask: np.ndarray) -> int:
    starts = _stance_starts(mask)
    return int(starts[0]) if len(starts) else 0


def _phase_offset_touchdown(
    ref_starts: np.ndarray,
    other_starts: np.ndarray,
    period: float,
) -> float:
    if period <= 0 or len(ref_starts) < 1 or len(other_starts) < 1:
        return 0.0
    phases = []
    for r in ref_starts:
        deltas = other_starts - r
        forward = deltas[deltas >= 0]
        if len(forward):
            d = float(forward.min())
        else:
            d = float((other_starts.max() - r) % period)
        phases.append((d / period) % 1.0)
    if not phases:
        return 0.0
    angles = 2.0 * np.pi * np.asarray(phases)
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float((mean_angle / (2.0 * np.pi)) % 1.0)


def _cyclic_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs(a - b)
    return float(np.minimum(d, 1.0 - d).mean())


def _cyclic_dist_max(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs(a - b)
    return float(np.minimum(d, 1.0 - d).max())


def _template_score(name: str, phases: np.ndarray, template: np.ndarray) -> float:
    if name == "Pronk":
        return _cyclic_dist_max(phases, template)
    return _cyclic_dist(phases, template)


def classify_gait(contact: np.ndarray) -> str:
    n_steps, n_feet = contact.shape
    if n_feet < 4:
        return "n/a"

    fl, fr, rl, rr = contact[:, 0], contact[:, 1], contact[:, 2], contact[:, 3]
    duty = float(contact.mean())

    fl_starts = _stance_starts(fl)
    fr_starts = _stance_starts(fr)
    rl_starts = _stance_starts(rl)
    rr_starts = _stance_starts(rr)

    no_stride = all(
        len(s) < MIN_STRIDE_STARTS
        for s in (fl_starts, fr_starts, rl_starts, rr_starts)
    )
    if duty > STAND_DUTY or no_stride:
        return "Stand"

    if len(fl_starts) < MIN_STRIDE_STARTS:
        return "n/a"
    period = float(np.mean(np.diff(fl_starts)))
    if period <= 0:
        return "n/a"
    if period > MAX_STRIDE_PERIOD:
        return "Stand"

    phases = np.array([
        0.0,
        _phase_offset_touchdown(fl_starts, fr_starts, period),
        _phase_offset_touchdown(fl_starts, rl_starts, period),
        _phase_offset_touchdown(fl_starts, rr_starts, period),
    ])

    scores = {
        name: _template_score(name, phases, np.array(t["phases"]))
        for name, t in GAIT_TEMPLATES.items()
    }
    best_name = min(scores, key=scores.get)
    if scores[best_name] > PHASE_TOLERANCE:
        return "Irregular"
    return best_name


def classify_gait_with_score(contact: np.ndarray) -> tuple[str, float]:
    n_steps, n_feet = contact.shape
    if n_feet < 4:
        return "n/a", float("inf")

    fl = contact[:, 0]
    duty = float(contact.mean())
    fl_starts = _stance_starts(fl)
    fr_starts = _stance_starts(contact[:, 1])
    rl_starts = _stance_starts(contact[:, 2])
    rr_starts = _stance_starts(contact[:, 3])

    no_stride = all(
        len(s) < MIN_STRIDE_STARTS
        for s in (fl_starts, fr_starts, rl_starts, rr_starts)
    )
    if duty > STAND_DUTY or no_stride:
        return "Stand", 0.0

    if len(fl_starts) < MIN_STRIDE_STARTS:
        return "n/a", float("inf")
    period = float(np.mean(np.diff(fl_starts)))
    if period <= 0:
        return "n/a", float("inf")
    if period > MAX_STRIDE_PERIOD:
        return "Stand", 0.0

    phases = np.array([
        0.0,
        _phase_offset_touchdown(fl_starts, fr_starts, period),
        _phase_offset_touchdown(fl_starts, rl_starts, period),
        _phase_offset_touchdown(fl_starts, rr_starts, period),
    ])
    scores = {
        name: _template_score(name, phases, np.array(t["phases"]))
        for name, t in GAIT_TEMPLATES.items()
    }
    best_name = min(scores, key=scores.get)
    best_score = scores[best_name]
    if best_score > PHASE_TOLERANCE:
        return "Irregular", best_score
    return best_name, best_score


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
    n_roll = contact.shape[0]
    err = np.abs(vx.mean(axis=1) - v_cmd)
    survives = np.array([bool(contact[i].any()) for i in range(n_roll)])
    if survives.any():
        idx = np.where(survives)[0]
        return int(idx[np.argmin(err[idx])])
    return int(np.argmin(err))


def _classify_majority(contact_all: np.ndarray) -> tuple[str, float]:
    labels: list[str] = []
    score_by_label: dict[str, list[float]] = {}
    for i in range(contact_all.shape[0]):
        if not bool(contact_all[i].any()):
            continue
        label, score = classify_gait_with_score(contact_all[i])
        labels.append(label)
        score_by_label.setdefault(label, []).append(score)
    if not labels:
        return "n/a", float("inf")
    most, _ = Counter(labels).most_common(1)[0]
    mean_score = float(np.mean(score_by_label[most])) if score_by_label.get(most) else float("inf")
    return most, mean_score


def _draw_observed(ax, contact_window: np.ndarray, sim_dt: float) -> None:
    n_show = contact_window.shape[0]
    t = np.arange(n_show) * sim_dt
    for foot_i in range(4):
        y_low = 3 - foot_i - 0.4
        y_high = 3 - foot_i + 0.4
        mask = contact_window[:, foot_i].astype(bool)
        if not mask.any():
            continue
        edges = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            e_clip = min(e, n_show - 1)
            if e_clip <= s:
                e_clip = min(s + 1, n_show - 1)
            ax.fill_between([t[s], t[e_clip]], y_low, y_high,
                            color=FOOT_COLORS[foot_i], lw=0)


def _draw_template(ax, gait: str, period_steps: float, fl_offset: int,
                   n_show: int, sim_dt: float) -> None:
    tmpl = GAIT_TEMPLATES.get(gait)
    if tmpl is None or period_steps < 4.0:
        return
    duty_steps = tmpl["duty"] * period_steps
    for foot_i, phase in enumerate(tmpl["phases"]):
        s0 = fl_offset + phase * period_steps
        y_low = 3 - foot_i - 0.42
        y_high = 3 - foot_i + 0.42
        for k in range(-3, 200):
            s = s0 + k * period_steps
            e = s + duty_steps
            if s >= n_show:
                break
            if e <= 0:
                continue
            sx = max(int(round(s)), 0)
            ex = min(int(round(e)), n_show - 1)
            if ex <= sx:
                continue
            ax.fill_between([sx * sim_dt, ex * sim_dt], y_low, y_high,
                            facecolor="none",
                            edgecolor=FOOT_COLORS[foot_i],
                            hatch="///", linewidth=0.6, alpha=0.55)


def plot_gait_classification(
    traces_dir: Path,
    out_path: Path,
    num_bins: int = 8,
    t_window_s: float = 1.5,
) -> None:
    cond_to_files = _find_traces(traces_dir)
    if not cond_to_files:
        raise FileNotFoundError(f"no trace npz files in {traces_dir}")

    conditions = [c for c in CONDITION_ORDER if c in cond_to_files]
    n_rows = len(conditions)
    n_cols = num_bins

    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols, 1.7 * n_rows),
        sharex=True, sharey=True,
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for ri, cond in enumerate(conditions):
        files = cond_to_files[cond]
        z = np.load(files[0])
        sim_dt = float(z["sim_dt"]) if "sim_dt" in z.files else 0.02
        for b in range(num_bins):
            ax = axes[ri, b]
            key_c = f"contact_b{b}"
            key_vx = f"vx_b{b}"
            key_vc = f"vcmd_b{b}"
            if key_c not in z.files:
                ax.set_axis_off()
                continue
            contact = z[key_c]
            vx = z[key_vx]
            v_cmd = float(z[key_vc]) if key_vc in z.files else 0.0
            r = _pick_rollout(contact, vx, v_cmd)
            gait, score = _classify_majority(contact)

            n_show = min(int(t_window_s / sim_dt), contact.shape[1])
            window = contact[r, :n_show].astype(bool)
            fl_window = window[:, 0]
            period_steps = _stride_period(fl_window)
            fl_offset = _first_stance(fl_window)

            if score < PHASE_TOLERANCE and gait in GAIT_TEMPLATES:
                _draw_template(ax, gait, period_steps, fl_offset, n_show, sim_dt)
            _draw_observed(ax, window, sim_dt)

            ax.set_xlim(0, t_window_s)
            ax.set_ylim(-0.7, 3.7)
            ax.set_yticks([3, 2, 1, 0])
            if b == 0:
                ax.set_yticklabels(FOOT_LABELS, fontsize=9)
                ax.set_ylabel(CONDITION_LABEL.get(cond, cond),
                              fontsize=11, fontweight="bold",
                              color=CONDITION_COLOR.get(cond, "#111827"))
            if ri == 0:
                ax.set_title(f"v={v_cmd:.2f} m/s\n{gait}", fontsize=10)
            else:
                ax.set_title(gait, fontsize=10)
            if ri == n_rows - 1:
                ax.set_xlabel("t (s)", fontsize=9)
                ax.tick_params(axis="x", labelsize=8)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            ax.grid(False)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#374151", label="observed contact"),
        Patch(facecolor="none", edgecolor="#374151", hatch="///", label="canonical template"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=10)

    fig.suptitle("Gait Classification per Velocity Bin (observed vs canonical template)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.06, right=0.99,
                        hspace=0.35, wspace=0.10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=Path("src/results/eval_traces"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/gait_classification.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--t-window", type=float, default=1.5)
    args = parser.parse_args(argv)
    plot_gait_classification(args.traces_dir, args.out, num_bins=args.num_bins,
                             t_window_s=args.t_window)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
