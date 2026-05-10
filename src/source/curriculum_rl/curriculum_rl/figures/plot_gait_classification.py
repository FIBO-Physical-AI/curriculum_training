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
    "Pace":                {"phases": [0.0, 0.5, 0.0,  0.5],  "duty": 0.50},
    "Walk-LS":             {"phases": [0.0, 0.5, 0.75, 0.25], "duty": 0.65},
    "Trot":                {"phases": [0.0, 0.5, 0.5,  0.0],  "duty": 0.50},
    "Walk-DS":             {"phases": [0.0, 0.5, 0.25, 0.75], "duty": 0.65},
    "Bound":               {"phases": [0.0, 0.0, 0.5,  0.5],  "duty": 0.40},
    "Pronk":               {"phases": [0.0, 0.0, 0.0,  0.0],  "duty": 0.30},
    "Half-bound-R":        {"phases": [0.0, 0.0, 0.55, 0.45], "duty": 0.40},
    "Half-bound-L":        {"phases": [0.0, 0.0, 0.45, 0.55], "duty": 0.40},
    "Canter-R":            {"phases": [0.0, 0.5, 0.4,  0.0],  "duty": 0.40},
    "Canter-L":            {"phases": [0.0, 0.5, 0.0,  0.4],  "duty": 0.40},
    "Transverse-gallop-R": {"phases": [0.0, 0.1, 0.6,  0.5],  "duty": 0.35},
    "Transverse-gallop-L": {"phases": [0.0, 0.1, 0.5,  0.6],  "duty": 0.35},
    "Rotary-gallop-R":     {"phases": [0.0, 0.1, 0.5,  0.6],  "duty": 0.35},
    "Rotary-gallop-L":     {"phases": [0.0, 0.1, 0.6,  0.5],  "duty": 0.35},
}

ASYMMETRIC_NAMES = [
    "Bound", "Pronk",
    "Half-bound-R", "Half-bound-L",
    "Canter-R", "Canter-L",
    "Transverse-gallop-R", "Transverse-gallop-L",
    "Rotary-gallop-R", "Rotary-gallop-L",
]

CANONICAL_POINTS: list[tuple[str, float]] = [
    ("Pace",    0.0),
    ("Walk-LS", 0.25),
    ("Trot",    0.5),
    ("Walk-DS", 0.75),
]

QUADRANTS: list[tuple[str, float, float]] = [
    ("LSLC", 0.0,  0.25),
    ("LSDC", 0.25, 0.5),
    ("DSDC", 0.5,  0.75),
    ("DSLC", 0.75, 1.0),
]

ASYMMETRIC_LABEL_MAP: dict[str, str] = {
    "Bound":               "Bound",
    "Pronk":               "Pronk",
    "Half-bound-R":        "Half-bound (R-lead)",
    "Half-bound-L":        "Half-bound (L-lead)",
    "Canter-R":            "Canter (R-lead)",
    "Canter-L":            "Canter (L-lead)",
    "Transverse-gallop-R": "Transverse gallop (R)",
    "Transverse-gallop-L": "Transverse gallop (L)",
    "Rotary-gallop-R":     "Rotary gallop (R)",
    "Rotary-gallop-L":     "Rotary gallop (L)",
}

WALK_LABEL_MAP: dict[str, str] = {
    "Pace":    "Pace (walking)",
    "Walk-LS": "Walk-LS",
    "Trot":    "Trot (walking)",
    "Walk-DS": "Walk-DS",
    "LSLC":    "LSLC walk",
    "LSDC":    "LSDC walk",
    "DSDC":    "DSDC walk",
    "DSLC":    "DSLC walk",
}

RUN_LABEL_MAP: dict[str, str] = {
    "Pace":    "Pace (running)",
    "Walk-LS": "Amble",
    "Trot":    "Trot (running, flying-trot)",
    "Walk-DS": "DS-amble",
    "LSLC":    "LSLC run",
    "LSDC":    "LSDC run",
    "DSDC":    "DSDC run",
    "DSLC":    "DSLC run",
}

STAND_DUTY = 0.95
MIN_STRIDE_STARTS = 2
MAX_STRIDE_PERIOD = 50
BETA_RUN_THRESHOLD = 0.50
PHI_TOLERANCE = 0.10
PHASE_TOLERANCE = PHI_TOLERANCE
LR_ASYMMETRY_THRESHOLD = 0.20
DAP_DISSOCIATION_THRESHOLD = 0.05


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


def _cyclic_diff_scalar(a: float, b: float) -> float:
    d = abs(float(a) - float(b))
    return min(d, 1.0 - d)


def _circular_mean_pair(a: float, b: float) -> float:
    angles = 2.0 * np.pi * np.asarray([a, b], dtype=float)
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float((mean_angle / (2.0 * np.pi)) % 1.0)


def _template_score(name: str, phases: np.ndarray, template: np.ndarray) -> float:
    if name == "Pronk":
        return _cyclic_dist_max(phases, template)
    return _cyclic_dist(phases, template)


def _lateral_phases(phases: np.ndarray) -> tuple[float, float]:
    fl, fr, rl, rr = phases
    phi_L = float((fl - rl) % 1.0)
    phi_R = float((fr - rr) % 1.0)
    return phi_L, phi_R


def _dap_subtag(phi_L: float, phi_R: float) -> str:
    dap_L = phi_L - 0.5
    dap_R = phi_R - 0.5
    if max(abs(dap_L), abs(dap_R)) < DAP_DISSOCIATION_THRESHOLD:
        return ""
    if dap_L > 0 and dap_R > 0:
        return " +DAP"
    if dap_L < 0 and dap_R < 0:
        return " -DAP"
    return " ±DAP"


def _beta_qualifier(beta: float, base: str) -> str:
    return (WALK_LABEL_MAP if beta > BETA_RUN_THRESHOLD else RUN_LABEL_MAP)[base]


def _symmetric_label(
    phi_LH: float, beta: float, phi_L: float, phi_R: float,
) -> tuple[str, str | None, float]:
    nearest_name: str | None = None
    nearest_dist = float("inf")
    for name, point in CANONICAL_POINTS:
        d = _cyclic_diff_scalar(phi_LH, point)
        if d < nearest_dist:
            nearest_dist = d
            nearest_name = name

    if nearest_dist <= PHI_TOLERANCE and nearest_name is not None:
        label = _beta_qualifier(beta, nearest_name)
        if nearest_name == "Trot":
            label = label + _dap_subtag(phi_L, phi_R)
        return label, nearest_name, nearest_dist

    for q_name, lo, hi in QUADRANTS:
        if lo < phi_LH < hi:
            return _beta_qualifier(beta, q_name), None, nearest_dist
    return "Irregular", None, nearest_dist


def _asymmetric_label(phases: np.ndarray) -> tuple[str, str | None, float]:
    best_name: str | None = None
    best_score = float("inf")
    for name in ASYMMETRIC_NAMES:
        tmpl = np.asarray(GAIT_TEMPLATES[name]["phases"])
        score = _template_score(name, phases, tmpl)
        if score < best_score:
            best_score = score
            best_name = name
    if best_score <= PHI_TOLERANCE and best_name is not None:
        return ASYMMETRIC_LABEL_MAP[best_name], best_name, best_score
    return "Irregular", best_name, best_score


def _phase_tuple(contact: np.ndarray) -> tuple[np.ndarray | None, float]:
    fl_starts = _stance_starts(contact[:, 0])
    fr_starts = _stance_starts(contact[:, 1])
    rl_starts = _stance_starts(contact[:, 2])
    rr_starts = _stance_starts(contact[:, 3])
    if len(fl_starts) < MIN_STRIDE_STARTS:
        return None, 0.0
    period = float(np.mean(np.diff(fl_starts)))
    if period <= 0:
        return None, 0.0
    phases = np.array([
        0.0,
        _phase_offset_touchdown(fl_starts, fr_starts, period),
        _phase_offset_touchdown(fl_starts, rl_starts, period),
        _phase_offset_touchdown(fl_starts, rr_starts, period),
    ])
    return phases, period


def classify_gait_2d(contact: np.ndarray) -> dict:
    n_steps, n_feet = contact.shape
    if n_feet < 4:
        return {
            "label": "n/a", "template_key": None, "score": float("inf"),
            "phases": None, "phi_L": None, "phi_R": None,
            "phi_LH": None, "beta": None, "dphi": None,
        }

    beta = float(contact.mean())
    fl_starts = _stance_starts(contact[:, 0])
    fr_starts = _stance_starts(contact[:, 1])
    rl_starts = _stance_starts(contact[:, 2])
    rr_starts = _stance_starts(contact[:, 3])
    no_stride = all(
        len(s) < MIN_STRIDE_STARTS
        for s in (fl_starts, fr_starts, rl_starts, rr_starts)
    )
    if beta > STAND_DUTY or no_stride:
        return {
            "label": "Stand", "template_key": None, "score": 0.0,
            "phases": None, "phi_L": None, "phi_R": None,
            "phi_LH": None, "beta": beta, "dphi": None,
        }

    phases, period = _phase_tuple(contact)
    if phases is None:
        return {
            "label": "n/a", "template_key": None, "score": float("inf"),
            "phases": None, "phi_L": None, "phi_R": None,
            "phi_LH": None, "beta": beta, "dphi": None,
        }
    if period > MAX_STRIDE_PERIOD:
        return {
            "label": "Stand", "template_key": None, "score": 0.0,
            "phases": phases, "phi_L": None, "phi_R": None,
            "phi_LH": None, "beta": beta, "dphi": None,
        }

    phi_L, phi_R = _lateral_phases(phases)
    phi_LH = _circular_mean_pair(phi_L, phi_R)
    dphi = _cyclic_diff_scalar(phi_L, phi_R)

    if dphi >= LR_ASYMMETRY_THRESHOLD:
        label, key, score = _asymmetric_label(phases)
    else:
        label, key, score = _symmetric_label(phi_LH, beta, phi_L, phi_R)

    return {
        "label": label, "template_key": key, "score": score,
        "phases": phases, "phi_L": phi_L, "phi_R": phi_R,
        "phi_LH": phi_LH, "beta": beta, "dphi": dphi,
    }


def classify_gait(contact: np.ndarray) -> str:
    return classify_gait_2d(contact)["label"]


def classify_gait_with_score(contact: np.ndarray) -> tuple[str, float]:
    res = classify_gait_2d(contact)
    return res["label"], float(res["score"])


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


def _classify_majority(
    contact_all: np.ndarray,
) -> tuple[str, float, str | None]:
    labels: list[str] = []
    keys: list[str | None] = []
    score_by_label: dict[str, list[float]] = {}
    for i in range(contact_all.shape[0]):
        if not bool(contact_all[i].any()):
            continue
        res = classify_gait_2d(contact_all[i])
        labels.append(res["label"])
        keys.append(res["template_key"])
        score_by_label.setdefault(res["label"], []).append(float(res["score"]))
    if not labels:
        return "n/a", float("inf"), None
    most, _ = Counter(labels).most_common(1)[0]
    mean_score = (
        float(np.mean(score_by_label[most]))
        if score_by_label.get(most) else float("inf")
    )
    rep_key: str | None = None
    for lab, k in zip(labels, keys):
        if lab == most and k is not None:
            rep_key = k
            break
    return most, mean_score, rep_key


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


def _draw_template(ax, template_key: str, period_steps: float, fl_offset: int,
                   n_show: int, sim_dt: float) -> None:
    tmpl = GAIT_TEMPLATES.get(template_key)
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
            gait, score, tmpl_key = _classify_majority(contact)

            n_show = min(int(t_window_s / sim_dt), contact.shape[1])
            window = contact[r, :n_show].astype(bool)
            fl_window = window[:, 0]
            period_steps = _stride_period(fl_window)
            fl_offset = _first_stance(fl_window)

            if score < PHI_TOLERANCE and tmpl_key in GAIT_TEMPLATES:
                _draw_template(ax, tmpl_key, period_steps, fl_offset, n_show, sim_dt)
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
                ax.set_title(f"v={v_cmd:.2f} m/s\n{gait}", fontsize=9)
            else:
                ax.set_title(gait, fontsize=9)
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
