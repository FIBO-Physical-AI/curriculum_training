from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import numpy as np


CONDITION_ALIASES = {
    "uniform": "uniform",
    "task_specific": "task_specific",
    "taskspec": "task_specific",
    "task-spec": "task_specific",
    "teacher": "teacher",
    "teacher_guided": "teacher",
}

CONDITION_ORDER = ["uniform", "task_specific", "teacher"]
CONDITION_COLOR = {"uniform": "#94a3b8", "task_specific": "#f59e0b", "teacher": "#10b981"}
CONDITION_LABEL = {"uniform": "Uniform", "task_specific": "Task-Specific", "teacher": "Teacher-Guided"}

HEATMAP_CMAP = "YlOrRd"


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#374151",
        "axes.labelcolor": "#111827",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.7,
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 160,
        "figure.constrained_layout.use": True,
    })


def infer_condition(experiment_name: str) -> str | None:
    n = experiment_name.lower()
    for key, canon in CONDITION_ALIASES.items():
        if key in n:
            return canon
    return None


def find_runs(
    logs_root: Path,
    seeds_per_condition: int = 3,
    min_rows: int = 1000,
) -> list[tuple[str, Path, Path]]:
    out: list[tuple[str, Path, Path]] = []
    for exp_dir in sorted(logs_root.glob("*")):
        if not exp_dir.is_dir():
            continue
        cond = infer_condition(exp_dir.name)
        if cond is None:
            continue
        candidates: list[tuple[float, Path, Path]] = []
        for ts_dir in exp_dir.glob("*"):
            csv_path = ts_dir / "curriculum.csv"
            if not csv_path.is_file():
                continue
            try:
                with csv_path.open() as f:
                    row_count = sum(1 for _ in f) - 1
            except OSError:
                continue
            if row_count < min_rows:
                continue
            candidates.append((ts_dir.stat().st_mtime, ts_dir, csv_path))
        candidates.sort(key=lambda t: t[0], reverse=True)
        for _mtime, ts_dir, csv_path in candidates[:seeds_per_condition]:
            out.append((cond, ts_dir, csv_path))
    return out


def read_curriculum_csv(csv_path: Path) -> dict[str, np.ndarray]:
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {"step": np.empty(0), "bin_idx": np.empty(0), "weight": np.empty(0), "mean_reward": np.empty(0)}
    return {
        "step": np.array([int(r["step"]) for r in rows]),
        "bin_idx": np.array([int(r["bin_idx"]) for r in rows]),
        "weight": np.array([float(r["weight"]) for r in rows]),
        "mean_reward": np.array([float(r["mean_reward"]) for r in rows]),
    }


def reshape_to_grid(data: dict[str, np.ndarray], num_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = np.sort(np.unique(data["step"]))
    weights = np.full((len(steps), num_bins), np.nan)
    rewards = np.full((len(steps), num_bins), np.nan)
    step_idx = {s: i for i, s in enumerate(steps)}
    for k in range(len(data["step"])):
        i = step_idx[int(data["step"][k])]
        b = int(data["bin_idx"][k])
        if b < num_bins:
            weights[i, b] = data["weight"][k]
            rewards[i, b] = data["mean_reward"][k]
    return steps, weights, rewards


def aggregate_runs_by_condition(
    runs: list[tuple[str, Path, Path]], num_bins: int
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    by_cond: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for cond, _ts, csv_path in runs:
        data = read_curriculum_csv(csv_path)
        if data["step"].size == 0:
            continue
        by_cond.setdefault(cond, []).append(reshape_to_grid(data, num_bins))

    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for cond, trios in by_cond.items():
        common_steps = trios[0][0]
        for steps, _, _ in trios[1:]:
            n = min(len(common_steps), len(steps))
            common_steps = common_steps[:n]
        n = len(common_steps)
        w_stack = np.stack([t[1][:n] for t in trios], axis=0)
        r_stack = np.stack([t[2][:n] for t in trios], axis=0)
        out[cond] = (common_steps, np.nanmean(w_stack, axis=0), np.nanmean(r_stack, axis=0))
    return out
