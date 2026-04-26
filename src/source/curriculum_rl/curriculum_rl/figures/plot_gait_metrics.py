from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
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


METRICS = [
    ("duty_factor", "Duty factor", (0.3, 1.05)),
    ("stride_freq_hz", "Stride frequency (Hz)", None),
    ("stride_length_m", "Stride length (m)", None),
]


def plot_gait_metrics(
    epte_csv: Path,
    out_path: Path,
    num_bins: int = 8,
    v_max: float = 4.0,
) -> None:
    if not epte_csv.is_file():
        raise FileNotFoundError(f"{epte_csv} not found")

    bin_width = v_max / num_bins
    v_cmds = np.array([(b + 0.5) * bin_width for b in range(num_bins)])

    by_cond: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    have_metric = {m: False for m, _, _ in METRICS}
    with epte_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row.get("condition") or infer_condition(row.get("experiment", ""))
            if cond is None:
                continue
            b = int(row["bin_idx"])
            for metric_key, _, _ in METRICS:
                val_s = row.get(metric_key)
                if val_s is None or val_s == "":
                    continue
                try:
                    val = float(val_s)
                except ValueError:
                    continue
                if not np.isfinite(val):
                    continue
                by_cond[cond][metric_key][b].append(val)
                have_metric[metric_key] = True

    if not any(have_metric.values()):
        raise ValueError(
            f"{epte_csv} contains no gait metric columns. "
            "Re-run eval_epte.py to regenerate the csv."
        )

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    conditions = [c for c in CONDITION_ORDER if c in by_cond]
    for ax, (metric_key, ylabel, ylim) in zip(axes, METRICS):
        for cond in conditions:
            means = np.full(num_bins, np.nan)
            stds = np.full(num_bins, np.nan)
            for b in range(num_bins):
                vals = by_cond[cond].get(metric_key, {}).get(b, [])
                if vals:
                    means[b] = float(np.mean(vals))
                    stds[b] = float(np.std(vals))
            ax.errorbar(
                v_cmds, means, yerr=stds,
                color=CONDITION_COLOR[cond], lw=2.0, marker="o", ms=6,
                capsize=3, elinewidth=1.0, label=CONDITION_LABEL[cond],
            )
        ax.set_xlabel("commanded velocity v_cmd (m/s)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(v_cmds)
        ax.set_xticklabels([f"{v:.2f}" for v in v_cmds], fontsize=8, rotation=30)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25, lw=0.6)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)

    fig.suptitle("Gait metrics per bin", fontsize=12, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epte-csv", type=Path, default=Path("src/results/epte_sp.csv"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/gait_metrics.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--v-max", type=float, default=4.0)
    args = parser.parse_args(argv)
    plot_gait_metrics(args.epte_csv, args.out, num_bins=args.num_bins, v_max=args.v_max)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
