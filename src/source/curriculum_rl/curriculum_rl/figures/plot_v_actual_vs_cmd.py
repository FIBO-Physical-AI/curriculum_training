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


def plot_v_actual_vs_cmd(
    epte_csv: Path,
    out_path: Path,
    num_bins: int = 8,
    v_max: float = 4.0,
) -> None:
    if not epte_csv.is_file():
        raise FileNotFoundError(f"{epte_csv} not found")

    bin_width = v_max / num_bins
    v_cmds = np.array([(b + 0.5) * bin_width for b in range(num_bins)])

    by_cond: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with epte_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row.get("condition") or infer_condition(row.get("experiment", ""))
            if cond is None:
                continue
            b = int(row["bin_idx"])
            v_signed = float(row.get("v_x_signed_mean", "nan"))
            if np.isnan(v_signed):
                continue
            by_cond[cond][b].append(v_signed)

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    lo = -0.2
    hi = v_max + 0.3
    ax.plot([lo, hi], [lo, hi], color="#9ca3af", lw=1.0, ls="--", label="perfect tracking (v_act = v_cmd)")
    ax.axhline(0.0, color="#d1d5db", lw=0.6, ls=":")

    conditions = [c for c in CONDITION_ORDER if c in by_cond]
    for cond in conditions:
        means = np.full(num_bins, np.nan)
        stds = np.full(num_bins, np.nan)
        for b in range(num_bins):
            vals = by_cond[cond].get(b, [])
            if vals:
                means[b] = float(np.mean(vals))
                stds[b] = float(np.std(vals))
        ax.errorbar(
            v_cmds, means, yerr=stds,
            color=CONDITION_COLOR[cond], lw=2.0, marker="o", ms=7,
            capsize=3, elinewidth=1.2, label=CONDITION_LABEL[cond],
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("commanded velocity v_cmd (m/s)")
    ax.set_ylabel("achieved forward velocity v_act (m/s)")
    ax.set_title("Velocity tracking: did the policy actually move at the commanded speed?",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(v_cmds)
    ax.set_xticklabels([f"{v:.2f}" for v in v_cmds], fontsize=9)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25, lw=0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epte-csv", type=Path, default=Path("src/results/epte_sp.csv"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/v_actual_vs_cmd.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--v-max", type=float, default=4.0)
    args = parser.parse_args(argv)
    plot_v_actual_vs_cmd(args.epte_csv, args.out, num_bins=args.num_bins, v_max=args.v_max)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
