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


def plot_action_rate(
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
            val_s = row.get("action_rate")
            if val_s is None or val_s == "":
                continue
            try:
                val = float(val_s)
            except ValueError:
                continue
            if not np.isfinite(val):
                continue
            b = int(row["bin_idx"])
            by_cond[cond][b].append(val)

    if not by_cond:
        raise ValueError(
            f"{epte_csv} contains no action_rate column. "
            "Re-run eval_epte.py to regenerate the csv."
        )

    apply_style()
    fig, (ax_line, ax_box) = plt.subplots(1, 2, figsize=(12, 4.5))

    conditions = [c for c in CONDITION_ORDER if c in by_cond]

    for cond in conditions:
        means = np.full(num_bins, np.nan)
        stds = np.full(num_bins, np.nan)
        for b in range(num_bins):
            vals = by_cond[cond].get(b, [])
            if vals:
                means[b] = float(np.mean(vals))
                stds[b] = float(np.std(vals))
        ax_line.errorbar(
            v_cmds, means, yerr=stds,
            color=CONDITION_COLOR[cond], lw=2.0, marker="o", ms=6,
            capsize=3, elinewidth=1.0, label=CONDITION_LABEL[cond],
        )
    ax_line.set_xlabel("commanded velocity v_cmd (m/s)")
    ax_line.set_ylabel("mean |a_t - a_{t-1}|")
    ax_line.set_xticks(v_cmds)
    ax_line.set_xticklabels([f"{v:.2f}" for v in v_cmds], fontsize=8, rotation=30)
    ax_line.grid(True, alpha=0.25, lw=0.6)
    ax_line.legend(loc="upper left", frameon=False, fontsize=9)

    n_cond = len(conditions)
    width = 0.8 / max(n_cond, 1)
    for idx, cond in enumerate(conditions):
        offsets = (idx - (n_cond - 1) / 2.0) * width
        positions = v_cmds + offsets * bin_width
        data = [by_cond[cond].get(b, []) for b in range(num_bins)]
        bp = ax_box.boxplot(
            data,
            positions=positions,
            widths=width * bin_width * 0.9,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
        )
        for box in bp["boxes"]:
            box.set(facecolor=CONDITION_COLOR[cond], edgecolor=CONDITION_COLOR[cond], alpha=0.55)
        for median in bp["medians"]:
            median.set(color="#111827", lw=1.2)
        for whisk in bp["whiskers"]:
            whisk.set(color=CONDITION_COLOR[cond], lw=1.0)
        for cap in bp["caps"]:
            cap.set(color=CONDITION_COLOR[cond], lw=1.0)
    ax_box.set_xlabel("commanded velocity v_cmd (m/s)")
    ax_box.set_ylabel("mean |a_t - a_{t-1}| (per rollout)")
    ax_box.set_xticks(v_cmds)
    ax_box.set_xticklabels([f"{v:.2f}" for v in v_cmds], fontsize=8, rotation=30)
    ax_box.grid(True, alpha=0.25, lw=0.6, axis="y")

    fig.suptitle("Action rate per bin", fontsize=12, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epte-csv", type=Path, default=Path("src/results/epte_sp.csv"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/action_rate.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--v-max", type=float, default=4.0)
    args = parser.parse_args(argv)
    plot_action_rate(args.epte_csv, args.out, num_bins=args.num_bins, v_max=args.v_max)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
