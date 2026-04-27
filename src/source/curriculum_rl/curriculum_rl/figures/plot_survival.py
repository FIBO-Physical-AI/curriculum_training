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


REASON_ORDER = ["time_out", "bad_orientation", "base_contact", "other"]
REASON_COLOR = {
    "time_out": "#10b981",
    "bad_orientation": "#ef4444",
    "base_contact": "#f59e0b",
    "other": "#6b7280",
}
REASON_LABEL = {
    "time_out": "survived (time-out)",
    "bad_orientation": "fell (orientation)",
    "base_contact": "fell (base contact)",
    "other": "other",
}


def _load(epte_csv: Path) -> tuple[list[dict], int]:
    rows = list(csv.DictReader(epte_csv.open()))
    if not rows:
        raise ValueError(f"{epte_csv} is empty")
    K = max(int(r["fall_step"]) for r in rows) + 1
    return rows, K


def plot_survival(
    epte_csv: Path,
    out_path: Path,
    num_bins: int = 8,
    sim_dt: float = 0.02,
) -> None:
    if not epte_csv.is_file():
        raise FileNotFoundError(f"{epte_csv} not found")
    rows, K = _load(epte_csv)

    by_cond_bin: dict[tuple[str, int], list[int]] = defaultdict(list)
    reasons_by_cond_bin: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cond = r.get("condition") or infer_condition(r.get("experiment", ""))
        if cond is None:
            continue
        b = int(r["bin_idx"])
        fs = int(r["fall_step"])
        reason = r.get("term_reason", "other")
        if reason not in REASON_ORDER:
            reason = "other"
        by_cond_bin[(cond, b)].append(fs)
        reasons_by_cond_bin[(cond, b)][reason] += 1

    conditions = [c for c in CONDITION_ORDER if any((c, b) in by_cond_bin for b in range(num_bins))]
    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False

    fig = plt.figure(figsize=(15.5, 8.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.55], hspace=0.45)
    gs_top = gs[0].subgridspec(2, 4, hspace=0.55, wspace=0.18)
    gs_bot = gs[1].subgridspec(1, 1)

    t = np.arange(K + 1) * sim_dt
    t_max = K * sim_dt

    for b in range(num_bins):
        ax = fig.add_subplot(gs_top[b // 4, b % 4])
        for cond in conditions:
            falls = np.asarray(by_cond_bin.get((cond, b), []), dtype=int)
            if falls.size == 0:
                continue
            sorted_falls = np.sort(falls)
            n = falls.size
            counts_at_or_before = np.searchsorted(sorted_falls, np.arange(K + 1), side="right")
            S = 1.0 - counts_at_or_before / n
            ax.plot(t, S, color=CONDITION_COLOR[cond], lw=2.0, alpha=0.95,
                    label=CONDITION_LABEL[cond])
        v_lo = b * 0.5
        v_hi = (b + 1) * 0.5
        ax.set_title(f"{v_lo:.1f} - {v_hi:.1f} m/s", fontsize=10, color="#111827", pad=4)
        ax.set_xlim(0, t_max)
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(length=3, labelsize=9)
        if b % 4 == 0:
            ax.set_ylabel("fraction alive", fontsize=10)
        if b // 4 == 1:
            ax.set_xlabel("time (s)", fontsize=10)

    handles = [
        plt.Line2D([0], [0], color=CONDITION_COLOR[c], lw=2.2, label=CONDITION_LABEL[c])
        for c in conditions
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(conditions),
               bbox_to_anchor=(0.5, 0.985), frameon=False, fontsize=10)

    ax_bar = fig.add_subplot(gs_bot[0])
    n_cond = len(conditions)
    bar_w = 0.8 / max(n_cond, 1)
    x = np.arange(num_bins)
    bottoms = {cond: np.zeros(num_bins) for cond in conditions}
    for reason in REASON_ORDER:
        for ci, cond in enumerate(conditions):
            heights = np.zeros(num_bins)
            for b in range(num_bins):
                counts = reasons_by_cond_bin.get((cond, b), {})
                total = sum(counts.values()) or 1
                heights[b] = counts.get(reason, 0) / total
            offset = (ci - (n_cond - 1) / 2) * bar_w
            ax_bar.bar(
                x + offset, heights, width=bar_w * 0.96,
                bottom=bottoms[cond], color=REASON_COLOR[reason],
                edgecolor=CONDITION_COLOR[cond], lw=0.8,
            )
            bottoms[cond] = bottoms[cond] + heights

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{i*0.5:.1f}-{(i+1)*0.5:.1f}" for i in range(num_bins)], fontsize=9)
    ax_bar.set_xlabel("velocity bin (m/s)", fontsize=10)
    ax_bar.set_ylabel("fraction of rollouts", fontsize=10)
    ax_bar.set_ylim(0, 1.10)
    ax_bar.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_bar.tick_params(length=3, labelsize=9)

    reason_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=REASON_COLOR[r], edgecolor="#374151",
                      lw=0.6, label=REASON_LABEL[r])
        for r in REASON_ORDER
    ]
    cond_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=CONDITION_COLOR[c],
                      lw=1.4, label=f"border = {CONDITION_LABEL[c]}")
        for c in conditions
    ]
    ax_bar.legend(
        handles=reason_handles + cond_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.55), ncol=max(len(REASON_ORDER), len(cond_handles)),
        frameon=False, fontsize=9,
    )

    fig.suptitle("Survival over time and termination breakdown",
                 fontsize=13, fontweight="bold", y=0.998)
    fig.subplots_adjust(top=0.91, bottom=0.13, left=0.06, right=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epte-csv", type=Path, default=Path("src/results/epte_sp.csv"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/survival.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--sim-dt", type=float, default=0.02)
    args = parser.parse_args(argv)
    plot_survival(args.epte_csv, args.out, num_bins=args.num_bins, sim_dt=args.sim_dt)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
