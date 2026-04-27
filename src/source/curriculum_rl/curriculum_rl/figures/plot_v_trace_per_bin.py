from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from curriculum_rl.figures._util import (
    CONDITION_COLOR,
    CONDITION_LABEL,
    CONDITION_ORDER,
    apply_style,
    infer_condition,
)


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


def plot_v_trace_per_bin(
    traces_dir: Path,
    out_path: Path,
    num_bins: int = 8,
    n_rollouts_to_plot: int = 5,
) -> None:
    cond_to_files = _find_traces(traces_dir)
    if not cond_to_files:
        raise FileNotFoundError(f"no trace npz files found in {traces_dir}")

    conditions = [c for c in CONDITION_ORDER if c in cond_to_files]
    n_cols = len(conditions)
    n_rows = num_bins

    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.4 * n_cols, 1.6 * n_rows),
        sharex=True, sharey="row",
    )
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    rng = np.random.default_rng(0)

    for ci, cond in enumerate(conditions):
        files = cond_to_files[cond]
        for b in range(num_bins):
            ax = axes[b, ci]
            traces_concat = []
            v_cmd = None
            for fpath in files:
                z = np.load(fpath)
                key_vx = f"vx_b{b}"
                key_vc = f"vcmd_b{b}"
                if key_vx not in z.files:
                    continue
                vx = z[key_vx]
                traces_concat.append(vx)
                if key_vc in z.files:
                    v_cmd = float(z[key_vc])
            if not traces_concat:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center", color="#9ca3af")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            vx_all = np.concatenate(traces_concat, axis=0)
            n_total = vx_all.shape[0]
            n_pick = min(n_rollouts_to_plot, n_total)
            idx = rng.choice(n_total, size=n_pick, replace=False)
            for j in idx:
                ax.plot(vx_all[j], color=CONDITION_COLOR[cond], lw=0.8, alpha=0.55)
            mean_trace = vx_all.mean(axis=0)
            ax.plot(mean_trace, color=CONDITION_COLOR[cond], lw=1.6, alpha=1.0)
            if v_cmd is not None:
                ax.axhline(v_cmd, color="#111827", lw=1.0, ls="--", alpha=0.7)
            ax.axhline(0.0, color="#d1d5db", lw=0.5, ls=":")
            ax.set_ylim(-1.0, 5.0)
            ax.grid(True, alpha=0.2, lw=0.5)
            if ci == 0:
                v_label = f"bin {b}\nv_cmd={v_cmd:.2f}" if v_cmd is not None else f"bin {b}"
                ax.set_ylabel(v_label, fontsize=9)
            if b == 0:
                ax.set_title(CONDITION_LABEL[cond], fontsize=11, fontweight="bold",
                             color=CONDITION_COLOR[cond])
            if b == n_rows - 1:
                ax.set_xlabel("step", fontsize=9)

    fig.suptitle("Per-rollout forward velocity v_x(t)",
                 fontsize=13, fontweight="bold", y=0.995)
    legend_handles = [
        Line2D([0], [0], color="#111827", lw=1.0, ls="--", label="v_cmd"),
        Line2D([0], [0], color="#6b7280", lw=1.6, label="mean v_x"),
        Line2D([0], [0], color="#6b7280", lw=0.8, alpha=0.55, label="rollouts"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.07, right=0.99,
                        hspace=0.35, wspace=0.10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=Path("src/results/eval_traces"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/v_trace_per_bin.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--n-rollouts", type=int, default=5)
    args = parser.parse_args(argv)
    plot_v_trace_per_bin(args.traces_dir, args.out, num_bins=args.num_bins, n_rollouts_to_plot=args.n_rollouts)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
