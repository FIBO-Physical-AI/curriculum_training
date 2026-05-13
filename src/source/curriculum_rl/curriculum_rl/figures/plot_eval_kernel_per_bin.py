from __future__ import annotations

import argparse
import sys
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


def _alive_window_kernel(vx: np.ndarray, fall_step: np.ndarray, v_cmd: float) -> np.ndarray:
    T = vx.shape[1]
    mask = np.arange(T)[None, :] < fall_step[:, None]
    sign_v = np.sign(v_cmd) if v_cmd != 0.0 else 1.0
    aligned = np.clip(vx * sign_v, 0.0, abs(v_cmd))
    denom = max(abs(v_cmd), 0.1)
    per_step = aligned / denom
    per_step_masked = np.where(mask, per_step, np.nan)
    return np.nanmean(per_step_masked, axis=1)


def plot_eval_kernel_per_bin(
    traces_dir: Path,
    out_path: Path,
    num_bins: int = 8,
) -> None:
    cond_to_files = _find_traces(traces_dir)
    if not cond_to_files:
        raise FileNotFoundError(f"no trace npz files in {traces_dir}")

    bin_width = 4.0 / num_bins

    per_cond: dict[str, dict[int, np.ndarray]] = {}
    for cond, files in cond_to_files.items():
        per_bin: dict[int, list[np.ndarray]] = {}
        for fpath in files:
            z = np.load(fpath)
            for b in range(num_bins):
                key_vx = f"vx_b{b}"
                key_fs = f"fall_step_b{b}"
                if key_vx not in z.files:
                    continue
                vx = z[key_vx]
                if key_fs in z.files:
                    fs = z[key_fs]
                else:
                    fs = np.full(vx.shape[0], vx.shape[1], dtype=np.int32)
                v_cmd = (b + 0.5) * bin_width
                k = _alive_window_kernel(vx, fs, v_cmd)
                per_bin.setdefault(b, []).append(k)
        per_cond[cond] = {
            b: np.concatenate(v) for b, v in per_bin.items() if v
        }

    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False

    fig, ax = plt.subplots(figsize=(9, 4.5))
    conditions = [c for c in CONDITION_ORDER if c in per_cond]
    n_cond = len(conditions)
    width = 0.8 / max(n_cond, 1)
    x = np.arange(num_bins)
    for ci, cond in enumerate(conditions):
        means = np.array([
            float(np.nanmean(per_cond[cond][b])) if b in per_cond[cond] else np.nan
            for b in range(num_bins)
        ])
        stds = np.array([
            float(np.nanstd(per_cond[cond][b])) if b in per_cond[cond] else np.nan
            for b in range(num_bins)
        ])
        offset = (ci - (n_cond - 1) / 2.0) * width
        ax.bar(
            x + offset, means, width=width,
            yerr=stds, capsize=2.5,
            color=CONDITION_COLOR[cond],
            label=CONDITION_LABEL[cond],
            alpha=0.92, edgecolor="white", linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b * bin_width:.1f}-{(b + 1) * bin_width:.1f}" for b in range(num_bins)],
                       rotation=0, fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("velocity bin (m/s)", fontsize=11)
    ax.set_ylabel(r"alive-window mean $r_{\mathrm{lin}}$", fontsize=11)
    ax.set_title("Eval-time per-bin tracking kernel (alive window only)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, lw=0.6)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=Path("src/results/eval_traces"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/eval_kernel_per_bin.png"))
    parser.add_argument("--num-bins", type=int, default=8)
    args = parser.parse_args(argv)
    plot_eval_kernel_per_bin(args.traces_dir, args.out, num_bins=args.num_bins)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
