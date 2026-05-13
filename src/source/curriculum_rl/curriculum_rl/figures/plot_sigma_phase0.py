from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from curriculum_rl.figures._util import apply_style


def _read_phase0_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sigmas: list[float] = []
    means: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            sigmas.append(float(row["sigma_en_x"]))
            means.append(float(row["mean_r_en"]))
            mins.append(float(row["min_r_en"]))
            maxs.append(float(row["max_r_en"]))
    return (
        np.asarray(sigmas),
        np.asarray(means),
        np.asarray(mins),
        np.asarray(maxs),
    )


def plot_sigma_phase0(csv_path: Path, out_path: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"phase0 csv not found: {csv_path}")

    sigmas, means, mins, maxs = _read_phase0_csv(csv_path)
    order = np.argsort(sigmas)
    sigmas = sigmas[order]
    means = means[order]
    mins = mins[order]
    maxs = maxs[order]

    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.fill_between(sigmas, mins, maxs, color="#94a3b8", alpha=0.35, label="min / max across envs and steps")
    ax.plot(sigmas, means, marker="o", color="#1f2937", linewidth=1.6, label=r"mean $r_{\mathrm{en}}$")
    ax.set_xscale("log")
    ax.set_xticks(sigmas)
    ax.set_xticklabels([f"{int(s)}" for s in sigmas], rotation=0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$\sigma_x$ (with $\sigma_z = \sigma_x / 2$)")
    ax.set_ylabel(r"$r_{\mathrm{en}}$ under random actions")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", axis="x", alpha=0.3, lw=0.6)
    ax.grid(True, axis="y", alpha=0.3, lw=0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("src/results/phase0_table.csv"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/sigma_phase0.png"))
    args = parser.parse_args(argv)
    plot_sigma_phase0(args.csv, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
