from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from curriculum_rl.figures._util import HEATMAP_CMAP, apply_style


NUM_BINS = 8


def _read_sweep_v4_csv(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                parsed = {
                    "sigma_en_x": float(row["sigma_en_x"]),
                    "std": float(row["std"]),
                    "alpha_en": float(row["alpha_en"]),
                }
            except (KeyError, ValueError):
                continue
            err_total = 0.0
            err_valid = True
            for b in range(NUM_BINS):
                v = row.get(f"err_b{b}", "")
                try:
                    err_total += float(v)
                except (TypeError, ValueError):
                    err_valid = False
                    break
            parsed["sum_err"] = err_total if err_valid else math.nan
            rows.append(parsed)
    return rows


def plot_sigma_phase_v4(csv_path: Path, out_path: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"sweep_v4 csv not found: {csv_path}")

    rows = _read_sweep_v4_csv(csv_path)
    if not rows:
        raise FileNotFoundError(f"sweep_v4 csv is empty: {csv_path}")

    sigmas = sorted({r["sigma_en_x"] for r in rows})
    combos = sorted({(r["std"], r["alpha_en"]) for r in rows})

    grid = np.full((len(combos), len(sigmas)), np.nan)
    for r in rows:
        try:
            i = combos.index((r["std"], r["alpha_en"]))
            j = sigmas.index(r["sigma_en_x"])
        except ValueError:
            continue
        if math.isfinite(r["sum_err"]):
            grid[i, j] = r["sum_err"]

    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    im = ax.imshow(grid, cmap=HEATMAP_CMAP, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(sigmas)))
    ax.set_xticklabels([f"{int(s)}" for s in sigmas])
    ax.set_yticks(np.arange(len(combos)))
    ax.set_yticklabels(
        [r"$\sigma_{{\mathrm{{lin}}}}={s}$, $w_{{\mathrm{{en}}}}={a}$".format(s=s, a=a) for (s, a) in combos]
    )
    ax.set_xlabel(r"$\sigma_x$")
    ax.set_ylabel(r"$(\sigma_{\mathrm{lin}}, w_{\mathrm{en}})$")
    for i in range(len(combos)):
        for j in range(len(sigmas)):
            v = grid[i, j]
            if math.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, color="#111827")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=10, color="#6b7280")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r"$\sum_b \overline{|v_x^{\mathrm{cmd}} - v_x|}_b$")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("src/results/sweep_v4.csv"))
    parser.add_argument("--out", type=Path, default=Path("src/results/figures/sigma_phase_v4.png"))
    args = parser.parse_args(argv)
    plot_sigma_phase_v4(args.csv, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
