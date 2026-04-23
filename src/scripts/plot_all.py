from __future__ import annotations

import argparse
import sys
from pathlib import Path

from curriculum_rl.figures.plot_epte_bars import plot_epte_bars
from curriculum_rl.figures.plot_epte_violin import plot_epte_violin
from curriculum_rl.figures.plot_iterations_to_mastery import plot_iterations_to_mastery
from curriculum_rl.figures.plot_learning_curves import plot_learning_curves
from curriculum_rl.figures.plot_sampling_heatmap import plot_sampling_heatmap
from curriculum_rl.figures.plot_sampling_heatmap_3d import plot_sampling_heatmap_3d


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, default=Path("unitree_rl_lab/logs/rsl_rl"))
    parser.add_argument("--out-dir", type=Path, default=Path("src/results/figures"))
    parser.add_argument("--epte-csv", type=Path, default=Path("src/results/epte_sp.csv"))
    parser.add_argument("--num-bins", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("learning_curves.png", lambda p: plot_learning_curves(args.logs_root, p, num_bins=args.num_bins)),
        ("sampling_heatmap.png", lambda p: plot_sampling_heatmap(args.logs_root, p, num_bins=args.num_bins)),
        ("sampling_heatmap_3d.png", lambda p: plot_sampling_heatmap_3d(args.logs_root, p, num_bins=args.num_bins)),
        ("iterations_to_mastery.png", lambda p: plot_iterations_to_mastery(args.logs_root, p, num_bins=args.num_bins)),
        ("epte_bars.png", lambda p: plot_epte_bars(args.epte_csv, p, num_bins=args.num_bins)),
        ("epte_violin.png", lambda p: plot_epte_violin(args.epte_csv, p, num_bins=args.num_bins)),
    ]

    ok = 0
    for name, fn in tasks:
        path = args.out_dir / name
        try:
            fn(path)
            print(f"[plot_all] wrote {path}")
            ok += 1
        except FileNotFoundError as e:
            print(f"[plot_all] skip {name}: {e}")
        except Exception as e:
            print(f"[plot_all] error in {name}: {type(e).__name__}: {e}")

    print(f"[plot_all] {ok}/{len(tasks)} figures produced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
