from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts-per-bin", type=int, default=100)
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--v-max", type=float, default=4.0)
    parser.add_argument("--out-csv", type=Path, default=Path("results/epte_sp.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    _ = args.checkpoint
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
