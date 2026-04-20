from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
