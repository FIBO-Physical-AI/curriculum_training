from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


CONDITION_TO_TASK = {
    "uniform": "Curriculum-Go2-Velocity-Uniform",
    "task_specific": "Curriculum-Go2-Velocity-TaskSpec",
    "teacher": "Curriculum-Go2-Velocity-Teacher",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = REPO_ROOT / "unitree_rl_lab" / "scripts" / "rsl_rl" / "train.py"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    parser.add_argument(
        "--condition",
        required=True,
        choices=list(CONDITION_TO_TASK.keys()),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args, passthrough = build_argparser().parse_known_args(argv)
    task_id = CONDITION_TO_TASK[args.condition]

    import curriculum_rl  # noqa: F401

    os.chdir(REPO_ROOT / "unitree_rl_lab")
    sys.argv = [str(UPSTREAM), "--task", task_id, *passthrough]
    runpy.run_path(str(UPSTREAM), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
