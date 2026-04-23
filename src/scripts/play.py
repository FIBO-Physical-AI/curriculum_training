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
UPSTREAM = REPO_ROOT / "unitree_rl_lab" / "scripts" / "rsl_rl" / "play.py"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    parser.add_argument(
        "--condition",
        required=True,
        choices=list(CONDITION_TO_TASK.keys()),
    )
    parser.add_argument(
        "--bin",
        type=int,
        default=None,
        help="Force every env to this bin index (bin center velocity) for the whole rollout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args, passthrough = build_argparser().parse_known_args(argv)
    task_id = CONDITION_TO_TASK[args.condition]

    if args.bin is not None:
        os.environ["CURRICULUM_PLAY_BIN"] = str(args.bin)

    import curriculum_rl  # noqa: F401

    os.chdir(REPO_ROOT / "unitree_rl_lab")
    sys.path.insert(0, str(UPSTREAM.parent))
    sys.argv = [str(UPSTREAM), "--task", task_id, *passthrough]
    runpy.run_path(str(UPSTREAM), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
