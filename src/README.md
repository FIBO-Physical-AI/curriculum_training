# curriculum_rl

Project implementation for the proposal *A Comparative Study of Curriculum Learning Strategies for Velocity-Tracking Policies on the Unitree Go2 Quadruped* (`../proposal.tex`).

This folder is an Isaac Lab extension that imports the upstream `unitree_rl_lab` package and registers three velocity-command curricula on top of its Go2 velocity env:

| Condition       | Task ID                               | Source                        |
| --------------- | ------------------------------------- | ----------------------------- |
| Uniform         | `Curriculum-Go2-Velocity-Uniform`     | baseline, no curriculum       |
| Task-specific   | `Curriculum-Go2-Velocity-TaskSpec`    | Margolis et al. 2022          |
| Teacher-guided  | `Curriculum-Go2-Velocity-Teacher`     | Li et al. 2026 (LP-ACRL)      |

## Layout

```
src/
├── source/curriculum_rl/        Isaac Lab extension (Python package)
│   ├── config/extension.toml
│   ├── pyproject.toml
│   └── curriculum_rl/
│       ├── curricula/           3 sampling strategies (proposal §2.2–2.3)
│       ├── envs/                Go2 velocity env + gym registrations
│       ├── eval/                EPTE-SP, per-bin return, sampling heatmap, iterations-to-mastery (§6)
│       └── figures/             plot scripts for the paper
├── configs/                     hyperparameters (proposal Appendix A Table 1)
├── scripts/                     train / eval / sweep / plot launchers
└── results/                     runs, checkpoints, eval CSVs, figures (gitignored)
```

## Install

From the repository root, with the Isaac Lab python environment active:

```bash
cd src/source/curriculum_rl
pip install -e .
```

This also picks up `../../../unitree_rl_lab` as a dependency; install that first if it is not already installed.

## Run

All launchers below `import curriculum_rl` (to trigger `gym.register`) and then hand off to the upstream `unitree_rl_lab/scripts/rsl_rl/{train,play}.py` via `runpy` in the same process. Nothing in `unitree_rl_lab/` is modified. Any flag not listed is passed through verbatim to the upstream script.

```bash
python scripts/train.py --condition uniform       --seed 0 --headless --max_iterations 3000
python scripts/train.py --condition task_specific --seed 0 --headless
python scripts/train.py --condition teacher       --seed 0 --headless

bash scripts/run_sweep.sh

python scripts/play.py --condition uniform
python scripts/play.py --condition teacher --video --video_length 400 --real-time

python scripts/eval_epte.py --checkpoint results/<run>/model.pt
python scripts/plot_all.py
```

Logs land in `unitree_rl_lab/logs/rsl_rl/<experiment>/` because the launcher chdirs there (matching upstream convention).

## Status

Scaffold only — module bodies are empty. Fill in curriculum update rules, env overrides, eval metrics, and plotting.
