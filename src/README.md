# curriculum_rl

Project implementation for the proposal *A Comparative Study of Curriculum Learning Strategies for Velocity-Tracking Policies on the Unitree Go2 Quadruped* (`../proposal.tex`).

This folder is an Isaac Lab extension that imports the upstream `unitree_rl_lab` package and registers three velocity-command curricula on top of its Go2 velocity env:

| Condition       | Task ID                               | Source                        |
| --------------- | ------------------------------------- | ----------------------------- |
| Uniform         | `Curriculum-Go2-Velocity-Uniform`     | baseline, no curriculum       |
| Task-specific   | `Curriculum-Go2-Velocity-TaskSpec`    | Margolis et al. 2022          |
| Teacher-guided  | `Curriculum-Go2-Velocity-Teacher`     | Li et al. 2026 (LP-ACRL)      |

Task space: `[0, 4.0]` m/s forward velocity, 8 bins of width 0.5 m/s. Set in [`envs/go2_velocity_base.py`](source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py) (`V_MAX`, `NUM_BINS`); mirrored in [`configs/task_space.yaml`](configs/task_space.yaml) and in the defaults of [`scripts/eval_epte.py`](scripts/eval_epte.py) and [`scripts/plot_all.py`](scripts/plot_all.py). The defaults in `envs/commands.py::BinnedVelocityCommandCfg` are authoritative for curriculum hyperparameters (ε floor, β, γ, seed bin); the yaml files under `configs/curricula/` are documentation-only.

## Reward overrides on top of upstream

All env configs inherit from `unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg.RobotEnvCfg`. `Go2VelocityBaseEnvCfg.__post_init__` then applies, in order:

- `_flatten_terrain` — plane terrain, no height scanner, no terrain curriculum (decouples curriculum signal from terrain difficulty).
- `_remove_yaw_tracking_reward` — drops `track_ang_vel_z`, adds a small `ang_vel_z_l2` penalty (forward-only task).
- `_tune_feet_air_time_for_fast_gaits` — air-time threshold 0.3 s.
- `_rebalance_penalties_for_fast_gaits` — `track_lin_vel_xy = 0.75`, soften torques/energy/action-rate/joint-vel, keep `base_linear_velocity (z-bounce) = −0.5`.
- `_add_gait_shaping` — `feet_gait` and `feet_clearance` (period 0.4 s, trot offset).
- `_trim_velocity_command_obs` — observe forward-speed command only (`vel_command_b[:, 0]`), not the full 3-vector.

Tracking combines a Gaussian term and a small linear bootstrap bonus:

- `track_lin_vel_xy` (weight **0.75**, Gaussian) — primary reward, peaks when actual speed matches the command, falls off fast for being wrong.
- `track_lin_vel_x_linear` (weight **0.15**, linear partial-credit of `v_act / v_cmd`) — kept only as a bootstrap aid. At weight 0.75 this term let a cruise-at-2-m/s policy collect ~57% of max reward on bin 7 and ignore the command (we confirmed this by removing it: uniform failed to learn to walk at all within 300 iters because the pure Gaussian gave near-zero gradient for a random policy). At weight 0.15 the cruise cheat pays only ~0.09 reward while matching the command pays 0.75 from the Gaussian — ~9× penalty for cruising — but the linear term still provides enough dense signal early in training for a random policy to discover that moving forward earns reward.

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

`run_sweep.sh` wraps train → `eval_epte.py` → per-bin `play.py` video → `plot_all.py` per (condition, seed). Honored env vars: `CONDITIONS`, `SEEDS`, `MAX_ITERATIONS`, `NUM_ENVS`, `NUM_BINS`, `V_MAX`, `RECORD_VIDEOS` (set to `0` to skip video recording). Per-bin videos are emitted by forcing the sampler via `CURRICULUM_PLAY_BIN`, set transparently when `play.py` is called with `--bin N`.
