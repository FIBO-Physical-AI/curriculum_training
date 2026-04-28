# curriculum_training

A comparative study of curriculum learning strategies for velocity-tracking policies on the Unitree Go2 quadruped.

Training a single policy to track the full velocity range — from slow walk to fast run — with PPO runs into a multi-task reward-imbalance problem: high-velocity commands produce near-zero reward at initialization and drown the gradient in uninformative trajectories. This project compares three command-sampling strategies under an otherwise identical pipeline:

| Condition       | Task ID                             | Source                                |
| --------------- | ----------------------------------- | ------------------------------------- |
| Uniform         | `Curriculum-Go2-Velocity-Uniform`   | baseline, no curriculum               |
| Task-specific   | `Curriculum-Go2-Velocity-TaskSpec`  | Margolis et al. 2022 (Box Adaptive)   |
| Teacher-guided  | `Curriculum-Go2-Velocity-Teacher`   | Li et al. 2026 (LP-ACRL)              |

Task space: `[0, 4.0]` m/s forward velocity, split into 8 bins of width 0.5 m/s. Lateral velocity and yaw rate are fixed at zero by the command class — the policy only ever sees a forward-velocity command. Set in [src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py](src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py) (`V_MAX`, `NUM_BINS`); the defaults in `envs/commands.py::BinnedVelocityCommandCfg` are authoritative for curriculum hyperparameters (γ, β, ε, seed bin, `min_episodes_per_bin`, `stage_length`). The yaml files under `src/configs/curricula/` and `src/configs/task_space.yaml` are documentation-only and not loaded by the training pipeline.

Evaluation follows proposal §6: per-bin mean return, EPTE-SP, task-sampling heatmap, iterations-to-mastery.

## Overrides on top of upstream

All env configs inherit from `unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg.RobotEnvCfg`. `Go2VelocityBaseEnvCfg.__post_init__` applies the following, in order:

- `_flatten_terrain` — plane terrain, no height scanner, no terrain curriculum (decouples curriculum signal from terrain difficulty).
- `_apply_sprint_retune` — overrides 7 reward/action parameters; see table below.
- Swap `commands.base_velocity` for `BinnedVelocityCommandCfg` (forward-only, lateral and yaw fixed at zero, 8 bins of width 0.5 m/s).
- Replace upstream `lin_vel_cmd_levels` curriculum term with `velocity_curriculum_step`, which dispatches to one of the three sampling strategies in `curricula/`.

The play env also applies `_lock_play_pose` (zero pose range on reset) and `_apply_play_camera` (fixed-asset chase camera) for stable video capture.

### `_apply_sprint_retune` overrides

| Parameter | Upstream | Current | Δ |
| --- | --- | --- | --- |
| `track_lin_vel_xy.params["std"]` | `math.sqrt(0.25)` | `0.5` | numerically identical (suspected no-op) |
| `action_rate.weight` | `-0.1` | `-0.005` | 20× weaker |
| `joint_acc.weight` | `-2.5e-7` | `-1e-7` | 2.5× weaker |
| `joint_torques.weight` | `-2e-4` | `-2e-5` | 10× weaker |
| `joint_vel.weight` | `-1e-3` | `-1e-4` | 10× weaker |
| `feet_air_time.params["threshold"]` | `0.5` s | `0.1` s | 5× shorter swing target |
| `JointPositionAction.scale` | `0.25` | `0.35` | 1.4× larger |

All other 15 reward terms (including `track_ang_vel_z` at weight `0.75`) and the PD gains, observation/critic terms, terminations, and PPO/network hyperparameters are inherited from upstream unchanged. Forward-only behaviour comes from the command class zeroing `lin_vel_y` and `ang_vel_z`, not from removing the yaw-tracking reward — the term is still active and rewards tracking `ω_z = 0`.

## Repository layout

| Path                                 | Role                                                                     |
| ------------------------------------ | ------------------------------------------------------------------------ |
| [src/](src/)                         | **Project implementation** — curricula, env overrides, configs, scripts |
| [unitree_rl_lab/](unitree_rl_lab/)   | Upstream Isaac Lab extension (Unitree robots, PPO runner, untouched)    |
| [unitree_model/](unitree_model/)     | Unitree robot USD/URDF assets                                           |

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
python src/scripts/train.py --condition uniform       --seed 0 --headless --max_iterations 6000
python src/scripts/train.py --condition task_specific --seed 0 --headless
python src/scripts/train.py --condition teacher       --seed 0 --headless

bash src/scripts/run_sweep.sh

python src/scripts/play.py --condition uniform
python src/scripts/play.py --condition teacher --video --video_length 400 --real-time

python src/scripts/eval_epte.py --checkpoint src/results/<run>/model.pt
python src/scripts/plot_all.py
```

Logs land in `unitree_rl_lab/logs/rsl_rl/<experiment>/` because the launcher chdirs there (matching upstream convention).

`run_sweep.sh` wraps train → `eval_epte.py` → per-bin `play.py` video → `plot_all.py` per (condition, seed). Honored env vars: `CONDITIONS`, `SEEDS`, `MAX_ITERATIONS` (default 6000), `NUM_ENVS` (default 2048), `STEPS_PER_ITER`, `NUM_BINS` (default 8), `V_MAX` (default 4.0), `RECORD_VIDEOS` (set to `0` to skip video recording). Per-bin videos are emitted by forcing the sampler via `CURRICULUM_PLAY_BIN`, set transparently when `play.py` is called with `--bin N`.

## Platform

- Isaac Lab + Isaac Sim 4.5
- Python 3.10, PyTorch
- RSL-RL 2.3.1+
- Unitree Go2 (12 actuated joints)
