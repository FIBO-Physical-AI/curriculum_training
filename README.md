# curriculum_training

A comparative study of curriculum learning strategies for velocity-tracking policies on the Unitree Go2 quadruped.

Training a single policy to track the full velocity range — from slow walk to fast run — with PPO runs into a multi-task reward-imbalance problem: high-velocity commands produce near-zero reward at initialization and drown the gradient in uninformative trajectories. This project compares three command-sampling strategies under an otherwise identical pipeline:

| Condition       | Task ID                             | Source                                |
| --------------- | ----------------------------------- | ------------------------------------- |
| Uniform         | `Curriculum-Go2-Velocity-Uniform`   | baseline, no curriculum               |
| Task-specific   | `Curriculum-Go2-Velocity-TaskSpec`  | Margolis et al. 2022 (Box Adaptive)   |
| Teacher-guided  | `Curriculum-Go2-Velocity-Teacher`   | Li et al. 2026 (LP-ACRL)              |

Task space: `[0, 4.0]` m/s forward-velocity, split into 8 bins of width 0.5 m/s.

Reward shape (on top of the upstream `unitree_rl_lab` Go2 velocity env): yaw-tracking disabled (forward-only task), Gaussian `track_lin_vel_xy` as the sole tracking signal, soft penalty rebalance for fast gaits, and gait shaping (`feet_air_time`, `feet_gait`, `feet_clearance`). No linear partial-credit on forward velocity — tracking reward drops sharply when the actual speed misses the command, so a cruise-at-comfortable-speed policy cannot collect reward across all bins.

Evaluation follows proposal §6: per-bin mean return, EPTE-SP, task-sampling heatmap, iterations-to-mastery.

## Repository layout

| Path                | Role                                                                     |
| ------------------- | ------------------------------------------------------------------------ |
| [proposal.tex](proposal.tex)         | Project proposal (FRA503 Deep Reinforcement Learning)     |
| [src/](src/)                         | **Project implementation** — curricula, env overrides, configs, scripts |
| [unitree_rl_lab/](unitree_rl_lab/)   | Upstream Isaac Lab extension (Unitree robots, PPO runner) |
| [unitree_model/](unitree_model/)     | Unitree robot USD/URDF assets                             |

See [src/README.md](src/README.md) for install and run instructions.

## Quick start

```bash
cd src/source/curriculum_rl
pip install -e .

cd ../../..
python src/scripts/train.py --condition uniform --seed 0 --headless --max_iterations 3000
python src/scripts/play.py  --condition uniform
```

## Platform

- Isaac Lab + Isaac Sim 4.5
- Python 3.10, PyTorch
- RSL-RL 2.3.1+
- Unitree Go2 (12 actuated joints)
