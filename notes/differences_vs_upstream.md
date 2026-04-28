# What we changed vs upstream `unitree_rl_lab`

Reference: upstream config at `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py`.

We never edit the upstream file. All overrides are applied at config-time inside our `Go2VelocityBaseEnvCfg` / `Go2VelocityBasePlayEnvCfg` classes (in `src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py`), which subclass upstream `RobotEnvCfg` / `RobotPlayEnvCfg`.

---

## 1. Reward weights (sprint-friendly retune)

Applied in `_apply_sprint_retune(cfg)`.

| Term | Upstream | Ours | Ratio | Why |
|---|---|---|---|---|
| `track_lin_vel_xy.params.std` | `sqrt(0.25) ≈ 0.5` | `0.5` | 1× (same) | Tight Gaussian — penalizes >0.3 m/s tracking error |
| `action_rate.weight` | `-0.1` | `-0.005` | **20× weaker** | Upstream cripples high-frequency control needed at sprint |
| `joint_acc.weight` | `-2.5e-7` | `-1e-7` | **2.5× weaker** | Quadratic in joint accel — punishes fast leg swings |
| `joint_torques.weight` | `-2e-4` | `-2e-5` | **10× weaker** | Quadratic — torques scale with sprint, dominates reward at v>2 m/s |
| `joint_vel.weight` | `-0.001` | `-1e-4` | **10× weaker** | Quadratic — joint speed scales with v_cmd at high bins |
| `feet_air_time.params.threshold` | `0.5` s | `0.1` s | **5× shorter** | Upstream rewards 0.5 s air time = 2 Hz stride; sprinting Go2 needs ~8 Hz (0.125 s) |
| `JointPositionAction.scale` | `0.25` | `0.35` | **1.4× larger** | Lets policy reach larger hip extensions for longer strides |

Everything else in `RewardsCfg` (track_ang_vel_z, base_linear_velocity, dof_pos_limits, energy, flat_orientation_l2, joint_pos, air_time_variance, feet_slide, undesired_contacts) is **unchanged** from upstream.

## 2. Terrain (flatten)

Applied in `_flatten_terrain(cfg)`.

| Item | Upstream | Ours |
|---|---|---|
| `scene.terrain.terrain_type` | `"generator"` (COBBLESTONE_ROAD_CFG) | `"plane"` |
| `scene.terrain.terrain_generator` | cobblestone w/ marble texture | `None` |
| `scene.height_scanner` | active 1.6×1.0 m grid raycaster | `None` (disabled) |
| `curriculum.terrain_levels` | enabled (`terrain_levels_vel`) | `None` |

We disabled terrain difficulty altogether — it's a 1D forward velocity task on flat ground.

## 3. Velocity command (binned)

Replaced upstream `UniformLevelVelocityCommandCfg` with our `BinnedVelocityCommandCfg`.

| Item | Upstream | Ours |
|---|---|---|
| Command class | `UniformLevelVelocityCommandCfg` | `BinnedVelocityCommandCfg` |
| `lin_vel_x` initial range | `(-0.1, 0.1)` | `(0.0, 4.0)` |
| `lin_vel_x` limit range | `(-1.0, 1.0)` | `(0.0, 4.0)` |
| `lin_vel_y` | `(-0.4, 0.4)` (limit) | `(0.0, 0.0)` — no lateral |
| `ang_vel_z` | `(-1.0, 1.0)` | `(0.0, 0.0)` — no turning |
| `resampling_time_range` | `(10.0, 10.0)` | `(20.0, 20.0)` |
| `rel_standing_envs` | `0.1` | `0.0` |
| `debug_vis` | `True` | `False` |
| Bins | n/a (continuous) | 8 bins, width 0.5 m/s |
| `curriculum_kind` | n/a | `"uniform"` / `"task_specific"` / `"teacher"` |

Forward-only, no turning. Upstream Go2 was tuned for ±1 m/s with turning; we extended forward to 4 m/s and removed lateral/turning entirely.

## 4. Curriculum

| Item | Upstream | Ours |
|---|---|---|
| `curriculum.terrain_levels` | `terrain_levels_vel` | `None` |
| `curriculum.lin_vel_cmd_levels` | `lin_vel_cmd_levels` | `None` |
| `curriculum.velocity_curriculum` | not present | `velocity_curriculum_step` (ours) |

Our `velocity_curriculum_step` (in `curriculum_rl/envs/mdp.py`) is what actually drives the binned curriculum each PPO step.

## 5. Play config (video orientation lock)

Applied in `_lock_play_pose(cfg)` — Play config only, training is unaffected.

| `events.reset_base.params.pose_range` | Upstream | Ours (Play only) |
|---|---|---|
| `x` | `(-0.5, 0.5)` | `(0.0, 0.0)` |
| `y` | `(-0.5, 0.5)` | `(0.0, 0.0)` |
| `yaw` | `(-3.14, 3.14)` | `(0.0, 0.0)` |

Every play episode now starts with the robot facing +x at the origin — needed for clean video recording.

## 6. Files we did NOT touch

We never modified any file in `unitree_rl_lab/`. The whole submodule is clean (`git status` confirms). All our customization lives in `src/source/curriculum_rl/`.

## 7. Files we own (new code, not overriding upstream)

- `src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py` — base env config with overrides above
- `src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_uniform.py` / `_taskspec.py` / `_teacher.py` — three curriculum variants
- `src/source/curriculum_rl/curriculum_rl/envs/commands.py` — `BinnedVelocityCommandCfg` + sampler
- `src/source/curriculum_rl/curriculum_rl/envs/mdp.py` — `velocity_curriculum_step` curriculum term
- `src/source/curriculum_rl/curriculum_rl/curricula/` — Margolis Box Adaptive (`task_specific`) + LP-ACRL (`teacher_guided`)
- `src/source/curriculum_rl/curriculum_rl/eval/iterations_to_mastery.py` — mastery metric
- `src/source/curriculum_rl/curriculum_rl/figures/` — all 6 plot scripts
- `src/scripts/` — train.py, play.py, eval_epte.py, plot_all.py, run_sweep.sh, play_per_bin.sh
- `src/configs/` — task_space.yaml, curricula/*.yaml
