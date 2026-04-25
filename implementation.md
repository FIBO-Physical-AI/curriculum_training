# 2026-04-23 — High-velocity tracking plan (target ≤ 4 m/s)
 
## Goal (recap)
 
1. **Train Go2 to track forward velocity commands up to 4 m/s**, 1D vx only (yaw kept at 0, no multi-dim scope change).
2. **Uniform is expected to fail on high bins** — that is the baseline proving the curriculum hypothesis — but it must at least *try* to walk/run, not collapse to standing.
3. **Task-Specific and Teacher-Guided should master as many bins as possible**, ideally all 8 up to 4 m/s. Realistic stretch given 1D scope + Go2 hardware: bins 0–6 (0–3 m/s) solid, bin 7 (3.5–4 m/s) stretch.
4. **Videos should show the policy at each bin's center velocity** (0.25, 0.75, 1.25, …, 3.75 m/s). Already implemented: `Go2VelocityBasePlayEnvCfg` uses `_make_binned_cmd` with `rel_standing_envs=0`, and `eval_epte.py` loops `v = (b + 0.5) * bin_width` per bin.
---
 
## Diagnosis of current state
 
| observation | current value | after Fix-A run |
|---|---|---|
| Uniform masters bins | bin 0 only | bins 0–2 (EPTE 0.41/0.12/0.26) |
| Task-Specific masters bins | bins 0–2 | bins 0–2 (EPTE 0.24/0.07/0.06) |
| Teacher-Guided masters bins | bins 0–2 partial | bins 0–2 (EPTE 0.39/0.09/0.06) |
| Bins 3–7 | all fail, EPTE=1.00 | still all fail, EPTE=1.00 |
 
**Root cause (confirmed by two research agents against legged_gym / walk-these-ways / DreamWaQ / HIMLoco codebases):** the upstream Unitree Go2 reward weights are tuned for ≤1 m/s walking. At 2+ m/s, three weights scale quadratically with velocity and become dominant penalties that exceed the tracking reward. The policy correctly converges to the *local reward optimum* — which is slow trot — and gets stuck there. No curriculum algorithm can rescue this.
 
| term | current | proven (Margolis / legged_gym) | multiplier |
|---|---:|---:|---:|
| `base_linear_velocity` (lin_vel_z²) | −2.0 | −0.02 (WTW) / −0.5 (safe) | 4–100× too strong |
| `base_angular_velocity` (ang_vel_xy²) | −0.05 | −0.001 (WTW) / −0.01 (safe) | 5–50× too strong |
| `joint_torques` (τ²) | −2e-4 | −1e-5 (legged_gym base) | **20× too strong** |
| `joint_vel` (q̇²) | −1e-3 | −1e-4 | 10× too strong |
| `action_rate` (Δa²) | −0.1 | −0.01 | 10× too strong |
| `energy` (|τ·q̇|) | −2e-5 | −1e-6 | 20× too strong |
| `feet_air_time` | +0.1 | +1.0 | 10× too weak |
 
**Secondary issue:** the upstream MDP has two powerful reward terms (`feet_gait`, `foot_clearance_reward`) that are wired into the G1 humanoid config but not into our Go2 config. Both are proven gait-shaping primitives.
 
**Existence proof that 1D velocity to 3+ m/s is achievable:**
- **HIMLoco** (Long et al. 2024): Go2, 1D velocity command, reaches ~3.5 m/s on flat terrain.
- **Rapid Locomotion** (Margolis 2022): Mini Cheetah, 3.9 m/s (multi-dim command, but single-axis velocity case documented).
- Neither paper uses a turnkey recipe — both require penalty rebalancing + gait shaping.
---
 
## The plan — four tiers
 
Each tier is independently verifiable. Apply bottom-up.
 
### Tier 1 — Penalty rebalance (mandatory, 10 min)
 
**File:** `src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py`
 
Add to the `__post_init__` of `Go2VelocityBaseEnvCfg` AND `Go2VelocityBasePlayEnvCfg`, after the existing helper calls:
 
```python
cfg = self
# --- soften penalties that scale with v² (the slow-gait trap) ---
cfg.rewards.base_linear_velocity.weight = -0.5     # was -2.0
cfg.rewards.base_angular_velocity.weight = -0.01   # was -0.05
cfg.rewards.joint_torques.weight = -1e-5           # was -2e-4  (20× reduction)
cfg.rewards.joint_vel.weight = -1e-4               # was -1e-3  (10× reduction)
cfg.rewards.action_rate.weight = -0.01             # was -0.1   (10× reduction)
cfg.rewards.energy.weight = -1e-6                  # was -2e-5  (20× reduction)
# --- boost the stride signal ---
cfg.rewards.feet_air_time.weight = 1.0             # was 0.1    (10× boost)
```
 
Keep `feet_air_time.params.threshold = 0.3` (already set by `_tune_feet_air_time_for_fast_gaits`).
 
**Why these numbers:** match legged_gym base defaults and Margolis walk-these-ways values that demonstrably reach 3+ m/s. Not arbitrary tuning.
 
### Tier 2 — Gait shaping (mandatory, 10 min)
 
**File:** same. Add after Tier 1 edits:
 
```python
from unitree_rl_lab.tasks.locomotion import mdp as upstream_mdp
from isaaclab.managers import SceneEntityCfg
 
cfg.rewards.feet_gait = RewTerm(
    func=upstream_mdp.feet_gait,
    weight=0.5,
    params={
        "period": 0.4,
        "offset": [0.0, 0.5, 0.5, 0.0],       # diagonal trot: FR, FL, RR, RL
        "threshold": 0.5,
        "command_name": "base_velocity",
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    },
)
cfg.rewards.feet_clearance = RewTerm(
    func=upstream_mdp.foot_clearance_reward,
    weight=0.5,
    params={
        "target_height": 0.08,
        "std": 0.05,
        "tanh_mult": 2.0,
        "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    },
)
```
 
Both functions are already defined in `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py`. No new code to write. G1 humanoid config (`robots/g1/29dof/velocity_env_cfg.py:299-309`) uses them with analogous parameters.
 
### Tier 3 — Widen the tracking Gaussian (optional, HIMLoco trick)
 
Only add if Tier 1 + Tier 2 don't get bins 5–7 working.
 
```python
import math
cfg.rewards.track_lin_vel_xy.params["std"] = 1.0    # was math.sqrt(0.25) = 0.5
```
 
Compensate the mastery threshold in `src/configs/curricula/task_specific.yaml`:
```
gamma: 0.5     # was 0.7  (phi=0.5 now corresponds to err=0.83 m/s under σ=1.0)
```
 
No changes needed for teacher-guided β — softmax temperature is relative.
 
### Tier 4 — Training budget and exploration (if still stuck)
 
- `MAX_ITERATIONS=6000` (was 3000–4000)
- `unitree_rl_lab/.../rsl_rl_ppo_cfg.py`: `entropy_coef = 0.015` (was 0.01) — only if you train less than 6k iters
### Tier 5 — Expand task space back to the full 0–4 m/s
 
If all previous tiers succeed on bins 0–5, revert:
- `V_MAX = 4.0`, `NUM_BINS = 8`
- `task_space.yaml`: `num_bins: 8`, `v_max: 4.0`
- `src/scripts/eval_epte.py` defaults: `num_bins=8, v_max=4.0`
- `src/scripts/plot_all.py` default: `num_bins=8`
Then re-run the full sweep for the final paper numbers.
 
---
 
## Expected outcomes
 
| tier applied | Uniform | Task-Specific | Teacher-Guided |
|---|---|---|---|
| **baseline (current)** | bin 0 | bins 0–2 | bins 0–2 |
| **+ Tier 1 + Tier 2** (V_MAX=3.0) | bins 0–2, partial 3 | bins 0–4 (~2.25 m/s), partial 5 | bins 0–3, partial 4–5 |
| **+ Tier 3** (wide σ) | bins 0–3 | all 6 bins (up to 3 m/s) | all 6 bins |
| **+ Tier 5** (V_MAX=4.0, full sweep) | fails bins 5–7 (baseline story) | bins 0–6 clean, partial 7 | bins 0–5 clean, partial 6–7 |
 
### What "Uniform should at least try" means
 
With Tier 1 + Tier 2, Uniform will walk — not stand. The softened `lin_vel_z` penalty no longer punishes vertical bouncing, so a bouncy-but-moving policy has higher reward than a static one. Uniform's failure mode becomes "masters low bins, degrades smoothly on high bins" — the correct paper baseline, not "degenerate stand-still."
 
### Videos per bin
 
Play env is already configured for this. Post-training:
 
```bash
python src/scripts/play.py --condition task_specific --video --video_length 400
```
 
With `num_envs=32` and `num_bins=8` the play env spreads ~4 envs per bin, each at the bin center (0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75 m/s). Video captures all bins simultaneously in one clip.
 
If you want one video per bin individually, modify `play.py` to accept a `--bin` flag that forces `cmd_term.vel_command_b[:, 0] = v_bin_center` for all envs. Same pattern as `eval_epte.py` lines 96–99.
 
---
 
## Execution order
 
1. **Apply Tier 1** (penalty rebalance). Open `go2_velocity_base.py`, paste the block into `Go2VelocityBaseEnvCfg.__post_init__` AND `Go2VelocityBasePlayEnvCfg.__post_init__`.
2. **Apply Tier 2** (gait + clearance). Same files.
3. **Sanity unit test** (no sim):
   ```bash
   python -c "
   import curriculum_rl
   from curriculum_rl.envs.go2_velocity_base import Go2VelocityBaseEnvCfg
   cfg = Go2VelocityBaseEnvCfg()
   print('torques:', cfg.rewards.joint_torques.weight)   # expect -1e-5
   print('action_rate:', cfg.rewards.action_rate.weight) # expect -0.01
   print('feet_gait weight:', cfg.rewards.feet_gait.weight)  # expect 0.5
   print('feet_clearance weight:', cfg.rewards.feet_clearance.weight)  # expect 0.5
   "
   ```
   Note: this requires Isaac Lab runtime; may need to run via `isaaclab.sh -p`. If pure-python check fails, skip and trust step 4.
4. **Seed-1 sanity run** (30 min):
   ```bash
   export CURRICULUM_STEPS_PER_ITER=24
   SEEDS="1" MAX_ITERATIONS=3000 bash src/scripts/run_sweep.sh
   ```
5. **Inspect bin 3–5 mastery** in `curriculum.csv`:
   ```bash
   LATEST=$(ls -td unitree_rl_lab/logs/rsl_rl/*taskspec*/* | head -1)
   awk -F, '$2==3 && NR>1 {print $1, $4}' "$LATEST/curriculum.csv" | tail -20
   awk -F, '$2==4 && NR>1 {print $1, $4}' "$LATEST/curriculum.csv" | tail -20
   awk -F, '$2==5 && NR>1 {print $1, $4}' "$LATEST/curriculum.csv" | tail -20
   ```
   Success criterion: bin 3 mean_reward > 0.3, bin 4 > 0.15, bin 5 > 0.05 by end of training.
6. **If bins 4–5 still near zero**: apply Tier 3 (widen σ), re-run sanity.
7. **Full 3-seed sweep** (6k iters, ~9 hours):
   ```bash
   for S in 0 1 2; do SEEDS="$S" MAX_ITERATIONS=6000 bash src/scripts/run_sweep.sh; done
   ```
8. **Smooth plots** (see `issue/2026-04-23_plot_smoothing.md` when written, or add rolling-mean window=20 to `plot_learning_curves.py`).
9. **If all above succeeds on bins 0–5**: apply Tier 5, re-run full sweep at V_MAX=4.0 for final paper numbers.
---
 
## What NOT to do (saves time)
 
- **Don't change `action_scale` from 0.25.** Margolis reaches 3.9 m/s with this scale; it's not the bottleneck.
- **Don't add multi-dim commands** (vy, ωz). Changes proposal scope.
- **Don't re-enable yaw tracking with varied yaw commands.** Not justified for vx-only study; benefit is indirect and speculative.
- **Don't tune `feet_air_time.threshold` below 0.2.** Threshold 0.3 is right for fast trot; lower values reward chaotic step timing.
- **Don't add `only_positive_rewards` flag.** May interact badly with curriculum reward; skip unless Tiers 1–4 fail.
- **Don't chase bin 7 (3.5–4.0 m/s) as a hard requirement.** HIMLoco reports 3.5 m/s max; 4 m/s with 1D command is unproven territory.
---
 
## Acceptance criteria
 
For the paper to be publishable as a "three-curriculum comparative study":
 
1. **Uniform** shows degraded-but-present learning across all bins (EPTE monotonically increases with bin index, doesn't plateau at 1.0 early).
2. **Task-Specific** masters strictly more bins than Uniform with statistically clear margin.
3. **Teacher-Guided** masters strictly more bins than Uniform, shows different sampling profile than Task-Specific.
4. At least one condition (ideally Task-Specific) reaches bin 5 (2.5–3.0 m/s) with EPTE < 0.3.
5. Three seeds per condition with non-overlapping confidence intervals on at least 2 bins.
If 1–4 are met, the paper has a defensible story. If 5 is met, the statistical claim is solid.
 
---
 
## References consulted
 
- **Margolis et al. 2022** "Rapid Locomotion via Reinforcement Learning" (arXiv:2205.02824). Code: [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways), [rapid-locomotion-rl](https://github.com/Improbable-AI/rapid-locomotion-rl).
- **legged_gym** (Rudin et al.) — base config `legged_gym/envs/base/legged_robot_config.py`.
- **unitree_rl_gym** (Unitree official) — `legged_gym/envs/go2/go2_config.py`.
- **DreamWaQ** (Ji et al. 2023, arXiv:2301.10602) — positive-shaping trick, penalty weights.
- **HIMLoco** (Long et al. 2024) — Go2 3.5 m/s with 1D velocity, no-move termination trick, σ=1.0.
- **LP-ACRL** (arXiv:2601.17428) — ANYmal D, 2.5 m/s, LP-based automatic curriculum.
## Files this plan touches
 
- `src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py` — Tier 1 + Tier 2
- `src/configs/curricula/task_specific.yaml` — Tier 3 (γ adjustment, only if applied)
- `unitree_rl_lab/.../rsl_rl_ppo_cfg.py` — Tier 4 (entropy bump, optional)
- `src/scripts/run_sweep.sh` or environment var — Tier 4 (MAX_ITERATIONS)
- All task_space / eval / plot defaults — Tier 5 (revert to 8 bins after bins 0–5 succeed)
## Rough time budget
 
| step | wall-clock |
|---|---|
| Tier 1 + Tier 2 edits | 20 min |
| Seed-1 sanity run @ 3k iter | 30 min |
| Tier 3 edits (if needed) | 5 min |
| Full 3-seed sweep @ 6k iter | ~9 h (overnight) |
| Plot smoothing + paper-quality figures | 30 min |
| Tier 5 re-run (if bins 5 succeeded and you want V_MAX=4.0) | ~12 h |
 
Total: 1 day of active work + 1 overnight run. 2 days if pushing for V_MAX=4.0.