# Gait Emergence Implementation V2: Additive Energy Reward (Liang 2024)

This document specifies the **second attempt** at implementing energy-driven gait emergence on Go2. Run 1 used Liang's multiplicative composite form `(R_motion + α_en·R_en) · exp(-R_aux)` and produced a "seal-shuffle" failure mode without gait emergence. Diagnosis: cutting the upstream reward stack and replacing it with a custom multiplicative composite stripped too much structural support. Run 2 restores the upstream reward stack and adds the energy term as a single additive contribution.

Read the entire document before writing code. The verification section is non-optional.

## 1. Branch Setup

Already on `gait-emergence` branch. No new branch needed. The existing files (`liang_composite_reward.py`, `verify_liang_composite.py`, modified `go2_velocity_base.py`) will be modified in place.

## 2. Paper Reference

**Liang B., Sun L., Zhu X., Zhang B., Xiong Z., Wang Y., Li C., Sreenath K., Tomizuka M.** "Adaptive Energy Regularization for Autonomous Gait Transition and Energy-Efficient Quadruped Locomotion." ICRA 2025. arXiv:2403.20001v2.

### 2.1 Energy Reward (Paper Eq. 4)

$$
R_{\text{en}} = \exp\!\left(-\frac{\sum_i |\tau_i|\,|\dot{q}_i|}{\sigma_{\text{en},x}\,|v_x| + \sigma_{\text{en},z}\,|\omega_z|}\right)
$$

Paper values: `σ_en,x = 1000`, `σ_en,z = 500`, `α_en = 1.0`. Hatted symbols are commands; unhatted are actual body velocities.

### 2.2 Floor for Denominator

Paper does not specify singularity handling. We add `eps = 0.1`:

$$
\text{denom} = \max(\sigma_{\text{en},x}\,|v_x| + \sigma_{\text{en},z}\,|\omega_z|,\ \varepsilon)
$$

## 3. Why Additive (Departure from Run 1)

Run 1 used multiplicative composition `R = (R_motion + α_en·R_en) · exp(-R_aux)`. Run 1 failed: policy converged to a low-power shuffle without gait differentiation across speeds. Diagnosis: stripping the upstream reward stack and replacing with custom composite removed structural support that the policy needed.

Liang paper itself acknowledges (Section IV-A): "Compared to ANYmal-C, we found that the energy reward R_en alone is insufficient to regularize Go1's behavior, which is likely due to the lighter weight compared to its motor power. Thus, following the settings in [Margolis 2022], we further add a fixed auxiliary reward R_aux to (2)..."

In practice: lighter quadrupeds need substantial auxiliary structure beyond just energy reward. The paper used the multiplicative form on Go1; on Go2 in Isaac Lab the multiplicative form did not transfer.

**Run 2 strategy:** keep the proven additive baseline that already tracks 0–4 m/s. Add `α_en · R_en` as a new additive term. Drop only the gait-biasing terms paper explicitly removed. Single-variable change vs. proven baseline. Cleaner experiment.

## 4. Files

### 4.1 Modify: `liang_composite_reward.py`

Add a new function `energy_cot` implementing Eq. 4. Keep the existing `composite_liang_energy_reward` function in place (unused but available for reference) — do not delete.

```python
def energy_cot(
    env: "ManagerBasedRLEnv",
    sigma_en_x: float = 1000.0,
    sigma_en_z: float = 500.0,
    eps: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    v_x = asset.data.root_lin_vel_b[:, 0]
    w_z = asset.data.root_ang_vel_b[:, 2]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    power = torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)
    denom = torch.clamp(
        sigma_en_x * torch.abs(v_x) + sigma_en_z * torch.abs(w_z),
        min=eps,
    )
    return torch.exp(-power / denom)
```

The function returns shape `(N,)` tensor with values bounded in `(0, 1]`.

### 4.2 Modify: `go2_velocity_base.py`

Rename `_apply_liang_composite_rewards` → `_apply_liang_additive_energy`.

The function should produce this final cfg state:

**Tracking (proven sprint_retune values):**
- `track_lin_vel_xy.weight = 1.5`
- `track_lin_vel_xy.params["std"] = 0.5`
- `track_ang_vel_z.weight = 0.75`

**Smoothness penalties (proven sprint_retune values — weakened from upstream to allow sprint speeds):**
- `action_rate.weight = -0.005`
- `joint_acc.weight = -1e-7`
- `joint_torques.weight = -2e-5`
- `joint_vel.weight = -1e-4`

**Body stability (upstream values, kept):**
- `base_linear_velocity.weight = -2.0`
- `base_angular_velocity.weight = -0.05`
- `flat_orientation_l2.weight = -2.5`
- `dof_pos_limits.weight = -10.0`
- `undesired_contacts.weight = -1.0`
- `feet_slide.weight = -0.1`
- `joint_pos.weight = -0.7`

**Action interface (proven sprint_retune):**
- `cfg.actions.JointPositionAction.scale = 0.35`

**Gait-biasing — DROP (paper-mandated):**
- `feet_air_time.weight = 0.0`
- `air_time_variance.weight = 0.0`

**Energy term — REPLACE function and update params:**
- `cfg.rewards.energy.func = energy_cot` (new function from `liang_composite_reward.py`)
- `cfg.rewards.energy.weight = 1.0` (paper's α_en)
- `cfg.rewards.energy.params = {"sigma_en_x": 1000.0, "sigma_en_z": 500.0, "eps": 0.1}`

**Composite term — DELETE:**
- Remove the `cfg.rewards.composite_liang = RewTerm(...)` line entirely. The composite is no longer registered.

**Logging-only motion weights — REMOVE the 1e-8 hack:**
- The 1e-8 weight was a workaround when motion was inside the composite. Now that motion (`track_lin_vel_xy`) is a real additive term at weight 1.5, the curriculum reads from it normally. No special logging hack needed.

The function should be called from `__post_init__` in both `Go2VelocityBaseEnvCfg` and `Go2VelocityBasePlayEnvCfg`, replacing the previous `_apply_liang_composite_rewards` call.

### 4.3 Files NOT Touched

- `unitree_rl_lab/` (any file)
- `curriculum_rl/curricula/` (any file)
- `curriculum_rl/eval/`, `curriculum_rl/figures/`
- `curriculum_rl/envs/mdp.py` (curriculum step logic)
- `curriculum_rl/envs/commands.py`
- All observation, termination, PPO config

## 5. Default Parameters Summary

| Parameter | Value | Source |
|---|---:|---|
| `σ_en,x` | 1000 | Paper Section IV-A |
| `σ_en,z` | 500 | Paper Section IV-A |
| `α_en` (energy weight) | 1.0 | Paper Section IV-A |
| `eps` | 0.1 | Our deviation (paper does not specify) |
| `track_lin_vel_xy.weight` | 1.5 | Sprint retune (proven baseline) |
| `track_lin_vel_xy.std` | 0.5 | Sprint retune |
| `track_ang_vel_z.weight` | 0.75 | Sprint retune |
| `action_rate.weight` | -0.005 | Sprint retune |
| `joint_acc.weight` | -1e-7 | Sprint retune |
| `joint_torques.weight` | -2e-5 | Sprint retune |
| `joint_vel.weight` | -1e-4 | Sprint retune |
| `feet_air_time.weight` | 0.0 | Paper drops |
| `air_time_variance.weight` | 0.0 | Paper drops |
| Action scale | 0.35 | Sprint retune |

All other R_aux terms keep their upstream values.

## 6. Verification

Run 1's `verify_liang_composite.py` tested the multiplicative composite. That composite is no longer registered, so the script's V3-V12 will fail (composite_liang no longer exists). Instead, do a lightweight verification of just the new `energy_cot` function plus a static cfg check.

### 6.1 Static cfg check

Add a small print block to verify the cfg state right before training. Either modify `verify_liang_composite.py` or write a new minimal script. The script should:

1. Construct the cfg
2. Print all reward term names and their weights
3. Verify that:
   - `track_lin_vel_xy.weight == 1.5`
   - `track_ang_vel_z.weight == 0.75`
   - `feet_air_time.weight == 0.0`
   - `air_time_variance.weight == 0.0`
   - `energy.weight == 1.0`
   - `energy.func == energy_cot`
   - `composite_liang` does NOT exist as a cfg field
   - `actions.JointPositionAction.scale == 0.35`
4. Confirm all R_aux term weights match upstream values

### 6.2 Runtime check

Construct an env, step 100 times with random actions, and assert:
- `energy_cot(env)` returns shape `(num_envs,)` tensor
- Values are in `(0, 1]` (allow tiny numerical slack)
- No NaN / Inf
- Mean value across 100 steps is in `(0.05, 0.95)` range — i.e., not stuck at 0 or 1

This catches gross implementation errors. The math is simpler than Run 1's composite, so heavy verification is not needed.

## 7. Smoke Test (100 iterations)

After verification passes:

```bash
python src/scripts/train.py --condition uniform --seed 0 --max_iterations 100 --headless
```

### 7.1 Expected TensorBoard Scalars at Iter 99

**Training health:**
- `Train/mean_episode_length` ≥ 200 steps and growing
- `Train/mean_reward` positive (range 5–50)

**Episode_Reward terms (which should be present and approximately what magnitudes):**
- `Episode_Reward/track_lin_vel_xy` — positive, growing, in range 50–500 by iter 99
- `Episode_Reward/track_ang_vel_z` — positive
- `Episode_Reward/energy` — positive, in range 100–800 (≈ R_en × ~1000 steps × weight 1.0)
- `Episode_Reward/action_rate` — negative, small
- `Episode_Reward/dof_pos_limits` — small or zero
- `Episode_Reward/flat_orientation_l2` — negative, small
- `Episode_Reward/feet_slide` — negative, small
- `Episode_Reward/undesired_contacts` — small or zero

**Episode_Reward terms that MUST be zero:**
- `Episode_Reward/feet_air_time` — exactly 0.0 (weight is 0)
- `Episode_Reward/air_time_variance` — exactly 0.0 (weight is 0)

**Episode_Reward terms that MUST NOT exist:**
- `Episode_Reward/composite_liang` — not in scalars list

**Termination:**
- `Episode_Termination/time_out` — non-zero (good, episodes reaching end)
- `Episode_Termination/bad_orientation` and `base_contact` — small or zero

### 7.2 Smoke Test Gates (Stop and Diagnose if Any Fail)

- `mean_episode_length` < 100 by iter 99 → policy collapsing, do not proceed
- `mean_reward` strongly negative → penalties dominating, check weights
- `composite_liang` still in scalars list → cfg not actually updated, did not delete the term
- NaN in any scalar → bug in `energy_cot`, check denominator floor

## 8. Full Training (Only After Smoke Passes)

```bash
python src/scripts/train.py --condition uniform --seed 0 --max_iterations 3000 --headless
```

Wall-clock target: ~30 min on RTX 4070 Ti SUPER (matches main-branch baseline since reward stack is now close to baseline).

If wall-clock exceeds 60 min, stop — something is slow that wasn't before.

After seed 0 completes, paste:
- Final `Train/mean_reward` and `Train/mean_episode_length`
- All `Episode_Reward/*` final values
- `Metrics/base_velocity/error_vel_xy` and `error_vel_yaw` final
- All `Episode_Termination/*` final
- Snapshot at iter 1500 of `Train/mean_reward` and `error_vel_xy` (mid-training health check)
- Wall-clock time

Do not start seed 1 until seed 0 is reviewed.

## 9. Acceptance Criteria

The implementation is acceptable if all of the following hold after Uniform 3000-iter run:

**Training health:**
- Exit code 0
- `Train/mean_episode_length` at iter 3000 ≥ 800 (out of 1000 max)
- No NaN at any iteration

**Tracking (this should match or beat the curriculum study baseline):**
- `error_vel_xy` ≤ 0.3 m/s on bins 0–4
- `error_vel_xy` ≤ 0.6 m/s on bins 5–7 (high-speed allowed to be worse)

**Gait emergence (the actual experimental question):**
- Visual inspection of `gait_diagram.png`: low-speed bins show different contact pattern than high-speed bins
- Duty factor (computed from foot contact data) decreases monotonically (or roughly so) with speed
- High-speed bins (≥ 2.5 m/s) show shorter stance fraction than low-speed bins
- At least one bin among 5/6/7 shows visible flight phase (all four feet airborne briefly)

**Hardware safety:**
- p95 joint velocity ≤ 28 rad/s on every bin

If gait emergence criteria fail but tracking is healthy → write up as honest negative result. Do not retune further. The energy reward in additive form on Go2 in Isaac Lab is what it is.

## 10. Workflow

1. **Read this entire document first.**
2. Modify `liang_composite_reward.py` — add `energy_cot` function. Keep existing composite function.
3. Modify `go2_velocity_base.py` — rename function, update body, delete composite_liang registration.
4. Run static cfg check (Section 6.1). Paste output. **Wait for review before continuing.**
5. Run runtime energy check (Section 6.2). Paste output. **Wait for review before continuing.**
6. Run smoke test (Section 7). Paste TensorBoard scalars. **Wait for review.**
7. If smoke passes, run full 3000-iter training (Section 8). Paste metrics. **Wait for review.**
8. **Do not commit.** All commits require explicit confirmation per CLAUDE.md.

## 11. Things That Will NOT Be Touched

- Three curriculum strategies (uniform / task_specific / teacher) — code unchanged, only Uniform exercised
- `mdp.py` curriculum step logic
- Observation / critic specifications
- Termination conditions
- PPO hyperparameters
- Domain randomization
- Sim parameters (dt, decimation)

If during implementation any of these appear to need changes, **stop and report back** rather than modifying.

## 12. Deviations From Paper (For Final Writeup)

1. **Floor on energy denominator (`eps = 0.1`).** Paper does not specify.
2. **Additive composition instead of multiplicative.** Paper uses `R = (R_motion + α_en·R_en) · exp(-R_aux)`; we use `R = R_motion + α_en·R_en + R_aux_additive`. Run 1 attempted faithful multiplicative form and failed. Run 2 uses additive. This is a meaningful methodological deviation that must be documented.
3. **R_aux contents.** We use Isaac Lab Go2 upstream stack with sprint_retune (12 active terms), which is in spirit similar to but not identical to Liang's R_aux from WTW. Paper's R_aux block is borrowed from Margolis 2022.
4. **Hardware platform.** Paper used Go1 + ANYmal-C with `α_en = 1.0`. We use Go2 with the same `α_en = 1.0`.
5. **Simulator.** Paper used IsaacGym; we use Isaac Lab (Isaac Sim 4.5 / PhysX). Documented performance regressions in this version may affect outcomes.

## 13. If Run 2 Also Fails

If after Section 9 acceptance check fails (tracking is fine but gait emergence absent), **do not retry**. Time budget exhausted. Write up as:

> "We implemented Liang 2024's energy-driven gait emergence on Go2 in Isaac Lab, both in the paper's faithful multiplicative form (Run 1) and an additive variant (Run 2). Tracking succeeded. Gait emergence did not transfer cleanly. Hypotheses for non-transfer: (i) platform sensitivity (paper used Go1 12kg + ANYmal-C 50kg; Go2 sits between but closer to Go1, the regime paper itself reports as marginal), (ii) simulator differences (paper used IsaacGym; we used Isaac Lab with documented PhysX regression in v4.5), (iii) α_en tuning may need platform-specific calibration not provided in paper."

That is a defensible scientific contribution. Negative results matter.