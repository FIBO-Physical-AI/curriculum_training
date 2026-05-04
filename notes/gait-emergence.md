# Gait Emergence Implementation: Liang 2024 Multiplicative Composite Reward

This document specifies the implementation of an energy-driven gait emergence experiment for the Go2 quadruped, following Liang, Sun, Zhu et al. ICRA 2025. Read the entire document before writing any code. The verification section is non-optional — implementation drift on the math is the primary risk and must be caught before training begins.

## 1. Branch Setup

Create a new branch from `main`:

```
git checkout main
git pull
git checkout -b gait-emergence
```

Do not branch from any prior gait-shaping branch (Path D, etc.). The point is a clean baseline where `main` is known to track 0–4 m/s without gait shaping.

This branch will not be pushed. All commits stay local until reviewed.

## 2. Paper Reference

**Citation.** Liang B., Sun L., Zhu X., Zhang B., Xiong Z., Wang Y., Li C., Sreenath K., Tomizuka M. "Adaptive Energy Regularization for Autonomous Gait Transition and Energy-Efficient Quadruped Locomotion." ICRA 2025. arXiv:2403.20001v2.

**Paper hardware:** Unitree Go1 + ANYmal-C. Same scaling constants used on both — the formulation transfers across platforms. Go2 is the direct successor of Go1.

### 2.1 Composition Equation (Paper Eq. 2)

$$
R = (R_{\text{motion}} + \alpha_{\text{en}} \cdot R_{\text{en}}(v_x, \omega_z)) \cdot \exp(-R_{\text{aux}})
$$

This is **multiplicative shielding**. The motion + energy block is non-negative; the auxiliary block attenuates that block multiplicatively in the range (0, 1]. The total reward `R` is therefore non-negative at every timestep regardless of how badly the robot is behaving.

### 2.2 Motion Reward (Paper Eq. 3)

$$
R_{\text{lin}} = \exp\!\left(-\frac{|v_x - \hat{v}_x|^2 + |v_y - \hat{v}_y|^2}{\sigma_v}\right)
$$

$$
R_{\text{ang}} = \exp\!\left(-\frac{|\omega_z - \hat{\omega}_z|^2}{\sigma_\omega}\right)
$$

$$
R_{\text{motion}} = R_{\text{lin}} + \alpha_{\text{ang}} \cdot R_{\text{ang}}
$$

Paper values: `σ_v = σ_ω = 0.25` (this is the divisor inside `exp`, not its square root), `α_ang = 0.5`. Hatted symbols are commands.

Note: `σ_v = 0.25` matches Isaac Lab's existing `track_lin_vel_xy_exp(std=0.5)` — that function divides by `std**2 = 0.25`. So the upstream tracking term parameters already match the paper. No change to the math; we re-implement inside the composite.

### 2.3 Energy Reward (Paper Eq. 4)

$$
R_{\text{en}} = \exp\!\left(-\frac{\sum_i |\tau_i|\,|\dot{q}_i|}{\sigma_{\text{en},x}\,|v_x| + \sigma_{\text{en},z}\,|\omega_z|}\right)
$$

Paper values: `σ_en,x = 1000`, `σ_en,z = 500`, `α_en = 1.0`. The denominator is "generalized distance" — translation contribution plus rotation contribution. Setting `σ_en,z = 0` reduces to the conventional Cost of Transport definition.

The numerator `Σ |τ_i| · |q̇_i|` is mechanical power magnitude per joint, summed. Already implemented in upstream `mdp.energy` — we reuse the formula but compose differently.

`v_x`, `ω_z` are the **actual** body linear and angular velocities (not commands). Read from `root_lin_vel_b[:, 0]` and `root_ang_vel_b[:, 2]`.

### 2.4 Floor for Denominator

Paper does not specify how they handle the singularity when `v_x ≈ 0` and `ω_z ≈ 0` (denominator → 0). We add a small floor `eps` to avoid division by zero:

$$
\text{denom} = \max(\sigma_{\text{en},x}\,|v_x| + \sigma_{\text{en},z}\,|\omega_z|,\ \varepsilon)
$$

Default `eps = 0.1`. Document this as a deviation from the paper.

### 2.5 Auxiliary Reward `R_aux`

Paper Section IV-A specifies `R_aux` contents: "collision avoidance, action rate control and trunk orientation regularization … penalizing limb-ground collision, out-of-range joint position, and high frequency joint action."

`R_aux` is the **non-negative** sum of penalty magnitudes. Each contributing term is a non-negative quantity (squared norm, distance, count) multiplied by the absolute value of its weight. Sign convention is critical: `R_aux ≥ 0` always; it goes inside `exp(-R_aux)` so larger means more attenuation.

## 3. Why Multiplicative

From the existing progress update, Section 1.1: at sprint commands the additive penalties exceed the 2.25 tracking ceiling, producing a standing-still local optimum. The current workaround (`_apply_sprint_retune`) weakens five penalty weights to recover a positive optimum at sprint speed. This workaround was flagged in the progress update Q2 as "not fully understood — how much of the bin-6/7 noise is the retune versus the curriculum itself?"

Multiplicative shielding eliminates the standing-still trap structurally:

- `R_motion + α_en · R_en ∈ [0, ~2.5]` always
- `exp(-R_aux) ∈ (0, 1]` always
- Therefore `R ≥ 0` always — penalties cannot invert the sign of the reward, only attenuate it

`_apply_sprint_retune` becomes obsolete and is removed. Action scale (currently 0.35, was 0.25 upstream) is kept at 0.35 because it is a platform setup parameter (joint range of motion needed for the sprint envelope), not a reward-shaping workaround.

## 4. Reward Bucket Assignment

Map the 16 existing reward terms in the upstream Go2 reward stack into the paper's three buckets, plus drops.

### `R_motion` (2 terms)

| Existing term | Notes |
|---|---|
| `track_lin_vel_xy` | Becomes `R_lin` inside composite |
| `track_ang_vel_z` | Becomes `α_ang · R_ang` with α_ang = 0.5 |

### `R_en` (replaces existing `energy`)

| Existing term | Notes |
|---|---|
| `energy` | Replace with new `energy_cot` formulation per Eq. 4 |

### `R_aux` (7 terms)

| Existing term | Why included |
|---|---|
| `dof_pos_limits` | Paper: out-of-range joint position |
| `action_rate` | Paper: high frequency joint action |
| `undesired_contacts` | Paper: limb-ground collision |
| `flat_orientation_l2` | Paper: trunk orientation regularization |
| `base_linear_velocity` (lin_vel_z_l2) | Vertical body stability — prevents pronk/bounce |
| `base_angular_velocity` (ang_vel_xy_l2) | Roll/pitch body stability |
| `feet_slide` | Foot dynamics safety, sim-to-real |

### Dropped (5 terms)

| Existing term | Why dropped |
|---|---|
| `feet_air_time` | Gait-biasing, paper Section IV-A explicitly removes this class of reward |
| `air_time_variance` | Gait-biasing, punishes legitimate 4-beat walk asymmetry |
| `joint_vel` (Σ q̇²) | Subsumed by energy term (energy already penalizes q̇ via τ·q̇ product) |
| `joint_acc` (Σ q̈²) | Not in paper R_aux; smoothness handled by action_rate |
| `joint_torques` (Σ τ²) | Subsumed by energy term |
| `joint_pos` (default-pose penalty) | Not in paper; biases against running posture, conflicts with energy minimization at high speed |

The 7 R_aux terms keep their **upstream** weights (not sprint_retune values). With multiplicative shielding, the sprint-retune weakening is no longer required.

## 5. Files

No hard-coded paths in this document — local session resolves them in the existing project structure. Files referenced by role:

### 5.1 New file: composite reward module

A new Python module under the curriculum_rl envs package containing one function: `composite_liang_energy_reward`. It computes `R_motion`, `R_en`, `R_aux` internally and returns the multiplicative composition.

### 5.2 Modified file: Go2 base velocity environment configuration

The file currently containing `_apply_sprint_retune` and `Go2VelocityBaseEnvCfg`. Modifications:

- Remove the call to `_apply_sprint_retune` (function can stay defined but unused, or delete — implementer's choice)
- Add a new function `_apply_liang_composite_rewards(cfg)` that:
  - Sets weights of dropped terms (`feet_air_time`, `air_time_variance`, `joint_vel`, `joint_acc`, `joint_torques`, `joint_pos`) to `0.0`
  - Sets the existing `energy` term's weight to `0.0` (it is replaced by the composite)
  - Sets `track_lin_vel_xy.weight = 1e-8` (logging-only; see Section 7 on curriculum compatibility)
  - Sets `track_ang_vel_z.weight = 1e-8` (logging-only)
  - Keeps R_aux terms (`dof_pos_limits`, `action_rate`, `undesired_contacts`, `flat_orientation_l2`, `base_linear_velocity`, `base_angular_velocity`, `feet_slide`) at their upstream weights — do not apply sprint-retune values to these
  - Adds a new reward term `composite_liang` calling the new function with weight `1.0`
- Action scale stays at `0.35` (kept as platform setup)
- Call `_apply_liang_composite_rewards(self)` from `__post_init__` of both `Go2VelocityBaseEnvCfg` and `Go2VelocityBasePlayEnvCfg`

### 5.3 Files NOT touched

- Any file under `unitree_rl_lab/`
- Any file under `curriculum_rl/curricula/`
- Any file under `curriculum_rl/eval/`
- Any file under `curriculum_rl/figures/`
- The MDP module `curriculum_rl/envs/mdp.py` (curriculum step logic stays as-is)
- All command, observation, termination, and PPO configuration

## 6. Composite Reward Function Specification

The function signature returns a `(num_envs,)` tensor of per-env reward values. The implementation must follow this exact math; deviations are bugs.

### 6.1 Inputs

- `env`: `ManagerBasedRLEnv`
- `command_name: str = "base_velocity"`
- `sigma_v: float = 0.25`
- `sigma_w: float = 0.25`
- `alpha_ang: float = 0.5`
- `sigma_en_x: float = 1000.0`
- `sigma_en_z: float = 500.0`
- `alpha_en: float = 1.0`
- `eps: float = 0.1`
- `r_aux_clip: float = 10.0`
- `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`

### 6.2 Computation Steps

In order:

1. **Resolve asset and command:**
   - `asset = env.scene[asset_cfg.name]`
   - `cmd = env.command_manager.get_command(command_name)` — shape `(N, 3)` for `[v_x_cmd, v_y_cmd, ω_z_cmd]`

2. **Read body-frame state:**
   - `v_b = asset.data.root_lin_vel_b` — shape `(N, 3)`
   - `w_b = asset.data.root_ang_vel_b` — shape `(N, 3)`
   - `v_x = v_b[:, 0]`, `v_y = v_b[:, 1]` — actual body-frame linear velocity components
   - `w_z = w_b[:, 2]` — actual body-frame yaw rate

3. **Compute R_lin (Eq. 3, linear):**

   $$R_{\text{lin}} = \exp\!\left(-\frac{(v_x - \hat{v}_x)^2 + (v_y - \hat{v}_y)^2}{\sigma_v}\right)$$

   - `v_err_sq = (v_x - cmd[:, 0])**2 + (v_y - cmd[:, 1])**2`
   - `R_lin = torch.exp(-v_err_sq / sigma_v)`
   - Bounded in (0, 1]

4. **Compute R_ang (Eq. 3, angular):**

   $$R_{\text{ang}} = \exp\!\left(-\frac{(\omega_z - \hat{\omega}_z)^2}{\sigma_\omega}\right)$$

   - `w_err_sq = (w_z - cmd[:, 2])**2`
   - `R_ang = torch.exp(-w_err_sq / sigma_w)`
   - Bounded in (0, 1]

5. **Compose R_motion:**
   - `R_motion = R_lin + alpha_ang * R_ang`
   - Bounded in (0, 1.5]

6. **Compute energy (numerator of Eq. 4):**
   - `qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]`
   - `qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]`
   - `power = torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)` — sum over joints, shape `(N,)`

7. **Compute denominator with floor (Eq. 4):**

   $$\text{denom} = \max(\sigma_{\text{en},x}\,|v_x| + \sigma_{\text{en},z}\,|\omega_z|,\ \varepsilon)$$

   - `denom = torch.clamp(sigma_en_x * torch.abs(v_x) + sigma_en_z * torch.abs(w_z), min=eps)`

8. **Compute R_en (Eq. 4):**
   - `R_en = torch.exp(-power / denom)`
   - Bounded in (0, 1]

9. **Compute R_aux:**

   The R_aux value is the sum of the 7 R_aux penalty terms' raw quantities multiplied by the absolute value of their cfg weights. Read these from `env.reward_manager`.

   This requires resolving the seven term names and pulling their per-term cfg weights at function-call time, OR hardcoding them at module load. The simpler implementation is to compute the seven penalty quantities directly inside the function using the same primitives as upstream `mdp.*` functions. See section 6.3.

10. **Clip R_aux:**
    - `R_aux = torch.clamp(R_aux, min=0.0, max=r_aux_clip)`
    - Without clip, severe violations can drive `R_aux` to >30, causing `exp(-R_aux)` underflow and dead gradient. Clip at 10 keeps `exp(-R_aux) ≥ 4.5e-5`.

11. **Final composite:**
    - `R = (R_motion + alpha_en * R_en) * torch.exp(-R_aux)`
    - Bounded in [0, ~2.5]
    - Return this tensor

### 6.3 R_aux Internal Computation

To avoid coupling to `RewardManager` internals (which would create circular dependencies — composite reward depending on other reward terms' computed values), compute the seven R_aux quantities directly. This means re-implementing the math of those penalty terms inside the composite. The upstream functions used are:

- `dof_pos_limits` magnitude: position clamped outside soft limits, summed L1 over joints
- `action_rate` magnitude: `||a_t - a_{t-1}||²` summed
- `undesired_contacts` magnitude: count or contact force magnitude on disallowed bodies
- `flat_orientation_l2` magnitude: `||projected_gravity_xy||²`
- `base_linear_velocity` magnitude: `v_z²`
- `base_angular_velocity` magnitude: `||ω_xy||²`
- `feet_slide` magnitude: foot xy velocity during contact

For each, multiply by the absolute value of the corresponding upstream weight (NOT the cfg-set weight, which is now 0). Hardcode these reference weights as constants inside the composite function or at module level. The reference weights are the **original upstream values**:

```
W_DOF_POS_LIMITS    = 10.0
W_ACTION_RATE       = 0.1
W_UNDESIRED_CONTACTS = 1.0
W_FLAT_ORIENTATION  = 2.5
W_BASE_LIN_VEL_Z    = 2.0
W_BASE_ANG_VEL_XY   = 0.05
W_FEET_SLIDE        = 0.1
```

Sum of `W_i * |raw_quantity_i|` gives R_aux. Document each constant at the top of the module with the upstream source so future drift can be traced.

## 7. Curriculum Compatibility

The existing curriculum step (`mdp.velocity_curriculum_step` in `curriculum_rl/envs/mdp.py`) reads per-bin tracking reward via:

```
env.reward_manager._episode_sums["track_lin_vel_xy"] / episode_length / weight
```

If `track_lin_vel_xy.weight = 0`, the division is `0 / 0` → NaN, and curriculum strategies (`uniform`, `task_specific`, `teacher`) all break.

**Fix:** keep `track_lin_vel_xy.weight = 1e-8` and `track_ang_vel_z.weight = 1e-8`. These weights:
- Contribute negligibly to the additive total (which is dominated by the new `composite_liang` term at weight 1.0)
- Keep `_episode_sums` populated with non-zero values
- Allow curriculum's `per_env = episode_sums / ep_len / weight` calculation to recover the per-step `R_lin` value (since `weight × R_lin / weight = R_lin`)

The existing curriculum step code does not require any modification.

## 8. Default Parameters Summary

| Parameter | Value | Source |
|---|---:|---|
| `σ_v` | 0.25 | Paper Section IV-A |
| `σ_ω` | 0.25 | Paper Section IV-A |
| `α_ang` | 0.5 | Paper Section IV-A (Eq. 3) |
| `σ_en,x` | 1000 | Paper Section IV-A |
| `σ_en,z` | 500 | Paper Section IV-A |
| `α_en` | 1.0 | Paper Section IV-A, Fig. 3a |
| `eps` | 0.1 | Our deviation (paper does not specify) |
| `r_aux_clip` | 10.0 | Our deviation (numerical safety) |
| Composite reward weight | 1.0 | The single active reward term |
| `track_lin_vel_xy.weight` | 1e-8 | Logging-only for curriculum |
| `track_ang_vel_z.weight` | 1e-8 | Logging-only for curriculum |
| Action scale | 0.35 | Kept from sprint-retune (platform setup) |

## 9. Verification Sequence

The greatest risk is implementation drift on the math. The verification sequence below must be executed before training. Implementer adds a small standalone test script (under the project's existing scripts or tests directory — implementer chooses) that constructs a single env and runs the checks. Do not skip checks — fix any failure before proceeding.

### 9.1 Static checks (no rollout)

**V1.** Import the module without errors. Confirm the function signature matches Section 6.1.

**V2.** Confirm in the cfg after `__post_init__`:
- `cfg.rewards.composite_liang.weight == 1.0`
- `cfg.rewards.track_lin_vel_xy.weight == 1e-8`
- `cfg.rewards.track_ang_vel_z.weight == 1e-8`
- All 6 dropped term weights are exactly `0.0`
- All 7 R_aux term weights match their upstream values (printed for human inspection)

### 9.2 Runtime checks (single env, 100 steps with random actions)

**V3.** Assert at every step: `R >= 0`. The multiplicative composition guarantees this; if violated, the implementation has a sign error.

**V4.** Assert at every step: `R_motion ∈ [0, 1.5 + 1e-6]` (allow numerical slack). If exceeded, the motion exp-form is wrong.

**V5.** Assert at every step: `R_en ∈ [0, 1.0 + 1e-6]`. If exceeded, the energy exp-form is wrong.

**V6.** Assert at every step: `R_aux ∈ [0, r_aux_clip]`. If negative, sign convention on a penalty quantity is wrong.

**V7.** Print the mean and max of `R_motion`, `R_en`, `R_aux`, and `R` over the 100 steps. Sanity ranges:
- `R_motion` mean in [0.1, 1.5] — depends on initial random tracking
- `R_en` mean in [0.05, 0.95] — depends on whether motion is happening
- `R_aux` mean in [0.05, 1.0] for normal operation; spikes on resets are fine
- `R` mean in [0.05, 2.0] — log this as the headline number

### 9.3 Edge case checks

**V8. Standing still with zero command.** Force one env to have `v_x = v_y = w_z = 0` (manually zero out the body velocities and command, one step). Expected:
- `R_lin ≈ 1.0` (perfect tracking)
- `R_ang ≈ 1.0` (perfect tracking)
- `R_motion ≈ 1.5`
- `denom = eps = 0.1` (floor active)
- `R_en = exp(-power / 0.1)`. Power should be near zero (no torque, no motion), so `R_en ≈ 1`.
- This confirms standing-still is no longer a free-reward trap; it gives `R ≈ 2.5 × exp(-R_aux)`. **Without the energy penalty, it would still be high — the trap is mitigated by walking giving similar reward, not by punishing standing.**

**V9. Sprint command with zero motion.** Force one env to have command `v_x = 4.0`, actual `v_x = 0`. Expected:
- `R_lin = exp(-16 / 0.25) = exp(-64) ≈ 0` (huge tracking error)
- `R_ang ≈ 1.0` (yaw OK)
- `R_motion ≈ 0.5` (only ang_track contributing)
- This is much lower than V8's 1.5. So with multiplicative composition, sprint-with-zero-motion is **worse** than standing-with-zero-command — the policy has incentive to actually move at sprint command. (In the additive system, both would be similar after sprint_retune; this is the qualitative improvement.)

**V10. Hardware-violation case.** Force one env into a joint-limit violation (set joint pos to limit + 0.5 rad). Expected:
- `dof_pos_limits` quantity > 0
- `R_aux` jumps significantly
- `R` drops via the `exp(-R_aux)` factor
- Confirms the multiplicative shielding works

### 9.4 Numerical stability checks

**V11.** Run 1000 random steps. Assert no NaN, no Inf in `R`, `R_motion`, `R_en`, `R_aux` at any step.

**V12.** Confirm `denom` floor is engaging at low actual velocity. Print `(denom == eps).float().mean()` over the 1000 steps. Expect non-zero fraction (should engage during stationary or near-stationary moments).

## 10. Smoke Test (Before Full Training)

After verification passes, run a 100-iteration training to confirm the system learns at all under the new reward.

```
python src/scripts/train.py --condition uniform --seed 0 max_iterations=100
```

(Implementer adapts argument syntax to existing entry point. The point is: 100 iterations on uniform sampler, single seed.)

**Expected at iteration 100:**

- Mean episode length grows toward `episode_length_s` (= 20 s = 1000 steps); should be at least 100 steps mean by iter 100.
- Mean total reward `R` per step is in [0.5, 2.5] range.
- Mean `R_lin` (logged via the 1e-8 weight on `track_lin_vel_xy`) is positive and trending upward.
- No NaN in any logged scalar.
- Curriculum step logs (in `curriculum.csv`) show non-zero `mean_reward` per bin.

If any of these fail, do not proceed. Diagnose first.

## 11. Full Training Run

After smoke test passes, run the full training. Single condition (Uniform), two seeds, 3000 iterations:

```
bash src/scripts/run_sweep.sh
```

(Or whatever invocation pattern the existing sweep script uses. The implementer ensures only the Uniform condition is enabled for this experiment, and only seeds 0 and 1 are run.)

Expected wall-clock: ~30 min × 2 seeds = ~60 min total on the user's RTX 4070 Ti SUPER, matching the main-branch throughput baseline.

If wall-clock exceeds 90 min, the composite reward function has a Python loop or host-device sync that needs to be diagnosed before further runs.

## 12. New Plots

**None.** The existing plotting infrastructure is reused without modification:
- `convergence.png`, `epte_bars.png`, `iterations_to_mastery.png`, `survival.png`, `v_actual_vs_cmd.png`, `sampling_heatmap.png`, `gait_diagram.png`, `action_rate.png`

The gait diagram (`plot_gait_diagram.py`) is the primary visual output for this experiment — it shows foot contact patterns over a 3-second window per (condition, bin), which is exactly what we need to verify gait emergence.

The CoT-vs-velocity plot from Liang Fig. 3a is a useful additional figure but is **not required** for this implementation. Do not add new plotting code unless asked.

## 13. Acceptance Criteria

The implementation is acceptable if all of the following hold after the full Uniform 3000-iter × 2-seed run:

**Training health:**
- Both seeds complete with exit code 0
- Mean episode length at iter 3000 ≥ 800 steps on bins 0–5
- No NaN in any logged scalar at any iteration

**Tracking:**
- `R_lin` at iter 3000 (mean over 2 seeds) ≥ 0.7 on bins 0–4
- `R_lin` at iter 3000 (mean over 2 seeds) ≥ 0.5 on bins 5–7
- Achieved velocity matches commanded velocity within 0.3 m/s on bins 0–5 in eval rollouts

**Gait emergence (the actual experimental question):**
- Gait diagram on bin 0 (0.25 m/s) shows visibly different contact pattern than bin 6 (3.25 m/s)
- High-speed bins (≥ 2.5 m/s) show shorter stance fraction than low-speed bins
- At least one of bins 5/6/7 shows visible flight phase (all four feet off ground simultaneously) for at least 20 ms per stride

If gait emergence criteria are not met but tracking is healthy, the fallback is one re-tune attempt: bump `α_en` from 1.0 to 1.5, watching for tracking degradation per the paper's Fig. 3a observation. If second attempt also fails, rollback this branch and reconsider design.

**Hardware safety:**
- p95 joint velocity over all rollouts ≤ 28 rad/s on every bin (Go2 hardware envelope)

## 14. Things That Will NOT Be Touched

To prevent scope creep:

- The three curriculum strategies (uniform / task_specific / teacher) — code unchanged, only Uniform is exercised in this experiment
- `mdp.py` curriculum step logic
- Observation / critic specifications
- Termination conditions
- PPO hyperparameters (network, learning rate, entropy coefficient, etc.)
- Domain randomization
- Action interface (joint position action with scale 0.35)

If during implementation any of these appear to need changes to make the composite reward work, **stop and report back** rather than modifying. That kind of cascading change is the warning sign of a wrong abstraction, not a license to refactor.

## 15. Deviations From Paper (Document for Submission)

The following deviations from Liang et al. 2024 are intentional and must be acknowledged in the write-up:

1. **Floor on energy denominator (`eps = 0.1`).** Paper does not specify singularity handling.
2. **Clip on R_aux (`r_aux_clip = 10`).** Paper does not specify; needed for numerical stability.
3. **Logging-only motion weights (`1e-8`).** Implementation artifact for compatibility with the existing curriculum-tracking infrastructure. Mathematically negligible.
4. **R_aux contents.** Paper specifies "collision avoidance, action rate control and trunk orientation regularization" — we include 7 terms covering these plus body stability. Document the 7 terms and their weights.
5. **Hardware platform.** Paper used Go1 + ANYmal-C with `α_en = 1.0`. We use Go2 with the same `α_en = 1.0` — Go2 is the direct successor of Go1 with similar mass and leg length, so the same scaling is expected to work. If empirically `α_en = 1.0` is too weak/strong on Go2, this is a tunable parameter and the deviation is benign.
6. **Composition at the framework level.** Paper applies multiplicative composition directly. Isaac Lab's `RewardManager` is additive; the multiplicative form is implemented inside a single composite reward function. This is a faithful re-expression, not a mathematical deviation.