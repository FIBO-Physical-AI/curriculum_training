# 2026-04-22 — Disabled `track_ang_vel_z` (and added replacement penalty)

## Status: Resolved (contradicts proposal §6.1; proposal text must be updated)

## Short summary

Under the unmodified `unitree_rl_lab` Go2 reward configuration, the policy converged to a "stand still" solution and never learned to track any forward velocity. Cause: the yaw-tracking reward `track_ang_vel_z` (weight `+0.75`) combined with the proposal's choice of fixing yaw command at zero creates a free per-step reward for doing nothing. The fix was to set its weight to `0.0` and add a small yaw-rate L2 penalty (`ang_vel_z_l2`, weight `-0.05`) to replace its implicit stability role.

The fix contradicts proposal §6.1, which states the upstream reward configuration is adopted "without modification". The proposal needs one sentence added.

## Symptom

After training uniform seed 0 for 3000 PPO iterations on flat terrain, the eval script `src/scripts/eval_epte.py` produced the following EPTE-SP values on 100 deterministic rollouts per bin:

| bin | $v_x^{\mathrm{cmd}}$ (m/s) | `fall_step` (mean) | `tracking_error` (mean) | `EPTE-SP` (mean) |
|---:|---:|---:|---:|---:|
| 0 | 0.25 | 999 (no fall) | 1.000 | 1.000 |
| 1 | 0.75 | 999 | 1.000 | 1.000 |
| 2 | 1.25 | 999 | 1.000 | 1.000 |
| ⋮ | ⋮ | ⋮ | ⋮ | ⋮ |
| 7 | 3.75 | 999 | 1.000 | 1.000 |

`fall_step = 999` for all 800 rollouts means no robot fell — the episode timed out. `tracking_error = 1.000` (saturated) means the mean per-step velocity error exceeded the commanded velocity, i.e. the robot's actual linear velocity was $\approx 0$ m/s the entire episode. **The policy's deterministic mean action is "stand still"**, for every bin.

## Why this happened — the reward landscape

### The task reward structure

Upstream `track_lin_vel_xy` and `track_ang_vel_z` are both exponential-kernel tracking rewards:

$$
r_{v_{xy}}^{\mathrm{cmd}}(s) \;=\; w_{v_{xy}} \cdot \exp\!\left( - \frac{\lVert \mathbf{v}^{\mathrm{cmd}}_{xy} - \mathbf{v}_{xy} \rVert^{2}}{\sigma_{v}^{2}} \right),
\qquad
r_{\omega_{z}}^{\mathrm{cmd}}(s) \;=\; w_{\omega_z} \cdot \exp\!\left( - \frac{(\omega_{z}^{\mathrm{cmd}} - \omega_{z})^{2}}{\sigma_{\omega}^{2}} \right)
$$

with $w_{v_{xy}} = 1.5$, $w_{\omega_z} = 0.75$, $\sigma_{v} = \sigma_{\omega} = \sqrt{0.25} = 0.5$.

Both are bounded in $[0, w]$ — maximum when error is zero, decaying sharply with squared error.

### The trap

Proposal §4 *In scope* fixes the yaw command at zero throughout training:

$$
\omega_{z}^{\mathrm{cmd}} \equiv 0.
$$

Under this constraint, consider a policy whose mean action keeps the robot stationary ($\mathbf{v}_{xy} \approx 0$, $\omega_{z} \approx 0$). Its per-step reward breakdown is:

| reward term | value when standing still |
|---|---|
| $r_{v_{xy}}$, bin $\zeta_i$ with $v_i = (i+0.5)\Delta v$ | $1.5 \cdot \exp(-v_i^2 / 0.25)$ |
| $r_{\omega_z}$ (with $\omega_z^{\mathrm{cmd}} = 0$ and $\omega_z = 0$) | $0.75 \cdot \exp(0) = 0.75$ (**constant maximum**) |
| penalty sum (`action_rate`, `joint_torques`, `flat_orientation`, …) | $\approx 0$ (no movement $\Rightarrow$ no penalty) |

The $r_{\omega_z}$ reward is **a constant free +0.75 per step** as long as the policy doesn't rotate — it has nothing to do with the tracking task defined in the proposal, because the command is fixed at zero, making this reward independent of the policy's tracking behavior.

Over an episode of $K = 1000$ steps, standing gives $\approx 750$ free reward. To beat this by walking, the policy must earn more from $r_{v_{xy}}$ plus accept the penalties $\sum_k |r_{\text{pen}}(s_k)|$. Since $\exp(-v_i^2 / 0.25)$ decays very fast (values of $v_i$: $0.25$ gives $0.78$; $0.75$ gives $0.105$; $1.25$ gives $0.002$; $\geq 1.75$ gives numerical zero), walking on hard bins yields $\approx 0$ $r_{v_{xy}}$ while incurring $\approx -200$ / episode from action-rate and stability penalties. Standing is strictly better than walking on 6 of 8 bins.

At 3000 iterations the optimizer discovers this local optimum and parks the mean action of the Gaussian policy at "don't move", using exploration noise (policy std $\approx 0.26$ at iter 2999) to occasionally stumble into partial tracking during training. At deterministic evaluation the exploration noise is removed, revealing the underlying motionless mean action.

### TensorBoard evidence (uniform, seed 0, 3000 iters, before fix)

| metric at iter 2999 | value |
|---|---:|
| `Train/mean_reward` | `584` |
| `Train/mean_episode_length` | `1000` (never falls) |
| `Episode_Termination/bad_orientation` | `0` |
| `Episode_Reward/track_ang_vel_z` | `703` (free reward, $\approx 0.75 \times 1000$ / episode) |
| `Episode_Reward/track_lin_vel_xy` | `260` / `1500` (17 % of ceiling) |
| `Metrics/base_velocity/error_vel_xy` | `1.85 m/s` (essentially not moving) |

Compare to a hypothetical perfectly-standing policy: reward $\approx 0 + 750 + 0 = 750$ / episode — **better than the actual 584**. The optimizer found the standing optimum and the remaining 164 of reward lost is the net cost of exploration noise during training.

## The fix

### Code change (very small)

Added `_remove_yaw_tracking_reward(cfg)` helper, invoked from both `Go2VelocityBaseEnvCfg.__post_init__` and `Go2VelocityBasePlayEnvCfg.__post_init__`, at [src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py](../src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py):

```python
def _remove_yaw_tracking_reward(cfg) -> None:
    if hasattr(cfg.rewards, "track_ang_vel_z"):
        cfg.rewards.track_ang_vel_z.weight = 0.0
    cfg.rewards.ang_vel_z_penalty = RewTerm(
        func=curriculum_mdp.ang_vel_z_l2,
        weight=-0.05,
    )
```

A new L2-kernel yaw-rate penalty was added to `curriculum_rl.envs.mdp`:

$$
r_{\omega_z}^{\mathrm{pen}}(s) \;=\; -0.05 \cdot \omega_z^{2}.
$$

Isaac Lab's built-in `ang_vel_xy_l2` only penalizes roll/pitch rates; nothing in the upstream reward list penalizes yaw rate directly, so this term is needed to retain the yaw-stability behavior that the (now-disabled) $r_{\omega_z}^{\mathrm{cmd}}$ was implicitly providing.

### Reward landscape after fix

For the same standing policy:

| term | before | after |
|---|---:|---:|
| $r_{\omega_z}^{\mathrm{cmd}}$ weight | $0.75$ | $0.0$ |
| standing per-step bonus | $0.75 / \text{step}$ | $0 / \text{step}$ |
| standing-episode floor | $\approx 750$ | $\approx 0$ (only the $\exp(-v_i^2/0.25)$ from $r_{v_{xy}}$ on low-$v$ bins) |
| incentive to move | net negative on bins 2–7 | net positive wherever gradient exists |

### Yaw-stability replacement

$r_{\omega_z}^{\mathrm{pen}} = -0.05 \,\omega_z^2$ is zero when the robot is not rotating and grows quadratically with yaw rate. It cannot produce a standing-is-optimal trap because its maximum (over any non-degenerate policy state) is zero — there is no free reward for sitting still, only a penalty for spinning. This replaces the stability role of the disabled tracking term without reintroducing the degeneracy.

## Verification (uniform, seed 0, 3000 iters, after fix)

| metric at iter 2999 | before | after |
|---|---:|---:|
| `Metrics/base_velocity/error_vel_xy` | `1.85 m/s` | **`0.04 m/s`** |
| `Episode_Reward/track_lin_vel_xy` | `260` / 1500 | `297` / 1500 |
| `Episode_Reward/track_ang_vel_z` | `703` | `0.00` (disabled ✓) |
| `Episode_Reward/ang_vel_z_penalty` | n/a | `-0.98` / episode |
| `Train/mean_episode_length` | `1000` | `190` (policy now falls while exploring) |
| `Policy/mean_noise_std` | `0.26` | `0.35` |

EPTE-SP under the post-fix policy:

| bin | $v_x^{\mathrm{cmd}}$ | before (mean EPTE) | after (mean EPTE) | after falls/100 |
|---:|---:|---:|---:|---:|
| 0 | 0.25 m/s | 1.000 | **0.359** | 0 |
| 1 | 0.75 m/s | 1.000 | **0.081** | 1 |
| 2 | 1.25 m/s | 1.000 | 1.000 | 100 (falls at step $\approx 9$) |
| 3 | 1.75 m/s | 1.000 | 1.000 | 100 |
| ≥ 4 | ≥ 2.25 m/s | 1.000 | 1.000 | 100 |

The policy now walks. Failure to track high-velocity commands is a separate issue (kernel $\exp(-\text{err}^2/\sigma^2)$ gradient vanishes past err $\approx 2\sigma$) and is the problem the curriculum is designed to address.

## Proposal §6.1 text update required

Replace:

> Reward, observation, termination, and PPO/network hyperparameters: adopted from the Unitree Go2 velocity-tracking task configuration in `unitree_rl_lab`~\cite{unitree_rllab} **without modification**. Using an off-the-shelf configuration removes reward shaping and observation design as confounds in the comparison.

with:

> Reward, observation, termination, and PPO/network hyperparameters: adopted from the Unitree Go2 velocity-tracking task configuration in `unitree_rl_lab`~\cite{unitree_rllab} with one modification. Because proposal §4 fixes the yaw command at zero throughout, the upstream yaw-tracking reward $r_{\omega_z}^{\mathrm{cmd}} = 0.75 \exp(-(\omega_z - 0)^2 / 0.25)$ collapses to a constant $0.75$ per step for any policy that does not rotate, producing a "standing is optimal" local optimum. We therefore set its weight to zero and add a small L2 yaw-rate penalty $r_{\omega_z}^{\mathrm{pen}} = -0.05\,\omega_z^2$ to retain yaw stability. All other reward, observation, termination, and PPO/network hyperparameters are adopted without modification. Using an off-the-shelf configuration otherwise removes reward shaping and observation design as confounds in the comparison.

## Files touched

- [src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py](../src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py) — added `_remove_yaw_tracking_reward` helper, called from both the training and play env configs.
- [src/source/curriculum_rl/curriculum_rl/envs/mdp.py](../src/source/curriculum_rl/curriculum_rl/envs/mdp.py) — added `ang_vel_z_l2(env, asset_cfg)` reward-function implementation; Isaac Lab ships `ang_vel_xy_l2` only.

## Risks / follow-ups

1. **Yaw drift not penalized, only yaw rate.** Nothing in the reward penalizes yaw *angle* drift. Policy may slowly rotate in place at a low rate without immediate penalty. Observed in practice: `Episode_Reward/ang_vel_z_penalty` stays near `-1` per episode, suggesting very low sustained yaw rate.
2. **Proposal text must be updated** before this fix can be considered consistent with the project plan.
3. **High-velocity bins still fail** after this fix. That is the proposal's target failure mode (proposal §7 Expected Results: "Uniform. Return curves plateau early and low on hard bins.") and is the problem the curricula must solve — it is not caused by this fix.
