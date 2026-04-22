# `unitree_rl_lab` Go2 velocity-tracking MDP — reference document

**Purpose.** This file documents the complete MDP — actions, observations, reward terms, terminations, events, commands, and PPO setup — as defined by the upstream `unitree_rl_lab` Go2 velocity-tracking task configuration at commit `4960b847`. This is the *unmodified* baseline from which our curriculum experiments inherit. Reading this file should answer *exactly what the robot sees, what it outputs, what it is rewarded for, and under what conditions an episode ends*.

**Primary source.** [`unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py)

**Secondary sources.** Function bodies in [`IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py`](/home/drl-68/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py), [`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`](/home/drl-68/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py), [`unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py).

---

## 1. Robot — Unitree Go2

Quadruped, 12 actuated joints (3 per leg), all revolute:

```
FR_hip_joint   FR_thigh_joint   FR_calf_joint
FL_hip_joint   FL_thigh_joint   FL_calf_joint
RR_hip_joint   RR_thigh_joint   RR_calf_joint
RL_hip_joint   RL_thigh_joint   RL_calf_joint
```

**Default joint positions** (used as the zero-action reference):

| joint pattern | default angle (rad) |
|---|---:|
| `.*R_hip_joint`   | $-0.1$ |
| `.*L_hip_joint`   | $+0.1$ |
| `F[L,R]_thigh_joint` | $+0.8$ |
| `R[L,R]_thigh_joint` | $+1.0$ |
| `.*_calf_joint`   | $-1.5$ |

**Initial base height:** $z_0 = 0.4$ m. **Initial base linear/angular velocity:** zero.

**Actuators:** all 12 joints use a single `UnitreeActuatorCfg_Go2HV` model with:

$$
\tau \;=\; k_p\,(q^{\mathrm{des}} - q) \;-\; k_d\,\dot{q} \;-\; f_s \operatorname{sign}(\dot{q}),
$$

with $k_p = 25.0$, $k_d = 0.5$, $f_s = 0.01$ (static/Coulomb friction). Actuator saturation handled by the `UnitreeActuatorCfg_Go2HV` model's effort and velocity limits.

Source: [`unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py:32-76`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py).

---

## 2. Time discretization

| quantity | symbol | value |
|---|---|---|
| Physics step | $\mathrm{d}t_{\text{sim}}$ | $0.005$ s |
| Decimation (actions applied every N physics steps) | $D$ | $4$ |
| Control step (RL action) | $\mathrm{d}t = D \cdot \mathrm{d}t_{\text{sim}}$ | $0.020$ s ($50$ Hz) |
| Episode length (wall-clock) | $T_{\text{ep}}$ | $20.0$ s |
| Episode length (control steps) | $K$ | $T_{\text{ep}} / \mathrm{d}t = 1000$ |

So one episode is 1000 policy calls, 4000 physics steps.

Source: [`velocity_env_cfg.py:385-388`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py).

---

## 3. Command space $C$

The command is a 3-vector $\mathbf{c} = (v_x^{\mathrm{cmd}},\, v_y^{\mathrm{cmd}},\, \omega_z^{\mathrm{cmd}})$ in the robot's body frame.

Upstream defaults (`UniformLevelVelocityCommandCfg`):

| component | initial range | final / limit range |
|---|---|---|
| $v_x^{\mathrm{cmd}}$ | $[-0.1, +0.1]$ m/s | $[-1.0, +1.0]$ m/s |
| $v_y^{\mathrm{cmd}}$ | $[-0.1, +0.1]$ m/s | $[-0.4, +0.4]$ m/s |
| $\omega_z^{\mathrm{cmd}}$ | $[-1.0, +1.0]$ rad/s | $[-1.0, +1.0]$ rad/s |

- **Resampling interval:** every $10.0$ s (i.e. twice per 20-s episode). Reset happens by drawing independently from the uniform distribution on the *current* active range.
- **Standing-env fraction:** `rel_standing_envs = 0.1`. On each resample, 10 % of parallel environments have $\mathbf{c}$ *zeroed* by `_update_command` to encourage the policy to also hold a stand.
- **Range curriculum:** the upstream `lin_vel_cmd_levels` curriculum expands the active range additively each episode when $\mathbb{E}[r_{v_{xy}}] > 0.8 \cdot w_{v_{xy}}$, up to the limit range. Our project replaces this with the binned `BinnedVelocityCommand` and disables `lin_vel_cmd_levels`.

Source: [`velocity_env_cfg.py:187-202`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py), [`unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py).

---

## 4. Action space $A$

Action is a 12-dim vector $\mathbf{a}_t \in \mathbb{R}^{12}$. Each component is an additive **delta from the default joint position**, scaled by $\alpha = 0.25$:

$$
q^{\mathrm{des}}_{t}[j] \;=\; q^{\mathrm{default}}[j] \;+\; \alpha \cdot a_t[j] \quad \text{for } j = 1, \dots, 12.
$$

The action is then clipped to $[-100, +100]$ per joint (effectively no clipping in normal operation), and the target $q^{\mathrm{des}}_t$ is passed to the actuator PD controller at $50$ Hz (i.e. each target is held for 4 physics steps).

The RL policy outputs $\mathbf{a}_t$ from a Gaussian distribution $\mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\sigma}^2)$ where $\boldsymbol{\mu}_t$ is the actor network output and $\boldsymbol{\sigma}$ is a learned diagonal (initialized to $1.0$).

**Action dimension: $|A| = 12$.**

Source: [`velocity_env_cfg.py:205-211`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py).

---

## 5. Observation space $O$

Two observation groups: a **policy** observation (visible to actor) and a **critic** observation (privileged, visible only to value network — enables asymmetric actor-critic training).

### 5.1 Policy observation (`ObservationsCfg.PolicyCfg`) — 45-dim

Concatenated in this exact order; terms use a uniform additive noise model for robustness (`enable_corruption=True`). Noise is sampled once per step per environment per term.

| term | symbol | dim | scale | noise ($\pm$) | formula / source |
|---|---|---:|---:|---|---|
| `base_ang_vel`       | $\boldsymbol{\omega}_b$ | 3 | $0.2$ | $[-0.2, +0.2]$ | $\boldsymbol{\omega}_b = $ root angular velocity in body frame |
| `projected_gravity`  | $\hat{\mathbf{g}}_b$    | 3 | $1.0$ | $[-0.05, +0.05]$ | $\hat{\mathbf{g}}_b = R_b^\top \hat{\mathbf{g}}_w$; upright $\Rightarrow \hat{\mathbf{g}}_b \approx (0,0,-1)$ |
| `velocity_commands`  | $\mathbf{c}$            | 3 | $1.0$ | none | $(v_x^{\mathrm{cmd}}, v_y^{\mathrm{cmd}}, \omega_z^{\mathrm{cmd}})$ |
| `joint_pos_rel`      | $\tilde{\mathbf{q}}$    | 12 | $1.0$ | $[-0.01, +0.01]$ | $\tilde q_j = q_j - q^{\mathrm{default}}_j$ |
| `joint_vel_rel`      | $\tilde{\dot{\mathbf{q}}}$ | 12 | $0.05$ | $[-1.5, +1.5]$ | $\tilde{\dot q}_j = \dot q_j - \dot q^{\mathrm{default}}_j = \dot q_j$ (default $\dot q^{\mathrm{default}} = 0$) |
| `last_action`        | $\mathbf{a}_{t-1}$      | 12 | $1.0$ | none | previous step's action |

**Total policy observation dimension: $|O_{\pi}| = 3 + 3 + 3 + 12 + 12 + 12 = 45$.**

### 5.2 Critic observation (`ObservationsCfg.CriticCfg`) — 51-dim privileged

Adds the (otherwise unobservable) base linear velocity $\mathbf{v}_b$ and joint torques:

| term | symbol | dim | noise |
|---|---|---:|---|
| `base_lin_vel`       | $\mathbf{v}_b$           | 3 | none |
| `base_ang_vel`       | $\boldsymbol{\omega}_b$  | 3 | none (scale 0.2) |
| `projected_gravity`  | $\hat{\mathbf{g}}_b$     | 3 | none |
| `velocity_commands`  | $\mathbf{c}$             | 3 | none |
| `joint_pos_rel`      | $\tilde{\mathbf{q}}$     | 12 | none |
| `joint_vel_rel`      | $\tilde{\dot{\mathbf{q}}}$ | 12 | none (scale 0.05) |
| `joint_effort`       | $\boldsymbol{\tau}$      | 12 | none (scale 0.01) |
| `last_action`        | $\mathbf{a}_{t-1}$       | 12 | none |

**Total critic observation dimension: $|O_{V}| = 3 + 3 + 3 + 3 + 12 + 12 + 12 + 12 = 60$.**

> **Why the noise on policy obs but not critic obs?** The policy must be robust at deployment (sensor noise is real). The critic is an advantage estimator at training time only, and sees a cleaner, privileged view to improve value accuracy — this is standard asymmetric actor-critic (sometimes called "teacher-student" or "privileged critic").

Source: [`velocity_env_cfg.py:214-265`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py).

---

## 6. Reward function $R$

Per-step reward is the weighted sum of 16 terms:

$$
r_t \;=\; \sum_{k=1}^{16} w_k \cdot \phi_k(s_t, a_t, s_{t+1})
$$

where $\phi_k$ is the per-step contribution from reward term $k$. Episodic return is $R_\tau = \sum_{t=0}^{K-1} r_t$. The following table lists every term, its function, its weight, and its per-step value.

### 6.1 Task rewards (positive shaping toward tracking)

| term | weight $w$ | function | per-step value $\phi$ |
|---|---:|---|---|
| `track_lin_vel_xy` | $+1.5$ | `mdp.track_lin_vel_xy_exp` | $\exp\!\left( -\dfrac{\lVert \mathbf{v}_{xy}^{\mathrm{cmd}} - \mathbf{v}_{xy} \rVert^{2}}{\sigma_{v}^{2}} \right)$, $\sigma_v = \sqrt{0.25}$ |
| `track_ang_vel_z`  | $+0.75$ | `mdp.track_ang_vel_z_exp` | $\exp\!\left( -\dfrac{(\omega_{z}^{\mathrm{cmd}} - \omega_{z})^{2}}{\sigma_{\omega}^{2}} \right)$, $\sigma_\omega = \sqrt{0.25}$ |

Both are bounded in $[0, 1]$. Maximum when error is zero, decays as Gaussian kernel. The parameter $\sigma = \sqrt{0.25} = 0.5$ means tracking error of $0.5$ m/s produces reward $\exp(-0.25/0.25) = e^{-1} \approx 0.37$; error of $1.0$ m/s produces $\exp(-4) \approx 0.018$.

### 6.2 Root stability penalties

| term | weight $w$ | function | per-step value $\phi$ (always $\geq 0$) |
|---|---:|---|---|
| `base_linear_velocity`  | $-2.0$   | `mdp.lin_vel_z_l2`      | $v_{b,z}^{2}$ — penalizes vertical bouncing |
| `base_angular_velocity` | $-0.05$  | `mdp.ang_vel_xy_l2`     | $\omega_{b,x}^{2} + \omega_{b,y}^{2}$ — penalizes roll/pitch rate |
| `flat_orientation_l2`   | $-2.5$   | `mdp.flat_orientation_l2`| $\hat g_{b,x}^{2} + \hat g_{b,y}^{2}$ — penalizes base tilt (1 when lying on side) |

### 6.3 Actuator / joint penalties

| term | weight $w$ | function | per-step value $\phi$ |
|---|---:|---|---|
| `joint_vel`            | $-0.001$  | `mdp.joint_vel_l2`        | $\sum_{j} \dot q_{j}^{2}$ |
| `joint_acc`            | $-2.5 \times 10^{-7}$ | `mdp.joint_acc_l2` | $\sum_{j} \ddot q_{j}^{2}$ |
| `joint_torques`        | $-2 \times 10^{-4}$ | `mdp.joint_torques_l2` | $\sum_{j} \tau_{j}^{2}$ |
| `action_rate`          | $-0.1$    | `mdp.action_rate_l2`      | $\sum_{j} (a_{t,j} - a_{t-1,j})^{2}$ — discourages jittery actions |
| `dof_pos_limits`       | $-10.0$   | `mdp.joint_pos_limits`    | sum of squared violations outside the soft joint limit range |
| `energy`               | $-2 \times 10^{-5}$ | `mdp.energy`       | $\sum_{j} \lvert \dot q_{j} \rvert \cdot \lvert \tau_{j} \rvert$ |

### 6.4 Posture-specific penalty

| term | weight $w$ | function | description |
|---|---:|---|---|
| `joint_pos` | $-0.7$ | `mdp.joint_position_penalty` | $\lVert \mathbf{q} - \mathbf{q}^{\mathrm{default}} \rVert_{2}$, **scaled by $5\times$ when the command is zero AND body speed $< 0.3$ m/s** (forces a specific "stand" pose when told to stand still) |

Source: [`unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py:64-73`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py).

### 6.5 Foot / gait rewards and penalties

| term | weight $w$ | function | description |
|---|---:|---|---|
| `feet_air_time` | $+0.1$ | `mdp.feet_air_time` | rewards each foot's swing duration over a threshold of $0.5$ s, measured at first-contact event. Zeroed when $\lVert \mathbf{v}_{xy}^{\mathrm{cmd}} \rVert < 0.1$ (no bonus for lifting feet while standing). Value: $\sum_{f \in \text{feet}} (\text{last\_air\_time}_{f} - 0.5) \cdot \mathbb{1}[\text{first\_contact}_{f}]$ |
| `air_time_variance` | $-1.0$ | `mdp.air_time_variance_penalty` | $\operatorname{Var}_f(\min(\text{last\_air\_time}_f, 0.5)) + \operatorname{Var}_f(\min(\text{last\_contact\_time}_f, 0.5))$ — forces symmetric gait across the four legs |
| `feet_slide` | $-0.1$ | `mdp.feet_slide` | $\sum_{f} \lVert \mathbf{v}^{\text{world}}_{f, xy} \rVert \cdot \mathbb{1}[\text{contact}_f]$ — penalizes a foot moving laterally while in contact |

### 6.6 Contact penalty

| term | weight $w$ | function | description |
|---|---:|---|---|
| `undesired_contacts` | $-1.0$ | `mdp.undesired_contacts` | Counts number of body parts in the set $\{ \text{Head}, \text{\_hip}, \text{\_thigh}, \text{\_calf} \}$ whose contact force exceeds $1.0$ N. Non-foot body contact is strongly discouraged |

### 6.7 Numerical worked example — standing still on commanded $v_x = 0.25$ m/s

| term | $\phi$ | $w$ | $w \cdot \phi$ |
|---|---:|---:|---:|
| `track_lin_vel_xy` | $\exp(-0.0625/0.25) = 0.779$ | $+1.5$ | $+1.17$ |
| `track_ang_vel_z`  | $\exp(0) = 1.0$ | $+0.75$ | $+0.75$ |
| `base_linear_velocity`  | $\approx 0$ | $-2.0$ | $\approx 0$ |
| `base_angular_velocity` | $\approx 0$ | $-0.05$ | $\approx 0$ |
| `flat_orientation_l2`   | $\approx 0$ | $-2.5$ | $\approx 0$ |
| `joint_vel` | $\approx 0$ | $-0.001$ | $\approx 0$ |
| `joint_acc` | $\approx 0$ | $-2.5 \times 10^{-7}$ | $\approx 0$ |
| `joint_torques` | small (holding pose) | $-2 \times 10^{-4}$ | small negative |
| `action_rate` | $\approx 0$ | $-0.1$ | $\approx 0$ |
| `dof_pos_limits` | $0$ | $-10.0$ | $0$ |
| `energy` | $\approx 0$ | $-2 \times 10^{-5}$ | $\approx 0$ |
| `joint_pos` | small | $-0.7$ | small negative |
| `feet_air_time` | $0$ (no steps) | $+0.1$ | $0$ |
| `air_time_variance` | $0$ | $-1.0$ | $0$ |
| `feet_slide` | $0$ | $-0.1$ | $0$ |
| `undesired_contacts` | $0$ | $-1.0$ | $0$ |
| **sum** | | | **$\approx +1.92$ per step** |

Over $K = 1000$ steps: $\approx +1920$ per episode. This is the "standing is optimal" floor that motivated our `track_ang_vel_z` fix (see [2026-04-22_disabled_track_ang_vel_z.md](2026-04-22_disabled_track_ang_vel_z.md)).

Source for the reward list: [`velocity_env_cfg.py:268-344`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py).

---

## 7. Terminations

A rollout ends when any of the three below trigger. Only `time_out` is "truncation" (does not zero the value bootstrap); the other two are true environment deaths.

| term | function | parameters | description |
|---|---|---|---|
| `time_out` | `mdp.time_out` | none (uses `episode_length_s`) | Wall-clock timeout at $K = 1000$ control steps. `time_out=True` → reported as truncation, not failure |
| `base_contact` | `mdp.illegal_contact` | sensor on `base`, threshold $1.0$ N | Any contact force on the base body → termination (robot flipped, belly down, etc.) |
| `bad_orientation` | `mdp.bad_orientation` | `limit_angle=0.8` rad ($\approx 45.8°$) | Base tilt exceeds $0.8$ rad from vertical → termination |

Source: [`velocity_env_cfg.py:346-355`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py).

---

## 8. Events (domain randomization + perturbations)

Events run at fixed lifecycle points; they are the main source of domain randomization.

### 8.1 Startup (once per scene creation)

| event | parameters |
|---|---|
| `physics_material` (friction) | static friction $\sim \mathcal{U}(0.3, 1.2)$, dynamic friction $\sim \mathcal{U}(0.3, 1.2)$, restitution $\sim \mathcal{U}(0.0, 0.15)$, sampled once per env across 64 buckets |
| `add_base_mass` | add $\sim \mathcal{U}(-1.0, +3.0)$ kg to the base body (default Go2 base mass ~5 kg) |

### 8.2 Reset (every episode boundary)

| event | parameters |
|---|---|
| `base_external_force_torque` | force_range=(0,0), torque_range=(0,0) — effectively disabled in default config |
| `reset_base` (pose)       | $\Delta x,\Delta y \sim \mathcal{U}(-0.5, +0.5)$ m, yaw $\sim \mathcal{U}(-\pi, +\pi)$. Linear/angular velocities reset to zero |
| `reset_robot_joints`      | joint positions scaled by factor $\mathcal{U}(1.0, 1.0)$ (no change), joint velocities $\sim \mathcal{U}(-1.0, +1.0)$ rad/s |

### 8.3 Interval (during episode)

| event | parameters |
|---|---|
| `push_robot` | Every $\sim\mathcal{U}(5, 10)$ s: impulse setting base velocity $\Delta v_x, \Delta v_y \sim \mathcal{U}(-0.5, +0.5)$ m/s |

These perturbations train policy robustness. They are **disabled** in the `RobotPlayEnvCfg` used for our evaluation runs (proposal §6.4 Phase 4: "Domain randomisation: disabled").

Source: [`velocity_env_cfg.py:115-184`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py).

---

## 9. Scene & terrain

- **Parallel environments:** 4096 (training) / 32 (play/eval); our eval uses 100.
- **Environment spacing:** 2.5 m (ignored for terrain-generator scenes, which tile the world).
- **Terrain:** `terrain_type="generator"` with `COBBLESTONE_ROAD_CFG`. The generator tiles a 10 × 20 grid of 8 × 8 m sub-terrains. Each sub-terrain is one of:
  - `flat` (proportion $0.1$) — smooth plane.
  - Commented-out in source: `random_rough`, `hf_pyramid_slope` (×2), `boxes`, `pyramid_stairs` (×2). In the shipped config, only `flat` is enabled, so in practice the terrain is a tiled flat marble surface with small heightmap variance.
- **Friction model:** multiply combine, $\mu_s = \mu_d = 1.0$ before randomization (`physics_material` event overrides per-env).
- **Height scanner:** $1.6 \text{ m} \times 1.0 \text{ m}$ grid at $0.1 \text{ m}$ resolution, attached to the robot base — **available but not currently included in the policy observation group** (commented out in `CriticCfg`).

Our curriculum project overrides the terrain to a single flat plane (`terrain_type="plane"`, no generator) to match proposal §4 "on flat ground". See [`src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py`](../src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py).

Source: [`velocity_env_cfg.py:24-113`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py).

---

## 10. Upstream curriculum terms (disabled in our project)

Upstream ships two curriculum terms, both of which mutate environment state during training:

| term | function | trigger | effect |
|---|---|---|---|
| `terrain_levels` | `terrain_levels_vel` | every `max_episode_length` steps | advances terrain difficulty per-env based on how far each robot walked during the episode; disabled in our project because we flatten the terrain |
| `lin_vel_cmd_levels` | `lin_vel_cmd_levels`  | every `max_episode_length` steps | grows `cfg.ranges.lin_vel_x` and `lin_vel_y` by $\pm 0.1$ m/s per advance when $\mathbb{E}[r_{v_{xy}}] > 0.8 \cdot w_{v_{xy}}$, up to `limit_ranges`; disabled because we replace the command with our binned sampler |

We **disable both** and attach our own `velocity_curriculum_step` term that updates the bin-sampling distribution.

Source: [`velocity_env_cfg.py:358-363`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py).

---

## 11. PPO hyperparameters (`BasePPORunnerCfg`)

Uses `rsl_rl.OnPolicyRunner`. Network: `RslRlPpoActorCriticCfg` feed-forward MLP (no recurrence).

| parameter | value |
|---|---:|
| `num_steps_per_env` (rollout length between updates) | $24$ |
| `max_iterations` (upstream default) | $50000$ |
| `save_interval` | every $100$ iterations |
| `empirical_normalization` | `False` (obs not normalized by a running mean) |

### 11.1 Policy / value networks

| parameter | value |
|---|---|
| `actor_hidden_dims` | $[512, 256, 128]$ |
| `critic_hidden_dims` | $[512, 256, 128]$ |
| `activation` | ELU |
| `init_noise_std` (policy $\boldsymbol{\sigma}$ at start) | $1.0$ (learnable per-dim) |

### 11.2 PPO algorithm

| parameter | value |
|---|---:|
| `value_loss_coef` | $1.0$ |
| `use_clipped_value_loss` | True |
| `clip_param` $\epsilon$ | $0.2$ |
| `entropy_coef` | $0.01$ |
| `num_learning_epochs` | $5$ |
| `num_mini_batches` | $4$ |
| `learning_rate` (initial) | $1 \times 10^{-3}$ |
| `schedule` | adaptive (KL-targeted) |
| `desired_kl` | $0.01$ |
| `gamma` ($\gamma$, discount) | $0.99$ |
| `lam` ($\lambda$, GAE) | $0.95$ |
| `max_grad_norm` | $1.0$ |

Optimizer: Adam (rsl_rl default).

**Adaptive KL scheduler rule:** after each PPO update, if observed mean-KL is more than $2 \times$ `desired_kl`, the learning rate is halved; if less than $0.5 \times$ `desired_kl`, the learning rate is doubled. Bounded in $[10^{-5}, 10^{-2}]$ by rsl_rl.

Source: [`unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`](../unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py).

---

## 12. End-to-end flow (one control step)

```
                ┌──────────────────────────────────────────────┐
                │  previous action  a_{t-1}                    │
                │  (stored by env)                             │
                └───────────────┬──────────────────────────────┘
                                ↓
     ┌──────────────────────────────────────────────────────────┐
     │  Observation:                                            │
     │    ω_b, ĝ_b, c, q̃, q̇̃, a_{t-1}  (45 dims, w/ noise)        │
     └─────────────────┬────────────────────────────────────────┘
                       ↓
       ┌───────────────────────────────────┐
       │  Actor MLP [512,256,128] → μ_t    │    critic MLP sees privileged obs
       │  sample a_t ~ N(μ_t, σ²)          │    → value estimate V(s_t)
       └───────────────┬───────────────────┘
                       ↓
       ┌───────────────────────────────────┐
       │  q_des = q_default + 0.25 · a_t    │
       └───────────────┬───────────────────┘
                       ↓
       ┌───────────────────────────────────┐
       │  PD actuator @500 Hz × 4 substeps: │
       │    τ = k_p(q_des - q) - k_d q̇ - fᵣ │
       └───────────────┬───────────────────┘
                       ↓
       ┌───────────────────────────────────┐
       │  Physics step(s) — PhysX 0.005 s   │
       └───────────────┬───────────────────┘
                       ↓
       ┌───────────────────────────────────────────────────────┐
       │  Compute reward r_t = Σ_k w_k · φ_k(s_t, a_t, s_{t+1}) │
       │  Check terminations  (time_out | base_contact | tilt)  │
       │  If done → _reset_idx → resample command (maybe)       │
       │  Event interval? → push_robot                          │
       └───────────────┬───────────────────────────────────────┘
                       ↓
                  next state s_{t+1},
                  next observation
```

---

## 13. Connection to this project's experiment

In our curriculum experiment, the following are **inherited unmodified** from the above:

- Robot, joints, PD controller settings
- Observations (both policy and critic groups, noise levels, scales)
- PPO hyperparameters (all of §11)
- Terminations
- Events (domain randomization)
- Reward terms **except** $r_{\omega_z}^{\mathrm{cmd}}$ (weight set to $0$ — see [2026-04-22_disabled_track_ang_vel_z.md](2026-04-22_disabled_track_ang_vel_z.md))
- Episode length, decimation, $\mathrm{d}t$

The following are **replaced** by project-specific code:

- Terrain (upstream cobblestone $\to$ flat plane, per proposal §4)
- Command sampling: upstream `UniformLevelVelocityCommandCfg` with `lin_vel_cmd_levels` $\to$ project's `BinnedVelocityCommandCfg` with `velocity_curriculum_step`
- $v_y^{\mathrm{cmd}}, \omega_z^{\mathrm{cmd}}$ fixed at $0$ (proposal §4) — enforced in the binned command config
- `rel_standing_envs` set to $0.0$ in the play-env config for eval (so 100 % of rollouts carry the bin-center command)
- One added reward term: `ang_vel_z_penalty` with weight $-0.05$, to retain yaw stability after disabling $r_{\omega_z}^{\mathrm{cmd}}$

Everything else stays byte-for-byte aligned with the upstream baseline. This is intentional — proposal §6.1's claim that "using an off-the-shelf configuration removes reward shaping and observation design as confounds in the comparison" still holds for all three conditions, because any such confound affects every condition identically.
