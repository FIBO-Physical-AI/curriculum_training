# Curriculum Learning for Velocity Tracking on the Unitree Go2 Quadruped

**Author:** Phakin Boonchanachai (66340500037)
**Course:** FRA503 Deep Reinforcement Learning

---

## Table of Contents

- [1. Introduction](#1-introduction)
  - [1.1 Motivation](#11-motivation)
  - [1.2 Problem statement](#12-problem-statement)
  - [1.3 Objectives](#13-objectives)
  - [1.4 Scope and limitations](#14-scope-and-limitations)
  - [1.5 Contributions](#15-contributions)
  - [1.6 Structure of the report](#16-structure-of-the-report)
- [2. Background](#2-background)
  - [2.1 PPO and Isaac Lab in brief](#21-ppo-and-isaac-lab-in-brief)
  - [2.2 Velocity-command curricula](#22-velocity-command-curricula)
    - [2.2.1 Box Adaptive curriculum (task-specific)](#221-box-adaptive-curriculum-task-specific)
    - [2.2.2 LP-ACRL (teacher-guided)](#222-lp-acrl-teacher-guided)
    - [2.2.3 Structural comparison](#223-structural-comparison)
  - [2.3 Reward design for wide-range velocity tracking](#23-reward-design-for-wide-range-velocity-tracking)
    - [2.3.1 Structure of a locomotion reward](#231-structure-of-a-locomotion-reward)
    - [2.3.2 The upstream IsaacLab Go2 16-term reward](#232-the-upstream-isaaclab-go2-16-term-reward)
  - [2.4 Synthesis](#24-synthesis)
- [3. Methodology](#3-methodology)
  - [3.1 Experimental setup and reward configuration](#31-experimental-setup-and-reward-configuration)
    - [3.1.1 Simulator, training budget, and parallelism](#311-simulator-training-budget-and-parallelism)
    - [3.1.2 Velocity command space](#312-velocity-command-space)
    - [3.1.3 Action interface and observation](#313-action-interface-and-observation)
    - [3.1.4 Reward configuration and the standing-still derivation](#314-reward-configuration-and-the-standing-still-derivation)
  - [3.2 The three curriculum conditions](#32-the-three-curriculum-conditions)
    - [3.2.1 Uniform (baseline)](#321-uniform-baseline)
    - [3.2.2 Box Adaptive (task-specific)](#322-box-adaptive-task-specific)
    - [3.2.3 LP-ACRL (teacher-guided)](#323-lp-acrl-teacher-guided)
  - [3.3 Evaluation protocol](#33-evaluation-protocol)
  - [3.4 Hyperparameter summary](#34-hyperparameter-summary)
- [4. Results](#4-results)
  - [4.1 Three curricula under the sprint-retune reward](#41-three-curricula-under-the-sprint-retune-reward)
    - [4.1.1 Per-bin tracking-reward convergence](#411-per-bin-tracking-reward-convergence)
    - [4.1.2 Per-bin EPTE-SP](#412-per-bin-epte-sp)
    - [4.1.3 Qualitative behaviour at evaluation](#413-qualitative-behaviour-at-evaluation)
  - [4.2 From sprint-retune to energy regularisation](#42-from-sprint-retune-to-energy-regularisation)
    - [4.2.1 The Liang energy-regularised reward](#421-the-liang-energy-regularised-reward)
    - [4.2.2 Calibrating the energy scale](#422-calibrating-the-energy-scale)
  - [4.3 Three curricula under the energy-regularised reward](#43-three-curricula-under-the-energy-regularised-reward)
    - [4.3.1 Per-bin tracking-reward convergence](#431-per-bin-tracking-reward-convergence)
    - [4.3.2 Per-bin EPTE-SP](#432-per-bin-epte-sp)
    - [4.3.3 Qualitative behaviour at evaluation](#433-qualitative-behaviour-at-evaluation)
    - [4.3.4 Gait classification](#434-gait-classification)
- [5. Conclusion](#5-conclusion)
- [References](#references)

---

## 1. Introduction

### 1.1 Motivation

The Unitree Go2 is a small quadruped robot. To be useful in the field, it needs a single controller that works across a wide speed range, from a standstill up to a sprint at around 4 m/s. A controller that only runs at one fixed speed is not practical for real deployment.

Modern locomotion controllers are usually trained with deep reinforcement learning (DRL). A neural-network policy is trained in simulation against a reward that depends on how well it tracks a commanded velocity, and the trained policy is then transferred to hardware (Rudin et al., 2022). Recent simulators such as Isaac Lab (Mittal et al., 2023) make this practical on a single workstation by running thousands of robots in parallel.

Training one policy across a wide speed range is not the same as training a policy at a single target speed. The control problem is harder at high speed than at low speed:

- At low speed, the robot can balance easily. The reward signal is dense and the policy learns from the very first iteration.
- At high speed, the dynamics change. Ground reaction forces grow, the gait pattern has to shift from walk to trot to run, and the robot is more likely to fall over.

If velocity commands are sampled uniformly across the full range from the start of training, most high-speed commands return almost no reward, because the policy cannot yet run and fails immediately. The gradient is dominated by these useless failures, and the policy ends up specialising on the easy low-speed end of the range while never learning the sprint end (Margolis et al., 2022).

This is the *sparse-signal problem* at high speed, and it motivates **curriculum learning**: actively choosing which velocity commands the policy sees during training, so that compute is spent where the policy can still make progress.

Two different curriculum philosophies have appeared in the recent quadrupedal-locomotion literature:

- **Box Adaptive** (Margolis et al., 2022). Start at low velocity. Expand the sampling window outward only after each bin's tracking reward crosses a mastery threshold. The rule only adds weight to bins; it never removes weight.
- **LP-ACRL** (Li, Li, & Hutter, 2026). Reallocate sampling mass continuously using a softmax over per-task learning progress. When a bin stops improving, its weight is automatically reduced and reallocated to bins that are still improving.

Although both rules have been published individually (Box Adaptive on the MIT Mini Cheetah, LP-ACRL on the ANYmal D), no matched comparison of the two has been reported on the Unitree Go2, to the author's knowledge.

> **Terminology used throughout this report.** Following the naming used in the project's implementation, the Box Adaptive rule of Margolis et al. (2022) is referred to as the *task-specific* curriculum, and the LP-ACRL rule of Li, Li, & Hutter (2026) as the *teacher-guided* curriculum. The baseline that gives every bin the same sampling probability is referred to as the *uniform* condition. All later chapters use these three names (*uniform*, *task-specific*, *teacher-guided*) when referring to the conditions of the matched comparison.

There is also a hardware-safety reason to care about which curriculum strategy is used. The Go2's joint motors have finite torque and velocity limits. A policy that solves the simulated tracking reward by spinning its legs at unphysical speeds, or by snapping its feet down at uncontrolled action rates, will damage the robot once deployed. The curriculum does not just decide *which* speeds are learned. It also shapes *how* they are learned.

### 1.2 Problem statement

This report trains and evaluates a single PPO policy (Schulman et al., 2017) on the Unitree Go2 to track forward velocity commands across the range $v_x^{\mathrm{cmd}} \in [0, 4]$ m/s in simulation.

- Lateral velocity command $v_y^{\mathrm{cmd}}$ is fixed at $0$.
- Yaw-rate command $\omega_z^{\mathrm{cmd}}$ is fixed at $0$.
- The forward range is split into $N = 8$ bins of width $\Delta v = 0.5$ m/s.

The central question: **which command-sampling strategy lets the single policy reach the high-velocity bins under a fixed training budget on a flat-ground simulation of the Go2?**

### 1.3 Objectives

- **O1.** Implement three velocity-command sampling strategies for a single PPO policy on the Go2 in Isaac Lab:
  - *Uniform*: a baseline that gives every bin the same sampling probability throughout training.
  - *Task-specific*: Box Adaptive (Margolis et al., 2022).
  - *Teacher-guided*: LP-ACRL (Li, Li, & Hutter, 2026).
- **O2.** Run a matched comparison of the three strategies across three independent random seeds on $[0, 4]$ m/s partitioned into $8$ bins. All non-curriculum components (reward, observations, terminations, PPO hyperparameters, domain randomisation) are held fixed across conditions.
- **O3.** Evaluate the three strategies using four per-bin metrics:
  - Mean tracking reward.
  - EPTE-SP: expected per-step tracking error, normalised by commanded speed.
  - Task-sampling distribution over training.
  - Iterations-to-mastery.

### 1.4 Scope and limitations

The work is bounded as follows.

- **Velocity axis.** Only the forward velocity command is studied. Lateral velocity and yaw rate are fixed at zero throughout training and evaluation.
- **Terrain.** Training and evaluation are on a flat ground plane. No terrain generator, no height scanner, no terrain-difficulty curriculum.
- **Simulation only.** All experiments run in Isaac Lab. No sim-to-real transfer or hardware deployment is attempted in this report.
- **Single platform.** Only the Unitree Go2 is used. No cross-platform validation against other quadrupeds.
- **Single algorithm.** PPO is the only RL algorithm used. No comparison against alternative policy-optimisation methods.
- **Single-run cells.** Each (curriculum, seed) cell is run once. The matched comparison uses three seeds per condition; broader hyperparameter sweeps are out of scope.

### 1.5 Contributions

This report contributes:

- A matched three-way comparison of the uniform baseline, task-specific (Box Adaptive; Margolis et al., 2022), and teacher-guided (LP-ACRL; Li, Li, & Hutter, 2026) sampling strategies on the Unitree Go2 across $[0, 4]$ m/s, with every non-curriculum component held fixed across conditions.
- A per-bin analysis of tracking accuracy and learning progress for each condition, including evidence on which strategies reach the high-velocity bins under the fixed iteration budget and which fail.
- A discussion of where the observed task-sampling distributions and iterations-to-mastery patterns agree or diverge from the hypotheses underlying the two published curriculum rules.

### 1.6 Structure of the report

The remainder of the report is organised as follows.

- **Chapter 2: Background.** Deep RL for legged locomotion. The two curriculum strategies under comparison. The gait-taxonomy framework used later in the report to interpret locomotion behaviour.
- **Chapter 3: Methodology.** Experimental design, robot and simulator setup, reward function, command space, curriculum operator parameters, PPO hyperparameters, and evaluation metrics.
- **Chapter 4: Results and discussion.** Quantitative results for the three curricula, followed by discussion.

---

## 2. Background

### 2.1 PPO and Isaac Lab in brief

**Proximal Policy Optimisation (PPO)** is the policy-gradient algorithm used to train every policy in this report (Schulman et al., 2017). PPO is an on-policy actor-critic method. The actor outputs an action distribution conditioned on the current observation; the critic estimates the value function used to compute the advantage of each action. At every update, PPO collects a fresh batch of trajectories from the current policy and maximises a clipped surrogate objective:

$$
L^{\mathrm{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\!\left( r_t(\theta)\, \hat{A}_t,\; \mathrm{clip}\!\left( r_t(\theta),\, 1 - \epsilon,\, 1 + \epsilon \right) \hat{A}_t \right) \right],
$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)$ is the importance-sampling ratio between the new and old policies, $\hat{A}_t$ is the estimated advantage, and $\epsilon$ is the clip range. The clip prevents updates from moving the policy too far in a single iteration, which trades a small amount of optimisation speed for substantially improved stability. In practice this makes PPO robust to hyperparameter choices and reliable enough to train large simulated systems without manual tuning of the trust region.

**Isaac Lab** is the simulator used for every experiment in this report (Mittal et al., 2023). It is a modular robot-learning framework layered over NVIDIA Isaac Sim that exposes thousands of parallel simulation environments on a single GPU. Each parallel environment runs an independent copy of the robot, the terrain, and the reward computation, and PPO collects gradient information from the union of all environments at every step. This parallelism is the reason a wide-range velocity-tracking policy can be trained on a single workstation in roughly an hour rather than over days (Rudin et al., 2022). Isaac Lab also provides standard interfaces for observation composition, command sampling, termination conditions, and domain randomisation, all of which are used in this work.

### 2.2 Velocity-command curricula

Both curriculum rules studied in this report operate on the same input: a per-task tracking-reward signal computed over recently observed trajectories. They differ in how that signal is consumed to produce the next sampling distribution over commands. This section presents each rule in the notation of its source paper.

#### 2.2.1 Box Adaptive curriculum (task-specific)

Margolis et al. (2022) consider a velocity command sampled at the start of each episode from a probability distribution over a two-dimensional command space spanned by the forward linear-velocity command $\mathbf{v}_x^{\mathrm{cmd}}$ and the yaw-rate command $\boldsymbol{\omega}_z^{\mathrm{cmd}}$. The command space is represented as a discrete grid with resolution $[0.5\ \mathrm{m/s},\ 0.5\ \mathrm{rad/s}]$ centered at $[0\ \mathrm{m/s},\ 0\ \mathrm{rad/s}]$. The sampling distribution at episode $k$ is denoted $p_{\mathbf{v}_x,\boldsymbol{\omega}_z}^{k}(\cdot, \cdot)$.

The **Box Adaptive** variant maintains the joint distribution as a product of independent marginals,

$$
p_{\mathbf{v}_x,\boldsymbol{\omega}_z}^{k}(\cdot, \cdot) = p_{\mathbf{v}_x}^{k}(\cdot)\, p_{\boldsymbol{\omega}_z}^{k}(\cdot),
$$

with the per-component update rules

$$
p_{\mathbf{v}_x}^{k+1}(\cdot) \leftarrow f_v\!\left( p_{\mathbf{v}_x}^{k}(\cdot),\, r_{v_x^{\mathrm{cmd}}} \right),
\qquad (8\text{a})
$$

$$
p_{\boldsymbol{\omega}_z}^{k+1}(\cdot) \leftarrow f_\omega\!\left( p_{\boldsymbol{\omega}_z}^{k}(\cdot),\, r_{\omega_z^{\mathrm{cmd}}} \right),
\qquad (8\text{b})
$$

where $r_{v_x^{\mathrm{cmd}}}$ and $r_{\omega_z^{\mathrm{cmd}}}$ are the velocity-tracking rewards on the most recent episode whose command was $\mathbf{v}_x^{\mathrm{cmd}},\, \boldsymbol{\omega}_z^{\mathrm{cmd}}$. The specific Box Adaptive update is

$$
p_{\mathbf{v}_x}^{k+1}(\mathbf{v}_x^n) \leftarrow
\begin{cases}
p_{\mathbf{v}_x}^{k}(\mathbf{v}_x^n) & r_{v_x^{\mathrm{cmd}}} < \gamma, \\
1 & \text{otherwise},
\end{cases}
\qquad (10\text{a})
$$

$$
p_{\boldsymbol{\omega}_z}^{k+1}(\boldsymbol{\omega}_z^n) \leftarrow
\begin{cases}
p_{\boldsymbol{\omega}_z}^{k}(\boldsymbol{\omega}_z^n) & r_{\omega_z^{\mathrm{cmd}}} < \gamma, \\
1 & \text{otherwise},
\end{cases}
\qquad (10\text{b})
$$

where $\gamma \in (0, 1)$ is a success threshold, and the neighbouring commands $\mathbf{v}_x^n,\, \boldsymbol{\omega}_z^n$ are the adjacent discretised cells on each axis:

$$
\mathbf{v}_x^n \in \{\mathbf{v}_x^{\mathrm{cmd}} - 0.5,\ \mathbf{v}_x^{\mathrm{cmd}} + 0.5\}, \qquad
\boldsymbol{\omega}_z^n \in \{\boldsymbol{\omega}_z^{\mathrm{cmd}} - 0.5,\ \boldsymbol{\omega}_z^{\mathrm{cmd}} + 0.5\}.
$$

The structural property of this rule is that probability mass is only added; it is never removed. Once a cell is unlocked, it remains in the sampling distribution for the rest of training. The curriculum therefore expands outward from the initialisation in a monotone fashion.

In this report, the Box Adaptive rule is implemented over a one-dimensional command axis (forward linear velocity only, with lateral velocity and yaw rate fixed at zero per §1.2) and is referred to as the **task-specific** curriculum.

#### 2.2.2 LP-ACRL (teacher-guided)

Li, Li, and Hutter (2026) formulate curriculum learning as an explicit task-sampling problem. The task space is denoted $\mathcal{T}$, and the curriculum is a sequence of task-sampling distributions

$$
\mathcal{C} = (c_0, c_1, \ldots, c_j, \ldots), \qquad c_j \in \Delta(\mathcal{T}),
\qquad (1)
$$

where $c_j$ is the task-sampling distribution at curriculum stage $j$. A task instance is denoted $\zeta \in \mathcal{T}$.

The agent's performance on task instance $\zeta$ is measured by the expected episodic reward when training under distribution $c_j$:

$$
R_{c_j}(\zeta) = \mathbb{E}_{\tau \sim c_j}\!\left[ R_\tau \right],
\qquad (5)
$$

where $R_\tau$ is the cumulative reward of a trajectory $\tau$ of length $H$.

**Learning progress** on a task instance is defined as the change in expected episodic reward between two consecutive curriculum stages:

$$
LP_{c_j}(\zeta) = R_{c_j}(\zeta) - R_{c_{j-1}}(\zeta).
\qquad (6)
$$

A task instance with positive $LP$ is improving; a task instance with $LP$ near zero has plateaued, either because it has been mastered or because it is currently unlearnable.

The task-sampling distribution at the next curriculum stage is then a softmax over learning progress:

$$
c_{j+1}(\zeta) = \frac{\exp\!\left( LP_{c_j}(\zeta) / \beta \right)}{\sum_{\zeta' \in \mathcal{T}} \exp\!\left( LP_{c_j}(\zeta') / \beta \right)},
\qquad (7)
$$

where $\beta$ is a temperature parameter that controls the sharpness of the distribution. Small $\beta$ produces a near winner-take-all distribution; large $\beta$ produces a near-uniform distribution.

The structural property of this rule is that probability mass is *reallocated* at every curriculum stage. When a task instance plateaus, its softmax weight collapses, and the freed mass is redistributed across whichever instances are currently improving fastest.

In this report, LP-ACRL is implemented on the same one-dimensional command axis as the Box Adaptive variant (forward linear velocity, 8 discrete task instances of width $0.5$ m/s) and is referred to as the **teacher-guided** curriculum.

#### 2.2.3 Structural comparison

The two rules differ in three properties that have direct consequences for the analysis in later chapters.

- **Reversibility.** Box Adaptive adds probability mass to cells and never removes it. LP-ACRL reweights the entire distribution at every stage based on current learning progress.
- **Trigger.** Box Adaptive uses a fixed-threshold trigger ($\gamma$): a cell unlocks the moment its tracking reward crosses $\gamma$. LP-ACRL has no fixed threshold; sampling weight is allocated proportionally to recent improvement.
- **Dilution.** Once $k$ cells are unlocked under Box Adaptive, each unlocked cell receives roughly $1/k$ of the sampling budget regardless of which cells are still improving. Under LP-ACRL, sampling weight follows learning progress, so a plateaued cell shares minimal budget regardless of how many other cells are being sampled.

These three differences appear repeatedly in the analysis of Chapter 4.

### 2.3 Reward design for wide-range velocity tracking

#### 2.3.1 Structure of a locomotion reward

A locomotion reward for velocity tracking is typically a weighted sum of three families of terms (Rudin et al., 2022).

- **Tracking terms** that produce a positive scalar when the actual body velocity matches the commanded velocity. The standard form is an exponential kernel on velocity error.
- **Smoothness and effort penalties** that produce a negative scalar when the policy commands jerky or high-torque actions. Typical members of this family penalise joint velocity, joint acceleration, joint torque, and the change in the policy's output action between consecutive control steps.
- **Bonus terms** that reward specific gait-pattern features, most commonly a feet-air-time bonus that pays out when a foot is in swing phase for at least a threshold duration.

The total reward at step $t$ is

$$
r_t = \sum_{i} w_i\, r_i(\zeta_t),
$$

where $r_i$ is the $i$-th term and $w_i$ is its weight. Tracking terms have $w_i > 0$, penalty terms have $w_i < 0$, and bonus terms have $w_i > 0$. The relative magnitudes of these weights determine which solution the policy converges to.

#### 2.3.2 The upstream IsaacLab Go2 16-term reward

The starting point of this work is the upstream IsaacLab Go2 reward function published by Unitree Robotics (2024), composed of 16 weighted terms. The two tracking terms are exponential kernels on linear and angular velocity error:

$$
r_{\mathrm{lin}} = \exp\!\left( -\, \frac{\| v_{xy}^{\mathrm{cmd}} - v_{xy} \|^2}{\sigma_{\mathrm{lin}}^{2}} \right), \qquad w_{\mathrm{lin}} = 1.5, \quad \sigma_{\mathrm{lin}} = 0.5\ \mathrm{m/s},
$$

$$
r_{\mathrm{ang}} = \exp\!\left( -\, \frac{(\omega_z^{\mathrm{cmd}} - \omega_z)^2}{\sigma_{\mathrm{ang}}^{2}} \right), \qquad w_{\mathrm{ang}} = 0.75, \quad \sigma_{\mathrm{ang}} = 0.25\ \mathrm{rad/s}.
$$

Both kernels saturate at $1$ when the actual velocity matches the command, so the maximum reward contribution from the two tracking terms combined is $w_{\mathrm{lin}} + w_{\mathrm{ang}} = 2.25$.

The smoothness and effort group is

$$
r_{\mathrm{smooth}} = -\,|w_{\dot{q}}|\, \|\dot{q}\|^2 \;-\; |w_{\ddot{q}}|\, \|\ddot{q}\|^2 \;-\; |w_\tau|\, \|\tau\|^2 \;-\; |w_{\Delta a}|\, \|a_t - a_{t-1}\|^2 \;\le\; 0,
$$

with all four weights $w_{\dot{q}},\, w_{\ddot{q}},\, w_\tau,\, w_{\Delta a}$ negative in the configuration, so that the leading sign of each term in $r_{\mathrm{smooth}}$ is negative.

The remaining ten terms cover orientation, height stability, contact bonuses (including feet-air-time), and termination penalties. The upstream values for all 16 weights, and the specific modifications adopted in this work, are reported in Chapter 3.

### 2.4 Synthesis

The three streams above set up the experimental setting of Chapter 3 and the comparisons of Chapter 4. PPO and Isaac Lab (§2.1) supply the optimisation algorithm and the parallel simulator under which a curriculum comparison is feasible. The two curriculum rules (§2.2) provide qualitatively different hypotheses about how training compute should be allocated across a wide command range: Box Adaptive grows the sampling support monotonically with measured per-bin success, while LP-ACRL reallocates probability mass continuously based on observed learning progress. The reward function (§2.3) defines what succeeding on a per-bin task actually means in scalar terms, and its 16 weighted components are the surface on which both tracking quality and locomotion style are eventually scored.

---

## 3. Methodology

This chapter specifies the concrete experimental setting under which the three curricula of Chapter 2 are compared. The setting has four parts: the simulator and command space, the reward configuration actually used in this work (which deviates from the upstream Go2 defaults of §2.3.2 for a reason that is derived below), the three curriculum conditions, and the evaluation protocol. A consolidated hyperparameter summary is given at the end of the chapter.

### 3.1 Experimental setup and reward configuration

#### 3.1.1 Simulator, training budget, and parallelism

All experiments are run in Isaac Lab on a single workstation with 4096 parallel environments collecting rollouts in lockstep. Each PPO update consumes a batch of 24 environment steps per env, and training runs for 3000 PPO iterations per (condition, seed). Three independent random seeds are drawn for every condition, so each curriculum is compared on three runs and reported statistics are taken across seeds.

The 3000-iteration budget is half of what a longer pilot run at 6000 iterations had used, and is short by the standard of comparable wide-range velocity-tracking studies. The choice is a wall-clock compromise: a three-condition, three-seed sweep at 3000 iterations at 4096 envs occupies one GPU for roughly nine hours per reward configuration (about 36 minutes per (condition, seed) cell at this batch size), and the experimental design of this report requires two such sweeps (one for the sprint-retune reward of §3.1.4, one for the energy-regularised reward of §4.2). Where the 3000-iteration budget visibly truncates a learning curve before it plateaus, this is flagged in §4.1 and §4.3.

#### 3.1.2 Velocity command space

The robot is given a command vector $(v_x^{\mathrm{cmd}}, v_y^{\mathrm{cmd}}, \omega_z^{\mathrm{cmd}})$. The lateral and yaw components are held at zero in both training and evaluation:

$$
v_y^{\mathrm{cmd}} = 0, \qquad \omega_z^{\mathrm{cmd}} = 0.
$$

The forward command is discretised into eight contiguous bins of width $0.5$ m/s spanning the full velocity envelope:

$$
B_0 = [0.0,\, 0.5),\ B_1 = [0.5,\, 1.0),\ \ldots,\ B_6 = [3.0,\, 3.5),\ B_7 = [3.5,\, 4.0].
$$

Bin 7 is closed on the right so that the upper bound $4.0$ m/s is sampleable. Each bin index $j \in \{0, 1, \ldots, 7\}$ defines a per-bin task in the sense of §2.2: the per-bin tracking reward, per-bin success threshold, and per-bin learning-progress signal used by the two curricula are all computed over the eight bins defined here. The command is resampled every 20 simulated seconds, and the fraction of environments holding a standing-still command is set to zero (the upstream default of 10 percent stand-still environments was removed to keep all training samples on the discrete bin grid).

#### 3.1.3 Action interface and observation

The policy emits a 12-dimensional joint position action $a_t$. Joint targets are computed as $q^{\mathrm{target}} = q^{\mathrm{default}} + s \cdot a_t$, with action scale $s$. The upstream scale of $0.25$ is retained for every joint except that the scalar $s$ itself is raised from the upstream value:

$$
s : \quad 0.25 \;\longrightarrow\; 0.35.
$$

The upstream scale of $0.25$ is too small for the leg-swing amplitude that a $4$ m/s sprint requires; raising it to $0.35$ gives the policy enough joint-angle range per step to produce a sprint-speed stride. PD gains, torque envelope, friction defaults, and action clip are all inherited from the upstream Unitree Go2 configuration without change.

The policy observation is six terms (base angular velocity, projected gravity, velocity command, joint position relative to default, joint velocity relative to default, last action), each scaled and noised exactly as in the upstream configuration. The critic observation adds two privileged terms (base linear velocity, joint effort). All observations are clipped to $\pm 100$. The domain-randomisation block (friction, restitution, base mass perturbation, reset pose, push events) is inherited from the upstream Go2 configuration unchanged.

#### 3.1.4 Reward configuration and the standing-still derivation

The 16-term reward of §2.3.2 is the starting point. Five of the 16 weights are modified for this work; the remaining eleven are inherited from upstream without change. The motivation for the five modifications is a structural failure of the upstream weighting under the stretched command range $[0, 4]$ m/s, derived below.

**The standing-still local optimum.** Recall from §2.3.2 that the per-step reward decomposes as

$$
r_t = w_{\mathrm{lin}}\, r_{\mathrm{lin}} + w_{\mathrm{ang}}\, r_{\mathrm{ang}} + r_{\mathrm{smooth}} + r_{\mathrm{rest}},
$$

with $w_{\mathrm{lin}} = 1.5$, $w_{\mathrm{ang}} = 0.75$, and $r_{\mathrm{smooth}} \le 0$ summing the four smoothness and effort penalties. The two tracking kernels each saturate at $1$, so the maximum tracking contribution is $w_{\mathrm{lin}} + w_{\mathrm{ang}} = 2.25$.

The smoothness magnitude $|r_{\mathrm{smooth}}|$ grows roughly with the square of the commanded velocity: faster commands force faster joint motion, larger torques, and larger step-to-step action changes. Under the upstream weights, $|r_{\mathrm{smooth}}|$ at sprint commands near $4$ m/s exceeds the tracking ceiling of $2.25$. The consequence is a binary comparison at sprint commands:

- *If the policy actually runs at the commanded speed.* The two tracking terms saturate at $r_{\mathrm{lin}} + r_{\mathrm{ang}} \approx 2.25$. The smoothness term satisfies $|r_{\mathrm{smooth}}| > 2.25$. Therefore $r_t < 0$.
- *If the policy stands still.* Joint velocities, accelerations, torques, and action rates are near zero, so $r_{\mathrm{smooth}} \approx 0$. The yaw command is zero (above), so $r_{\mathrm{ang}}$ saturates at $1$ even when the body is stationary. Therefore $r_t \approx w_{\mathrm{ang}} \cdot 1 = 0.75 > 0$.

The standing-still solution is preferred by the policy gradient over the sprint solution, and a uniform initial sampling over $[0, 4]$ m/s converges to a policy that stands still under every command. The fix is to reduce the magnitudes of the four smoothness weights so that the smoothness penalty at the commanded sprint speed no longer exceeds the tracking ceiling.

**Sprint-retune weights.** The four smoothness weights are reduced one order of magnitude, and the feet-air-time threshold is reduced from $0.5$ s to $0.1$ s. The fifth modification (feet-air-time threshold) is included here because the upstream threshold of $0.5$ s rewards only foot swings longer than half a second, which is incompatible with the higher stride frequency that running near $4$ m/s requires.

| Reward term | Upstream weight or threshold | Sprint-retune value | What this term penalises |
|---|---:|---:|---|
| Joint velocity squared | $-1 \times 10^{-3}$ | $-1 \times 10^{-4}$ | $\|\dot{q}\|^2$ |
| Joint acceleration squared | $-2.5 \times 10^{-7}$ | $-1 \times 10^{-7}$ | $\|\ddot{q}\|^2$ |
| Joint torque squared | $-2 \times 10^{-4}$ | $-2 \times 10^{-5}$ | $\|\tau\|^2$ |
| Action rate squared | $-0.1$ | $-0.005$ | $\|a_t - a_{t-1}\|^2$ |
| Feet-air-time threshold | $0.5$ s | $0.1$ s | minimum swing duration for the air-time bonus |

*Table 3.1: The five reward modifications relative to the upstream Go2 configuration. The remaining eleven of the 16 terms (linear and angular velocity tracking, orientation, height stability, contact bonuses other than feet-air-time, and termination penalties) are kept at upstream weights.*

The combined effect of the four weight reductions is that $|r_{\mathrm{smooth}}|$ at sprint commands is brought back below the tracking ceiling of $2.25$, so that an actually-running policy receives a positive total reward and the standing-still solution is no longer the gradient-preferred local optimum. Every experiment reported in Chapter 4 is run on this modified reward configuration.

### 3.2 The three curriculum conditions

All three conditions share §3.1 exactly. They differ only in how the sampling distribution $c_j(\zeta)$ over the eight bins evolves during training. A shared heartbeat is used: every 48 environment steps (one *tick*, equal to two PPO iterations at rollout length 24), the per-bin tracking reward buffers are flushed to the active curriculum operator and the sampling distribution for the next tick is recomputed.

#### 3.2.1 Uniform (baseline)

The sampling distribution is fixed at the uniform distribution over all eight bins for the full 3000 iterations:

$$
c_j(\zeta) = \frac{1}{N} = \frac{1}{8}, \quad j = 0, 1, \ldots, 7.
$$

No mastery signal is computed and no support expansion or reallocation occurs. This is the no-curriculum baseline.

#### 3.2.2 Box Adaptive (task-specific)

The Box Adaptive operator of §2.2.1 is initialised with active support $\mathcal{A}_0 = \{0\}$, the first bin only. On every tick, the smoothed mean tracking reward of every active bin is compared against the success threshold

$$
\gamma = 0.7.
$$

When the active bin $b$ at the top of the support crosses $\gamma$, the support is grown to $\mathcal{A} \cup \{b-1, b, b+1\}$ following eqs. (10a) and (10b) of Margolis et al. (2022). Sampling within $\mathcal{A}$ is uniform. To prevent the threshold-crossing rule from triggering on a single lucky episode, the mastery check is gated by a minimum of 50 episodes collected in the candidate bin before the per-bin mean is consulted. This min-episode gate is an implementation detail of the operator and does not appear in the Margolis paper.

#### 3.2.3 LP-ACRL (teacher-guided)

The LP-ACRL operator of §2.2.2 buffers per-bin reward signals on every tick but recomputes the sampling distribution only every $M = 50$ ticks. With $M = 50$ ticks of 48 env steps each at rollout length 24, one re-weight cycle equals exactly 100 PPO iterations, matching the stage length used by Li et al. (2026). At each re-weight, the per-bin learning progress $\mathrm{LP}_j$ is computed from the difference between the current and the previous stage's per-bin reward, the softmax weights are formed from $\mathrm{LP}_j$ at temperature

$$
\beta = 0.05,
$$

and a uniform-mixture floor is applied at

$$
\varepsilon = 0.15
$$

so that the final sampling distribution is $c_j = (1 - \varepsilon)\, \mathrm{softmax}_j(\mathrm{LP}/\beta) + \varepsilon / N$. With $N = 8$, the floor sets the minimum per-bin probability at $\varepsilon / N \approx 1.9\%$.

### 3.3 Evaluation protocol

After training, every policy is evaluated under deterministic action selection (mean of the policy Gaussian, no noise) on each of the eight bins independently. For each bin the policy is rolled out for a fixed number of episodes. The unit of comparison is the per-bin pair (condition, bin). With 8 bins, 3 conditions, and 3 seeds, the sweep produces 72 per-bin evaluation cells in total. The following scalar quantities are recorded from each rollout.

- **Per-bin tracking reward.** The episode-averaged value of $r_{\mathrm{lin}}$ in bin $j$, denoted $R_j$. Reported as the mean and standard deviation across the three seeds.
- **EPTE-SP (Episodic Percentage Tracking Error with Stability Penalty).** Defined by Li et al. (2026, eq. 8). For a single rollout of length $T$ with commanded velocity $v^{\mathrm{cmd}}$ and measured forward velocity $v_x(t)$, the episodic tracking error percentage is the time-averaged absolute relative error, and the stability penalty adds a fixed penalty for any episode that terminates by falling. EPTE-SP is reported per bin (mean across seeds and across episodes).
- **Sampling heatmap.** For each curriculum condition, the matrix $c_j(\mathrm{iter})$ giving the per-bin sampling probability at every PPO iteration is logged during training. The heatmap is the visual signature of how each operator allocates compute.
- **Iterations-to-mastery.** For each (condition, bin) cell, the first PPO iteration at which the smoothed per-bin $R_j$ crosses $\gamma = 0.7$. Bins that never cross the threshold within the 3000-iteration budget are recorded as "not mastered." This is the metric on which Box Adaptive and LP-ACRL are scored against the uniform baseline at sprint commands.

### 3.4 Hyperparameter summary

Table 3.2 collects every hyperparameter whose value matters for reproducibility. Values inherited from the upstream Go2 configuration without change are marked "upstream." Values that differ from upstream are listed explicitly.

| Group | Hyperparameter | Value |
|---|---|---:|
| Simulator | Parallel environments | $4096$ |
| Simulator | Rollout length $T$ (env steps per PPO update) | $24$ |
| Simulator | Total PPO iterations | $3000$ |
| Simulator | Seeds per condition | $3$ |
| Command | Bins $N$, bin width $\Delta v$, $V_{\max}$ | $8$, $0.5$ m/s, $4.0$ m/s |
| Command | Lateral and yaw command | $v_y^{\mathrm{cmd}} = 0$, $\omega_z^{\mathrm{cmd}} = 0$ |
| Command | Resampling interval | $20$ s |
| Command | Stand-still environment fraction | $0$ |
| Action | Joint position action scale $s$ | $0.35$ (upstream $0.25$) |
| Action | PD gains, torque envelope, action clip | upstream |
| Reward | $w_{\mathrm{lin}}$, $\sigma_{\mathrm{lin}}$ | $1.5$, $0.5$ m/s |
| Reward | $w_{\mathrm{ang}}$, $\sigma_{\mathrm{ang}}$ | $0.75$, $0.25$ rad/s |
| Reward | $w_{\dot{q}}$ (joint velocity squared) | $-1 \times 10^{-4}$ (upstream $-1 \times 10^{-3}$) |
| Reward | $w_{\ddot{q}}$ (joint acceleration squared) | $-1 \times 10^{-7}$ (upstream $-2.5 \times 10^{-7}$) |
| Reward | $w_{\tau}$ (joint torque squared) | $-2 \times 10^{-5}$ (upstream $-2 \times 10^{-4}$) |
| Reward | $w_{\Delta a}$ (action rate squared) | $-0.005$ (upstream $-0.1$) |
| Reward | Feet-air-time threshold | $0.1$ s (upstream $0.5$ s) |
| Reward | Other 11 of 16 weights | upstream |
| PPO | Clip $\varepsilon_{\mathrm{PPO}}$, learning rate, GAE $\lambda$, discount $\gamma_{\mathrm{disc}}$, epochs, mini-batches | upstream |
| Curriculum (shared) | Tick (env steps between operator updates) | $48$ |
| Curriculum (task-specific) | Success threshold $\gamma$ | $0.7$ |
| Curriculum (task-specific) | Seed bin | bin $0$ |
| Curriculum (task-specific) | Minimum episodes before mastery check | $50$ |
| Curriculum (teacher-guided) | Temperature $\beta$ | $0.05$ |
| Curriculum (teacher-guided) | Stage length $M$ (ticks per re-weight) | $50$ |
| Curriculum (teacher-guided) | Uniform mixture floor $\varepsilon$ | $0.15$ |

*Table 3.2: Consolidated hyperparameter summary. The five reward modifications, the action scale, and all curriculum operator knobs are listed explicitly; everything else not listed inherits the upstream Unitree Go2 configuration.*

---

## 4. Results

Chapter 4 reports the matched comparison in two halves. §4.1 reports the three-condition comparison under the sprint-retune reward of §3.1.4. §4.2 introduces the energy-regularised reward of Liang et al. (2024) and the calibration procedure used to set its scale parameters on the Go2. §4.3 repeats the matched comparison of §4.1 under the energy-regularised reward.

### 4.1 Three curricula under the sprint-retune reward

A three-condition (uniform, task-specific, teacher-guided) sweep with three independent seeds per condition was run to $3000$ PPO iterations using exactly the configuration of §3.1. Every policy was then evaluated under the deterministic protocol of §3.3, with $100$ rollouts of $1000$ steps per (condition, seed, bin) cell ($300$ rollouts per (condition, bin) cell after aggregating over seeds).

#### 4.1.1 Per-bin tracking-reward convergence

Table 4.1 reports the smoothed per-bin tracking reward $R_j$ at the end of training, averaged across the three seeds. Numbers are taken from the last 200 PPO iterations of each run.

| Bin | $v_x^{\mathrm{cmd}}$ | Uniform | Task-specific | Teacher-guided |
|---:|---:|---:|---:|---:|
| 0 | 0.25 m/s | $0.927 \pm 0.000$ | $0.925 \pm 0.001$ | $0.922 \pm 0.001$ |
| 1 | 0.75 m/s | $0.921 \pm 0.000$ | $0.917 \pm 0.001$ | $0.918 \pm 0.001$ |
| 2 | 1.25 m/s | $0.914 \pm 0.001$ | $0.911 \pm 0.001$ | $0.914 \pm 0.001$ |
| 3 | 1.75 m/s | $0.904 \pm 0.001$ | $0.901 \pm 0.001$ | $0.904 \pm 0.001$ |
| 4 | 2.25 m/s | $0.892 \pm 0.001$ | $0.888 \pm 0.001$ | $0.890 \pm 0.001$ |
| 5 | 2.75 m/s | $0.820 \pm 0.007$ | $0.864 \pm 0.001$ | $0.868 \pm 0.002$ |
| 6 | 3.25 m/s | $0.058 \pm 0.010$ | $0.827 \pm 0.002$ | $0.830 \pm 0.003$ |
| 7 | 3.75 m/s | $0.000 \pm 0.000$ | $0.703 \pm 0.003$ | $0.698 \pm 0.010$ |

*Table 4.1: Per-bin tracking reward $R_j$ at end of training, sprint-retune reward of §3.1.4, 3000 PPO iterations per cell. Each entry is the mean $\pm$ standard deviation over the last 200 PPO iterations of the curve obtained by averaging the three seeds.*

For bins 0 through 4 with commands from $0$ to $2.5$ m/s, every condition saturates the per-bin tracking kernel at the same plateau in the range $0.89$ to $0.93$ and there is no separation between the three rules. The plateau is bin-monotone decreasing within each condition. At higher commands the same absolute velocity error translates into a smaller value of $\exp(-|\Delta v|^2/\sigma_{\mathrm{lin}}^2)$, so the ceiling falls from about $0.93$ at bin 0 to about $0.89$ at bin 4 even when the policy tracks the command well.

At bin 5 with commands from $2.5$ to $3.0$ m/s the three rules begin to separate. Uniform sits at $0.82$ while task-specific and teacher-guided sit at $0.87$.

At bins 6 and 7 with commands from $3.0$ to $4.0$ m/s the separation becomes large, and it splits the three conditions into *two* groups rather than three. Task-specific and teacher-guided both clear the mastery threshold $\gamma = 0.7$ at bins 6 and 7, reaching $0.83 / 0.70$ for task-specific and $0.83 / 0.70$ for teacher-guided. Uniform collapses entirely, reaching $0.06$ at bin 6 and $0.00$ at bin 7. Figure 4.1 plots the per-bin learning curves under the sprint-retune reward. Uniform stalls in the first quarter of training at bins 6 and 7 and never recovers. Task-specific and teacher-guided both continue climbing through iteration 3000 at those bins, with no large per-bin gap between the two operators.

![Per-bin tracking-reward learning curves, sprint-retune reward, three seeds per condition.](../src/results_phase1_rerun/figures/learning_curves_rlin.png)

*Figure 4.1: Per-bin tracking reward $R_j$ over training, sprint-retune reward. Bands are $\pm 1$ std across three seeds. Bottom-row bins are the ones that separate uniform from the two curriculum-driven conditions.*

The 3000-iteration budget is short relative to standard practice in this literature. Margolis et al. (2022) trained at the equivalent of about 12000 iterations on this scale of robot. The slope-at-end of every cell in Table 4.1 is at most $10^{-5}$ per iteration, including the bin-6 and bin-7 cells of uniform, so the policies have stopped learning at this budget rather than merely being interrupted partway through. The bin-6 and bin-7 failure of uniform is not a standing-still local optimum. The policy never learned how to take a stride at sprint speed, so it cannot step forward when given the command and loses balance instead (cf. §4.1.3). Training and deterministic evaluation use the same environment and reward and differ only in whether the policy's action noise is on. The failure mode is the same under both protocols.

The reason uniform fails at the sprint bins under the §3.1.4 reward is the design of the reward itself. The sprint-retune reward contains tracking, smoothness, and feet-air-time terms, but no contact-coordination term, no flight-phase term, and no power term. The policy therefore has to discover a stable sprint gait through exploration alone, and stable sprint near $4$ m/s requires a stride frequency and contact pattern that cannot be reached by simply scaling up a walking gait. Discovering it is sample-hungry. Under uniform sampling each of the eight bins gets $1/8$ of the compute, which is not enough attempts at the sprint bins to stumble on a successful trajectory within $3000$ iterations. Under task-specific sampling, the sprint bins are added to the support only after the lower bins cross $\gamma$, and once added are sampled at the same probability as every other active bin. In this sweep that proves sufficient. By the time bins 6 and 7 unlock there are still on the order of $1500$ iterations left, which is enough to bring the per-bin reward across the mastery threshold. Teacher-guided sampling, with the same reward, also clears bins 6 and 7 within budget, with a similar per-bin reward to task-specific but a different time profile in which the operator concentrates samples on bins where learning progress is detectable. The conclusion at this stage is that *both* curricula succeed under the §3.1.4 reward while uniform does not. §4.2 patches the reward directly by adding the energy-regularised term of Liang et al. (2024), and §4.3 re-runs the comparison under the patched reward to separate the operator effect from the reward effect.

The iterations-to-mastery metric (first PPO iteration at which the seed-averaged $R_j$ crosses $\gamma = 0.7$) is summarised in Table 4.2.

| Bin | $v_x^{\mathrm{cmd}}$ | Uniform | Task-specific | Teacher-guided |
|---:|---:|---:|---:|---:|
| 0 | 0.25 | 30 | 20 | 22 |
| 1 | 0.75 | 214 | 134 | 196 |
| 2 | 1.25 | 280 | 186 | 286 |
| 3 | 1.75 | 340 | 232 | 370 |
| 4 | 2.25 | 534 | 284 | 466 |
| 5 | 2.75 | 984 | 334 | 576 |
| 6 | 3.25 | not mastered (0/3) | 510 (3/3) | 836 (3/3) |
| 7 | 3.75 | not mastered (0/3) | 1100 (3/3) | 1742 (3/3) |

*Table 4.2: Iterations-to-mastery per (condition, bin) cell, computed as the first PPO iteration at which the seed-averaged $R_j$ crosses $\gamma = 0.7$. The fraction of seeds that reached mastery individually is shown in parentheses where it is below 3/3.*

Bins 0 through 5 are mastered by every condition. Task-specific reaches every one of these bins faster than uniform, because once a bin's mastery threshold is crossed, the support has already expanded to include it and the operator focuses sampling on the new edge. Teacher-guided is comparable to uniform on bins 1 through 4 because the LP-ACRL temperature keeps probability mass spread across all bins as long as multiple bins still show positive learning progress, and it pulls ahead on bins 5, 6, and 7 once the lower bins plateau. Task-specific reaches bins 6 and 7 earlier than teacher-guided, at $510$ versus $836$ at bin 6 and $1100$ versus $1742$ at bin 7, but both operators master both sprint bins within the budget. Uniform masters neither.

![Iterations-to-mastery, sprint-retune reward. Bars truncated at 3000 indicate the bin was not mastered within the budget.](../src/results_phase1_rerun/figures/iterations_to_mastery.png)

*Figure 4.2: Iterations-to-mastery per (condition, bin), sprint-retune reward. Uniform's bars at bins 6 and 7 are missing because no seed crossed $\gamma$ within the budget.*

Figure 4.3 shows the per-bin sampling distribution $c_j(\mathrm{iter})$ as a heatmap. The three rows make the operator behaviour explicit. Uniform is a flat $1/8$ across the eight bins for the full 3000 iterations. Task-specific expands monotonically from bin 0 outward, with every newly-unlocked bin receiving the same probability as the bins below it. Teacher-guided concentrates its mass on the bin with the highest current learning progress, which moves upward across the velocity range as the policy improves at each bin in turn.

![Sampling heatmap per condition, sprint-retune reward.](../src/results_phase1_rerun/figures/sampling_heatmap.png)

*Figure 4.3: Per-bin sampling probability $c_j(\mathrm{iter})$ over training, sprint-retune reward. Bin index on the vertical axis, PPO iteration on the horizontal axis, sampling probability on the colour scale.*

#### 4.1.2 Per-bin EPTE-SP

Table 4.3 reports the per-bin EPTE-SP from the deterministic evaluation rollouts of §3.3, together with the fraction of rollouts that terminated early by falling. EPTE-SP is bounded above at $1.0$ by construction. An EPTE-SP of $1.0$ together with a $100\%$ fall fraction indicates that the deterministic policy is not able to take a single survivable step under that command.

| Bin | $v_x^{\mathrm{cmd}}$ | Uniform EPTE-SP / fall% | Task-specific EPTE-SP / fall% | Teacher EPTE-SP / fall% |
|---:|---:|---:|---:|---:|
| 0 | 0.25 | $0.424$ / $0\%$ | $0.421$ / $1\%$ | $0.368$ / $2\%$ |
| 1 | 0.75 | $0.117$ / $0\%$ | $0.103$ / $0\%$ | $0.082$ / $0\%$ |
| 2 | 1.25 | $0.078$ / $0\%$ | $0.074$ / $0\%$ | $0.052$ / $0\%$ |
| 3 | 1.75 | $0.063$ / $0\%$ | $0.062$ / $0\%$ | $0.052$ / $0\%$ |
| 4 | 2.25 | $0.069$ / $0\%$ | $0.070$ / $0\%$ | $0.064$ / $0\%$ |
| 5 | 2.75 | $0.139$ / $9\%$ | $0.067$ / $0\%$ | $0.066$ / $0\%$ |
| 6 | 3.25 | $0.982$ / $98\%$ | $0.537$ / $50\%$ | $0.187$ / $13\%$ |
| 7 | 3.75 | $1.000$ / $100\%$ | $0.536$ / $50\%$ | $0.228$ / $17\%$ |

*Table 4.3: Per-bin EPTE-SP and fall fraction at deterministic evaluation, averaged across three seeds and 300 rollouts per cell, sprint-retune reward.*

The story across bins 0 through 5 is consistent with Table 4.1. All three conditions track the command with single-digit-percent error and essentially no falls. The small $9\%$ fall fraction for uniform at bin 5 reflects the policy already wobbling at $2.5$ to $3.0$ m/s.

**EPTE-SP formula.** For a rollout of length $T$ steps with commanded velocity $v^{\mathrm{cmd}}$ that terminates at step $T' \leq T$ (either by reaching the end without falling, $T' = T$, or by falling at $T' < T$), EPTE-SP is defined as

$$\mathrm{EPTE\text{-}SP} = \frac{1}{T} \left[ \sum_{t=1}^{T'} \min\!\left(\frac{|v_x(t) - v^{\mathrm{cmd}}|}{v^{\mathrm{cmd}}},\, 1\right) + (T - T') \cdot 1 \right].$$

The first sum is the time-averaged tracking-error percentage over the survived steps, each clipped to at most $1$. The second term fills the remaining $T - T'$ steps after a fall with a penalty of $1$ per step. By construction, every term inside the brackets is in $[0, 1]$, so EPTE-SP $\in [0, 1]$.

Two limiting cases connect this formula to the extremes of Table 4.3:

- **Perfect tracking, no fall** ($T' = T$, $v_x(t) = v^{\mathrm{cmd}}$ throughout): all per-step errors are $0$, so $\mathrm{EPTE\text{-}SP} = 0$.
- **Immediate fall** ($T' \approx 0$): essentially all $T$ steps are penalised, so $\mathrm{EPTE\text{-}SP} \to 1.0$. This is exactly the uniform condition at bin 7 (fall fraction $100\%$, EPTE-SP $= 1.000$).

The $50\%$ fall fraction of task-specific at bins 6 and 7 puts it between these extremes: surviving episodes contribute their tracking-error term, falling episodes contribute close to $1$, and the average of the two groups lands near $0.54$.

**Why bin 0 reads $\sim 0.4$.** The normalisation $1/v^{\mathrm{cmd}}$ in the tracking term means the same absolute velocity error is amplified more at slow commands:

$$\frac{|\Delta v_x|}{v^{\mathrm{cmd}}} \bigg|_{b_0,\, v^{\mathrm{cmd}}=0.25} = \frac{|\Delta v_x|}{0.25} \;\gg\; \frac{|\Delta v_x|}{v^{\mathrm{cmd}}} \bigg|_{b_3,\, v^{\mathrm{cmd}}=1.75} = \frac{|\Delta v_x|}{1.75}.$$

With the mean measured forward velocity at bin 0 between $0.18$ and $0.19$ m/s against a $0.25$ m/s command, the mean absolute error is roughly $0.06$–$0.07$ m/s. At the bin-0 command, this normalises to $0.06/0.25 = 24\%$; the step-to-step variance in $v_x(t)$ pushes the time-average above the static estimate, landing near $0.4$. At bin 3, the same $0.06$ m/s error normalises to $0.06/1.75 = 3.4\%$, consistent with the single-digit EPTE-SP in Table 4.3. This is an artifact of the normalisation rather than a control failure.

The collapse at bins 6 and 7 mirrors the training-time tracking-reward separation but is *finer-grained* than it. Under deterministic evaluation, task-specific is intermediate rather than lumped with uniform. Uniform falls $98\%$ at bin 6 and $100\%$ at bin 7. Task-specific falls $50\%$ at both sprint bins, and teacher-guided falls $13\%$ at bin 6 and $17\%$ at bin 7. EPTE-SP mirrors the same ranking. Two factors separate task-specific from teacher-guided here despite their similar training-time $R_j$ on bins 6 and 7. First, teacher-guided invested more total samples on each sprint bin once it reached them, so the underlying policy is more robust. Second, under deterministic evaluation the action noise is removed, and a policy that succeeded stochastically by sometimes-lucky strides regresses more than one that converged tightly.

Figure 4.4 plots the per-bin mean forward velocity at evaluation against the command. At bins 6 and 7, uniform stops at $0.21$ m/s and $0.08$ m/s against commands of $3.25$ and $3.75$ m/s, which is the same failure pattern described above. The policy never learned how to step forward at sprint speed, so it cannot move at the commanded velocity and loses balance instead. Task-specific reaches $1.73$ m/s and $1.81$ m/s in the same bins, roughly half the commanded sprint speed, which is a real running gait that survives half the deterministic rollouts. Teacher-guided reaches $2.85$ m/s and $2.97$ m/s in the same bins, closer to but still below the commanded velocity, and falls in only about $15\%$ of rollouts. The EPTE-SP rank at bins 6 and 7 in which uniform is much larger than task-specific which is in turn larger than teacher-guided is therefore driven by both factors above, namely how fast the policy tracks the command and how often it survives the rollout.

![Measured vs commanded forward velocity per bin, sprint-retune reward.](../src/results_phase1_rerun/figures/v_actual_vs_cmd.png)

*Figure 4.4: Mean measured forward velocity vs commanded velocity, sprint-retune reward. Diagonal is the perfect-tracking reference. Uniform's points at bins 6 and 7 sit near $\bar v_x = 0$; task-specific reaches roughly half-speed at the sprint bins; teacher-guided is the closest to the diagonal at bins 6 and 7.*

![EPTE-SP per (condition, bin), sprint-retune reward.](../src/results_phase1_rerun/figures/epte_bars.png)

*Figure 4.5: EPTE-SP per (condition, bin), sprint-retune reward. The saturated bar at bin 7 of uniform corresponds to the $100\%$-fall column of Table 4.3; task-specific's bin-6 and bin-7 bars at $\sim 0.54$ reflect its $50\%$ fall fraction at the sprint bins.*

#### 4.1.3 Qualitative behaviour at evaluation

The visual signature of the sprint-retune reward depends on which bin is being evaluated. Two qualitatively different failure modes appear at the extremes of the velocity range and are visible in Figure 4.6.

![Per-foot stance/swing strips at deterministic evaluation, sprint-retune reward.](../src/results_phase1_rerun/figures/gait_diagram.png)

*Figure 4.6: Gait diagram at deterministic evaluation, sprint-retune reward. Columns are conditions (uniform / task-specific / teacher-guided), rows are velocity bins. Coloured blocks mark stance (foot in contact); white gaps mark swing. Per cell the plotted rollout is the best-surviving rollout across the three seeds, with ties broken by closest mean forward velocity to the command.*

**Low bins. Irregular contact at slow command speed.** At a commanded velocity of $0.25$ m/s the Phase 1 gait classifier labels all three conditions as *Irregular* in Figure 4.6. The per-foot contact pattern does not match any canonical walk template, and the duty factor and stride frequency reported in §4.3.3 for the same bin under the energy-regularised reward are absent here because no consistent stride cycle is being produced. The policy nevertheless meets the slow commanded forward velocity at about $0.19$ m/s against the $0.25$ m/s command, and the chassis stays upright with a fall fraction below $2\%$ across the three conditions in Table 4.3. The qualitative outcome is that the policy never learned a real walking gait at the slow command. The command is slow enough that any non-falling foot motion satisfies the tracking term $r_{\mathrm{lin}}$, and the §3.1.4 reward contains no term that would penalise the absence of a canonical gait pattern, so the policy settles in an irregular contact pattern that scores well without learning to walk properly.

**High bins. Failure to learn under uniform.** At $v_x^{\mathrm{cmd}} \ge 3.0$ m/s under uniform, Figure 4.6 shows the rollout terminating early. The bin-6 and bin-7 cells in the uniform column are short stance/swing fragments that end before a full stride completes, and the gait classifier reports "no valid window" at bin 6 and *n/a* at bin 7. The accompanying video shows the same outcome. The policy never learned how to step forward at sprint speed, so it cannot move at the commanded velocity and loses balance before the rollout ends (cf. the $98$–$100\%$ fall column of Table 4.3). This is not a reward hack. It is a learning failure within the $3000$-iteration budget at these bins. The task-specific and teacher-guided columns at the same bins show all four feet cycling through stance and swing for the full evaluation window (duty $\beta \approx 0.52$–$0.54$, diagonal pair near-synchronous), so the high-bin failure is specific to the operator that under-samples the sprint bins, not to the reward.

**Why the §3.1.4 reward allows the low-bin failure.** The tracking term $r_{\mathrm{lin}}$ scores forward velocity indifferent to how that velocity is generated. The feet-air-time bonus with threshold $0.1$ s in Table 3.2 is the only term in §3.1.4 designed to shape the contact pattern, and it scores each foot independently. Nothing in the reward depends on the coordination *between* feet. An irregular four-foot contact pattern therefore scores just as well as a clean canonical walk as long as the body drifts forward at the slow commanded speed, and the policy lands on whichever contact pattern is gradient-cheapest to reach. The same absence of contact-coordination is the reason uniform never escapes the high-bin failure either. The reward gives no gradient signal for *which* contact structure to use at sprint, so the only way for the operator to find a sprint gait that actually steps forward is to stumble on it by exploration, and uniform does not do this within $3000$ iterations at the $1/8$ sampling share each sprint bin receives.

**Deployment implication.** The project goal is a single PPO policy that tracks forward-velocity commands across $[0,\,4]$ m/s. The sprint-retune reward partially fails this goal in two different ways. Every condition produces an irregular contact pattern at the low bin, and uniform cannot step forward at the sprint bins at all. Task-specific and teacher-guided both reach the sprint bins under this reward, but they still produce the low-bin irregularity, so the curriculum alone does not rescue the low-bin failure. A different reward shape is needed. §4.2 introduces one in which the contact coordination between feet enters the reward through an energy-cost-of-transport term.

### 4.2 From sprint-retune to energy regularisation

The reward term added to fix the §4.1.3 observation has to depend on the relative timing of foot contacts. Liang et al. (2024) report that a single energy-regularised term, added to a tracking objective, causes the policy to autonomously transition between walking, trotting, and fly-trotting as the command rises, without any contact-phase reference being supplied during training. The term depends on contact timing indirectly through its cost-of-transport interpretation. It penalises mechanical power normalised by command magnitude. The four-foot bound carries a higher power cost per unit of forward motion than an alternating gait does, so under the energy term the alternating pattern scores higher. This project adopts Liang's approach because it requires no hand-coded gait template and because the existing §3.1.4 reward already supplies the tracking block that the Liang reward composes with. Liang's claim is empirical, so the same gait family has to be verified on the Go2. That verification is §4.3.4.

The remainder of this section reproduces the Liang reward in its source notation (§4.2.1) and explains the procedure used to fix its scale parameters on the Go2 (§4.2.2). The matched three-condition comparison under the calibrated reward is reported in §4.3.

#### 4.2.1 The Liang energy-regularised reward

Liang et al. (2024) consider a single PPO policy on the Unitree Go1 quadruped tracking forward-velocity and yaw-rate commands. The total per-step reward is written as a multiplicative composition of three blocks,

$$
R = (R_{\mathrm{motion}} + R_{\mathrm{energy}}) \cdot f(R_{\mathrm{aux}}),
\qquad (1)
$$

where $R_{\mathrm{motion}}$ scores command tracking, $R_{\mathrm{energy}}$ scores energetic efficiency, and $f(R_{\mathrm{aux}})$ is a positive multiplicative wrapping of an auxiliary term $R_{\mathrm{aux}}$ that aggregates the smoothness and safety penalties.

The specific choice of $f$ in the paper is the exponential, so that the auxiliary block acts as a unit-bounded multiplier on the motion-plus-energy sum,

$$
R = \big( R_{\mathrm{motion}} + \alpha_{\mathrm{en}}\, R_{\mathrm{en}}(v_x, \omega_z) \big) \cdot \exp\!\big( - R_{\mathrm{aux}} \big),
\qquad (2)
$$

where $\alpha_{\mathrm{en}} \ge 0$ is the scalar weight on the energy block.

The motion block decomposes into a linear-velocity term and an angular-velocity term,

$$
R_{\mathrm{motion}} = R_{\mathrm{lin}} + \alpha_{\mathrm{ang}}\, R_{\mathrm{ang}},
$$

$$
R_{\mathrm{lin}} = \exp\!\left( - \frac{ |v_x - \hat v_x|^2 + |v_y - \hat v_y|^2 }{\sigma_v} \right),
$$

$$
R_{\mathrm{ang}} = \exp\!\left( - \frac{ |\omega_z - \hat\omega_z|^2 }{\sigma_\omega} \right),
\qquad (3)
$$

where $\hat v_x, \hat v_y$ are the commanded linear-velocity components, $\hat\omega_z$ is the commanded yaw rate, $v_x, v_y, \omega_z$ are the measured values, and $\sigma_v, \sigma_\omega$ are scale parameters that control the width of the tracking kernel.

The energy block is the new ingredient. Liang et al. (2024) define it as

$$
R_{\mathrm{en}} = \exp\!\left( - \frac{ \sum_i \, |\tau_i| \, |\dot q_i| }{ \sigma_{\mathrm{en},x}\, |v_x| + \sigma_{\mathrm{en},z}\, |\omega_z| } \right),
\qquad (4)
$$

where $\tau_i$ is the applied joint torque on actuator $i$, $\dot q_i$ is the corresponding joint velocity, the sum runs over all actuated joints, and $\sigma_{\mathrm{en},x},\, \sigma_{\mathrm{en},z}$ are scale parameters that set the cost-of-transport scale on linear and angular motion respectively.

Three design choices in (4) are non-obvious and they each have a behavioural consequence.

- **Absolute values inside the sum.** The numerator uses $|\tau_i|\,|\dot q_i|$ rather than the signed product $\tau_i \dot q_i$. The signed product would be the net mechanical power flowing out of the actuator, which can be negative when the joint is braking. The absolute-value form charges the policy for braking energy as if it were positive energy expenditure, on the grounds that the physical motor does not regenerate that energy back into the battery. This is what the paper means by treating $\sum_i |\tau_i||\dot q_i|$ as a proxy for instantaneous power draw rather than as net mechanical work.
- **Denominator as a command-magnitude scale.** The denominator $\sigma_{\mathrm{en},x}|v_x| + \sigma_{\mathrm{en},z}|\omega_z|$ is proportional to the measured speed of motion. The ratio inside the exponential is therefore (power) / (speed-scale), which has units of force times a per-axis scale, i.e. a cost of transport. The implication is that the same power draw is penalised less harshly when the robot is moving fast than when it is moving slow. The reward is shaped around energy *per unit of useful motion*, not energy per unit of time.
- **Exponential wrapping.** The exponential makes $R_{\mathrm{en}} \in (0, 1]$, so it integrates additively into the motion block of (2) without dominating the tracking reward at any one timestep. There is no separate contact-schedule reward. The only thing pushing the policy toward an alternating gait is the cost-of-transport interpretation of the denominator.

The reported defaults for the Go1 are $\sigma_{\mathrm{en},x} = 1000$, $\sigma_{\mathrm{en},z} = 500$, $\alpha_{\mathrm{en}} = 1.0$, and $\sigma_v = \sigma_\omega = 0.25$, on the forward-velocity range $v_x \in [0,\, 2.5]$ m/s. Liang et al. report an ablation in which setting $\alpha_{\mathrm{en}} = 0$ (energy block disabled) yields the bouncing four-foot synchronised gait described above, while $\alpha_{\mathrm{en}} = 1.0$ yields a walking pattern at low command speed, a two-beat trot at moderate command speed, and a fly-trotting pattern with a flight phase at high command speed, with no contact reference supplied at any point during training.

This project's energy-regularised reward configuration uses (4) as written, with $\sigma_{\mathrm{en},x}$, $\sigma_{\mathrm{en},z}$, $\alpha_{\mathrm{en}}$, and $\sigma_v$ fixed by the calibration of §4.2.2 and a small clamping $\max(\cdot,\,\varepsilon)$ added to the denominator of (4) to bound the gradient at very low commanded speed.

#### 4.2.2 Calibrating the energy scale

The reward of (4) has one knob that matters. That knob is $\sigma_{\mathrm{en},x}$, the linear-speed entry in the denominator, with $\sigma_{\mathrm{en},z} = 0.5\,\sigma_{\mathrm{en},x}$ fixed by the ratio Liang et al. (2024) used. The behaviour of $R_{\mathrm{en}} = \exp(-x)$ depends on where this knob places the argument $x$. If $\sigma_{\mathrm{en},x}$ is too small, $x$ is large and $R_{\mathrm{en}}$ collapses near zero. The energy term then swamps the tracking term and the policy stops tracking. If $\sigma_{\mathrm{en},x}$ is too large, $x$ is small and $R_{\mathrm{en}}$ saturates near one. The energy term is then a near-constant offset on the policy gradient and the policy ignores it. Only a middle $\sigma_{\mathrm{en},x}$ gives the policy a usable gradient from $R_{\mathrm{en}}$.

Liang et al. (2024) report $\sigma_{\mathrm{en},x} = 1000$ for the Unitree Go1 on $v_x \in [0,\, 2.5]$ m/s. This project uses a different robot (Go2) and a wider command range ($v_x \in [0,\, 4]$ m/s). The joint-power numerator and the command-magnitude denominator of $R_{\mathrm{en}}$ both depend on the robot and the range, so Liang's value is not guaranteed to fall in the middle regime on the Go2. The right $\sigma_{\mathrm{en},x}$ has to be found by calibration.

The picture to keep in mind is the shape of $R_{\mathrm{en}} = \exp(-x)$ as a function of its argument $x = P / (\sigma_{\mathrm{en},x}|v_x| + \sigma_{\mathrm{en},z}|\omega_z|)$. The exponential is steep only in a narrow middle region. Outside that region it is either pinned near $0$ or pinned near $1$ and barely moves when $x$ changes. Training works only if $\sigma_{\mathrm{en},x}$ places the typical $x$ inside the steep region, so that small changes in the policy translate into changes in $R_{\mathrm{en}}$ that the gradient can act on. Picking $\sigma_{\mathrm{en},x}$ is therefore not a search for the "right" reward value. It is a search for the value that keeps the gradient alive.

The calibration is split into two cheap stages.

**Step 1, no training. Find which $\sigma_{\mathrm{en},x}$ values keep the gradient alive at all.** Run the Go2 in Isaac Lab for a fixed number of steps under a uniformly-random joint-action policy. The robot will twitch in place without making real forward motion. At each step record the joint power proxy $P = \sum_i |\tau_i|\,|\dot q_i|$, the forward speed $|v_x|$, and the yaw rate $|\omega_z|$. Plug each candidate $\sigma_{\mathrm{en},x}$ from a logarithmic grid into the $R_{\mathrm{en}}$ formula, compute the mean of $R_{\mathrm{en}}$ across the rollout, and keep only the $\sigma_{\mathrm{en},x}$ values for which this mean lies in $[0.3,\, 0.75]$. That range is the steep middle region of the exponential. Values of $\sigma_{\mathrm{en},x}$ that produce a mean below it have already saturated $R_{\mathrm{en}}$ near zero, so the energy term would swamp tracking. Values above it have saturated near one, so the energy term acts as a constant offset that the policy ignores.

The reason random actions are used as the test signal is that they are the worst case the policy will ever see. Power draw is high and forward motion is near zero, so $x$ is at its largest and $R_{\mathrm{en}}$ at its smallest. A $\sigma_{\mathrm{en},x}$ that lands the random-action mean inside the steep region is guaranteed to keep $R_{\mathrm{en}}$ in or above the steep region for the entire rest of training, because every improvement in the policy reduces $x$ and pushes $R_{\mathrm{en}}$ upward. The energy term gently fades out as the gait improves, which is the desired behaviour.

There are three things Step 1 cannot do. It cannot tell us how the energy block should be weighted against the tracking block (the parameter $\alpha_{\mathrm{en}}$ in equation 2). It cannot tell us how strict the tracking kernel should be (the parameter $\sigma_v$). And it cannot verify that the live PPO gradient actually drives the policy to a walking or trotting gait once training starts, rather than to some hack that satisfies the reward without locomotion. All three gaps are closed by a small training experiment.

**Step 2, short training. Pick the configuration with the best tracking and a meaningful gait differential.** Take three $\sigma_{\mathrm{en},x}$ values from the Step 1 bracket (one near the low edge, one in the middle, one near the high edge) and combine them with three choices of $(\sigma_v,\,\alpha_{\mathrm{en}})$. The result is a $3 \times 3$ grid of nine combinations. For each combination, train a single uniform-sampling policy under the energy-regularised reward for a reduced number of iterations chosen so the entire grid finishes in a manageable wall-clock. After training, run each of the nine policies through the standard §3.3 evaluation on all eight velocity bins and record two scalars per run. The first scalar is the sum of per-bin tracking error across the eight bins. The second is the duty-factor differential $\beta_{b_0} - \beta_{b_4}$ between the slowest and the middle bin. Tracking error measures whether the policy actually reaches the commanded speed at each bin. The duty-factor differential is a coarse but cheap proxy for whether the contact pattern changes with command speed at all. A real walk-to-trot transition is accompanied by the duty factor dropping as the command speed rises, so a non-zero differential is evidence that the policy is using its feet differently at different speeds rather than running one fixed pattern across the entire range. The cell with the lowest summed tracking error, subject to a non-degenerate duty differential, is the one whose $(\sigma_{\mathrm{en},x},\,\sigma_{\mathrm{en},z},\,\sigma_v,\,\alpha_{\mathrm{en}})$ tuple is frozen and reused unchanged for the matched three-condition sweep of §4.3.

In one sentence. Step 1 throws away the $\sigma_{\mathrm{en},x}$ values where the reward gradient is dead before training even starts, and Step 2 trains the survivors and keeps the one with the best tracking error across the eight bins.

**Step 1 result.** Table 4.4 reports the Step 1 random-action diagnostic for seven candidate $\sigma_{\mathrm{en},x}$ values on a logarithmic grid. The mean instantaneous power proxy was $100.2$ W, the mean $|v_x|$ was $0.10$ m/s, and the mean $|\omega_z|$ was $0.35$ rad/s.

| $\sigma_{\mathrm{en},x}$ | mean $R_{\mathrm{en}}$ | std $R_{\mathrm{en}}$ | regime |
|---:|---:|---:|---|
| $50$ | $0.013$ | $0.046$ | too strong (saturated near $0$) |
| $100$ | $0.063$ | $0.097$ | strong |
| $250$ | $0.240$ | $0.191$ | strong |
| $500$ | $0.436$ | $0.223$ | useful gradient |
| $1000$ | $0.627$ | $0.207$ | useful gradient |
| $2000$ | $0.775$ | $0.163$ | weak gradient |
| $5000$ | $0.895$ | $0.099$ | weak gradient |

*Table 4.4: Step 1 random-action diagnostic on the Go2 under the §3 simulation configuration, $1000$ random-action steps per candidate $\sigma_{\mathrm{en},x}$. Values inside the $[0.3,\,0.75]$ band are those for which $R_{\mathrm{en}}$ is in the steep region of its argument; values outside the band give a flat gradient.*

![Step 1 random-action diagnostic of mean $R_{en}$ versus $\sigma_{en,x}$.](../src/results_phase2_rerun/figures/sigma_phase0.png)

*Figure 4.7: Step 1 random-action diagnostic. Mean $R_{\mathrm{en}}$ across the rollout (dark line) and per-environment min/max envelope (grey band) as a function of $\sigma_{\mathrm{en},x}$ on the logarithmic grid of Table 4.4.*

Only $\sigma_{\mathrm{en},x} \in \{500, 1000\}$ fall inside the target band. The value $250$ is just below at $0.240$ and $2000$ is just above at $0.775$. The Step 2 grid is anchored on the in-band pair and reaches into the just-below band to verify that the calibration does not lose tracking quality at the low end. The three $\sigma_{\mathrm{en},x}$ values selected for the grid are $\{500,\, 750,\, 1000\}$, and they are crossed with $(\sigma_v,\,\alpha_{\mathrm{en}}) \in \{(0.5, 1.0),\, (1.0, 1.0),\, (1.0, 0.5)\}$.

**Step 2 result.** Table 4.5 reports the $3 \times 3$ calibration grid evaluated under the deterministic protocol of §3.3. Each grid cell was trained for $1500$ PPO iterations under uniform sampling. The three columns track different aspects of the resulting policy. The first column is the summed per-bin tracking error $\sum_j \mathrm{err}_j$ across all eight bins. The second column is the duty-factor differential between the lowest and the mid bin ($\beta_{b_0} - \beta_{b_4}$), which is a coarse proxy for whether the gait pattern changes with command magnitude. The third column is the wall-clock cost.

| Run | $\sigma_{\mathrm{en},x}$ | $\sigma_v$ | $\alpha_{\mathrm{en}}$ | $\sum_j$ err | $\beta_{b_0} - \beta_{b_4}$ | wall-clock (min) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | $500$ | $0.5$ | $1.0$ | $2.922$ | $0.155$ | $32.4$ |
| 2 | $750$ | $0.5$ | $1.0$ | $2.333$ | $0.088$ | $33.8$ |
| 3 | $1000$ | $0.5$ | $1.0$ | $2.487$ | $0.099$ | $32.9$ |
| 4 | $500$ | $1.0$ | $1.0$ | $1.787$ | $0.105$ | $36.4$ |
| 5 | $750$ | $1.0$ | $1.0$ | $1.750$ | $0.183$ | $34.5$ |
| 6 | $1000$ | $1.0$ | $1.0$ | $1.627$ | $0.143$ | $33.7$ |
| **7** | **$500$** | **$1.0$** | **$0.5$** | **$1.537$** | **$0.143$** | **$37.5$** |
| 8 | $750$ | $1.0$ | $0.5$ | $1.798$ | $0.137$ | $38.6$ |
| 9 | $1000$ | $1.0$ | $0.5$ | $1.585$ | $0.104$ | $32.0$ |

*Table 4.5: Step 2 calibration grid. Each row is a separate uniform-sampling training run at $1500$ PPO iterations under the energy-regularised reward, with the listed $(\sigma_{\mathrm{en},x},\,\sigma_v,\,\alpha_{\mathrm{en}})$ tuple. Tracking-quality column is the sum of per-bin tracking error across all eight bins. The fifth column is the duty-factor differential between the bottom and the middle of the velocity range. Run 7 is highlighted as the selected configuration. $\sigma_{\mathrm{en},z}$ is fixed at $0.5\,\sigma_{\mathrm{en},x}$ throughout.*

![Step 2 calibration grid: per-bin tracking error and duty factor across the nine runs.](../src/results_phase2_rerun/figures/sigma_phase_v4.png)

*Figure 4.8: Step 2 calibration grid. Per-bin tracking error (top) and per-bin duty factor (bottom) for each of the nine runs of Table 4.5. Run 7 (selected configuration) is highlighted.*

Three observations on Table 4.5. (i) Across all nine cells, no run masters bin 7 ($3.5$–$4.0$ m/s) within the $1500$-iteration grid budget. This is consistent with §4.1.1, since the grid uses uniform sampling, and uniform alone does not reach the sprint bin even under the energy-regularised reward (cf. §4.3.1). (ii) The $\sigma_v = 0.5$ rows (Runs 1–3) have summed tracking error in the range $2.3$–$2.9$, well above the $\sigma_v = 1.0$ rows ($1.5$–$1.8$). The wider tracking kernel matters more than the choice of $\sigma_{\mathrm{en},x}$ in this grid, because the narrower kernel collapses to near zero at velocity errors that the wider kernel still gives a usable gradient on. (iii) Within the $\sigma_v = 1.0$ rows, the best summed tracking error is Run 7 ($1.537$), and the best gait differential is Run 5 ($0.183$). Run 7 was chosen over Run 5 because the tracking advantage is larger than the gait-differential gap, and because reducing $\alpha_{\mathrm{en}}$ from $1.0$ to $0.5$ leaves more headroom for the curriculum to be the binding constraint on sprint progress (§4.3).

The chosen tuple is therefore

$$
\sigma_{\mathrm{en},x} = 500, \quad \sigma_{\mathrm{en},z} = 250, \quad \sigma_v = 1.0, \quad \alpha_{\mathrm{en}} = 0.5,
$$

and it is reused unchanged for the matched three-condition sweep of §4.3.

### 4.3 Three curricula under the energy-regularised reward

A second three-condition (uniform, task-specific, teacher-guided) sweep with three independent seeds per condition was run to $3000$ PPO iterations under the energy-regularised reward of §4.2.1, with the scale parameters fixed at the values chosen by the calibration of §4.2.2. The remaining experimental configuration (curriculum operators, evaluation protocol, command range, episode length, PPO hyperparameters) is identical to §4.1, so the two halves of this chapter are directly comparable on a per-cell basis.

#### 4.3.1 Per-bin tracking-reward convergence

Table 4.6 reports the smoothed per-bin tracking reward $R_j$ at the end of training under the energy-regularised reward, averaged across the three seeds.

| Bin | $v_x^{\mathrm{cmd}}$ | Uniform | Task-specific | Teacher-guided |
|---:|---:|---:|---:|---:|
| 0 | 0.25 m/s | $0.976 \pm 0.001$ | $0.975 \pm 0.001$ | $0.975 \pm 0.000$ |
| 1 | 0.75 m/s | $0.971 \pm 0.001$ | $0.970 \pm 0.000$ | $0.970 \pm 0.000$ |
| 2 | 1.25 m/s | $0.958 \pm 0.001$ | $0.960 \pm 0.000$ | $0.957 \pm 0.001$ |
| 3 | 1.75 m/s | $0.946 \pm 0.002$ | $0.947 \pm 0.001$ | $0.945 \pm 0.001$ |
| 4 | 2.25 m/s | $0.934 \pm 0.001$ | $0.935 \pm 0.001$ | $0.935 \pm 0.001$ |
| 5 | 2.75 m/s | $0.911 \pm 0.002$ | $0.917 \pm 0.002$ | $0.917 \pm 0.001$ |
| 6 | 3.25 m/s | $0.811 \pm 0.005$ | $0.867 \pm 0.004$ | $0.859 \pm 0.002$ |
| 7 | 3.75 m/s | $0.001 \pm 0.001$ | $0.706 \pm 0.006$ | $0.696 \pm 0.005$ |

*Table 4.6: Per-bin tracking reward $R_j$ at end of training, energy-regularised reward of §4.2.1 with the Run-7 tuple, 3000 PPO iterations per cell. Each entry is the mean $\pm$ standard deviation over the last 200 PPO iterations of the curve obtained by averaging the three seeds.*

**What changed from Table 4.1.** The plateau on bins 0 through 6 lifts substantially under the energy-regularised reward and the three conditions become indistinguishable there. Every condition reaches $0.97$–$0.98$ at bin 0 and decreases monotonically with the command to $0.81$–$0.87$ at bin 6, with cross-condition gaps below the seed-to-seed noise. The lift is mainly the wider tracking kernel ($\sigma_v = 1.0$ in the Liang reward vs $\sigma_v = 0.5$ in §3.1.4), which makes $R_{\mathrm{lin}} = \exp(-|\Delta v|^2/\sigma_v)$ saturate near $1$ over a much wider velocity-error window. The bin-monotone decrease is what remains of the same Jensen-type artefact noted in §4.1.1. The kernel $R_{\mathrm{lin}}$ is concave in $\Delta v$, and at higher commands the same fractional error costs more reward.

**Bin 7 is now the only separator.** Task-specific and teacher-guided both clear $\gamma = 0.7$ at bin 7 ($0.71$ and $0.70$ respectively), and uniform collapses to $0.001$. This is the same qualitative collapse uniform had under the sprint-retune reward, but now isolated to the single sprint bin. The energy-regularised reward closes most of the bin-6 gap that uniform had in Phase 1 (uniform reaches $0.81$ at bin 6 here vs $0.06$ in Table 4.1) because the cost-of-transport gradient now actively pushes the policy toward a coordinated gait. But at bin 7 even the energy reward is not enough. Uniform's $1/8$ share at the highest command leaves the operator with too few iterations to find a gait that can sustain $3.75$ m/s. Curriculum-driven sampling restores it, and at bin 7 task-specific and teacher-guided are within seed-noise of each other in $R_j$.

![Per-bin tracking-reward learning curves, energy-regularised reward.](../src/results_phase2_rerun/figures/learning_curves_rlin.png)

*Figure 4.9: Per-bin tracking-reward curves, energy-regularised reward, three seeds per condition. Rows are velocity bins, columns are curriculum conditions. The mastery threshold $\gamma = 0.7$ is marked as a horizontal reference.*

The iterations-to-mastery metric is summarised in Table 4.7.

| Bin | $v_x^{\mathrm{cmd}}$ | Uniform | Task-specific | Teacher-guided |
|---:|---:|---:|---:|---:|
| 0 | 0.25 | 24 | 22 | 22 |
| 1 | 0.75 | 66 | 96 | 72 |
| 2 | 1.25 | 254 | 196 | 266 |
| 3 | 1.75 | 324 | 252 | 350 |
| 4 | 2.25 | 374 | 302 | 442 |
| 5 | 2.75 | 462 | 386 | 558 |
| 6 | 3.25 | 614 | 526 | 706 |
| 7 | 3.75 | not mastered (0/3) | 1232 (3/3) | 1318 (3/3) |

*Table 4.7: Iterations-to-mastery per (condition, bin) cell, energy-regularised reward, computed as the first PPO iteration at which the seed-averaged $R_j$ crosses $\gamma = 0.7$. The fraction of seeds that reached mastery individually is shown in parentheses where it is below 3/3.*

Bins 0 through 6 are mastered by every condition. Task-specific is the fastest on every bin from 2 through 6 because its monotone-expansion rule concentrates samples on the unlocked edge, which coincides with the bins that have the steepest remaining learning progress. Teacher-guided is the slowest of the three on bins 2 through 6 because the LP-ACRL temperature keeps probability mass spread across all still-improving bins rather than concentrating on the current edge, so each individual bin receives fewer samples per iteration. At bin 7 the separation sharpens: uniform fails to master at all, while task-specific masters first ($1232$) and teacher-guided follows at $1318$. The two curriculum operators are within seed-to-seed noise of each other at the sprint bin, and the ordering is the same as under the sprint-retune reward, where task-specific also reached bin 7 before teacher-guided.


![Iterations-to-mastery per (condition, bin), energy-regularised reward.](../src/results_phase2_rerun/figures/iterations_to_mastery.png)

*Figure 4.10: Iterations-to-mastery per (condition, bin), energy-regularised reward. Each bar is the mean across three seeds.*

![Per-bin sampling weight across training, energy-regularised reward.](../src/results_phase2_rerun/figures/sampling_heatmap.png)

*Figure 4.11: Per-bin sampling weight across training, energy-regularised reward. Rows are velocity bins, columns are training iterations. Uniform is flat by construction; task-specific expands its support monotonically; teacher-guided reallocates sampling toward bins with the largest learning progress.*

#### 4.3.2 Per-bin EPTE-SP

Table 4.8 reports the EPTE-SP and fall fraction from the deterministic-evaluation protocol of §3.3, exactly the same metric as Table 4.3 of §4.1.2 but on the energy-regularised reward.

| Bin | $v_x^{\mathrm{cmd}}$ | Uniform EPTE-SP / fall% | Task-specific EPTE-SP / fall% | Teacher EPTE-SP / fall% |
|---:|---:|---:|---:|---:|
| 0 | 0.25 | $0.432$ / $1\%$ | $0.425$ / $0\%$ | $0.414$ / $2\%$ |
| 1 | 0.75 | $0.121$ / $0\%$ | $0.121$ / $0\%$ | $0.122$ / $0\%$ |
| 2 | 1.25 | $0.062$ / $0\%$ | $0.061$ / $0\%$ | $0.064$ / $0\%$ |
| 3 | 1.75 | $0.054$ / $0\%$ | $0.056$ / $0\%$ | $0.057$ / $0\%$ |
| 4 | 2.25 | $0.054$ / $0\%$ | $0.051$ / $0\%$ | $0.056$ / $0\%$ |
| 5 | 2.75 | $0.057$ / $0\%$ | $0.055$ / $0\%$ | $0.057$ / $0\%$ |
| 6 | 3.25 | $0.092$ / $0\%$ | $0.079$ / $0\%$ | $0.077$ / $0\%$ |
| 7 | 3.75 | $0.857$ / $0\%$ | $0.385$ / $0\%$ | $0.126$ / $0\%$ |

*Table 4.8: Per-bin EPTE-SP and fall fraction, energy-regularised reward, three seeds and 300 rollouts per cell.*

Three changes from the Phase 1 row of the same metric (Table 4.3) are visible. (i) The fall column has collapsed to zero across the board. Under the energy-regularised reward, none of the three conditions falls at any of the eight bins in any meaningful fraction of rollouts. The previously catastrophic bin-6 and bin-7 collapse of uniform, and the $50\%$ fall fraction of task-specific at the sprint bins under §3.1.4, is gone. (ii) Bins 0 through 6 are within $\sim 10\%$ tracking error for all three conditions (interpreting the $0.4$–$0.5$ EPTE-SP at bin 0 as the normalisation artefact discussed in §4.1.2). (iii) At bin 7 the three conditions separate sharply. Teacher-guided reaches $0.126$, task-specific reaches $0.385$, and uniform reaches $0.857$.

The bin-7 separation is the central observation of this section. The mean forward velocity at bin 7 under deterministic evaluation, taken from Figure 4.12, is $\bar v_x = 0.54$ m/s for uniform against the $3.75$ m/s command, $\bar v_x = 2.31$ m/s for task-specific, and $\bar v_x = 3.29$ m/s for teacher-guided. Uniform is essentially refusing the sprint command, since the policy walks forward at well below the commanded speed. Task-specific produces a roughly two-thirds-speed walking trot (duty $\beta = 0.59$, no flight phase, see Table 4.10). Only teacher-guided tracks near the commanded velocity. None of the three falls. The bin-7 EPTE-SP rank is therefore a rank in *what velocity the policy will accept the command at*, not in survival.

![Measured vs commanded forward velocity per bin, energy-regularised reward.](../src/results_phase2_rerun/figures/v_actual_vs_cmd.png)

*Figure 4.12: Mean measured forward velocity vs commanded velocity, energy-regularised reward. Diagonal is the perfect-tracking reference. Uniform sits near $\bar v_x = 0.5$ m/s at bin 7 against the $3.75$ m/s command; only teacher-guided follows the diagonal through the sprint bin.*

![EPTE-SP per (condition, bin), energy-regularised reward.](../src/results_phase2_rerun/figures/epte_bars.png)

*Figure 4.13: EPTE-SP per (condition, bin), energy-regularised reward. Bin-7 bars separate the three conditions: teacher-guided $0.126$, task-specific $0.385$, uniform $0.857$.*

#### 4.3.3 Qualitative behaviour at evaluation

The low-bin irregular contact pattern of §4.1.3 is no longer visible under the energy-regularised reward. The irregular four-foot pattern that the §3.1.4 reward had allowed at slow command is replaced by a coordinated Trot (walking) under all three conditions at bin 0 (Table 4.9). The sprint-bin collapse that uniform showed at bins 6 and 7 is replaced at bins 0 through 6 by a sustained forward gait. What separates the three conditions at evaluation is the *style* of locomotion at each commanded speed and whether that style matches the speed it is being asked to track at the sprint bin.

The per-foot stance/swing strips at deterministic evaluation are reproduced in Figure 4.14 for every (condition, bin) cell. The gait-classifier labels of §4.3.4 (Table 4.9) are referred to here only when convenient. The qualitative description below is read directly off the contact pattern in Figure 4.14 and the mean forward velocities of §4.3.2 (Figure 4.12).

![Per-foot stance/swing strips at deterministic evaluation, energy-regularised reward.](../src/results_phase2_rerun/figures/gait_diagram.png)

*Figure 4.14: Gait diagram at deterministic evaluation, energy-regularised reward. Columns are conditions (uniform / task-specific / teacher-guided), rows are velocity bins. Coloured blocks mark stance (foot in contact); white gaps mark swing. Per cell the plotted rollout is the best-surviving rollout across the three seeds, with ties broken by closest mean forward velocity to the command.*

- **Uniform.** All eight bins show a clean diagonal-pair Trot pattern, with duty factor $\beta$ declining from $0.67$ at bin 0 to $0.55$ at bins 5–6 (Table 4.10). At bin 7 stride frequency drops and duty factor rises back to $0.65$, but the diagonal Trot structure is maintained. The policy cannot track the sprint command — measured velocity is $0.54$ m/s against $3.75$ m/s. Uniform solves survival at the sprint bin (the robot does not fall) but not velocity (it produces a slow Trot instead of sprinting).
- **Task-specific.** Bins 0–1 show a diagonal-pair Trot. From bin 2 the diagonal synchrony weakens and the contact pattern transitions to a four-beat Walk, which holds through bin 4. From bin 5 the pattern shifts to a fore-pair/hind-pair alternation (Bound, walking, $\beta > 0.5$ throughout), which holds through bin 7 ($\beta = 0.59$, $\bar v_x = 2.31$ m/s against the $3.75$ m/s command). This Trot → Walk → Bound progression is observed in seed 2 (the best-surviving rollout at the high bins); seeds 0 and 1 show Trot throughout at all bins (§4.3.4).
- **Teacher-guided.** All eight bins show a clean diagonal-pair Trot throughout. Stride frequency and duty factor scale monotonically with the command from bin 0 ($\beta = 0.67$, $3.4$ Hz) through bin 7 ($\beta = 0.54$, $5.7$ Hz). The diagonal Trot contact pattern is maintained at the sprint bin, and measured velocity at bin 7 is $\bar v_x = 3.29$ m/s against the $3.75$ m/s command. No flight phase occurs in any bin ($\beta > 0.5$ throughout, Table 4.10).

The dominant contact pattern across all three conditions and all eight bins is Trot (walking). Teacher-guided confirms that the energy-regularised reward, *under a curriculum that allocates compute to the harder bins*, maintains a coordinated Trot through the sprint bin. Task-specific (best-surviving seed) shows a Trot → Walk → Bound progression at high bins, but this is a single-seed observation within a condition that is otherwise Trot (§4.3.4). Uniform maintains Trot throughout but cannot achieve sprint velocity: without curriculum-driven sampling at $b_7$, the policy settles in a slow-Trot optimum at the highest command.

| Bin | $v_x^{\mathrm{cmd}}$ | Uniform $\beta$ / freq (Hz) | Task-spec $\beta$ / freq (Hz) | Teacher $\beta$ / freq (Hz) |
|---:|---:|---:|---:|---:|
| 0 | 0.25 m/s | $0.67$ / $3.4$ | $0.66$ / $3.2$ | $0.67$ / $3.4$ |
| 1 | 0.75 m/s | $0.60$ / $4.3$ | $0.60$ / $4.4$ | $0.61$ / $4.4$ |
| 2 | 1.25 m/s | $0.59$ / $4.7$ | $0.59$ / $4.7$ | $0.59$ / $4.8$ |
| 3 | 1.75 m/s | $0.57$ / $5.0$ | $0.58$ / $5.1$ | $0.57$ / $5.1$ |
| 4 | 2.25 m/s | $0.56$ / $5.2$ | $0.57$ / $5.5$ | $0.55$ / $5.5$ |
| 5 | 2.75 m/s | $0.55$ / $5.4$ | $0.55$ / $5.7$ | $0.54$ / $5.6$ |
| 6 | 3.25 m/s | $0.55$ / $5.4$ | $0.55$ / $5.8$ | $0.54$ / $5.7$ |
| 7 | 3.75 m/s | $0.65$ / $3.9$ | $0.59$ / $5.3$ | $0.54$ / $5.7$ |

*Table 4.10: Duty factor and stride frequency per (condition, bin) cell, energy-regularised reward, averaged across three seeds and 300 rollouts per cell. All cells satisfy $\beta > 0.5$ (no flight phase), so the contact patterns are *walking* gaits in the Hildebrand sense.*

#### 4.3.4 Gait classification

Gait labels are assigned using two complementary sources: visual inspection of the raw foot-contact diagram (Figure 4.14), and a per-leg contact correlation analysis computed from the stored rollout traces. The correlation analysis computes the mean Pearson correlation across diagonal leg pairs (FL+RR, FR+RL; high positive = Trot) and across fore/hind pairs (FL+FR, RL+RR; high positive = Bound) for the best-surviving rollout per cell, using the same selection criterion as Figure 4.14. A cell is labelled **Trot (walking)** if the diagonal correlation exceeds $+0.5$, **Bound (walking)** if the fore/hind correlation exceeds $+0.5$, and **Walk** if both fall below $0.1$. All labels carry $\beta > 0.5$ throughout (Table 4.10) — no flight-phase gait occurs in any cell. The three labels are:

| Label | $\beta$ | Footfall pattern | Speed range |
|---|:---:|---|---|
| Walk | $> 0.5$ | Four-beat, each foot offset by ~¼ stride | Slowest |
| Trot (walking) | $> 0.5$ | Diagonal pairs (FL+RR, FR+RL) alternating, no flight phase | Medium |
| Bound (walking) | $> 0.5$ | Fore pair (FL+FR) and hind pair (RL+RR) alternating, no flight phase | Faster |

Per (condition, bin) cell, the label is taken from the best-surviving rollout using the same criterion as Figure 4.14. For task-specific, two seeds (seeds 0 and 1) show Trot throughout at all bins; the pattern below reflects seed 2, which is the best-surviving rollout at high-speed bins and shows a different progression (marked †):

| Bin | $v_x^{\mathrm{cmd}}$ | Uniform | Task-specific | Teacher-guided |
|---:|---:|---|---|---|
| 0 | 0.25 m/s | Trot (walking) | Trot (walking)    | Trot (walking) |
| 1 | 0.75 m/s | Trot (walking) | Trot (walking)    | Trot (walking) |
| 2 | 1.25 m/s | Trot (walking) | Walk†             | Trot (walking) |
| 3 | 1.75 m/s | Trot (walking) | Walk†             | Trot (walking) |
| 4 | 2.25 m/s | Trot (walking) | Walk†             | Trot (walking) |
| 5 | 2.75 m/s | Trot (walking) | Bound (walking)†  | Trot (walking) |
| 6 | 3.25 m/s | Trot (walking) | Bound (walking)†  | Trot (walking) |
| 7 | 3.75 m/s | Trot (walking) | Bound (walking)†  | Trot (walking) |

*Table 4.9: Gait label per (condition, bin) cell at deterministic evaluation, energy-regularised reward, from the best-surviving rollout per cell (matching Figure 4.14 selection). Labels confirmed by per-leg contact correlation (see §4.3.4). †Single-seed observation (seed 2 only); seeds 0 and 1 for task-specific show Trot (walking) at all bins.*

Three features of Table 4.9 are notable.

**Teacher-guided: Trot throughout.** The diagonal-pair contact pattern holds across all eight bins. Stride frequency and duty factor continue scaling with the command through bin 7 (β = 0.54, 5.7 Hz), and measured velocity is 3.29 m/s against the 3.75 m/s command. Teacher-guided is the only condition that holds the Trot pattern with sprint-speed tracking.

**Task-specific: Trot → Walk → Bound (single-seed observation).** The best-surviving rollout (seed 2) shows Trot at bins 0–1, a four-beat Walk from bins 2–4 (diagonal correlation weakens to $+0.09$–$+0.40$, below the Trot threshold), and fore-pair/hind-pair alternation from bin 5 onward (Bound, walking, $\beta > 0.5$), reaching $\bar v_x = 2.31$ m/s against the $3.75$ m/s command at bin 7. Seeds 0 and 1 show Trot throughout at all bins. The Walk and Bound labels at bins 2–7 are therefore a within-condition, single-seed observation rather than a condition-level finding.

**Uniform: Trot throughout but speed-limited.** The diagonal-pair pattern is visible across all bins. At bin 7 stride frequency drops and duty factor rises to 0.65, yet the Trot footfall structure is maintained. The policy cannot track the sprint command — measured velocity is only 0.54 m/s against 3.75 m/s. Uniform produces the Trot contact pattern but not sprint velocity.

Under a fixed energy-regularised reward and fixed budget, the dominant contact pattern is Trot (walking) across all three conditions. Teacher-guided maintains Trot with sprint-speed tracking. Task-specific (in the best-seed rollout) shows Trot → Walk → Bound at bins 4–7, but the other two seeds show Trot throughout; the Bound observation is a within-condition seed-level variant. Uniform maintains Trot but cannot achieve sprint velocity. A curriculum that allocates non-trivial compute to the highest bin is required for sprint-speed tracking, regardless of which contact pattern it produces.

---

## 5. Conclusion

Three velocity-command sampling rules were compared on a single PPO policy on the Unitree Go2 across $[0,\,4]$ m/s, under two rewards. The three rules are uniform, task-specific (Box Adaptive, Margolis et al. 2022), and teacher-guided (LP-ACRL, Li, Li, & Hutter, 2026). The two rewards are the *sprint-retune* reward of §3.1.4 and the *energy-regularised* reward of Liang et al. (2024) with Go2-calibrated scales (§4.2).

Findings:

- **Separation lives at the sprint bins. Bins 0 through 5 are indistinguishable.** Under both rewards, all three conditions reach the same per-bin tracking-reward plateau on bins 0–4 (commands up to $2.5$ m/s) and are within seed-noise of each other on per-bin EPTE-SP through bin 5. The operator effect is concentrated at $b_6$ and $b_7$.
- **Curriculum decides survival. Reward decides velocity at the sprint bin.** Under the sprint-retune reward, uniform is the only condition that does not master $b_6$ or $b_7$ (Tables 4.1, 4.2) and falls in $98$–$100\%$ of deterministic rollouts there (Table 4.3). Task-specific and teacher-guided both clear $\gamma = 0.7$ at both sprint bins, though task-specific still falls $50\%$ of the time at $b_6$/$b_7$ while teacher-guided falls only $13\%$/$17\%$. Under the energy-regularised reward the fall fraction collapses to zero across all three conditions at every bin (Table 4.8). What separates the operators is then the velocity the policy will accept the command at. At $b_7$ the deterministic measured velocity is $\bar v_x = 3.29$ m/s for teacher-guided, $2.31$ m/s for task-specific, and $0.54$ m/s for uniform, against a $3.75$ m/s command (§4.3.2). Uniform under the energy reward solves survival at the sprint bin but not the velocity. It produces a slow Trot instead of sprinting (§4.3.4).
- **The split is curriculum-vs-uniform, not task-spec-vs-teacher.** On both rewards, task-specific and teacher-guided are within seed-noise on per-bin $R_j$ at $b_6$ and $b_7$ (Tables 4.1, 4.6). Their iterations-to-mastery profiles differ. Task-specific reaches the unlocked edge faster on bins 0 through 6, and reaches $b_7$ marginally earlier than teacher-guided under the energy reward as well ($1232$ vs $1318$, Table 4.7), though the two operators are within seed-to-seed noise at the sprint bin. Teacher-guided holds Trot (walking) at the sprint bin under the energy reward; task-specific shows Trot in two of three seeds, with the best-surviving seed exhibiting Bound (walking) at the sprint bins (§4.3.4). Uniform is the outlier under both rewards.
- **Gait family on the Go2 within budget: Trot (walking) dominant, all walking.** Every (condition, bin) cell under the energy-regularised reward has duty factor $\beta > 0.5$ (Table 4.10), so all contact patterns are *walking* gaits in the Hildebrand sense. The dominant pattern across all three conditions and all eight bins is Trot (walking) — diagonal-pair alternation with no flight phase. One seed of task-specific curriculum (seed 2) exhibits a Trot → Walk → Bound progression at bins 4–7, with the fore-pair/hind-pair Bound pattern persisting to the sprint bin ($\bar v_x = 2.31$ m/s, $\beta = 0.59$); the other two seeds show Trot (walking) throughout.

A few caveats apply. Each (condition, seed) is a single run, and spreads are across three seeds, not three replicates. The 3000-iteration budget is short relative to standard practice in this literature (Margolis et al., 2022, trained at the equivalent of $\sim 12000$ iterations on this scale of robot), so the uniform-at-$b_7$ collapse is a budget-bound failure, not necessarily asymptotic. All experiments are flat-ground simulation on a single platform. Transfer to terrain, other quadrupeds, or hardware is not claimed.

Within those bounds, the curriculum operator is the binding constraint on whether a single PPO policy on the Go2 covers $[0,\,4]$ m/s within budget. Any sampler that allocates non-trivial compute to the highest bin, whether Box Adaptive or LP-ACRL, is sufficient to do so. Uniform sampling is not.

---

## References

Li, Z., Li, C., & Hutter, M. (2026). Scaling rough terrain locomotion with automatic curriculum reinforcement learning. *arXiv:2601.17428*. <https://arxiv.org/abs/2601.17428>

Liang, Z., Sun, Y., Zhu, K., Zhang, Y., Xiong, S., Wang, Z., Li, R., Sreenath, K., & Tomizuka, M. (2024). Adaptive energy regularization for autonomous gait transition and energy-efficient quadruped locomotion. *arXiv:2403.20001v2*. <https://arxiv.org/abs/2403.20001>

Margolis, G. B., Yang, G., Paigwar, K., Chen, T., & Agrawal, P. (2022). Rapid locomotion via reinforcement learning. *Robotics: Science and Systems*. <https://arxiv.org/abs/2205.02824>

Mittal, M., Yu, C., Yu, Q., Liu, J., Rudin, N., Hoeller, D., Yuan, J. L., Singh, R., Guo, Y., Mazhar, H., Mandlekar, A., Babich, B., Birchfield, S., Hutter, M., & Garg, A. (2023). Orbit: A unified simulation framework for interactive robot learning environments. *IEEE Robotics and Automation Letters*, 8(6), 3740-3747. <https://arxiv.org/abs/2301.04195>

Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022). Learning to walk in minutes using massively parallel deep reinforcement learning. *Conference on Robot Learning* (pp. 91-100). <https://arxiv.org/abs/2109.11978>

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*. <https://arxiv.org/abs/1707.06347>

Unitree Robotics. (2024). *unitree_rl_lab* [Software]. GitHub. <https://github.com/unitreerobotics/unitree_rl_lab>
