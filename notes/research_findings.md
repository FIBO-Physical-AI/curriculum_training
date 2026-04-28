# Research Findings: Why Go2 Can't Learn High-Velocity Tracking

Date: 2026-04-24
Goal: compare our setup to Li 2026 LP-ACRL paper (the target we're replicating), identify what broke, propose fixes.

---

## TL;DR

1. **We stripped and rewrote so many upstream reward weights that our env barely resembles either the paper or the upstream Isaac Lab baseline.**
2. **Three specific regressions are probably the culprit:**
   - We zeroed out `track_ang_vel_z` (paper weight = 0.5 · dt). Without yaw tracking, robot has no incentive to face forward or walk straight at high speeds.
   - We weakened `action_rate` penalty 10x (upstream -0.1 → our -0.01; paper uses -0.25 · dt). Without smooth actions, fast locomotion becomes violent and unstable.
   - We changed `feet_air_time` threshold (upstream 0.5 → our 0.3) and inverted the direction of the signal. Paper uses 0.5 threshold, weight 2 · dt.
3. **Our added terms (linear bootstrap, feet_clearance) are not in the paper and may be distorting the landscape.**
4. **Physical ceiling:** Go2 real-world max speed is ~3.7 m/s peak (Unitree spec). Li's 4 m/s was on ANYmal D (~50 kg, 0.6m legs), not Go2 (~15 kg, 0.3m legs). Reaching 4 m/s on Go2 is at the edge of physical possibility; 3 m/s is a more realistic target.

---

## 1. Li 2026 LP-ACRL reward setup (Table III, Appendix)

| Term | Definition | Weight |
|---|---|---|
| Linear velocity tracking | φ(v*_xy − v_xy), σ=0.5 | **1.0 · dt** |
| Angular velocity tracking | φ(ω*_z − ω_z), σ=0.5 | **0.5 · dt** |
| Vertical velocity penalty | −v²_z | 4 · dt |
| Angular velocity penalty | −\|ω_xy\|² | 0.05 · dt |
| Joint motion penalty | −\|q̇\|² − \|q̈\|² | 0.001 · dt |
| Joint torque penalty | −\|τ\|² | 2e-5 · dt |
| Action rate penalty | −\|Δa\|² | **0.25 · dt** |
| Collision penalty | −n_collisions | 0.001 · dt |
| Feet air time reward | Σ(t_air,f − 0.5) | **2.0 · dt** |

Where φ(x) = exp(−‖x‖² / 0.25). So σ² = 0.25 ⇒ σ = 0.5.

**NOT present in paper:**
- `flat_orientation_l2` (paper uses only vertical vel penalty)
- `joint_pos` with `stand_still_scale`
- `feet_slide`
- `air_time_variance`
- `feet_clearance` / `feet_gait` / `foot_clearance_reward`
- `dof_pos_limits` (tolerance only; paper uses nominal joint action clip)
- Any linear-bootstrap tracking term (ours: `track_lin_vel_x_linear`)

**Robot/env specs:**
- ANYmal D, 12-DOF
- 50 Hz control
- Command: \|v*_x\| ∈ [0, 4] m/s, 8 bins of width 0.5, signs uniform ⇒ effective range [−4, 4]
- Observation: base_lin_vel(3), base_ang_vel(3), gravity(3), cmd(3), joint_pos(12), joint_vel(12), prev_action(12), height(275 — excluded in flat exp)
- Trains to 4 m/s at 3000 iterations

---

## 2. Upstream `unitree_rl_lab` Go2 default rewards

From [unitree_rl_lab/source/.../go2/velocity_env_cfg.py:273-343](unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py#L273-L343)

| Term | Weight (upstream) |
|---|---|
| track_lin_vel_xy (Gaussian σ=0.5) | **+1.5** |
| track_ang_vel_z (Gaussian σ=0.5) | **+0.75** |
| base_linear_velocity (lin_vel_z_l2) | -2.0 |
| base_angular_velocity (ang_vel_xy_l2) | -0.05 |
| joint_vel | -0.001 |
| joint_acc | -2.5e-7 |
| joint_torques | -2e-4 |
| action_rate | **-0.1** |
| energy | -2e-5 |
| dof_pos_limits | -10.0 |
| flat_orientation_l2 | -2.5 |
| joint_pos (stand_still_scale=5.0) | -0.7 |
| feet_air_time (threshold=0.5) | **+0.1** |
| air_time_variance | -1.0 |
| feet_slide | -0.1 |
| undesired_contacts | -1 |

**Upstream specs:**
- 50 Hz control (200 Hz sim / decimation 4)
- Action scale 0.25, Kp 25, Kd 0.5
- Default initial cmd ±0.1 m/s, limit ±1.0 m/s
- Episode 20s = 1000 steps at 50 Hz
- Domain rand: friction [0.3, 1.2], added base mass [-1, 3] kg, push interval 5-10s

---

## 3. Our overrides (go2_velocity_base.py)

| Term | Paper | Upstream | Ours | Multiplier vs upstream |
|---|---|---|---|---|
| **track_lin_vel_xy** | 1.0 dt | +1.5 | **+0.75** | 0.5x |
| **track_ang_vel_z** | 0.5 dt | +0.75 | **0.0 (removed!)** | 0x |
| ang_vel_z_penalty (we ADDED) | (none — paper uses tracking) | (none) | -0.05 | new |
| base_linear_velocity (v_z²) | 4 dt | -2.0 | **-0.5** | 0.25x |
| base_angular_velocity (ω_xy²) | 0.05 dt | -0.05 | -0.01 | 0.2x |
| joint_vel | 0.001 dt | -0.001 | -1e-4 | 0.1x |
| joint_torques | 2e-5 dt | -2e-4 | -1e-5 | 0.05x |
| **action_rate** | **0.25 dt** | **-0.1** | **-0.01** | **0.1x** |
| energy | — | -2e-5 | -1e-6 | 0.05x |
| **feet_air_time** (threshold) | 0.5 | 0.5 | **0.3** | |
| **feet_air_time** (weight) | **2.0 dt** | +0.1 | **1.0** | 10x upstream but still 2x smaller than paper |
| flat_orientation_l2 | (none) | -2.5 | -2.5 | 1x |
| joint_pos (stand_still) | (none) | -0.7 | -0.7 | 1x |
| dof_pos_limits | (none) | -10.0 | -10.0 | 1x |
| air_time_variance | (none) | -1.0 | -1.0 | 1x |
| feet_slide | (none) | -0.1 | -0.1 | 1x |
| undesired_contacts | 0.001 dt (collisions) | -1 | -1 | much stronger than paper |
| feet_clearance (we ADDED) | (none) | (none) | +0.5 | new |
| **track_lin_vel_x_linear (we ADDED)** | (none) | (none) | +0.35 | new, replaces some of the tracking reward with linear-in-velocity signal |
| feet_gait (we ADDED, then REMOVED) | (none) | (none) | 0 | removed commit 50cfd0b |

### Observation & command overrides:
- `velocity_commands` trimmed from 3-dim to 1-dim (forward only) — our `curriculum_mdp.forward_velocity_command`.
- Command resample every 20s (upstream 10s).
- `rel_standing_envs = 0` (upstream 0.1).
- Full command range [0, 4] sampled immediately (upstream curriculum starts ±0.1 m/s).
- `lin_vel_cmd_levels` disabled (we use our own binned curriculum).

---

## 4. Discrepancy analysis — likely root causes

Ranked by likely impact:

### R1 — Removed yaw tracking (severe)
We zero out `track_ang_vel_z` and replace with a `−0.05` penalty on ω_z. But zero yaw command + no tracking reward = the policy only sees a weak penalty for ANY yaw motion. There's no reward to actively track yaw = 0. At high forward speeds where the body wants to veer, the policy has no positive signal to counter the drift — it's easier to stand still than to oscillate between "heading slightly off → penalty" and "correcting → penalty".

**Paper explicitly weights yaw tracking at 0.5 · dt** — half the linear tracking weight. That's load-bearing.

### R2 — Action rate penalty 10x too weak (severe)
Upstream -0.1, paper -0.25, we're at -0.01. High-speed locomotion demands fast but SMOOTH actions. With a tiny action_rate penalty, the policy learns jerky actions, which prevents stable high-frequency gaits.

This is probably why bins 5-7 fail: at cmd 3+ m/s, the policy would need to oscillate fast, but each "try it" exploration is violent enough to either terminate the episode (via `undesired_contacts` or orientation) or trigger the `air_time_variance` penalty for inconsistent stepping. Standing still is safer.

### R3 — Linear bootstrap reward confuses the landscape
At cmd=3.75 m/s, `aligned / |v_cmd| = v_act/3.75`. So moving at 0.5 m/s gives reward 0.5/3.75 × 0.35 = 0.047. Moving at 3 m/s gives 0.80 × 0.35 = 0.28. Gaussian tracking at v_act=3 is exp(-(0.75)²/0.25) × 0.75 = 0.05. 

So at high commands, our bootstrap dominates the Gaussian. The reward surface near the target is almost flat (bootstrap saturates). No pull toward exact tracking. Could be why EPTE stays poor even when bin 5 "cracks".

### R4 — feet_air_time inverted signal
Upstream defines `feet_air_time` such that reward is proportional to (t_air − threshold) when foot is in contact — encouraging feet to stay in the air longer than threshold before touching down. Threshold 0.5 means reward for 0.5s+ swing time. We changed to 0.3, encouraging shorter swings.

Paper uses 0.5 AND weight 2.0 · dt (ours 1.0). Paper wants LONG air time + strong reward ⇒ bigger strides ⇒ faster achievable velocities. We moved the exact opposite direction.

### R5 — Missing/extra terms vs paper
Paper doesn't have:
- `flat_orientation_l2` (−2.5): forces body flat; at high speeds there may be a natural pitch forward. Penalizing it discourages fast gaits.
- `joint_pos` with `stand_still_scale=5.0` (−0.7): penalizes joint-position deviation from default × 5 when velocity < 0.3. Actually this should discourage standing still. BUT it also penalizes any unusual pose when commanded to go slow, which could shape the early curriculum.
- `dof_pos_limits` (−10.0): OK for safety but weight is enormous.
- `air_time_variance` (−1.0): penalizes variance in air time across feet. At high speeds, asymmetric gaits (gallop) have high variance; this penalty forbids them.
- `feet_slide` (−0.1): anti-slip; fine but unnecessary on plane terrain.
- `undesired_contacts` (−1.0) with body_names including hip, thigh, calf, head: terminates on body contact. At high speeds, bumping a hip against the ground is easy → terminates → contributes to "stand still is safer".

### R6 — Physical ceiling
Go2 max speed (Unitree spec): 2.5 m/s continuous, 3.7 m/s peak. Li 2026 reached 4 m/s on ANYmal D, which has ~3x the mass and 2x the leg length. **Asking Go2 to track 4 m/s is asking near its physical limit.**

Margolis 2022 (Rapid Locomotion) reached 3.9 m/s on Mini Cheetah (similar size to Go2) — but that used aggressive hardware and a sharply tuned reward. Default Isaac Lab unitree_rl_lab weights are tuned for stable walking ≤1 m/s, not high-speed running.

---

## 5. Recommended fixes, prioritized

### Phase A: restore paper/upstream alignment (HIGH IMPACT, low risk)

These are "undo things we shouldn't have changed":

1. **Restore yaw tracking.** Delete `_remove_yaw_tracking_reward`. Keep upstream `track_ang_vel_z` at +0.75 (or lower to paper's 0.5 if we want to exactly match).
2. **Restore action_rate to upstream -0.1** (or paper's -0.25, but -0.1 is safer).
3. **Restore feet_air_time to upstream: threshold 0.5, weight +0.1.** OR go to paper's 2.0 weight. Do NOT invert to 0.3 threshold.
4. **Restore track_lin_vel_xy weight to 1.0** (between paper's 1.0 and upstream's 1.5).
5. **Delete the linear bootstrap reward** (`track_lin_vel_x_linear`). Paper doesn't need it.
6. **Delete `feet_clearance`.** Paper uses just air_time.

### Phase B: tune for Go2 physical reality (MEDIUM IMPACT)

7. **Lower V_MAX from 4.0 to 3.0 m/s.** Honest target. Bin width 0.375 instead of 0.5.
   Or: keep 4.0 but accept that bins 7 (3.5-4.0) will always be hard.
8. **Consider relaxing `undesired_contacts`** — remove hip/thigh/calf from body_names, only terminate on head. At speed, leg contacts are normal.
9. **Consider relaxing `flat_orientation_l2`** weight from -2.5 to -1.0. Allow some pitch for speed.
10. **Consider relaxing `dof_pos_limits`** — robot may need to use its joint range at high speed.

### Phase C: curriculum tuning (if phases A+B don't fully solve it)

11. Tune teacher β: currently 0.05. If weights still collapse, try 0.02 (sharper softmax).
12. Tune stage_length: currently 50 iters. Could try 30 for more responsive LP signal.

---

## 6. Proposed experimental protocol

**Goal:** confirm Phase A restoration fixes the stand-still pathology before investing in a 6k × 3 sweep.

1. Revert `go2_velocity_base.py` to a minimal override that only:
   - flattens terrain
   - installs `BinnedVelocityCommandCfg`
   - wires in `velocity_curriculum_step` curriculum hook
   - keeps `rel_standing_envs = 0`
   
   Leave ALL upstream reward weights as-is. No rebalance, no remove-yaw, no bootstrap, no gait shaping.

2. Smoke test: 3 conditions × 1 seed × 3000 iters, V_MAX=3.0.

3. Check:
   - Uniform bins 5-6 (2.0-2.5 m/s) EPTE < 0.9 (should be achievable at Go2 continuous max)
   - Teacher outperforms uniform on bins 4-6
   - Failure mode at highest bin = "tried and fell" (EPTE 0.6-0.8), not "stand still" (EPTE = 1.0)

4. If Phase A smoke passes → run full 6k × 3 seeds × V_MAX ∈ {3.0, 4.0} and compare.

5. If Phase A smoke fails → inspect whether `undesired_contacts` / `flat_orientation_l2` are terminating episodes prematurely → move to Phase B.

---

## 7. Open questions

- What β did Li use? Not stated in main text. Appendix B referenced but needs to be read.
- Is the Go2 action scale 0.25 enough for fast gaits? ANYmal D PD stiffness/scale may differ.
- Does Li's PPO hyperparameter setup (LR, batch, horizon) differ from our `BasePPORunnerCfg`?
- What is `mdp.feet_air_time` actually computing in upstream? (Sign convention matters.)
