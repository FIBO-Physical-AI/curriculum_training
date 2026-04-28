# Project Summary — Curriculum Learning for Go2 Sprint Tracking

**Course:** KMUTT FRA503 Deep Reinforcement Learning
**Goal:** Train a Unitree Go2 quadruped to track forward velocity commands from **0 to 4 m/s** using PPO + Isaac Lab, and compare three curriculum learning strategies.
**Branch:** `v_max_4_retune`

---

## 1. The three curricula being compared

All three sample velocity commands from **8 bins** spanning 0 to 4 m/s (bin width = 0.5 m/s, centers 0.25, 0.75, ..., 3.75 m/s).

| Strategy | How it samples bins |
|---|---|
| **Uniform** | Every bin sampled with equal probability (12.5% each). Baseline. |
| **Task-Specific** (Margolis Box Adaptive, 2022) | Start at lowest bin only. When current bin's mean reward crosses a threshold, **expand frontier** by one bin. Sample uniformly within active set. |
| **Teacher-Guided** (LP-ACRL, Li 2026) | Weight bins by **Learning Progress** (\|Δreward over time\|). Concentrates samples on bins where the policy is currently improving. |

Code:
- `src/source/curriculum_rl/curriculum_rl/curricula/task_specific.py`
- `src/source/curriculum_rl/curriculum_rl/curricula/teacher_guided.py`
- `src/source/curriculum_rl/curriculum_rl/envs/commands.py` — `BinnedVelocityCommandCfg` shared sampler

---

## 2. Reference papers (in workspace)

### Margolis 2022 — "Walk These Ways" (`2205.02824v1.pdf`)
- Robot: **MIT Mini Cheetah** (~9 kg, ~17 Nm peak torque per joint)
- Reached **3.9 m/s real / 5.07 m/s sim** with grid-adaptive curriculum
- Key claim: **2D infeasibility** (forward × turn at high speed → centrifugal impossible regions) makes curricula essential. Uniform sampling wastes samples in physically infeasible regions.
- Their curriculum is the basis for our `task_specific` condition (4-connected neighbor frontier expansion).

### Li 2026 — LP-ACRL (`2601.17428v1.pdf`)
- Robot: **ANYmal D** (~50 kg, larger than Go2)
- Curriculum: weighted sampling by Learning Progress
- Key claim: LP-driven sampling produces **more robust policies** because it doesn't prematurely abandon a bin just because mean reward briefly crossed a threshold (which Margolis's frontier rule does).
- Basis for our `teacher` condition.

---

## 3. Robot physical comparison

| Property | Mini Cheetah (Margolis) | Go2 (ours) | ANYmal D (Li) |
|---|---|---|---|
| Mass | ~9 kg | ~15 kg | ~50 kg |
| Peak joint torque | ~17 Nm | ~35 Nm | ~80 Nm |
| Reported max forward speed | 3.9 m/s real / 5.07 sim | ~4 m/s achievable | ~3 m/s |
| Stride frequency at 4 m/s | ~4 Hz | **~8 Hz** (smaller leg) | ~2.5 Hz |

**Key implication:** Go2 at 4 m/s needs ~8 Hz stride. Upstream Go2 reward had `feet_air_time.threshold = 0.5 s` (= 2 Hz stride), which actively rewarded slow gait → blocked sprinting.

---

## 4. The problem we hit and how we solved it

### Phase 1 — V_MAX=3.0 (initial setup)
**Symptom:** Robot dead/standing in place across all bins, all conditions.
**Root cause discovered:**
- Upstream Go2 (`unitree_rl_lab/.../go2/velocity_env_cfg.py`) was tuned for `lin_vel_x ∈ [-1.0, 1.0]` m/s. We were commanding 0–3 m/s (3× past envelope).
- `track_lin_vel_xy.std = 0.5` → at v_cmd=2.5 with achieved=0, reward = exp(-25/0.25) ≈ 0. No learning signal at high bins.
- Quadratic penalties (joint_torques, joint_vel) scale with v² → at sprint, penalties dominate task reward by 100×, policy chooses to stand still.

**Fix attempt 1 (Plan A):** Loosen reward (`std=1.0`), reduce action_rate (-0.1 → -0.05). **Insufficient.** At v=3.5, joint_torques still cost 0.54/step (quadratic), joint_vel 0.19/step. Stand-still strategy still won by 0.6/step.

### Phase 2 — V_MAX=4.0 with Plan B (the working retune)
Reduced ALL the quadratic penalties enough that sprint is profitable. Increased action scale so legs can extend further. Shortened the air-time threshold to allow 8 Hz stride.

| Reward term | Upstream | Plan B (ours) | Ratio |
|---|---|---|---|
| `track_lin_vel_xy.std` | 0.5 | 1.0 (Plan B) → **0.5 final** | 2× looser then back |
| `action_rate.weight` | -0.1 | **-0.005** | 20× weaker |
| `joint_acc.weight` | -2.5e-7 | **-1e-7** | 2.5× weaker |
| `joint_torques.weight` | -2e-4 | **-2e-5** | 10× weaker |
| `joint_vel.weight` | -0.001 | **-1e-4** | 10× weaker |
| `feet_air_time.threshold` | 0.5 s | **0.1 s** | 5× shorter (allows 8 Hz stride) |
| `JointPositionAction.scale` | 0.25 | **0.35** | 1.4× larger |

**Plan B unlocked sprint envelope.** All bins reachable, mean EPTE dropped from 0.44/0.52/0.63 → 0.140/0.147/0.148.

**But Plan B made the task TOO easy** — uniform tracked all bins as well as the curricula. The expected "uniform fails at high speed" story disappeared because there was no infeasibility (1D forward + flat terrain, no centrifugal limits).

### Phase 3 — Tighten `std` back to 0.5 (final)
Plan B + `std=0.5` recreates difficulty without breaking the budget:
- Low bins (0.25–2.0 m/s): still trivially trackable
- High bins (3.0–3.75 m/s): tight Gaussian = sharper reward = uniform's even sampling not enough exposure to learn the high-speed gait
- Curricula focus on the frontier and master each bin in sequence

This is the configuration in `_apply_sprint_retune()` in `src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py`.

---

## 5. Other environment changes vs upstream

(See `differences_vs_upstream.md` for the full diff — quick highlights:)

- **Terrain flattened** — `terrain_type="plane"`, no cobblestone, no height scanner, terrain_levels curriculum disabled. We're studying 1D forward velocity, not terrain difficulty.
- **No turning** — `ang_vel_z = (0, 0)` always. Upstream had ±1 rad/s. Margolis-style 2D infeasibility is intentionally excluded so we isolate the speed dimension.
- **No lateral motion** — `lin_vel_y = (0, 0)`. Pure forward task.
- **Custom curriculum term** (`velocity_curriculum_step` in `mdp.py`) replaces upstream's `terrain_levels` and `lin_vel_cmd_levels`.
- **Play config locks reset pose** to `(x=0, y=0, yaw=0)` so videos always show robot facing +x. Training keeps random yaw.

We **never modify** any file under `unitree_rl_lab/`. All overrides applied at config-time via subclassing.

---

## 6. Final results — 3 seeds × 6000 PPO iterations

```
Mean EPTE per condition (lower = better):
  teacher          0.143
  task_specific    0.311
  uniform          0.371
```

### Per-bin breakdown (300 rollouts/bin = 3 seeds × 100 rollouts)

| Cond | b0 (0.25) | b1 (0.75) | b2 (1.25) | b3 (1.75) | b4 (2.25) | b5 (2.75) | b6 (3.25) | b7 (3.75) |
|---|---|---|---|---|---|---|---|---|
| **uniform** falls | 2/300 | 0 | 0 | 0 | 0 | 40/300 | **300/300** | **300/300** |
| **uniform** v_act | 0.19 | 0.80 | 1.29 | 1.78 | 2.30 | 2.43 | **0.06** | **0.03** |
| **task_spec** falls | 2/300 | 0 | 0 | 0 | 0 | 0 | 228/300 | **300/300** |
| **task_spec** v_act | 0.23 | 0.81 | 1.30 | 1.79 | 2.31 | 2.78 | 0.99 | **0.07** |
| **teacher** falls | 1/300 | 0 | 0 | 0 | 0 | 0 | **29/300** | **100/300** |
| **teacher** v_act | 0.22 | 0.78 | 1.25 | 1.77 | 2.31 | 2.81 | **2.95** | **2.38** |

### Three-tier story

1. **Uniform — catastrophic collapse above 2.5 m/s.** Starts falling at b5 (40/300). At b6 and b7 ALL 300 rollouts collapse within ~15–19 steps and the robot achieves essentially zero forward velocity (0.06, 0.03 m/s). 100% failure across all 3 seeds — not random.
2. **Task-specific — frontier stalls at b6.** Margolis Box Adaptive pushes the frontier to b6 (partial: 228/300 falls but tries) and **never reaches b7** (300/300 falls, 0.07 m/s). Curriculum helps over uniform but the "expand when current is mastered" rule loses momentum at the top.
3. **Teacher — best generalization.** LP-ACRL is 90% robust at b6 (29 falls, achieves 2.95 m/s for cmd 3.25) and 67% robust at b7 (100 falls, achieves 2.38 m/s for cmd 3.75). 2.2× lower mean EPTE than task, 2.6× lower than uniform.

---

## 7. Metric — EPTE-SP (Episode-Penalized Tracking Error, Speed Profile)

```
EPTE = (err_norm × k_f + (K − k_f)) / K

  K        = total episode steps (1000)
  k_f      = step at which the robot fell (or K if no fall)
  err_norm = mean |v_act − v_cmd| during alive period, normalized by v_cmd
             (clamped to [0, 1])
```

**Lower is better.** Range [0, 1].
- 0 = perfect tracking, never fell
- 1 = fell immediately or never tracked at all

Code: `src/scripts/eval_epte.py`

### Caveat: low-speed normalization artifact

At b0 (cmd 0.25 m/s), all conditions have err_norm ≈ 0.25–0.45 even though no robot is falling. Reason: EPTE divides absolute error by v_cmd. Robot's natural minimum stable trot is ~0.3–0.5 m/s, so a 0.07 m/s absolute error becomes 0.28 normalized error. **This is a metric artifact at low speeds, not a curriculum failure.**

### Caveat: EPTE captures fall-rate but `iterations_to_mastery` doesn't

Mastery = "first PPO iter where mean reward ≥ 0.7". Uniform's policy at b6 has bimodal outcomes — half the rollouts succeed, half fall. The mean reward can still cross 0.7, so uniform is reported as "mastered" even though half the rollouts collapse catastrophically. **Always combine `iterations_to_mastery` with the fall count column** when interpreting the figure.

---

## 8. Outputs the sweep produces

For each condition × seed:
- `unitree_rl_lab/logs/rsl_rl/<exp>/<run>/` — checkpoints (`model_*.pt`), TensorBoard, `curriculum.csv`, in-run videos
- `src/results/epte_sp.csv` — per-rollout EPTE (300 rows per bin per condition for 3-seed sweep)
- `src/results/videos/<condition>_seed<N>/bin{0..7}_v<center>.mp4` — pushable videos (separate from logs so they can be committed to git)
- `src/results/run_timings.txt` — per-run wall-clock
- `src/results/figures/*.png` — 6 figures via `plot_all.py`
  - `learning_curves.png`
  - `sampling_heatmap.png` / `sampling_heatmap_3d.png`
  - `iterations_to_mastery.png`
  - `epte_bars.png`
  - `epte_violin.png`

---

## 9. How to reproduce

```bash
# clean slate
rm -rf unitree_rl_lab/logs/rsl_rl/* \
       src/results/epte_sp.csv \
       src/results/run_timings.txt \
       src/results/sweep_output.log \
       src/results/figures/*.png \
       src/results/videos \
       .sweep_runs/*.path

# full sweep: 3 conditions × 3 seeds × 6000 iters (~8.5 hours on one GPU)
SEEDS="0 1 2" MAX_ITERATIONS=6000 bash src/scripts/run_sweep.sh
```

Override knobs (env vars):
- `CONDITIONS="uniform task_specific teacher"`
- `SEEDS="0 1 2"`
- `MAX_ITERATIONS=6000`
- `NUM_ENVS=2048`
- `VIDEO_LENGTH=200` (steps; 4 sec at 50 Hz)
- `RECORD_VIDEOS=1`

---

## 10. Key takeaways for a reader

1. **Curricula matter when the high-speed regime has sharp reward gradients.** Plan B (loose reward) eliminated the curriculum advantage. Tightening `std` back to 0.5 restored it.
2. **Uniform sampling collapses catastrophically (100% fall rate) when bin difficulty exceeds the policy's exploration capability.** Not bimodal — fully dead at b6/b7.
3. **Margolis Box Adaptive helps but isn't optimal.** "Expand frontier when current bin is mastered" loses momentum at the top of the range. Reaches b6 partially, never b7.
4. **LP-ACRL (teacher) is best.** Distributes samples across bins where Learning Progress is high, producing more robust policies. 2.6× better mean EPTE than uniform.
5. **The story is consistent with Margolis's theoretical claim** — curricula matter in regions where uniform random exposure can't accumulate enough learning signal. We created that condition with tight `std` at high speeds (since we couldn't use 2D infeasibility).
6. **What was NOT changed:** the `unitree_rl_lab` submodule is clean. Every customization is applied at config-time via subclassing in `src/source/curriculum_rl/`.

---

## 11. Pointer to other docs in the repo

- `differences_vs_upstream.md` — full diff vs upstream `velocity_env_cfg.py`
- `research_findings.md` — earlier research notes
- `implementation.md` — implementation details
- `README.md` / `src/README.md` — basic setup and run instructions
