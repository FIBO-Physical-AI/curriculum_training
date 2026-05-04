# Gait Emergence Run 4: Parameter Sweep for Velocity Tracking + Gait Emergence

This document specifies a **parameter sweep** to find the configuration of Liang 2024's energy reward that produces both (a) velocity tracking up to 4 m/s and (b) some gait differentiation across speed bins on Go2 in Isaac Lab.

Run 2 used paper defaults (σ_en_x=1000, std=0.5, α_en=1.0). Result: tracking works on bins 0-4 (up to 2.25 m/s), collapses on bins 5-7 (sprint speeds). Gait stays as flat trot at all speeds.

This run sweeps three parameters: `σ_en_x` (energy denominator scale), `track_lin_vel_xy.std` (tracking reward sharpness), and `energy.weight` (energy reward weight α_en). The sweep tests the trade-off between energy gradient strength and tracking pressure.

Read the entire document before writing code.

## 1. Branch Setup

Continue on `gait-emergence` branch. Run 2 state is the starting point. No new branch.

## 2. Background

### 2.1 The Trade-Off

Run 2 revealed two competing objectives:
- **Gait emergence** wants energy reward strong enough to differentiate gaits at low speed (push policy toward walk over trot)
- **Velocity tracking** wants tracking reward strong enough to dominate at high speed (push policy toward sprint over slow-trot)

The paper's defaults (σ_en_x=1000, std=0.5, α_en=1.0) sit at one point in this trade-off space. The sweep tests other points to find one that hits both objectives.

### 2.2 The Three Knobs

**Knob 1: `σ_en_x`** — energy denominator scaling. Liang Eq. 4: `R_en = exp(-power / (σ_en_x · |v_x| + σ_en_z · |ω_z|))`. Smaller σ_en_x makes the energy reward gradient steeper (higher penalty for high power). Default 1000 may be too large for Go2 in Isaac Lab — energy gradient may be too weak to differentiate at high speed.

**Knob 2: `track_lin_vel_xy.std`** — tracking reward sharpness. Reward is `exp(-err² / std²)`. Default std=0.5 means at err=3 m/s (sprint failure), reward ≈ 10⁻¹⁶ — effectively zero gradient toward higher speed. Wider std=1.0 keeps a gradient at large errors, pushing the policy toward better tracking.

**Knob 3: `energy.weight`** — energy reward weight α_en. Liang's value 1.0 may overpay the policy for being slow at sprint commands. Reduced 0.5 lets tracking dominate at high speed.

## 3. Phase 0: σ_en_x Diagnostic Pilot

Goal: measure mean R_en under random actions across σ_en_x candidates. Identify which σ values land R_en in the gradient-rich middle range (0.3, 0.7).

### 3.1 New File

`src/scripts/sweep_sigma_diagnostic.py`. Single Python file, no training, no PPO.

### 3.2 What It Does

1. Construct the existing Uniform-curriculum env. `num_envs=512`.
2. Reset environment.
3. For 1000 steps, take uniformly random actions in [-1, 1]^12 and call `env.step(action)`.
4. At each step, cache power, |v_x|, |ω_z| once, then compute R_en for all candidate σ_en_x values from the same data.
5. Aggregate mean / std / min / max of R_en for each σ value across the rollout.

σ candidates:
```python
SIGMA_X_VALUES = [50, 100, 250, 500, 1000, 2000, 5000]
SIGMA_Z_RATIO = 0.5   # σ_en_z = σ_en_x × 0.5
EPS = 0.1
```

### 3.3 Output Table

```
σ_en_x     mean(R_en)   std(R_en)   min       max       gradient regime
─────────────────────────────────────────────────────────────────────────
50         X.XXX        X.XXX       X.XXX     X.XXX     <classify>
100        X.XXX        X.XXX       X.XXX     X.XXX     <classify>
...
```

Classify by mean(R_en):
- mean < 0.05 → "too strong (saturated near 0)"
- mean ∈ [0.05, 0.30) → "strong"
- mean ∈ [0.30, 0.75) → "useful gradient ✓"
- mean ∈ [0.75, 0.95) → "weak gradient"
- mean ≥ 0.95 → "saturated near 1, no gradient"

Also print: mean power, mean |v_x|, mean |ω_z| as sanity check.

### 3.4 Stop and Wait

After Phase 0, paste the table. **Wait for human review.** Human picks **3 σ_en_x values** for Phase 1 grid based on the table:
- One in "useful gradient" regime closest to mean R_en = 0.5 → call this `SIGMA_MID`
- One step lower (more aggressive energy pressure) → call this `SIGMA_LOW`
- One step higher (closer to or at paper's 1000) → call this `SIGMA_HIGH`

If all candidates are "too strong" or "weak", extend the σ range and re-run Phase 0.

## 4. Phase 1: Parameter Grid Training Sweep

After Phase 0 selects σ values, run a **9-run grid** over (σ, std, α_en):

| Run # | σ_en_x | std | α_en | Purpose |
|---|---|---|---|---|
| 1 | SIGMA_LOW | 0.5 | 1.0 | Stronger energy, narrow tracking, paper α_en |
| 2 | SIGMA_MID | 0.5 | 1.0 | (closest to paper baseline if SIGMA_MID = 1000) |
| 3 | SIGMA_HIGH | 0.5 | 1.0 | Weaker energy, narrow tracking, paper α_en |
| 4 | SIGMA_LOW | 1.0 | 1.0 | Stronger energy, wide tracking, paper α_en |
| 5 | SIGMA_MID | 1.0 | 1.0 | Mid energy, wide tracking, paper α_en |
| 6 | SIGMA_HIGH | 1.0 | 1.0 | Weaker energy, wide tracking, paper α_en |
| 7 | SIGMA_LOW | 1.0 | 0.5 | Stronger energy, wide tracking, reduced α_en |
| 8 | SIGMA_MID | 1.0 | 0.5 | Mid energy, wide tracking, reduced α_en |
| 9 | SIGMA_HIGH | 1.0 | 0.5 | Weaker energy, wide tracking, reduced α_en |

`σ_en_z = σ_en_x × 0.5` for all runs (keep paper's 2:1 ratio).

### 4.1 Per-Run Configuration

Each run uses identical config to Run 2 except for the three swept parameters:

```python
cfg.rewards.track_lin_vel_xy.params["std"] = <std for this run>
cfg.rewards.track_lin_vel_xy.weight = 1.5   # unchanged across all runs
cfg.rewards.energy.params = {
    "sigma_en_x": <σ_en_x for this run>,
    "sigma_en_z": <σ_en_x × 0.5>,
    "eps": 0.1,
}
cfg.rewards.energy.weight = <α_en for this run>
```

All other parameters unchanged from Run 2:
- Uniform curriculum
- 3000 iterations per run
- 4096 envs
- Single seed (seed 0) per run
- All other reward weights as in Run 2 final state
- Action scale 0.35

Total wall-clock: 9 × ~30 min = **~4.5 hours** (depending on machine state).

### 4.2 Sweep Script

`src/scripts/sweep_param_grid.py`. Takes the σ values selected after Phase 0 as input, runs all 9 configurations sequentially, logs to a single CSV.

Pseudo-code:

```python
SIGMA_LOW, SIGMA_MID, SIGMA_HIGH = <from Phase 0 selection>
configs = [
    (SIGMA_LOW, 0.5, 1.0),
    (SIGMA_MID, 0.5, 1.0),
    (SIGMA_HIGH, 0.5, 1.0),
    (SIGMA_LOW, 1.0, 1.0),
    (SIGMA_MID, 1.0, 1.0),
    (SIGMA_HIGH, 1.0, 1.0),
    (SIGMA_LOW, 1.0, 0.5),
    (SIGMA_MID, 1.0, 0.5),
    (SIGMA_HIGH, 1.0, 0.5),
]

for run_id, (sigma, std, alpha_en) in enumerate(configs, 1):
    # train 3000 iter with this config
    log_dir = train_one_run(run_id, sigma, std, alpha_en)
    # eval and extract metrics
    metrics = eval_per_bin(log_dir)
    append_to_csv(SWEEP_CSV, run_id, sigma, std, alpha_en, metrics)
    save_gait_diagram(log_dir, f"sweep_run_{run_id}.png")
```

The sweep script should be **resumable** — if a run fails, other runs continue. Failed run logged with NaN metrics.

### 4.3 Output Per Run

For each run, log to a single CSV (`src/results/sweep_v4.csv`) with columns:

```
run_id, sigma_en_x, std, alpha_en,
mean_reward_final, mean_ep_length_final,
err_b0, err_b1, err_b2, err_b3, err_b4, err_b5, err_b6, err_b7,
v_act_b0, v_act_b1, ..., v_act_b7,
duty_b0, duty_b1, ..., duty_b7,
freq_b0, freq_b1, ..., freq_b7,
stride_b0, ..., stride_b7,
flight_b0, ..., flight_b7,    # binary 0/1
wall_clock_min
```

Per-bin metrics computed identically to Run 2 evaluation:
- 100 deterministic rollouts per bin
- Mean velocity, tracking error, duty factor, stride frequency, stride length
- Flight phase indicator: 1 if foot contact data shows a moment of all-four-feet-airborne, else 0

### 4.4 Comparison Table (Auto-Generated)

After all 9 runs complete, generate a markdown table from the CSV. Save as `src/results/sweep_v4_comparison.md`. Format:

```markdown
# Sweep V4 — Comparison Table

| run | σ_en_x | std | α_en | mean_R | ep_len | b0 err | b4 err | b7 err | b0 duty | b4 duty | b7 duty | flight 5/6/7 | walltime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | 0/0/0 | ...m |
| 2 | ... |
...

## Per-Bin Tracking Error (matrix view)

| run | b0   | b1   | b2   | b3   | b4   | b5   | b6   | b7   |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| 2 | ...

## Per-Bin Duty Factor (matrix view)

| run | b0   | b1   | b2   | b3   | b4   | b5   | b6   | b7   |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
...
```

Each gait diagram saved as `src/results/figures/sweep_v4_run_<id>_gait.png`.

## 5. Acceptance Criteria

A run is "winning" if all three hold:

**A. High-speed tracking:**
- `track_err` on bin 7 ≤ 0.5 m/s (vs Run 2's 0.85)
- `track_err` on bin 6 ≤ 0.5 m/s

**B. Low-speed tracking maintained:**
- `track_err` on bins 0-4 ≤ 0.4 m/s for all 5 bins

**C. Some gait differentiation visible:**
- `duty(b0) - duty(b4) ≥ 0.05` (walk-trot differentiation), OR
- Flight phase visible at bin 6 or 7

If multiple runs win → pick the one with smallest mean tracking error across all bins.

If no run wins on all three → report which run wins on (A) and (B) (tracking-focused result), and acknowledge gait emergence remains absent. This is still a defensible result: "parameter sweep extended Liang's velocity tracking range from 2.25 m/s to X m/s on Go2 in Isaac Lab via parameter recalibration; gait emergence beyond efficient trot did not transfer."

## 6. Files

### 6.1 New Files

- `src/scripts/sweep_sigma_diagnostic.py` (Phase 0)
- `src/scripts/sweep_param_grid.py` (Phase 1 sweep driver)
- `src/results/sweep_v4.csv` (output, auto-created)
- `src/results/sweep_v4_comparison.md` (output, auto-created)
- `src/results/figures/sweep_v4_run_<id>_gait.png` × 9 (output, auto-created)

### 6.2 Modified Files

`src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py` — add a way to override σ_en_x, σ_en_z, std, energy.weight from environment variables or argparse, so the sweep script can set them per-run without editing code. Implementer's choice on mechanism (env vars or kwargs).

### 6.3 Files NOT Touched

- `liang_composite_reward.py` (energy_cot function unchanged)
- All curriculum code
- `mdp.py`, observations, terminations, PPO config, action interface, sim parameters

## 7. Workflow

```
1. Write src/scripts/sweep_sigma_diagnostic.py (Phase 0 script)
2. Write src/scripts/sweep_param_grid.py (Phase 1 driver, prepared but not run yet)
3. Run Phase 0 → paste table → WAIT for review
4. Human selects (SIGMA_LOW, SIGMA_MID, SIGMA_HIGH) → tells you the values
5. Pass selected σ values to sweep_param_grid.py
6. Run Phase 1 sweep (9 runs, ~4.5 hours total)
7. After all 9 complete, generate sweep_v4_comparison.md and gait diagrams
8. Paste comparison table → WAIT for review
9. Human picks winning run (or declares "no winner") → next steps
```

**Do not commit anything during the sweep.** Per CLAUDE.md, all commits require explicit confirmation.

**Do not start Phase 1 until Phase 0 is reviewed.** The σ values matter — wrong σ wastes 4.5 hours.

## 8. Resumability

The sweep script must handle failures gracefully:

- If any individual run crashes (PhysX, OOM, etc.), log the failure with NaN metrics, continue to next run.
- If the sweep is interrupted (Ctrl+C, system reboot), `sweep_param_grid.py` should detect existing runs in the CSV and skip those, resuming from where it stopped.

This is non-negotiable — 4.5 hours is too long to lose to a single crash.

## 9. Things That Will NOT Be Touched

- Curriculum strategies (Uniform only, as in Run 2)
- Reward stack composition beyond the three sweep knobs
- Action scale (stays 0.35)
- PPO hyperparameters
- Domain randomization
- Sim parameters (dt, decimation, num_envs)
- Episode length, command resampling time

If during implementation any of these appear to need changes, **stop and report back**.

## 10. What Local Session Reports After All 9 Runs

A single markdown file `src/results/sweep_v4_comparison.md` containing:

1. The full comparison table (Section 4.4)
2. Top 3 runs by tracking quality (sorted by sum of |track_err| across bins)
3. Top 3 runs by gait differentiation (sorted by duty(b0) - duty(b4))
4. The "winner" — single run that satisfies all three acceptance criteria, or "no winner" with explanation
5. The 9 gait diagrams as embedded images

Plus the raw CSV for further analysis if needed.

## 11. Time Budget Honest Note

9 runs × 30 min = 4.5 hours of training. Add ~30 min for eval/plotting per run = ~7 hours total wall-clock for the full sweep. The implementer should run it overnight or during a long break, not synchronously while watching.

If wall-clock per run exceeds 45 min, stop the sweep — something has slowed down vs Run 2's baseline. Diagnose throughput before continuing.

## 12. Post-Sweep Decisions

Three possible outcomes:

**Outcome A: One run wins all three criteria.**
→ Train seed 1 with that config to verify reproducibility (~30 min). If reproduces, write up as positive result.

**Outcome B: Best run wins on tracking but not gait.**
→ Reframe writeup: "Parameter recalibration extends tracking range; gait emergence remains an open problem."

**Outcome C: No run improves over Run 2.**
→ The parameter space within Liang's framework is exhausted on Go2 in Isaac Lab. Write up Run 2 + sweep as comprehensive negative result. Hypothesize further: action-space or simulator-level limitations.

All three outcomes are defensible. The sweep removes ambiguity.