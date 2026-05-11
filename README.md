# curriculum_training

Curriculum-learning study for velocity-tracking on the Unitree Go2 quadruped, built on Isaac Lab + RSL-RL. Three command-sampling strategies are compared under an otherwise identical pipeline.

| Condition       | Task ID                              | Source                              |
| --------------- | ------------------------------------ | ----------------------------------- |
| Uniform         | `Curriculum-Go2-Velocity-Uniform`    | baseline, no curriculum             |
| Task-specific   | `Curriculum-Go2-Velocity-TaskSpec`   | Margolis et al. 2022 (Box Adaptive) |
| Teacher-guided  | `Curriculum-Go2-Velocity-Teacher`    | Li et al. 2026 (LP-ACRL)            |

Task space: forward velocity in `[0, V_MAX] = [0, 4.0]` m/s, split into `NUM_BINS = 8` bins of width `0.5` m/s. Lateral velocity and yaw rate are forced to zero by the command class — the policy only ever sees a forward-velocity command.

## Install

From the repository root, with the Isaac Lab python environment active:

```bash
cd src/source/curriculum_rl
pip install -e .
```

Depends on `unitree_rl_lab/` (sibling directory) being installed first. Do not modify `unitree_rl_lab/`; all overrides are applied in-process from `curriculum_rl`.

## Run

All launchers `import curriculum_rl` (which triggers `gym.register`) and then hand off to upstream `unitree_rl_lab/scripts/rsl_rl/{train,play}.py` via `runpy` in the same process. Any flag not listed below is passed through verbatim.

### Train one condition × seed

```bash
python src/scripts/train.py --condition uniform       --seed 0 --headless --max_iterations 6000
python src/scripts/train.py --condition task_specific --seed 0 --headless
python src/scripts/train.py --condition teacher       --seed 0 --headless
```

Logs land under `unitree_rl_lab/logs/rsl_rl/<experiment>/<run_id>/` (the launcher chdirs there to match upstream convention). Each run also writes [`curriculum.csv`](src/source/curriculum_rl/curriculum_rl/envs/mdp.py) with per-bin `(weight, mean_reward, n_samples, r_lin, r_en)` per PPO iteration.

### Sweep — train + eval + per-bin video + ramp + plot

```bash
bash src/scripts/run_sweep.sh
```

Pipeline per `(condition, seed)`: `train.py` → `eval_epte.py` → per-bin `play.py --bin B --video` → after all conditions, pick best seed per condition → `eval_ramp.py` → `plot_all.py`.

| Env var | Default | Effect |
| --- | --- | --- |
| `CONDITIONS` | `"uniform task_specific teacher"` | space-separated list |
| `SEEDS` | `"0"` | space-separated list |
| `MAX_ITERATIONS` | `3000` | passed to `train.py` |
| `NUM_ENVS` | `4096` | passed to `train.py` |
| `STEPS_PER_ITER` | `48` | exported as `CURRICULUM_STEPS_PER_ITER` (curriculum update cadence) |
| `NUM_BINS` | `8` | task-space bin count for video sweep |
| `V_MAX` | `4.0` | top of velocity range for video sweep |
| `RECORD_VIDEOS` | `1` | `0` to skip per-bin and ramp video capture |
| `VIDEO_ENVS` | `1` | env count during play |
| `VIDEO_LENGTH` | `200` | frames per per-bin video |

Outputs: `src/results/run_timings.txt` (per-step START/STOP markers), `src/results/epte_sp.csv` (per condition × seed × bin), `src/results/videos/<cond>_seed<N>/binB_v<vc>.mp4`, `src/results/ramp_<cond>_seed<N>.npz`, `src/results/figures/*.png`. Markers in `.sweep_runs/<cond>_seed<N>.path` record the chosen log directory per run.

### Reward-shaping sweep (σ_en_x × std × α_en grid)

[`src/scripts/run_sweep_v4.sh`](src/scripts/run_sweep_v4.sh) — 9-run grid over `(σ_en_x, std, α_en)` for the `uniform` condition only:

```bash
bash src/scripts/run_sweep_v4.sh SIGMA_LOW SIGMA_MID SIGMA_HIGH
RESUME=1 bash src/scripts/run_sweep_v4.sh 250 500 1000        # skip runs whose eval CSV exists
```

| Run | σ_en_x | std | α_en |
| ---:| ------ | --- | ---- |
| 1   | LOW    | 0.5 | 1.0  |
| 2   | MID    | 0.5 | 1.0  |
| 3   | HIGH   | 0.5 | 1.0  |
| 4   | LOW    | 1.0 | 1.0  |
| 5   | MID    | 1.0 | 1.0  |
| 6   | HIGH   | 1.0 | 1.0  |
| 7   | LOW    | 1.0 | 0.5  |
| 8   | MID    | 1.0 | 0.5  |
| 9   | HIGH   | 1.0 | 0.5  |

`σ_en_z` is fixed at `0.5 · σ_en_x`. Each run:
1. exports `SWEEP_SIGMA_EN_X`, `SWEEP_SIGMA_EN_Z`, `SWEEP_TRACK_STD`, `SWEEP_ENERGY_WEIGHT`
2. trains for `MAX_ITERATIONS` (default `3000`) with `NUM_ENVS` (default `4096`)
3. evals `ROLLOUTS_PER_BIN` (default `100`) per bin via `eval_epte.py`
4. appends a row to `src/results/sweep_v4.csv` via `sweep_param_grid.py --append-run`

After all runs: `sweep_param_grid.py --compare` regenerates `src/results/sweep_v4_comparison.md`.

Three modes of `sweep_param_grid.py`:

| Mode | Flags | Purpose |
| --- | --- | --- |
| default | `--sigma-low <σ> --sigma-mid <σ> --sigma-high <σ> [--resume]` | full 9-run train+eval orchestrator (alternate to `run_sweep_v4.sh`) |
| append-run | `--append-run --run-id <i> --sigma <σ> --std <s> --alpha-en <α> --eval-csv <p> --trace-npz <p> --wall-sec <t> --ckpt <p>` | called from `run_sweep_v4.sh` per run |
| compare | `--compare` | regenerate markdown table from existing `sweep_v4.csv` |

### σ_en_x diagnostic (pick σ before sweeping)

```bash
python src/scripts/sweep_sigma_diagnostic.py --num_envs 512 --num_steps 1000
```

Runs random-action rollouts and prints `mean(R_en)` for `σ_en_x ∈ {50, 100, 250, 500, 1000, 2000, 5000}` (with `σ_en_z = 0.5 σ_en_x`, `ε = 0.1`). Pick σ values whose `mean(R_en)` lands in `[0.30, 0.75]` ("useful gradient" regime). Output → `src/results/phase0_table.txt`.

### Per-bin playback

```bash
python src/scripts/play.py --condition uniform
python src/scripts/play.py --condition teacher --bin 7 --video --video_length 400
```

`--bin N` sets `CURRICULUM_PLAY_BIN=N` so the sampler returns `v = (N + 0.5) * bin_width`.

### Eval — EPTE-SP per (condition, seed, bin)

```bash
python src/scripts/eval_epte.py --condition teacher --seed 0 \
    --checkpoint unitree_rl_lab/logs/rsl_rl/curriculum_go2_velocity_teacher/<run_id>/model_*.pt
```

Appends rows to `src/results/epte_sp.csv` and writes per-trace npz to `src/results/eval_traces/`.

### Ramp evaluation (0 → 4 m/s linear ramp)

```bash
python src/scripts/eval_ramp.py --condition teacher --seed 2 \
    --checkpoint <path-to-model.pt> \
    --v-start 0.0 --v-end 4.0 --duration 30.0 --hold-duration 10.0 \
    --video
```

Writes `src/results/ramp_<condition>_seed<N>.npz` (contact, joint, base-vel traces) consumed by [`plot_gait_transition.py`](src/source/curriculum_rl/curriculum_rl/figures/plot_gait_transition.py).

### Reward-shaping diagnostics

```bash
python src/scripts/dump_reward_weights.py                                             # prints reward weights for Curriculum-Go2-Velocity-Uniform (hardcoded)
python src/scripts/verify_liang_composite.py --num_envs 16 --num_steps 100
python src/scripts/sweep_sigma_diagnostic.py --num_envs 512 --num_steps 1000
python src/scripts/sweep_param_grid.py --sigma-low <σ> --sigma-mid <σ> --sigma-high <σ> --resume
```

`verify_liang_composite.py` recomputes `R_motion`, `R_en`, `R_aux` term-by-term in Python and asserts they match the in-env tensor; `sweep_sigma_diagnostic.py` prints `power / (σ_x|v_x| + σ_z|ω_z|)` statistics over a rollout to pick `σ_en_x`, `σ_en_z`.

### Plot all figures

```bash
python src/scripts/plot_all.py
```

## Equations and where they live

### Active total reward (additive form, current default)

Configured in [`go2_velocity_base.py:54 _apply_liang_additive_energy`](src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py#L54). Total per-env per-step reward is the weighted sum of the upstream `RewardManager` terms after the override:

$$r_t = \underbrace{1.5 \cdot e^{-\|v_{xy} - v_{xy}^{\text{cmd}}\|^2 / \text{std}^2}}_{\text{track\_lin\_vel\_xy}}
      + \underbrace{0.75 \cdot e^{-(\omega_z - \omega_z^{\text{cmd}})^2 / \text{std}^2}}_{\text{track\_ang\_vel\_z}}
      + \underbrace{\alpha_{\text{en}} \cdot R_{\text{en}}(\sigma^{en}_x, \sigma^{en}_z)}_{\text{energy (Liang CoT)}}
      + \sum_k w_k\, q_k$$

The energy term `R_en` uses [`liang_composite_reward.py:25 energy_cot`](src/source/curriculum_rl/curriculum_rl/envs/liang_composite_reward.py#L25):

$$R_{\text{en}} = \exp\!\left(-\frac{\sum_j |\dot q_j|\,|\tau_j|}{\max(\sigma^{en}_x |v_x| + \sigma^{en}_z |\omega_z|,\;\varepsilon)}\right),\quad \varepsilon = 0.1$$

The remaining (negative) auxiliary terms `w_k q_k` are pass-through from upstream with re-tuned weights (table below). The fully **multiplicative** Liang form `(R_lin + α_ang R_ang + α_en R_en) · e^(-R_aux)` is implemented in `composite_liang_energy_reward` for reference and verification but is **not** wired into the live config — only `energy_cot` from that file is plugged into `cfg.rewards.energy.func`.

Hyperparameters (set per-run via env vars, read at `__post_init__` time):

| Env var | Default | Symbol | Used by |
| --- | --- | --- | --- |
| `SWEEP_SIGMA_EN_X` | `1000.0` | `σ_en_x` | `R_en` denominator |
| `SWEEP_SIGMA_EN_Z` | `500.0`  | `σ_en_z` | `R_en` denominator |
| `SWEEP_TRACK_STD`  | `0.5`    | `std`    | tracking Gaussian width |
| `SWEEP_ENERGY_WEIGHT` | `1.0` | `α_en`   | `cfg.rewards.energy.weight` |

### Velocity tracking — linear (curriculum signal only, NOT the live reward)

[`mdp.py:31 track_lin_vel_x_linear`](src/source/curriculum_rl/curriculum_rl/envs/mdp.py#L31)

$$r_{\text{lin}}^{\text{linear}} = \frac{\min\!\bigl(\max(0,\; v_x \cdot \mathrm{sign}(v_x^{\text{cmd}})),\; |v_x^{\text{cmd}}|\bigr)}{\max(|v_x^{\text{cmd}}|,\; 0.1)}$$

This is a 0–1 progress signal consumed by [`mdp.py:101 velocity_curriculum_step`](src/source/curriculum_rl/curriculum_rl/envs/mdp.py#L101) for the curriculum update; the **live PPO reward** uses the Gaussian `track_lin_vel_xy` term shown in "Active total reward" above.

### Liang composite reward — full multiplicative form (reference, not active)

[`liang_composite_reward.py:45 composite_liang_energy_reward`](src/source/curriculum_rl/curriculum_rl/envs/liang_composite_reward.py#L45)

$$R_{\text{lin}} = \exp\!\left(-\frac{(v_x - v_x^{\text{cmd}})^2 + (v_y - v_y^{\text{cmd}})^2}{\sigma_v}\right)$$

$$R_{\text{ang}} = \exp\!\left(-\frac{(\omega_z - \omega_z^{\text{cmd}})^2}{\sigma_\omega}\right)$$

$$R_{\text{en}} = \exp\!\left(-\frac{\sum_j |\dot q_j| \cdot |\tau_j|}{\max(\sigma^{en}_x |v_x| + \sigma^{en}_z |\omega_z|,\; \varepsilon)}\right)$$

$$R_{\text{aux}} = \mathrm{clip}\!\Bigl(\sum_k w_k\, q_k,\; 0,\; r_{\text{aux}}^{\max}\Bigr) \quad\text{(7 terms)}$$

$$R = \bigl(R_{\text{lin}} + \alpha_{\text{ang}} R_{\text{ang}} + \alpha_{\text{en}} R_{\text{en}}\bigr)\, e^{-R_{\text{aux}}}$$

Defaults: `σ_v = σ_ω = 0.25`, `α_ang = 0.5`, `σ_en_x = 1000`, `σ_en_z = 500`, `α_en = 1.0`, `ε = 0.1`, `r_aux_clip = 10`. The 7 auxiliary terms (`dof_pos_limits`, `action_rate`, `undesired_contacts`, `flat_orientation`, `base_lin_vel_z`, `base_ang_vel_xy`, `feet_slide`) and their weights `W_DOF_POS_LIMITS = 10.0`, `W_ACTION_RATE = 0.1`, `W_UNDESIRED_CONTACTS = 1.0`, `W_FLAT_ORIENTATION = 2.5`, `W_BASE_LIN_VEL_Z = 2.0`, `W_BASE_ANG_VEL_XY = 0.05`, `W_FEET_SLIDE = 0.1` are at the top of the same file.

Verify the Python implementation matches the in-env tensor:

```bash
python src/scripts/verify_liang_composite.py --num_envs 16 --num_steps 100
```

### Auxiliary reward weights (the `Σ w_k q_k` part of the active reward)

Set inside [`go2_velocity_base.py:54 _apply_liang_additive_energy`](src/source/curriculum_rl/curriculum_rl/envs/go2_velocity_base.py#L54):

| Term (`cfg.rewards.<name>`) | Weight | vs upstream |
| --- | ---: | --- |
| `track_lin_vel_xy` | `+1.5` | Gaussian, `std = SWEEP_TRACK_STD` (default `0.5`) |
| `track_ang_vel_z` | `+0.75` | inherited |
| `energy` (`func = energy_cot`) | `+SWEEP_ENERGY_WEIGHT` (default `1.0`) | new term, `σ` from `SWEEP_SIGMA_EN_X / _Z` |
| `action_rate` | `−0.005` | 20× weaker |
| `joint_acc` | `−1e-7` | 2.5× weaker |
| `joint_torques` | `−2e-5` | 10× weaker |
| `joint_vel` | `−1e-4` | 10× weaker |
| `feet_air_time` | `0.0` | disabled |
| `air_time_variance` | `0.0` | disabled |
| `JointPositionAction.scale` | `0.35` | 1.4× larger action range |

All other upstream reward terms and PPO/network/observation/termination configs pass through unchanged.

### Curriculum updates

[`curricula/base.py:9 CurriculumBase`](src/source/curriculum_rl/curriculum_rl/curricula/base.py#L9) — abstract base; `sample(n, rng)` draws bins by `weights / sum(weights)` and returns bin centers.

[`curricula/uniform.py:12 UniformCurriculum.update`](src/source/curriculum_rl/curriculum_rl/curricula/uniform.py#L12) — no-op; weights stay uniform.

[`curricula/task_specific.py:25 TaskSpecificCurriculum.update`](src/source/curriculum_rl/curriculum_rl/curricula/task_specific.py#L25) — Margolis Box Adaptive: when bin `b` has both `episode_count[b] ≥ min_episodes_per_bin` AND `r_b ≥ γ`, raise `weights[b], weights[b±1]` by `0.2` (capped at `1.0`). Defaults: `γ = 0.7`, `seed_bin = 0`, `min_episodes_per_bin = 50`.

[`curricula/teacher_guided.py:46 TeacherGuidedCurriculum.update`](src/source/curriculum_rl/curriculum_rl/curricula/teacher_guided.py#L46) — Li LP-ACRL. Every `stage_length` PPO iterations:

$$\text{LP}_b = \bar r_b^{\text{stage}} - \bar r_b^{\text{prev}}, \qquad
w = (1-\varepsilon)\,\mathrm{softmax}\!\left(\frac{\text{LP}}{\beta}\right) + \frac{\varepsilon}{N_{\text{bins}}}$$

Runtime defaults (passed via `BinnedVelocityCommandCfg`): `β = 0.05`, `ε = 0.15`, `stage_length = 50`, `seed_bin = 0` initialized with `w[0] = 1.0`, `w[1] = 0.5`. (The `TeacherGuidedCurriculum.__init__` fallback for `ε` is `0.05`, but the cfg-level `eps = 0.15` is what actually reaches the curriculum at runtime — see [`commands.py:101`](src/source/curriculum_rl/curriculum_rl/envs/commands.py#L101).)

### Curriculum driver (called by Isaac Lab manager)

[`mdp.py:101 velocity_curriculum_step`](src/source/curriculum_rl/curriculum_rl/envs/mdp.py#L101) — accumulates per-bin `r_lin` and `R_en` from `_episode_sums / episode_length / weight`, fires `curriculum.update(per_bin, step)` every `CURRICULUM_STEPS_PER_ITER` steps (default 48), syncs back to GPU weights, and appends a row to `curriculum.csv`.

### Binned sampling

[`commands.py:27 BinnedVelocityCommand`](src/source/curriculum_rl/curriculum_rl/envs/commands.py#L27) overrides `UniformVelocityCommand`. On `_resample_command` it samples bin index from `weights`, sets `vel_command_b[:, 0] = bin_centers[idx]`, and zeros lateral and yaw. `CURRICULUM_PLAY_BIN` env var overrides sampling (used by `play.py --bin N`).

### EPTE-SP metric

[`eval/epte_sp.py:15 epte_sp_one`](src/source/curriculum_rl/curriculum_rl/eval/epte_sp.py#L15)

$$\mathrm{EPTE\text{-}SP} = \frac{\epsilon \cdot k_f + (K - k_f)}{K},\quad
\epsilon = \mathrm{clip}(\mathrm{tracking\_error},\;0,\;1)$$

with `K = episode_length`, `k_f = fall_step` (or `K` if no fall). Higher = worse: a fall before time `K` adds `1` per post-fall step.

### Hildebrand 2D gait classification

[`figures/plot_gait_classification.py:253 classify_gait_2d`](src/source/curriculum_rl/curriculum_rl/figures/plot_gait_classification.py#L253). Pipeline:

1. `_phase_tuple(contact)` extracts per-foot phase offsets `[FL=0, FR, RL, RR]` from stance starts and the FL stride period.
2. `_lateral_phases` computes `φ_L = (FL − RL) mod 1`, `φ_R = (FR − RR) mod 1`, then `φ_LH = circular_mean(φ_L, φ_R)`.
3. `_symmetric_label` snaps `φ_LH` to the nearest of `CANONICAL_POINTS` (Pace, Walk-LS, Trot, Walk-DS) within `PHI_TOLERANCE = 0.10`, otherwise falls into the matching quadrant `LSLC / LSDC / DSDC / DSLC`.
4. `_asymmetric_label` (when cyclic distance `dφ = cyclic(φ_L, φ_R) ≥ LR_ASYMMETRY_THRESHOLD = 0.20`) picks the closest of 14 templates in `GAIT_TEMPLATES` (Bound, Pronk, Half-bound, Canter, Transverse / Rotary gallop, ...).
5. `_beta_qualifier(β)` appends "(walking)" if duty `β > 0.5`, "(running)" otherwise.
6. `_dap_subtag(φ_L, φ_R)` adds `+DAP / -DAP / ±DAP` when `max(|φ_L − 0.5|, |φ_R − 0.5|) ≥ DAP_DISSOCIATION_THRESHOLD = 0.05` — i.e. at least one diagonal pair is dissociated from the perfect-trot point `0.5` (sign of both `φ_L − 0.5` and `φ_R − 0.5` decides `+/−/±`).

Public API: `classify_gait(contact) -> label`, `classify_gait_with_score(contact) -> (label, score)`.

### Repository layout

| Path                                                                                   | Role                                            |
| -------------------------------------------------------------------------------------- | ----------------------------------------------- |
| [src/source/curriculum_rl/](src/source/curriculum_rl/)                                 | Isaac Lab extension package                     |
| [src/source/curriculum_rl/curriculum_rl/curricula/](src/source/curriculum_rl/curriculum_rl/curricula/) | 3 sampling strategies                |
| [src/source/curriculum_rl/curriculum_rl/envs/](src/source/curriculum_rl/curriculum_rl/envs/)           | Go2 env cfgs, command class, MDP, Liang reward |
| [src/source/curriculum_rl/curriculum_rl/eval/](src/source/curriculum_rl/curriculum_rl/eval/)           | EPTE-SP, per-bin return, sampling heatmap, iters-to-mastery |
| [src/source/curriculum_rl/curriculum_rl/figures/](src/source/curriculum_rl/curriculum_rl/figures/)     | plot scripts                                   |
| [src/scripts/](src/scripts/)                                                           | train / eval / sweep / play / ramp launchers   |
| [src/results/](src/results/)                                                           | gait-emergence (update 2) results              |
| [src/results_update1/](src/results_update1/)                                           | progress update 1 results (snapshot)           |
| [unitree_rl_lab/](unitree_rl_lab/)                                                     | upstream Isaac Lab extension (untouched)       |
| [unitree_model/](unitree_model/)                                                       | Unitree robot USD/URDF assets                  |

## Platform

- Isaac Lab + Isaac Sim 4.5
- Python 3.10, PyTorch (CUDA)
- RSL-RL ≥ 2.3.1
- Unitree Go2 (12 actuated joints)
