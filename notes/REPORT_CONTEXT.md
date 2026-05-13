# Report Writing Session Context

**Last updated:** 2026-05-13
**Active file:** `notes/Full_Final_Report.md`
**Status:** Chapters 1, 2, 3 drafted. Chapter 4 partially drafted but blocked on re-run.

---

## What the report is

Senior project for **FRA503 Deep Reinforcement Learning**, KMUTT.

**Author:** Phakin Boonchanachai (Student ID 66340500037).

**Topic:** Comparing three velocity-command sampling strategies for a single PPO policy on the Unitree Go2 quadruped, tracking forward velocity 0 to 4 m/s in Isaac Lab.

**The three conditions:**
- **Uniform** — baseline, every bin gets equal probability throughout training.
- **Task-specific** — Box Adaptive (Margolis et al., 2022, RSS). Monotonic add-only support expansion above a success threshold gamma.
- **Teacher-guided** — LP-ACRL (Li, Li, Hutter, 2026). Softmax over per-bin learning progress with temperature beta, plus uniform mixture floor epsilon.

Eight contiguous bins of width 0.5 m/s covering [0, 4] m/s. Three seeds per condition. 2048 parallel envs. PPO via rsl_rl.

**Two phases of training:**
- **Phase 1** — sprint-retune reward (five weight changes to the upstream Unitree Go2 16-term reward). Found a tracking failure at bins 6 and 7 for uniform and task-specific.
- **Phase 2** — energy-based reward added (Liang et al., 2024, cost-of-transport kernel). 3000 iterations instead of Phase 1's 6000 because the new reward plateaus earlier.

**Narrative (Option 2):** the report tells the actual journey. Phase 1 results land first, then the section that introduces gait failure as the motivation to redesign the reward, then Phase 2 results, then discussion.

**Thesis (C-prime):** reward design dominates the outcome; curriculum choice is a residual factor that only shows up at the top bin (bin 7) once the reward is fixed.

**M1 motivation:** hardware-deployable single-policy controllers. A controller usable on the real Go2 must cover the full speed range without changing controllers and without unphysical joint dynamics.

---

## Report structure

```
Chapter 1: Introduction
  1.1 Motivation
  1.2 Problem statement
  1.3 Objectives
  1.4 Scope and limitations
  1.5 Contributions
  1.6 Structure of the report

Chapter 2: Background
  2.1 PPO and Isaac Lab in brief
  2.2 Velocity-command curricula
    2.2.1 Box Adaptive (task-specific) - eqs from Margolis 2022
    2.2.2 LP-ACRL (teacher-guided) - eqs from Li 2026
    2.2.3 Structural comparison
  2.3 Reward design
    2.3.1 Structure of a locomotion reward
    2.3.2 The upstream IsaacLab Go2 16-term reward
  2.4 Synthesis

Chapter 3: Methodology
  3.1 Experimental setup and reward configuration
    3.1.1 Simulator, training budget, parallelism
    3.1.2 Velocity command space
    3.1.3 Action interface and observation
    3.1.4 Reward configuration and the standing-still derivation
  3.2 The three curriculum conditions
  3.3 Evaluation protocol
  3.4 Hyperparameter summary (Table 3.2)

Chapter 4: Results (in progress)
  4.1 Phase 1: three curricula under sprint-retune
  4.2 (pending) Gait failure discovery -> motivation for reward redesign
  4.3 (pending) Energy reward redesign (Liang 2024); 6000 -> 3000 iter justification
  4.4 (pending) Phase 2 results; Hildebrand sidebar pending explicit approval
  4.5 (pending) Discussion
  4.6 (pending) Conclusion

References
```

---

## Current status

**Drafted and reviewed:** Chapters 1, 2, 3 are complete and approved by the user.

**Drafted, partially wrong, BLOCKED:** Section 4.1 was drafted with mechanism explanations under each table and figure. During verification against the actual eval data (`src/results_update1/epte_sp.csv`, `eval_traces/*.npz`, `figures/convergence.txt`):

| Claim in current §4.1 | Real value | Status |
|---|---|---|
| Uniform bin 5 survival ~75% | 88% (264/300 time-out, 36/300 fell) | wrong |
| Task-specific bin 5 survival ~95% | 100% | wrong |
| Uniform/task-specific bin 6 fall within 1 second | uniform 0.38s, task-specific 5s | wrong (task-specific) |
| Achieved v_x at bin 6 for task-specific "<= 0.2 m/s" | 3.48 m/s peak (near command), then falls | wrong |
| Teacher bin 6 survival ~100% | 91% (27/300 fell) | wrong |
| Teacher bin 7 survival ~67% | 67% (200/300 time-out) | correct |
| Teacher masters bins 5/6/7 before bin 4 | 200, 205, 225 vs 335 iters | correct |
| 0.4 floor on uniform/task-specific bins 6/7 | real number from convergence.txt; eval kernel = 0.000 | real but interpreted as training-buffer artefact (working hypothesis from notes/project/full_report.md line 555) |

**Decided:** the §4.1 numbers are wrong enough, and the underlying training logs are gone, that fixing in-place is not enough. Re-run is the cleanest path. Also re-run Phase 2 for the same reason (the same training-time signal vs eval-time discrepancy exists there too per notes/project/full_report.md lines 551-557).

**Banned topic for now:** the word *gait* (and *trot*, *gallop*, *bound*, *regime boundary* as a phenomenon) is removed from §4.1 until the user explicitly authorises reintroducing it. Hildebrand gait taxonomy is provisionally planned for §4.4 but only if explicitly approved.

---

## Rules and agreements (durable, observe across sessions)

### From CLAUDE.md (project root)

- **Never push.** Never add Claude as author or co-author. Ask before every commit. Exclude CLAUDE.md, *.pdf, *.tex from all commits.
- **No comments in code.** When generating or editing code, write no comments.
- **Figures plot data only.** No text on the figure that interprets, judges, or explains the data. No "lower is better," no threshold callouts, no causal text. Interpretation goes in the chat reply, never on the figure surface.
- **Verify before asserting.** Do not guess. Do not claim a task is complete without verification. Cite sources or check the code.

### Report style

- **No em dash** anywhere in the report.
- Simple wording. Detail preserved but no wall of text.
- **No code in the markdown.** No backtick formatting for module or parameter names. Write them in plain text.
- **Exact paper notation** when citing eqs from Margolis 2022 and Li 2026. Do not paraphrase notation.
- **Header:** Author "Phakin Boonchanachai (66340500037)" and Course "FRA503 Deep Reinforcement Learning". No date in the header.
- **Naming convention throughout:** Box Adaptive is referred to as the **task-specific** curriculum. LP-ACRL is referred to as the **teacher-guided** curriculum. Baseline is **uniform**. These names are introduced in §1.1 and used consistently after that.
- **Option 2 narrative:** Phase 1 results land first in §4.1, then §4.2 introduces the gait failure as the discovery that motivates redesign, then §4.3 redesigns the reward, then §4.4 shows Phase 2 results.
- **Caveman tag:** when the user types `/caveman`, they want the answer in simple short language with no jargon. Used to test explanations are honest and grounded.

### Citation conventions

- Citations are in-line `(Author Year)`.
- References at end of report, alphabetical, with arXiv links.
- Already in references: Li 2026, Margolis 2022, Mittal 2023, Rudin 2022, Schulman 2017, Unitree Robotics 2024.
- Pending references: Liang et al. 2024 (energy reward), arXiv:2403.20001.

### Honesty rules (learned this session)

- Do not assert a mechanism from intuition. If the data does not support it, say so.
- Do not narrate survival or fall numbers from memory. Read them from `src/results_update1/epte_sp.csv`.
- Where the project's prior writeup (`notes/project/full_report.md`) already documents a hypothesis (for example the pre-crash transient spike explanation for the 0.4 floor), attribute the hypothesis to that prior writeup rather than restating it as fact.

---

## Decisions made about the re-run

The user authorised a full re-run of Phase 1 and Phase 2 because:
1. Several §4.1 claims contradict the saved eval data.
2. The raw `curriculum.csv` training logs were deleted, so claims about mechanism (the 0.4 floor, teacher's wiggly curve) cannot be verified against the logs.
3. The same artefact contaminates Phase 2 per `notes/project/full_report.md` lines 551-557, so Phase 2 also needs re-logging.

**Before re-running**, the user wants me to **verify the logging code** so the next run captures everything the report needs.

Re-run scope to verify before kicking off:
- The training writes a `curriculum.csv` that contains: PPO step, per-bin sampling weight at that step, per-bin tracking signal (mean per-bin kernel value), per-bin sample count.
- Additionally needed for this report (to be confirmed): per-bin episode count and fall count during training (not just at eval), so we can compare training fall rate to eval fall rate and properly explain the floor without speculation.
- Eval pipeline writes `epte_sp.csv` and `eval_traces/*.npz` cleanly.
- All figures regenerate from these logs without manual surgery.

---

## Outstanding task list

1. **Verify logging code** for Phase 1 and Phase 2 re-runs. Confirm `curriculum.csv` columns, eval CSV columns, npz keys. Decide if any new diagnostic logging (e.g., per-bin training fall count) is needed and add it before running.
2. **Re-run Phase 1** (uniform, task-specific, teacher; 3 seeds each; 6000 iters). Archive logs and figures to `src/results_phase1_rerun/`.
3. **Re-run Phase 2** (same conditions, 3000 iters under energy reward). Archive to `src/results_phase2_rerun/`.
4. **Re-generate all figures** from the new logs.
5. **Rewrite §4.1** using the real numbers, with the gait-language ban still in effect (do not reintroduce gait words until user permission).
6. **Write §4.2** (gait failure discovery; first place the gait taxonomy may enter, only with explicit permission).
7. **Write §4.3** (energy reward redesign; include the 6000 to 3000 iteration justification, the Phase 0 sigma diagnostic, and the Phase v4 3x3 training grid).
8. **Write §4.4** (Phase 2 results; Hildebrand sidebar only if explicitly approved).
9. **Write §4.5** (discussion) and **§4.6** (conclusion).
10. **Add Liang et al. 2024** to References (arXiv:2403.20001).
11. **Update Table of Contents** with full §4 subsection list once §4 is finalised.

---

## Files to consult at the start of next session

- `notes/Full_Final_Report.md` — current report draft.
- `notes/project/update_progress_1.md` — Phase 1 source-of-truth tables (Table 6 reward, Table 7 EPTE-SP).
- `notes/project/update_progress_2.md` — Phase 2 source (energy reward equation, sigma calibration).
- `notes/project/full_report.md` — prior consolidated writeup; the pre-crash transient-spike hypothesis for the 0.4 floor is at line 555.
- `src/results_update1/figures/convergence.txt` — per-bin convergence status, the source of Table 4.1 numbers.
- `src/results_update1/epte_sp.csv` — per-rollout termination cause and metrics. Use this for survival numbers, NOT npz alive-length.
- `src/results_update1/eval_traces/*.npz` — per-step velocity and contact traces. npz pads to 1000 steps; use `fall_step` from epte_sp.csv to find the actual alive window.
- `src/source/curriculum_rl/curriculum_rl/envs/mdp.py` — `velocity_curriculum_step` function (line 101) is what writes `curriculum.csv`.
- `src/source/curriculum_rl/curriculum_rl/curricula/teacher_guided.py` — LP-ACRL implementation; verifies the "wiggle = LP softmax taking turns" mechanism structurally.

---

## What is not in scope for this report

- Hardware deployment, sim-to-real.
- Terrain (flat ground only).
- Lateral velocity or yaw rate (held at zero).
- Cross-platform validation (Go2 only).
- Alternative algorithms (PPO only).
- Hyperparameter sweeps beyond the three curricula and three seeds.
