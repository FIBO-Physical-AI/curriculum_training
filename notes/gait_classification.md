# Gait classification — canonical rules and glossary

This document defines how the Go2 quadruped's gait is classified from the
contact-trace data produced by `eval_epte.py` and `eval_ramp.py`. It is
the single source of truth for `plot_gait_classification.py` and
`plot_gait_transition.py`, and the canonical glossary for what each
named gait means in this project. When a code or paper term conflicts
with an entry here, this file wins. Any change here is a deliberate
decision; do not edit thresholds in code without updating this file.

## Reference

The whole classifier is anchored to **one** expert source:

> Hildebrand, M. (1989). *The quadrupedal gaits of vertebrates.*
> BioScience 39(11): 766–775. https://doi.org/10.2307/1310845

All gait names, all canonical phase tuples, and the tolerance value come
from this paper. Where Hildebrand does not specify a value (Stand
criterion, minimum stride count, window length), we use a stated
engineering threshold and mark it explicitly as "not from Hildebrand."

We do **not** mix in Robilliard 2007, Cartmill 2002, Margolis 2022, or
Liang 2024. Those are good papers, but blending them with Hildebrand
introduces decisions of our own that are then hard to defend. One
source, one taxonomy.

## Background

In Hildebrand's framework a quadruped gait is identified by two
stride-cycle parameters per foot.

- **Duty factor (β)** — fraction of the stride cycle that a foot is on
  the ground.
- **Limb phase (φ)** — the offset, expressed as a fraction of one stride
  cycle, between when each foot touches down and when the reference foot
  touches down. Phases live in [0, 1) modulo 1.

Hildebrand uses a 2D diagram (duty factor vs limb phase) with named
regions for each gait. Trained Go2 RL policies do not span the full duty
axis — duty stays in 0.50–0.67 across the entire 0→4 m/s ramp — so the
**limb-phase pattern** is the discriminating signal in our setting. We
match observed phases to the nearest canonical Hildebrand template and
do not use duty factor to override the phase-based label.

## Foot indexing

The contact array column order, set by Isaac Lab's body lookup
(`find_bodies(".*_foot")`) on Go2, is:

`contact[t, i]` is a boolean: foot `i` in stance at timestep `t`.

| i | Code label | Anatomical |
|---|------------|------------|
| 0 | FL         | Front-left  |
| 1 | FR         | Front-right |
| 2 | RL         | Rear-left   |
| 3 | RR         | Rear-right  |

All templates and rules below assume this order. FL is the reference
foot: its phase is defined as 0.

## Core measurements

- **Duty factor (β)** — fraction of stride one foot is on the ground.
  The classifier uses **mean β across all 4 feet** for the Stand check;
  phase classification does not depend on β.
- **Stride period** — number of timesteps between successive stance
  onsets on FL.
- **Limb phase (φ)** — for foot `i ≠ FL`, lag of foot `i`'s first
  touchdown after FL's, normalised by stride period, into `[0, 1)`.
  Estimated via touchdown-event timing (circular mean across strides).
- **Phase vector** — `[0, FR_phase, RL_phase, RR_phase] ∈ [0, 1)^4`.
- **Cyclic distance** — element-wise `min(|a-b|, 1-|a-b|)`, mean across
  feet. Distance from observed phase vector to a canonical template.

## Gait taxonomy

Five named gaits from Hildebrand 1989 plus two engineering necessities
(`Stand`, `Irregular`) and one data-quality flag (`n/a`). Phase tuples
are `[FL, FR, RL, RR]`.

| Gait          | Phase template            | Footfall meaning                          |
|---------------|---------------------------|-------------------------------------------|
| **Walk**      | `[0, 0.5, 0.75, 0.25]`    | 4-beat lateral sequence (RL→FL→RR→FR)     |
| **Trot**      | `[0, 0.5, 0.5,  0]`       | 2-beat diagonal pairs (FL+RR) (FR+RL)     |
| **Pace**      | `[0, 0.5, 0,    0.5]`     | 2-beat ipsilateral pairs (FL+RL) (FR+RR)  |
| **Bound**     | `[0, 0,   0.5,  0.5]`     | 2-beat front pair + rear pair             |
| **Pronk**     | `[0, 0,   0,    0]`       | all four feet together                    |
| **Stand**     | n/a                       | mean β > 0.95, or no detectable stride    |
| **Irregular** | n/a                       | best template cyclic distance > 0.10      |
| **n/a**       | n/a                       | < 2 stance starts on FL (insufficient data) |

Gait identity is determined by the **limb-phase pattern**. Duty factor
is not used to override the phase-based label. The only role of duty
factor in classification is the Stand check.

## Classification procedure

For one window of contact data of shape `(T, 4)` with sim timestep `dt`:

1. **Stand check** (engineering, not from Hildebrand).
   Classify as **Stand** if any of:
   - mean duty factor across all four feet > 0.95,
   - fewer than 2 stance-starts on every foot (no detectable stride),
   - **or** detected stride period > 1.0 s (50 frames at sim_dt = 0.02).
   The period cap rejects standstill bobbing: a quadruped robot in
   genuine locomotion has stride period < 1 s; longer "strides" are
   stationary weight-shifting that the framework should not label as a
   gait.

2. **Stride period.**
   Detect stance-start frames on FL (rising edges of the contact mask).
   If fewer than 2 starts on FL, return `n/a`.
   Period (in frames) = mean inter-start interval on FL.

3. **Limb phases by touchdown timing.**
   For each non-reference foot X ∈ {FR, RL, RR}:
   - Find the first stance-start of X at or after each FL stance-start.
   - Phase per stride = (X_start − FL_start) / period, taken modulo 1.
   - Foot phase = circular mean across strides.

4. **Template matching.**
   Compute cyclic distance between observed phases and each of the 5
   canonical Hildebrand templates.
   - For Walk, Trot, Pace, Bound: **mean** per-foot cyclic distance.
   - For Pronk: **max** per-foot cyclic distance. Pronk is the
     all-synchronous gait — any single off-phase foot disqualifies it.
     Without this strict rule the cyclic-distance metric can be fooled
     by phases near 1.0 (which wraps to ≈ 0) on three of four feet.

   The gait is the template with the smallest score.

5. **Irregular fallback.**
   If best mean cyclic distance > **0.10**, return **Irregular**.

## Thresholds

| Symbol              | Value  | Source                                              |
|---------------------|--------|-----------------------------------------------------|
| `PHASE_TOLERANCE`   | 0.10   | Hildebrand 1989: ±5% bin half-width per axis on the 2D diagram, translated to mean cyclic phase distance |
| `STAND_DUTY`        | 0.95   | Engineering (Hildebrand does not define Stand)      |
| `MAX_STRIDE_PERIOD` | 50 frames (1.0 s) | Engineering (rejects standstill bobbing as gait) |
| `MIN_STRIDE_STARTS` | 2      | Engineering (one full stride needed to estimate phase) |
| `WINDOW_S`          | 2.0 s  | Engineering (≥ 2 strides at the slow end of ramp)   |
| `STRIDE_S`          | 0.25 s | Engineering (sliding-window stride for transition plot) |
| `CONTACT_THRESHOLD` | 1.0 N  | Engineering (Isaac Lab contact-force noise floor)   |

## Aerial phase

Aerial fraction = fraction of frames in the window where all four feet
are off the ground. Hildebrand 1989 does **not** subdivide gaits by
aerial phase (no "Fly Trot" class), so we do not either. Aerial phase
is visible directly in the gait_classification ribbons as gaps between
all four contact bars; readers can identify a running trot by inspection.

## What this does NOT classify

Hildebrand 1989 also names gallops (transverse, rotary), half-bound,
canter, amble, and several others. We do not include them as templates
because:

- The Go2 RL policies evaluated here have not produced clean asymmetric
  gaits.
- Adding more templates without observed examples invites false
  positives near the tolerance boundary.

If a future policy genuinely produces these patterns, expect them to
land on `Irregular` or on the closest of the 5 templates. Re-introduce
the missing templates explicitly at that point.
