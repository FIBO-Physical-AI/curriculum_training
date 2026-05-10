# Gait classification — canonical rules and glossary

This document defines how the Go2 quadruped's gait is classified from
contact-trace data produced by `eval_epte.py` and `eval_ramp.py`. It is
the single source of truth for `plot_gait_classification.py` and
`plot_gait_transition.py`, and the canonical glossary for what each
named gait means in this project. When a code or paper term conflicts
with an entry here, this file wins. Any change here is a deliberate
decision; do not edit thresholds in code without updating this file.

## References

The classifier uses Hildebrand's 2D `(β, φ_LH)` representation, with
region boundaries and quadrant naming taken from successor papers that
formalised what Hildebrand drew. Diagonal-dissociation phenomenology is
named per the equine biomechanics literature.

- Hildebrand, M. (1989). *The quadrupedal gaits of vertebrates.*
  BioScience 39(11): 766–775.
  <https://doi.org/10.2307/1310845>
- Hildebrand, M. (1977). *Analysis of asymmetrical gaits.*
  Journal of Mammalogy 58(2): 131–156.
  (asymmetric-gait extension; canter / gallop / bound / half-bound)
- Cartmill, M., Lemelin, P., Schmitt, D. (2002).
  *Support polygons and symmetrical gaits in mammals.*
  Zool. J. Linn. Soc. 136: 401–420.
  (formalises lateral- vs diagonal-sequence quadrant boundaries)
- Janis, C. M., Shoshitaishvili, B., Kambic, R., Figueirido, B. (2021).
  *Evolutionary history of quadrupedal walking gaits shows mammalian
  release from locomotor constraint.* Proc. R. Soc. B 288: 20210937.
  <https://royalsocietypublishing.org/doi/10.1098/rspb.2021.0937>
  (LSLC / LSDC / DSDC / DSLC quadrants on the symmetric-gait circle)
- Robilliard, J. J., Pfau, T., Wilson, A. M. (2007).
  *Gait characterisation and classification in horses.*
  J. Exp. Biol. 210: 187–197.
  <https://journals.biologists.com/jeb/article/210/2/187/17107>
  (footfall-pairing definitions for trot / pace / canter / gallop /
  walk / tölt; LDA-validated 95% accuracy)
- Starke, S. D. & Clayton, H. M. (2015).
  *An exploration of the influence of diagonal dissociation and
  moderate changes in speed on locomotor parameters in trotting
  horses.* PeerJ. <https://pmc.ncbi.nlm.nih.gov/articles/PMC4933092>
  (diagonal advanced placement; named feature of trot, not a separate
  gait)
- Audoin-Maddison et al. (2009).
  *Walk–run classification of symmetrical gaits in the horse: a
  multidimensional approach.* J. R. Soc. Interface 6: 35–47.
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC2658658>
  (empirical β cutoff between walking and running gaits)

The taxonomy below is derived from these sources. Where multiple
sources agree, we take the value uniformly. Where they disagree (e.g.
horse vs robot β at trot), we note the disagreement and choose
according to the platform.

## Background — the 2D `(β, φ_LH)` representation

Hildebrand 1989 plots quadruped gait on a 2D plane:

- **Hindlimb duty factor β** — fraction of stride a hind foot is on
  the ground. Vertical axis. Walking gaits sit at β > 0.5; running
  gaits at β ≤ 0.5.
- **Lateral limb phase φ_LH** — fraction of stride by which a forelimb
  lags the ipsilateral hindlimb on the same side, in `[0, 1)`.
  Horizontal axis. Quoting Hildebrand: *"the proportion of stride
  interval that footfall of forefoot follows hind on same side."*

Canonical points on the φ axis:

| `φ_LH` | Symmetric gait at this point                       |
|-------:|----------------------------------------------------|
|   0.00 | **Pace** (ipsilateral pairs synchronous)           |
|   0.25 | **Lateral-sequence singlefoot** (4-beat, even)     |
|   0.50 | **Trot** (diagonal pairs synchronous)              |
|   0.75 | **Diagonal-sequence singlefoot** (4-beat, even)    |
|   1.00 | **Pace** (wraps to 0)                              |

Quadrants between canonical points (Cartmill 2002; Janis 2021):

| `φ_LH` range  | Quadrant | Meaning                                   |
|---------------|----------|-------------------------------------------|
| `(0,    0.25)`| LSLC     | Lateral-sequence, lateral-couplet (close-paced) |
| `(0.25, 0.50)`| LSDC     | Lateral-sequence, diagonal-couplet (close-trot, the canonical horse walk) |
| `(0.50, 0.75)`| DSDC     | Diagonal-sequence, diagonal-couplet (some primates) |
| `(0.75, 1.00)`| DSLC     | Diagonal-sequence, lateral-couplet (rare in nature) |

Asymmetric gaits (canter, gallop, bound, pronk, half-bound) do not
live on the symmetric `(β, φ_LH)` plane — their footfalls are not
left-right symmetric. They are detected by an asymmetry test (see
classification procedure §6) and labelled by 4-foot template matching
in the spirit of Hildebrand 1977.

## Foot indexing

The contact array column order, set by Isaac Lab's body lookup
(`find_bodies(".*_foot")`) on Go2, is:

| i | Code | Anatomical |
|---|------|------------|
| 0 | FL   | Front-left  |
| 1 | FR   | Front-right |
| 2 | RL   | Rear-left   |
| 3 | RR   | Rear-right  |

`contact[t, i]` is a boolean: foot `i` in stance at timestep `t`. All
templates and rules below assume this order. **FL is the reference
foot**: its phase is defined as 0.

## Core measurements

For one window of contact data of shape `(T, 4)` with sim timestep
`dt`:

- **Stride period.** Mean inter-stance-onset interval on FL. If fewer
  than 2 onsets on FL, the window is `n/a`.
- **Per-foot phase φ_i.** For each non-reference foot, the lag of its
  first stance onset after each FL onset, divided by stride period,
  taken modulo 1, circular-mean across strides. Phase tuple
  `[φ_FL, φ_FR, φ_RL, φ_RR] ∈ [0, 1)^4` with `φ_FL = 0`.
- **Mean duty factor β.** Mean across all four feet of (frames in
  stance) / (total frames). Robot proxy for Hildebrand's hindlimb β;
  the two are within 0.02 in our data.
- **Lateral limb phase φ_LH.** Circular mean of the two same-side fore-after-hind lags:
  `φ_L = (φ_FL − φ_RL) mod 1`, `φ_R = (φ_FR − φ_RR) mod 1`,
  `φ_LH = circular_mean(φ_L, φ_R)`.
- **LR asymmetry δφ.** Cyclic distance `min(|φ_L − φ_R|, 1 − |φ_L − φ_R|)`.
  Symmetric gaits sit near 0; asymmetric gaits well above.
- **DAP per side.** `DAP_L = φ_L − 0.5`, `DAP_R = φ_R − 0.5`.
  Positive = hind-first (positive diagonal advanced placement);
  negative = fore-first. Used as a sub-label on Trot.
- **Aerial-phase fraction.** Fraction of frames where all four feet
  are off the ground.

## Gait taxonomy

### Symmetric gaits (12 cells: 4 quadrants × walking/running × pace)

A symmetric gait has δφ < `LR_ASYMMETRY_THRESHOLD` (see §Thresholds).
The label is `(quadrant or canonical name) + (walking | running)
qualifier`:

| Label                          | β     | φ_LH      | Phase template (4-tuple)    |
|--------------------------------|-------|-----------|-----------------------------|
| Pace (walking)                 | > 0.5 | ≈ 0 / 1   | `[0, 0.5, 0,    0.5]`       |
| Pace (running)                 | ≤ 0.5 | ≈ 0 / 1   | `[0, 0.5, 0,    0.5]`       |
| LSLC walk                      | > 0.5 | (0, 0.25) | between Pace and Walk-LS    |
| LSLC run                       | ≤ 0.5 | (0, 0.25) | between Pace and Walk-LS    |
| Walk-LS (lateral singlefoot)   | > 0.5 | ≈ 0.25    | `[0, 0.5, 0.75, 0.25]`      |
| Amble (lateral-sequence run)   | ≤ 0.5 | ≈ 0.25    | `[0, 0.5, 0.75, 0.25]`      |
| LSDC walk (canonical horse walk) | > 0.5 | (0.25, 0.50) | close-trot LS walk      |
| LSDC run                       | ≤ 0.5 | (0.25, 0.50) | close-trot LS run         |
| Trot (walking)                 | > 0.5 | ≈ 0.50    | `[0, 0.5, 0.5,  0]`         |
| Trot (running, flying-trot)    | ≤ 0.5 | ≈ 0.50    | `[0, 0.5, 0.5,  0]`         |
| DSDC walk (primate walk)       | > 0.5 | (0.50, 0.75) | close-trot DS walk      |
| DSDC run                       | ≤ 0.5 | (0.50, 0.75) | close-trot DS run         |
| Walk-DS (diagonal singlefoot)  | > 0.5 | ≈ 0.75    | `[0, 0.5, 0.25, 0.75]`      |
| DS-amble                       | ≤ 0.5 | ≈ 0.75    | `[0, 0.5, 0.25, 0.75]`      |
| DSLC walk (rare)               | > 0.5 | (0.75, 1.00) | close-paced DS walk     |
| DSLC run (very rare)           | ≤ 0.5 | (0.75, 1.00) | close-paced DS run      |

**Trot DAP sub-label.** When |DAP_L| > `DAP_DISSOCIATION_THRESHOLD`
or |DAP_R| > `DAP_DISSOCIATION_THRESHOLD`, the Trot label carries a
sub-tag indicating dissociation direction:

| sub-tag | meaning                                                 |
|---------|---------------------------------------------------------|
| `+DAP`  | positive diagonal advanced placement (hind-first)       |
| `-DAP`  | negative diagonal advanced placement (fore-first)       |
| `±DAP`  | one diagonal positive, the other negative (LR asym low) |

Following Starke & Clayton 2015 / PMC4933092: dissociation of up to
~50 ms (≈ 0.10 cycle at trot stride period) is a normal feature of
trot in sound horses, **not a separate gait**. Magnitudes well above
that range remain Trot but flag a quality concern.

### Asymmetric gaits (templates from Hildebrand 1977)

When δφ ≥ `LR_ASYMMETRY_THRESHOLD`, classify against asymmetric
templates by 4-tuple cyclic distance:

| Label                  | Phase template `[FL, FR, RL, RR]` | Footfall meaning              |
|------------------------|-----------------------------------|-------------------------------|
| Bound                  | `[0, 0,   0.5,  0.5]`             | front pair + rear pair        |
| Pronk                  | `[0, 0,   0,    0]`               | all four together             |
| Half-bound (R-lead)    | `[0, 0,   0.55, 0.45]`            | front sync, rear ~sync        |
| Half-bound (L-lead)    | `[0, 0,   0.45, 0.55]`            | front sync, rear ~sync        |
| Canter (R-lead, 3-beat)| `[0, 0.5, 0.4,  0]`               | RH+(LH+RF)+LF                 |
| Canter (L-lead, 3-beat)| `[0, 0.5, 0,    0.4]`             | LH+(RH+LF)+RF                 |
| Transverse gallop (R)  | `[0, 0.1, 0.6,  0.5]`             | LH→RH→LF→RF                   |
| Transverse gallop (L)  | `[0, 0.1, 0.5,  0.6]`             | RH→LH→RF→LF                   |
| Rotary gallop (R)      | `[0, 0.1, 0.5,  0.6]`             | LH→RH→RF→LF                   |
| Rotary gallop (L)      | `[0, 0.1, 0.6,  0.5]`             | RH→LH→LF→RF                   |

Asymmetric templates above are derived from Robilliard 2007 Table 1
footfall sequences re-expressed as `[FL, FR, RL, RR]` phases relative
to FL. Pronk's per-foot synchronous-test rule is preserved (max
cyclic distance, not mean) for the same wrap-around reason as before.

### Admin labels

| Label     | Meaning                                                  |
|-----------|----------------------------------------------------------|
| **Stand** | β > `STAND_DUTY` OR no detectable stride OR period > `MAX_STRIDE_PERIOD_S` |
| **Irregular** | δφ asymmetric and best asymmetric-template cyclic distance > `PHI_TOLERANCE`; or symmetric and φ_LH does not fall in any quadrant within tolerance |
| **n/a**   | < 2 stance starts on FL (insufficient data)              |

## Classification procedure

For one window of contact data of shape `(T, 4)`:

1. **Stand check.** If β > `STAND_DUTY`, or no detectable stride on
   any foot, or measured stride period > `MAX_STRIDE_PERIOD_S` →
   **Stand**.
2. **Stride period.** If fewer than `MIN_STRIDE_STARTS` onsets on
   FL → **n/a**. Otherwise period = mean inter-onset interval on FL.
3. **Phase tuple.** Compute per-foot phases by touchdown timing,
   circular mean across strides.
4. **Lateral phase.** Compute φ_L, φ_R, φ_LH (circular mean of the
   two), and δφ.
5. **Asymmetric branch.** If δφ ≥ `LR_ASYMMETRY_THRESHOLD`:
   a. Compute cyclic distance from observed phase tuple to each
      asymmetric template (Pronk uses max-distance rule).
   b. If smallest distance ≤ `PHI_TOLERANCE` → that asymmetric label.
   c. Else → **Irregular**.
6. **Symmetric branch.** Otherwise:
   a. Find canonical-point distance:
      - Pace at φ = 0 (or 1)
      - LS singlefoot at φ = 0.25
      - Trot at φ = 0.50
      - DS singlefoot at φ = 0.75
   b. If φ_LH within `PHI_TOLERANCE` of a canonical point → that
      named gait, with β qualifier:
      - β > `BETA_RUN_THRESHOLD` → walking version
      - β ≤ `BETA_RUN_THRESHOLD` → running version
   c. Else → quadrant label (LSLC / LSDC / DSDC / DSLC) with β
      qualifier.
7. **Trot DAP sub-label.** If best label is Trot, also report DAP
   sub-tag based on |DAP_L|, |DAP_R| against
   `DAP_DISSOCIATION_THRESHOLD`.

## Thresholds

| Symbol                       | Value             | Source                                                              |
|------------------------------|-------------------|---------------------------------------------------------------------|
| `STAND_DUTY`                 | 0.95              | Engineering (Hildebrand does not define Stand)                      |
| `MAX_STRIDE_PERIOD_S`        | 1.0 s             | Engineering (rejects standstill bobbing as gait)                    |
| `MIN_STRIDE_STARTS`          | 2                 | Engineering (one full stride needed to estimate phase)              |
| `BETA_RUN_THRESHOLD`         | 0.50              | Hildebrand 1989 (β > 0.5 walking, ≤ 0.5 running). Empirical horse cutoff 0.518 (Audoin-Maddison 2009 / PMC2658658); we use the canonical 0.5 since Go2 sits very close to 0.5 in trot |
| `PHI_TOLERANCE`              | 0.10              | Half-quadrant width on the Cartmill 2002 / PMC8370795 25/50/75 boundary scheme — gives non-overlapping bins around each canonical point |
| `LR_ASYMMETRY_THRESHOLD`     | 0.20              | Engineering — 2 × `PHI_TOLERANCE`. Below this, the gait is treated as left-right symmetric and classified by φ_LH |
| `DAP_DISSOCIATION_THRESHOLD` | 0.05              | Engineering — corresponds to ~10 ms at typical trot period 200 ms; below this the trot is reported as synchronous |
| `WINDOW_S`                   | 2.0 s             | Engineering (≥ 2 strides at the slow end of ramp)                   |
| `STRIDE_S`                   | 0.25 s            | Engineering (sliding-window stride for transition plot)             |
| `CONTACT_THRESHOLD_N`        | 1.0 N             | Isaac Lab contact-force noise floor                                 |

## Note on this project's observations

The Go2 RL policies trained here populate only a small region of the
Hildebrand plane: β ∈ [0.50, 0.67] and φ_LH ∈ [0.20, 0.55] across the
0–4 m/s velocity range. Specifically:

- **Walk-LS** (φ_LH ≈ 0.25, β ≈ 0.65) at low velocities
- **Trot** (φ_LH ≈ 0.50, β ≈ 0.53–0.55) across most of the range
- No flying-trot (β stays > 0.5), no asymmetric gaits, no pace, no
  diagonal-sequence walks

At high velocities the trot persists with non-zero diagonal
dissociation (DAP magnitudes of ~0.05–0.15 phase, ~10–30 ms at
T ≈ 175 ms). This is **dissociated trot, not a new gait.** The 4-tuple
classifier with strict `PHI_TOLERANCE = 0.10` against the canonical
synchronous template `[0, 0.5, 0.5, 0]` was rejecting these windows
as Irregular even though Hildebrand's 2D `(β, φ_LH)` representation
places them at the trot canonical point. The 2D classifier above
resolves this artefact.

Why this region of the Hildebrand plane and not a wider one — i.e.,
why the policy does not reach pace, bound, gallop, or flying trot —
is an analysis question for the report, not a classifier question.
