# Why current results don't match expected (v_max_4_retune, 1k iters)

## Expected (the idealized plot)

- **Task-Specific**: clean staircase / diagonal — masters bin 0 → 1 → 2 → … → 7 in sequence, late bins eventually catch up.
- **Teacher-Guided**: same shape, slightly faster / smoother gradient.
- **Uniform**: lags everywhere, especially high bins.

## Actual (current 1k-iter sweep)

- **Task-Specific**: looks almost identical to Uniform after ~iter 250. EPTE-SP at b6/b7 is barely better than Uniform.
- **Teacher-Guided**: actually focuses on hard bins, pulls clear of the other two at b5/b6/b7.
- **Uniform**: as expected — flat sampling, mediocre everywhere.

## Why Task-Specific cannot produce the expected diagonal

This is **structural**, not a hyperparameter problem. Margolis Box-Adaptive is a *permission* curriculum, not a *focus* curriculum.

### 1. The update rule only ever ADDS weight

`src/source/curriculum_rl/curriculum_rl/curricula/task_specific.py:36-41`:

```python
if bin_rewards[b] >= self.gamma:        # γ = 0.7
    new_weights[b]   = min(w[b]   + 0.2, 1.0)
    new_weights[b-1] = min(w[b-1] + 0.2, 1.0)
    new_weights[b+1] = min(w[b+1] + 0.2, 1.0)
```

There is no decay, no down-weighting, no redistribution. Once a bin's weight reaches 1.0 it stays at 1.0 for the rest of training.

### 2. The box saturates to uniform very quickly

Low bins (0, 1, 2, 3) cross γ=0.7 within the first few hundred iters because slow walking is easy. Each success cascades +0.2 to its neighbours. By ~iter 250 the weight vector is saturated:

```
w = [1, 1, 1, 1, 1, 1, 1, 1]
```

From that point on, sampling is **1/8 across all bins — algorithmically identical to the Uniform condition**. So Task-Specific is just "Uniform with a 250-iter warmup".

### 3. There is no mechanism to focus on the frontier

Bin 6 needs maybe 60–80% of training samples to ever cross γ=0.7. Box-adaptive can only ever give it 12.5%. Result: at 1k iters, b6 sits at low reward indefinitely. More iters won't fix this — the algorithm is allocating budget wrong by design.

### 4. `min_episodes_per_bin = 50` makes the asymmetry worse

`src/source/curriculum_rl/curriculum_rl/curricula/task_specific.py:34`:

```python
if self._episode_counts[b] < self.min_episodes_per_bin:
    continue
```

This is a noise filter — it prevents a single lucky episode in bin 3 from prematurely expanding the box. But the floor is **per-bin absolute**, not relative to that bin's sampling rate. So:

- Bin 0 (weight 1.0, ~12.5% of envs once box opens) hits 50 episodes in a few iters → ratifies fast.
- Bin 6 (weight 0.2 then 1.0) hits 50 episodes much later, even after it's "unlocked".

The high bins are gated twice: once by the box not reaching them, and again by the episode-count floor. Even when the policy *could* succeed at b6, the curriculum is slow to ratify it.

### 5. Teacher-Guided is the only one that focuses mass

`src/source/curriculum_rl/curriculum_rl/curricula/teacher_guided.py:65-71`:

```python
lp     = stage_avg - self.prev_rewards
scaled = lp / β                          # β = 0.05  → sharp softmax
softmax = exp(scaled) / sum(exp(scaled))
weights = (1 - ε) * softmax + ε * uniform   # ε = 0.15
```

This actively concentrates probability on whichever bin is improving fastest. β=0.05 makes the softmax sharp — typically one or two bins get 60–80% of the mass for several stages. That is what produces the expected high-bin progress.

## Why the expected staircase is fundamentally not in this algorithm's repertoire

The expected plot (Task-Specific making a clean diagonal, mastering each bin in turn) describes a **focus** curriculum that *concentrates* on the next hardest bin. Margolis Box-Adaptive is a **permission** curriculum that *unlocks* bins. Once unlocked, every bin gets the same sampling rate — the algorithm has no concept of "now train hard on bin 5".

So:

- **Task-Specific** in this codebase will never produce the expected diagonal, regardless of iter count or γ.
- The diagonal pattern in the expected plot is what **Teacher-Guided** is supposed to do, and roughly does.
- The fix for Task-Specific is not more iters — it is replacing or augmenting the algorithm (e.g. Margolis-with-decay, or a difficulty-weighted variant).

## What the iter count actually buys

The PPO config is identical to upstream `BasePPORunnerCfg` and identical across all three conditions — see `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`. Only the `BinnedVelocityCommand` curriculum differs.

The 1k-iter run is enough to see the shape of each curriculum's behaviour. Going to 6k iters will:

- **Uniform**: improve b6/b7 modestly (more samples, no focus).
- **Task-Specific**: behave like Uniform from iter ~250 onward, so the extra 5k iters are also "Uniform-with-extra-iters" — small additional gain at high bins.
- **Teacher-Guided**: pull clearly ahead at high bins — extra iters compound because the LP softmax keeps reallocating mass to whichever bin is currently improving.

So 6k iters will widen the Teacher vs the-other-two gap, but it will **not** turn Task-Specific into the expected diagonal. That requires a different algorithm.
