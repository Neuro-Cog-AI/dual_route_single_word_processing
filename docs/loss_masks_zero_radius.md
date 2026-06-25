# Loss Masks, Zero-Error Radius, and Loss Decomposition

This document describes exactly how the training loss is computed for a repetition trial: the shape and semantics of the motor targets, the mask, the dead zone, the loss decomposition, and the output-positive weight.

---

## 1. BCE loss shape and reduction

The base loss function is PyTorch's binary cross-entropy:

```python
raw_motor = F.binary_cross_entropy(motor_outputs, motor_targets, reduction="none")
# shape: (2T, 39)
```

`reduction="none"` returns a per-element loss tensor of shape `(n_ticks, motor_size) = (2T, 39)`. Each element is:

```
BCE(p, y) = -[ y * log(p) + (1 - y) * log(1 - p) ]
```

where `p ∈ (0, 1)` is the sigmoid motor output and `y ∈ {0, 1}` is the target.

The final scalar loss is obtained by masking (and optionally weighting) and then summing:

```python
loss = (raw_motor * alive.float()).sum()
```

The default `loss_reduction = "sum"` means the total loss is the **sum over all active (tick, unit) pairs**. The alternative `loss_reduction = "mean_active"` divides by the count of active pairs, but this is not the training default.

---

## 2. motor_targets for repetition

`make_repetition_trial` constructs:

```python
motor_targets = cat([zeros(T, 39), phon_tensor], dim=0)   # (2T, 39)
```

- **Input phase (ticks 0..T-1):** target = `0.0` for all 39 units. The model is required to be silent.
- **Output phase (ticks T..2T-1):** target = `phon_tensor[t-T]` for tick `t`. For the active phoneme unit, `target = 1.0`; for all other 38 units, `target = 0.0`.

At any output tick, there is exactly **one** unit with `target = 1.0` (the phoneme being produced) and **38** units with `target = 0.0`. This is a structural consequence of one-hot encoding.

---

## 3. motor_loss_mask and broadcasting

`make_repetition_trial` sets:

```python
motor_loss_mask = ones(2 * T, dtype=bool)   # (2T,) — all True
```

All `2T` ticks are supervised — both input-phase silence and output-phase phoneme production contribute to the loss. This is stated in the paper: the motor layer must be silent during listening.

In `compute_trial_loss_breakdown`:

```python
alive_motor = trial.motor_loss_mask.unsqueeze(-1).expand_as(raw_motor)
# (2T,) → (2T, 1) → (2T, 39) by broadcasting
```

The 1D tick mask is broadcast to the 2D `(2T, 39)` shape. Every unit at every supervised tick is included before the dead zone is applied.

---

## 4. zero_error_radius / dead-zone behavior

If `zero_error_radius > 0.0` (paper value: 0.1):

```python
dead_motor = (motor_outputs.detach() - motor_targets).abs() < zero_error_radius
# (2T, 39) bool — True where unit is "close enough"
alive_motor = alive_motor & ~dead_motor
```

A unit is excluded from the loss (dead-zoned) if:

```
|prediction − target| < zero_error_radius
```

For `zero_error_radius = 0.1`:
- A unit with `target = 0.0` is dead-zoned when `prediction < 0.1` — the motor is sufficiently silent.
- A unit with `target = 1.0` is dead-zoned when `prediction > 0.9` — the phoneme is sufficiently produced.

**Key properties:**

- `.detach()` on the dead-zone computation: the dead-zone mask is computed without gradient. The mask acts as a hard gate — units inside the radius receive no gradient at all, units outside receive full gradient.
- Dead-zoned units contribute exactly **zero loss and zero gradient**. They never "count" against the loss.
- Once a unit enters the dead zone and stays there throughout an epoch, the loss for that trial can fall to zero even if the weights are not perfect. This is why `avg_loss = 0.0` is achievable without perfect predictions in the strict (non-radius) sense.
- The dead zone is applied **before** the `output_positive_weight` scaling. Weighting never resuscitates a dead-zoned unit.

---

## 5. Loss decomposition

The training script logs a decomposition of the eval-mode loss into three components. These are computed by `compute_trial_loss_breakdown` and reported separately (without `output_positive_weight`):

### Input BCE

The BCE contribution from input-phase ticks (ticks 0..T-1):

```
input_BCE = sum of BCE(motor_output[t, u], 0.0) for all t < T, all u, where alive
```

All input-phase targets are zero (silence). Good input suppression → small input BCE.

### Output-positive BCE

The BCE contribution from the one positive unit per output tick:

```
output_pos_BCE = sum of BCE(motor_output[T+t, pos_u], 1.0) for t in [0, T), where alive
```

There is exactly 1 positive unit per output tick, so there are T positive (tick, unit) pairs per trial. High positive activation → small positive BCE.

### Output-negative BCE

The BCE contribution from the 38 negative units per output tick:

```
output_neg_BCE = sum of BCE(motor_output[T+t, u], 0.0) for t in [0, T), all u ≠ pos_u, where alive
```

There are `T × 38` negative output pairs per trial. Low negative activation → small negative BCE.

**Implication:** Even if the model perfectly suppresses negative units (`output_neg_BCE ≈ 0`) while failing on positive units (`output_pos_BCE` remains high), it can still reduce total loss substantially — because negative units are 38× more numerous. This is the core reason the model can appear to make "progress" (loss decreasing) while failing to actually produce correct phonemes.

---

## 6. output_positive_weight

A diagnostic multiplier for the output-phase positive units:

```python
if output_positive_weight != 1.0 and trial.task == Task.REPETITION:
    T = trial.phon_tensor.shape[0]
    output_phase_mask = zeros_like(motor_targets, dtype=bool)
    output_phase_mask[T:, :] = True                     # ticks T..2T-1
    positive_mask = motor_targets >= 0.5                # target == 1.0
    weighted_mask = output_phase_mask & positive_mask
    weight = where(weighted_mask, full(output_positive_weight), ones)
    motor_loss = (raw_motor * alive.float() * weight).sum()
```

`output_positive_weight` applies only to units satisfying **all three conditions**:
1. Task is REPETITION
2. Tick is in the output phase (t ≥ T)
3. Target unit has `target >= 0.5` — i.e., the positive unit (the one that should be 1.0)

This is the **one active phoneme unit per output tick**. Input-phase targets are all zero, so `target >= 0.5` is never true during input; the weight never applies to input-phase ticks. Padded output ticks (if batching is introduced) with zero targets are also unaffected.

In practice: with `output_positive_weight = 3.0`, the loss contribution of the positive unit at each output tick is tripled. This compensates for the 38:1 numerical imbalance between negative and positive targets in the one-hot encoding.

---

## 7. The 38:1 imbalance in one-hot encoding

For a 39D one-hot encoding, each output tick has:
- 1 positive-target unit (`target = 1.0`)
- 38 negative-target units (`target = 0.0`)

In a T=5 phoneme word with `reduction="sum"`:
- Total positive output (tick, unit) pairs: 5 × 1 = 5
- Total negative output (tick, unit) pairs: 5 × 38 = 190
- Ratio: 38:1

Plus the input phase contributes 5 × 39 = 195 additional zero-target pairs.

Without correction, the model can minimize loss by simply suppressing all motor output (making all 39 units close to 0.0). This reduces loss for 38/39 of the output units at every tick while only increasing it for 1/39 units. The net loss reduction is strongly in favor of suppression.

Strategies to counteract this:
- `output_positive_weight` — increases the gradient contribution of positive units
- `zero_error_radius` — once negative units are suppressed below 0.1, they stop contributing gradient, which shifts the remaining gradient entirely toward positive units
- `loss_reduction="mean_active"` — normalizes by active unit count, reducing the reward for negative suppression relative to sum

---

## 8. Why the model can reduce BCE by suppressing motor outputs

The BCE formula for a zero-target unit:

```
BCE(p, 0) = -log(1 - p)
```

This is minimized when `p → 0`. For a one-target unit:

```
BCE(p, 1) = -log(p)
```

This is minimized when `p → 1`.

Starting from sigmoid(bias = -1) ≈ 0.27, the network has an initial activation well above zero for all units. The gradient `d/dp BCE(p, 0) = 1/(1-p)` pushes the network to reduce all outputs toward zero. Because there are 38 zero-target units for every one-target unit per output tick, the global gradient direction in weight space initially favors suppression.

The result is that during early training, the model often learns to suppress motor output (reducing `output_neg_BCE`) before it learns to selectively activate the correct phoneme unit (reducing `output_pos_BCE`). The loss curve descends without meaningful phoneme production. This is the documented pathology in the 1200-word diagnostic experiment (see [repetition_only_training_note.md](repetition_only_training_note.md)).
