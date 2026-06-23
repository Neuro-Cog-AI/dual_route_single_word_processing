# Repetition-Only Training Note (Phase 3e)

This note documents the first real training run of the Lichtheim 2 model on
English word-repetition data. It is intended as a reference for the LSCP/ENS
project supervised by Yair Lakretz and as a record of every non-trivial
implementation decision.

---

## 1. Goal of Phase 3e

Phase 3e is the first end-to-end repetition-only pipeline validation with real English data.

The goal is deliberately narrow:

- train on the **repetition task only**;
- use **real English words** with proper one-hot phoneme encodings;
- confirm that **BCE loss is finite and decreasing**;
- save metrics and predictions so training dynamics can be inspected;
- expose all non-obvious design choices as `[Open]` items rather than
  silently resolving them.

This is not a production training run. It is a scientific checkpoint: "can the
Lichtheim 2 architecture learn to repeat at all, and does the pipeline hold up
end-to-end?"

---

## 2. Architecture: full Lichtheim 2, not dorsal-only

### Option A (current): full architecture, repetition trials only

The **full `Lichtheim2Model`** is used unchanged. Both the dorsal pathway
(sound → iSMG → motor) and the ventral pathway
(sound → mSTG → aSTG → vATL → triangularis → motor) are present, connected,
and computed at every tick. Weight updates during repetition training affect
all parameters — including ventral-pathway weights — because
`triangularis_to_motor` contributes to the motor output that is directly
supervised by the repetition loss.

This matches the paper's architecture (Figure 1): both pathways converge on
the insular-motor output regardless of the task presented.

### Option B (deferred): dorsal-only sanity model

Option B would create a model with only the dorsal pathway (removing mSTG,
aSTG, vATL, triangularis layers and their connections) to produce a clean
isolation of the phonological-loop mechanism. This would make the repetition
training more interpretable in isolation, at the cost of architectural
faithfulness.

**`[Open #1]`** — Option B is deferred. The current implementation follows
Option A. Switching to Option B would require changes to `model.py` and is a
separate design decision that should be confirmed with Yair before doing.

### Why is this not dorsal-only, even though only repetition trials are used?

During repetition, `build_trial_inputs` never sets `clamp_vATL_in` (it passes
`None`). So `vATL_input_used = state.vATL_context` (the default copy-back).
The ventral pathway still runs:

1. Sound input → `mSTG` (via `sound_to_mSTG`)
2. `mSTG` → `aSTG` (via `mSTG_to_aSTG` + `vATL_in_to_aSTG(vATL_context)`)
3. `aSTG` → `triangularis` (via `aSTG_to_triangularis`)
4. `triangularis` → **motor** (via `triangularis_to_motor`)

Step 4 means the ventral pathway contributes to the motor output even in
repetition. The repetition loss (`motor_loss`) therefore sends gradient through
`triangularis_to_motor`, `aSTG_to_triangularis`, `mSTG_to_aSTG`, and
`sound_to_mSTG` at every training step.

---

## 3. Data loading pipeline

```
data/raw/nwr_swp/phonemes.csv   →  PhonemeInventory  (ordered symbol list)
data/raw/nwr_swp/wfe.csv        →  list[WordItem]    (word + phon_tensor)
```

1. **`load_phoneme_inventory(phonemes.csv)`**
   Reads the `Phoneme` column. Row order is the canonical one-hot index order.
   The English NWR inventory has 39 phonemes → `sound_input_size = 39`.

2. **`load_word_items(wfe.csv, inventory)`**
   For each row: parses the `No_Stress` column (Python-list syntax, e.g.
   `['AH', 'T', 'EH', 'N']`) with `ast.literal_eval`; checks that parsed
   length matches the `Length` column; encodes with `encode_phoneme_sequence`.

3. **`encode_phoneme_sequence(phonemes, inventory) → Tensor(T, N)`**
   Produces a one-hot float tensor of shape `(T, N)` where `T` is the number
   of phonemes and `N = len(inventory)`. Row `t` has a single `1.0` at
   `inventory.symbol_to_index[phoneme[t]]`. Raises `KeyError` if any phoneme
   is not in the inventory.

4. **Sampling**: `rng.sample(all_items, max_items)` takes a random subset
   controlled by `--max-items` and `--seed`.

---

## 4. Phoneme encoding pipeline

The phoneme encoding is **English one-hot** over a 39-symbol inventory derived
from the `phonemes.csv` file.

**`[Open #6]`** — The original Ueno et al. 2011 model used Japanese mora
features: a 21-dimensional binary code describing phonological properties (place
of articulation, manner, voicing, etc.) rather than pure one-hot identity
vectors. The current English one-hot encoding has the same shape but no shared
phonological structure between similar phonemes (e.g. `/p/` and `/b/` are
orthogonal vectors despite sharing place of articulation). Whether this affects
learning dynamics needs to be discussed with Yair.

---

## 5. Repetition trial construction

`make_repetition_trial(phon_tensor, motor_size)` creates a `SupervisedTrial`
with:

| Field | Value |
|-------|-------|
| `task` | `Task.REPETITION` |
| `n_ticks` | `2T` |
| `phon_tensor` | `(T, sound_size)` — input phoneme sequence |
| `motor_targets` | `cat([zeros(T, motor), phon_tensor], dim=0)` — shape `(2T, motor)` |
| `motor_loss_mask` | `ones(2T, dtype=bool)` — loss active at ALL ticks |
| `semantic_targets` | `None` — not evaluated in repetition |
| `sem_input` | `None` — no semantic clamping for repetition |

**`[Open #2]`** — The `motor_loss_mask` is `True` for all `2T` ticks,
including the `T` input-phase ticks where the motor target is zero (silence).
The initial state already has `motor = 0` (from `init_state`), so
input-phase loss starts near zero and does not dominate the gradient.
However, `docs/replication_spec.md` §5.1 originally said "ticks 1–3: motor
output is not evaluated." These two readings of the paper differ. The current
implementation follows `trials.py`'s own citation "[Paper: motor required to
be silent during input phase, meaning it has a zero target with gradient]".
This should be confirmed with Yair.

---

## 6. `run_trial()` and `forward_tick()` tick unfolding

`model.run_trial(Task.REPETITION, phon_tensor, sem_zeros, cfg)` calls
`build_trial_inputs` then iterates `forward_tick` for each tick:

**Input phase (ticks 0..T-1):**
- `sound_in` = `phon_tensor[t]` — one-hot phoneme vector, clamped directly
  as input, **not** passed through sigmoid.
  **`[Note #4]`** AUD/sound is a hard-clamped raw tensor; the activation
  function is only applied to the hidden/output net inputs inside `forward_tick`.
- `clamp_vATL_in = None` → `vATL_input_used = state.vATL_context`

**Output phase (ticks T..2T-1):**
- `sound_in` = zeros
- `clamp_vATL_in = None` → `vATL_input_used = state.vATL_context`

**At every tick**, `forward_tick` computes (in order):

```
# Dorsal
iSMG_net  = sound_to_iSMG(sound_in) + iSMG_elman(iSMG_context) + motor_copy_to_iSMG(motor_context)
new_iSMG  = sigmoid(iSMG_net)

# Ventral
new_mSTG  = sigmoid(sound_to_mSTG(sound_in))
aSTG_net  = mSTG_to_aSTG(new_mSTG) + vATL_in_to_aSTG(vATL_context)
new_aSTG  = sigmoid(aSTG_net)
new_vATL  = sigmoid(aSTG_to_vATL(new_aSTG))
new_tri   = sigmoid(aSTG_to_triangularis(new_aSTG))

# Motor (both pathways converge)
motor_net = iSMG_to_motor(new_iSMG) + triangularis_to_motor(new_tri)
new_motor = sigmoid(motor_net)

# Copy-back
iSMG_context  ← new_iSMG   (Elman self-recurrence)
motor_context ← new_motor   (motor copy-back to iSMG next tick)
vATL_context  ← new_vATL   (vATL copy-back to aSTG next tick)
```

**`[Open #3]`** — The copy-back is two-stage: an identity copy
(`motor_context ← new_motor`) followed by a **learned linear projection**
(`motor_copy_to_iSMG`, shape `(motor_size, iSMG_size)`, init `[−0.5, 0.5]`).
The term "copy-back" describes the mechanism (copy of activation + projection
back into the next tick's computation), not a weight-free identity connection.
Supp Fig S1 describes an "invisible copy layer fed back to iSMG" — the current
learned-weight realization is `[Inferred]`. Needs paper-level confirmation.

---

## 7. Loss computation

```python
loss = compute_trial_loss(tick_results, trial, zero_error_radius, loss_reduction)
```

Steps inside `compute_trial_loss_breakdown`:

1. Stack motor outputs: `motor_outputs = stack([r.state.motor for r in results])` → `(2T, 39)`
2. Compute raw BCE per element: `raw_motor = BCE(motor_outputs, motor_targets, reduction="none")` → `(2T, 39)`
3. Build alive mask from `motor_loss_mask` (all-True for repetition).
4. If `zero_error_radius > 0`: compute dead zone
   `dead = |motor_outputs.detach() - motor_targets| < radius`;
   alive mask = mask & ~dead.
   The `.detach()` is used only to compute the boolean mask — it does NOT
   truncate gradient flow through `raw_motor`.
5. `motor_loss = (raw_motor * alive.float()).sum()`
6. Semantic loss = 0 for repetition (no `semantic_targets`).
7. `total_loss = motor_loss`

**`[Open #5]`** — `zero_error_radius`:
- Script default: `0.0` (disabled — good for debugging, full gradient always).
- Paper value: `0.1`.
- `train_repetition_real_data.py` uses `0.1` by default.
- For a sanity run, `0.0` is safer (no risk of accidentally masking too much).
  Use `0.1` when trying to reproduce paper training dynamics.

---

## 8. Backpropagation through time (BPTT)

`run_trial` chains `forward_tick` calls without any `.detach()` or
`torch.no_grad()`. The `ModelState` tensors (`iSMG`, `motor`, `vATL_context`,
etc.) are passed from tick to tick as regular tensors with full gradient
tracking.

`train_step` calls `loss.backward()` **once**, after `run_trial` completes.
PyTorch unrolls the entire 2T-tick computation graph at that point and
backpropagates gradient to all parameters through all ticks.

**`[Open #7]`** — Full BPTT through all `2T` ticks within a trial. The
paper `[Supp]` describes online weight updates after each word, but does not
specify the BPTT scope. LENS (the original training framework) updates after
each pattern presentation using the full unrolled network. The current
implementation matches this. Whether truncated BPTT (e.g. stopping gradient
at the input/output phase boundary) would be more faithful is `[Open]`.

---

## 9. What outputs are saved

Each run creates a timestamped directory:
`outputs/repetition_only/<YYYYMMDD_HHMMSS>/`

| File | Content |
|------|---------|
| `run_config.json` | All CLI args + model config fields |
| `metrics.csv` | Per-epoch: `epoch`, `avg_loss`, `min_loss`, `max_loss`, `n_trials`, `all_finite` |
| `predictions_before.json` | Per-item output-phase predictions **before training** |
| `predictions_after.json` | Per-item output-phase predictions **after training** |
| `loss_curve.png` | Per-epoch avg/min/max loss plot (requires matplotlib) |

**No model checkpoints** are saved. If a checkpoint is needed, the model can
be saved manually with `torch.save(model.state_dict(), ...)` after the run.

**Prediction record fields:**

| Field | Description |
|-------|-------------|
| `label` | Item label (e.g. `word:tank`) |
| `item_id` | Row index in source CSV |
| `T` | Phoneme count |
| `n_ticks` | Total ticks (= 2T for repetition) |
| `target_phonemes` | Argmax of one-hot motor targets at output-phase ticks |
| `predicted_phonemes` | Argmax of model motor outputs at output-phase ticks |
| `phoneme_accuracy` | Fraction of output-phase ticks with correct argmax |
| `threshold_accuracy` | Fraction of (tick, unit) pairs where `|output − target| < 0.1` |
| `exact_match` | True if all output-phase phoneme argmaxes are correct |

---

## 10. How to run

### Check arguments

```bash
PYTHONPATH=src python scripts/train_repetition_only.py --help
```

### Smoke run (5 words, 2 epochs, ~seconds)

```bash
PYTHONPATH=src python scripts/train_repetition_only.py \
  --data-dir data/raw/nwr_swp \
  --config configs/english_nwr.yaml \
  --source words \
  --max-items 5 \
  --epochs 2 \
  --lr 0.01 \
  --device cpu \
  --seed 0 \
  --zero-error-radius 0.1 \
  --loss-reduction sum
```

Expected: all losses finite; BCE may or may not decrease visibly depending on seed, sampled items, and run length.

### Short diagnostic run (20 words, 20 epochs)

After the smoke run succeeds:

```bash
PYTHONPATH=src python scripts/train_repetition_only.py \
  --data-dir data/raw/nwr_swp \
  --config configs/english_nwr.yaml \
  --source words \
  --max-items 20 \
  --epochs 20 \
  --lr 0.01 \
  --device cpu \
  --seed 0 \
  --zero-error-radius 0.1 \
  --loss-reduction sum
```

Expected: BCE decreasing over epochs; phoneme accuracy should increase above
chance (1/39 ≈ 0.026 for pure random).

---

## 11. What can be shown to Yair

After a successful short run, the following artifacts are ready for review:

1. **`metrics.csv`** — show that BCE loss decreases epoch by epoch; plot the
   `avg_loss` column as a simple learning curve.

2. **`predictions_before.json` vs `predictions_after.json`** — for each word,
   compare `target_phonemes` vs `predicted_phonemes` before and after training.
   Even with 20 items × 20 epochs, phoneme accuracy should start to exceed
   chance.

3. **`loss_curve.png`** — a quick visual of learning dynamics (avg ± min/max
   envelope per epoch).

4. **The pipeline itself** — `scripts/train_repetition_only.py` is designed to
   be read top-to-bottom, with each step numbered and commented. It can be
   walked through line by line in a meeting.

**What to highlight to Yair:**
- The full dual-route architecture is intact (not a simplified model).
- Only the repetition task is trained; the ventral pathway still runs (see §2).
- The loss is standard BCE on sigmoid outputs — one candidate translation of
  the paper's cross-entropy criterion.
- English one-hot phonemes are a simplification vs the original Japanese
  mora-feature encoding (Open #6).
- No accuracy metric existed before this phase; phoneme argmax and threshold
  accuracy are now computed from first principles.

---

## 12. Open audit issues

### `[Open #1]` Option A vs Option B

**Full architecture (current) vs dorsal-only sanity model.**

Option A trains the full dual-route model on repetition. Option B would build a
stripped dorsal-only model (no mSTG, aSTG, vATL, triangularis) to validate the
phonological loop in isolation. Option B requires changes to `model.py` and
should only be done after agreement with Yair on the scientific motivation.

### `[Open #2]` Input-phase loss supervision

`motor_loss_mask` is `True` for all `2T` ticks. During the `T` input-phase
ticks, the motor target is zero (silence). The initial state already sets
`motor = 0`, so this term starts near zero and does not create a large initial
gradient. However, `docs/replication_spec.md` §5.1 originally stated "motor
output is not evaluated during input phase." The two documents disagree. Needs
confirmation from the paper or from Yair.

### `[Open #3]` Motor→iSMG copy-back: state copy + learned projection

The copy-back mechanism is two-stage:
1. **State copy**: `motor_context ← new_motor` (no weights, identity copy).
2. **Learned projection**: `motor_copy_to_iSMG(motor_context)` — a `Linear(motor_size, iSMG_size)` weight matrix initialized in `[−0.5, 0.5]`.

The term "copy-back" in the codebase refers to the overall mechanism. The
connection to iSMG is a learned weight, not a fixed identity matrix. Supp Fig
S1 describes an "invisible copy layer fed back to iSMG" — whether this implies
a fixed (non-learned) mapping or a learned projection is `[Inferred]`.

### `[Note #4]` AUD/sound input is clamped raw input, not sigmoid-computed

**This is correct behavior, not a bug.** The phoneme tensor (one-hot, values
0/1) is passed directly into `sound_to_iSMG` and `sound_to_mSTG` as the input
to those linear layers. Sigmoid is applied to the linear output (the net
activation), not to the input. This matches §6 of the replication spec:
"Input units are clamped directly (no activation function applied)."

If anyone sees `sigmoid` and worries that sound input is not sigmoided: it is
not, and it should not be.

### `[Open #5]` `zero_error_radius`: paper value vs debug default

- **Paper value**: `0.1` — units within 0.1 of target contribute zero loss.
- **Script default**: `0.0` — all units contribute loss (no dead zone).
- **`train_repetition_real_data.py` default**: `0.1`.

For debugging (checking that loss decreases at all), `0.0` is safer. For
replicating paper training dynamics, use `0.1`. In a real 200-epoch run, the
dead zone matters because it prevents small residual errors from driving
unnecessary gradient updates after approximate convergence.

### `[Open #6]` English one-hot encoding vs Japanese 21-bit mora features

The original Ueno et al. 2011 model used 21-dimensional **mora feature
vectors** for Japanese syllables, encoding phonological properties (place,
manner, voicing) as binary features. The current implementation uses
English phonemes encoded as **one-hot vectors** of dimension 39. Two
consequences:
- `sound_input_size = 39` vs the paper's `21`.
- Phonologically similar phonemes (e.g. `/p/` and `/b/`) are orthogonal in
  one-hot space; in the original coding they would share most feature bits.

This may affect how easily the model generalizes phonological patterns during
training. Discuss with Yair whether a richer English phoneme encoding is needed
before drawing conclusions.

### `[Open #7]` BPTT scope: full through all 2T ticks

The current implementation uses full BPTT: `loss.backward()` unrolls the
entire 2T-tick computation graph. No truncation at the input/output phase
boundary. This matches LENS-style online training (full unroll per pattern
presentation) and is the natural PyTorch approach.

Whether the paper intended truncated BPTT (e.g. blocking gradient at the tick
boundary where sound goes silent) is not specified in the supplement. This is
unlikely to matter in practice for 6–10 ticks but should be noted as an
assumption.

### `[Open #8]` Positive/negative unit imbalance in the BCE loss

For each output-phase tick, there is 1 positive unit (target = 1.0) and 38
negative units (target = 0.0). Across a 2T-tick repetition trial, the ratio
is approximately 77 zero-target pairs per positive-target pair (~98.7 % of all
supervised (tick, unit) pairs have a zero target).

Gradient descent first suppresses the many negative units (cheap, fast) before
the positive unit becomes the dominant learning signal. This is the mechanistic
explanation for the Phase 3e diagnostic results: BCE drops quickly while phoneme
argmax accuracy stays near chance.

One engineering mitigation is **class-balanced BCE** (up-weight positive units
by a factor of ~38 to equalize the gradient contribution) or **focal loss**
(down-weight easy negatives after they are suppressed). However, applying such
reweighting would **change the training objective** relative to the original
paper and might not replicate the paper's training dynamics. This is a
**scientific decision that should be confirmed with Yair** before
implementation. It is **not implemented in Phase 3f**.

### `[Open #9]` Exact paper scoring convention for word-level accuracy

Ueno et al. 2011 (Figure 2) evaluates repetition accuracy using a zero-error
radius, but the exact scoring convention is not fully specified:

1. Whether word correctness is evaluated on the output phase only (ticks T..2T-1)
   or across all 2T ticks (including input-phase silence).
2. Whether the radius matches the training `zero_error_radius` (0.1) or is
   defined independently.
3. Whether word correctness should be derived from an all-units-within-radius
   criterion or from another implementation-specific convention remains open.

The current implementation uses output-phase-only, all-units-within-radius at
`eval_radius=0.1` (default) as the candidate metric
(`output_all_units_within_radius`). This can differ from `zero_error_radius`
(the training dead-zone). **This should be confirmed with Yair before reporting
any result as a replication of Figure 2.**

### `[Open #10]` Dense sound input projection — experimental adaptation (Phase 3h)

`sound_proj_size` (CLI: `--sound-proj-size N`) adds a shared linear projection
`Linear(39 → N, bias=False)` applied to the raw phoneme vector before both
`sound_to_iSMG` (dorsal) and `sound_to_mSTG` (ventral). When `sound_proj_size`
is `null` (default), the model behaves exactly as in Ueno et al. 2011.

This projection is **not in the paper**. It was motivated by the fact that the
current English 39D one-hot encoding is itself already a provisional adaptation
of the original Japanese 21-bit mora feature setup: unlike mora features,
one-hot phonemes encode no phonological structure — `/p/` and `/b/` are
orthogonal vectors despite sharing place of articulation. The projection creates
a learned dense phoneme embedding space that could, in principle, recover
articulatory feature structure during training.

Key design decisions:
- `bias=False`: downstream layers (`sound_to_iSMG`, `sound_to_mSTG`) carry biases;
  a projection bias would be redundant. With `bias=False`, row `k` of
  `sound_proj.weight` is exactly the learned embedding of phoneme `k`.
- Zero sound input → zero projected output (correct for silent ticks).
- Weight init: uniform(−1, 1), same as all feedforward layers [Inferred].
- Shared across dorsal and ventral pathways (one matrix, not two).

**Any run with `sound_proj_size != null` is a model variant and is NOT a
replication of Ueno et al. 2011.** Discuss with Yair before interpreting
projection-enabled results in a comparative context.

---

## 13. Run results interpretation (June 2026)

### First diagnostic runs (Phase 3e)

Two runs were conducted with `--zero-error-radius 0.1 --loss-reduction sum
--lr 0.01`:

| Run | Items | Epochs | BCE start → end | phoneme_acc | threshold_acc | exact_match |
|-----|-------|--------|-----------------|-------------|---------------|-------------|
| Smoke | 5 | 2 | 145→24 | 0.094→0.094 | 0.39→0.90 | 0/5→0/5 |
| Diagnostic | 20 | 20 | 89→21 | 0.065→0.084 | 0.39→0.84 | 0/20→0/20 |

### Why BCE drops strongly but phoneme argmax accuracy stays flat

The motor target for each output-phase tick is one-hot (39 dimensions):
- **1 positive unit**: target = 1.0
- **38 negative units**: target = 0.0

For a T-phoneme word, across 2T total ticks:
- **T** positive-target (tick, unit) pairs — the units we want the model to activate.
- **77T** zero-target pairs (input-phase all-zero + output-phase 38 negatives).

With `zero_error_radius = 0.1`: sigmoid outputs start near 0.5. Pushing each
zero-target unit from 0.5 toward 0 enters the dead zone after a few gradient
steps. Once dead-zoned, that unit contributes no further loss or gradient. With
77× more zero-target units than positive-target units, the total BCE collapses
once the negative units are suppressed.

The positive unit (target=1, output≈0.5 initially) has `|output − target| ≈ 0.5`,
well above the dead-zone threshold of 0.1, so **it never enters the dead zone**.
Its loss remains active throughout training. But early in training, its gradient
is small relative to the 77 negative-unit gradients per tick.

**Phoneme argmax**: even when all 39 outputs are near-zero, `argmax` still
selects the highest value among 39 suppressed units — essentially random.
Argmax accuracy stays near 1/39 ≈ 0.026 chance. The small observed improvement
to ~0.08–0.09 suggests mild differential suppression across units but no
reliable phoneme-specific activation.

### Why the original threshold_accuracy is misleading

`threshold_accuracy` = fraction of output-phase (tick, unit) pairs where
`|output − target| < 0.1`. For a model that drives all outputs to near-0:

- 38/39 negative units: `|0 − 0| < 0.1` → PASS ✓
- 1/39 positive unit: `|0 − 1| = 1.0 > 0.1` → FAIL ✗
- Expected per-tick threshold accuracy: 38/39 ≈ 0.974

The observed rise to ~0.84–0.90 after training is explained almost entirely by
negative-unit suppression. This metric **does not indicate phoneme production**.

### How to interpret the new split metrics (Phase 3f)

Phase 3f adds `_compute_repetition_metric_breakdown()` to `_evaluate_one_trial()`.
The new metrics appear in `predictions_before.json` and `predictions_after.json`,
and are printed in the console evaluation summary.

| Metric | What it measures | After suppression only | After phoneme learning begins |
|--------|-----------------|------------------------|-------------------------------|
| `input_silence_threshold_acc` | Fraction of input-phase units below 0.1 | High (~1.0) | High (~1.0) |
| `output_negative_threshold_acc` | Fraction of negative units below 0.1 | High (~0.97) | High (~0.97) |
| `output_positive_threshold_acc` | Fraction of positive units above 0.9 | ~0.0 | Rising toward 1.0 |
| `mean_positive_output` | Mean activation of the correct phoneme unit | ~0.05–0.15 | Rising above 0.5 |
| `mean_negative_output` | Mean activation of negative units | ~0.03–0.08 | Low (~0.03–0.08) |
| `mean_max_motor_output` | Mean per-tick maximum motor activation | Low (~0.1) | Rising |

### Why `mean_positive_output` is more informative than `output_positive_threshold_acc`

`output_positive_threshold_acc` uses a strict threshold of 0.9. The positive
unit must exceed 0.9 — close to saturation — before it contributes to this
metric. Even if the model is learning to activate the correct unit, it may take
many epochs to push the sigmoid output above 0.9.

`mean_positive_output` is a continuous measure: any upward trend in the correct
unit's activation (e.g. from 0.1 to 0.3 over 20 epochs) is immediately visible,
even when `output_positive_threshold_acc` remains zero. It is therefore the
**earliest available signal** of phoneme production learning.

---

## 14. Evaluation metrics: what each one measures

The script computes and reports several overlapping metrics. This table clarifies
what each one actually tests.

| Metric | Phase(s) | Unit scope | Criterion | Primary purpose |
|--------|----------|-----------|-----------|-----------------|
| `phoneme_accuracy` | Output | Per-tick argmax | argmax(output) == argmax(target) | Phoneme identification rate |
| `exact_match` | Output | Whole sequence | All ticks phoneme_accuracy correct | Strict whole-word argmax match |
| `threshold_accuracy` | Output | All units (mixed) | \|output−target\| < 0.1 | **Misleading** — dominated by 38/39 negative units; kept for backward compatibility |
| `input_silence_threshold_acc` | Input | All units | output < 0.1 | Fraction of input-phase units suppressed |
| `output_negative_threshold_acc` | Output | Negative units only | output < 0.1 | Zero-target suppression rate (38/39 units per tick) |
| `output_positive_threshold_acc` | Output | Positive unit only | output > 0.9 | Near-saturation phoneme activation (strict) |
| `mean_positive_output` | Output | Positive unit only | mean sigmoid activation | Earliest continuous signal of phoneme learning |
| `mean_negative_output` | Output | Negative units only | mean sigmoid activation | Baseline suppression level |
| `mean_max_motor_output` | Output | Max per tick | mean of per-tick max | Whether any unit is being "selected" |
| `output_all_units_within_radius` | Output | All units | \|output−target\| < eval_radius | **Candidate paper-like word accuracy** [Open #9] |
| `input_all_silent_within_radius` | Input | All units | output < eval_radius | Word-level input silence criterion |
| `trial_all_supervised_units_within_radius` | Both | All units | Both above True | Strictest word criterion [Open #9] |

### Key distinctions

**`exact_match` vs `output_all_units_within_radius`:**
These are not equivalent. `exact_match` checks only 1 unit per output tick (the argmax).
`output_all_units_within_radius` checks all 39 motor units against their targets.
A word can satisfy `exact_match` but still fail `output_all_units_within_radius` if a
negative unit remains above `eval_radius`. Conversely, `output_all_units_within_radius`
is technically less strict for the positive unit (it requires `|output−1| < radius`,
not `argmax correctness`), but much stricter globally because of the 38 negative units.

**`threshold_accuracy` vs `output_negative_threshold_acc`:**
`threshold_accuracy` mixes all 39 output-phase units and is dominated by the 38/39
negative units — it rises to ~0.9 even when no phoneme is produced. Use
`output_negative_threshold_acc` to track suppression specifically, and
`output_positive_threshold_acc` / `mean_positive_output` to track phoneme production.

**`output_all_units_within_radius` is not confirmed Figure 2 replication:**
See `[Open #9]` in §12. The paper's exact scoring convention is not fully specified.
Do not report this metric as a Figure 2 result without confirmation from Yair.

---

## 15. Phase 3h/3i experiments: dense sound projection and loss decomposition (June 2026)

### 15.1 Technical implementation

#### Optional dense sound input projection [Adapted — not in Ueno et al. 2011]

Phase 3h adds an opt-in linear projection inside `Lichtheim2Model`:

```
raw phoneme input (39D one-hot)
  → sound_proj: Linear(39 → N, bias=False)
  → sound_to_iSMG (dorsal) and sound_to_mSTG (ventral)
```

The projection is enabled by passing `--sound-proj-size N` (e.g. `--sound-proj-size 20`).
When the flag is omitted, `sound_proj = None` and the model behaves exactly as in
Ueno et al. 2011 — the raw 39D one-hot is fed directly into `sound_to_iSMG` and
`sound_to_mSTG`.

**`bias=False`** — zero sound input must produce zero projected output (correct for
silent output-phase ticks). Downstream layers (`sound_to_iSMG`, `sound_to_mSTG`)
already carry biases; a projection bias would be redundant. With `bias=False`, each
row of `sound_proj.weight` is the learned dense embedding for one phoneme.

**What does not change with the projection enabled:**
- `phon_tensor` remains 39D one-hot (unchanged from baseline).
- Motor targets remain 39D one-hot (unchanged from baseline).
- Motor output remains 39D (unchanged from baseline).
- Sigmoid activations, copy-back mechanisms, and the full dorsal + ventral
  architecture are all unchanged.
- Repetition-only training only.

The projection is **not a replication of any mechanism described in Ueno et al. 2011**.
It was motivated by the fact that English one-hot phonemes encode no phonological
structure — `/p/` and `/b/` are orthogonal vectors — while the paper's Japanese mora
features encode articulatory properties. See `[Open #10]` in §12.

#### Per-epoch loss decomposition diagnostic [Phase 3i]

Phase 3i adds an optional eval-mode pass after each training epoch that decomposes
motor BCE into four components:

| Component | Phase | Units | What it tracks |
|-----------|-------|-------|----------------|
| `avg_eval_input_bce` | Input (ticks 0..T-1) | All motor units | Whether the model suppresses motor during input (all targets = 0) |
| `avg_eval_output_neg_bce` | Output (ticks T..2T-1) | Negative units (38/39) | Zero-target unit suppression |
| `avg_eval_output_pos_bce` | Output (ticks T..2T-1) | Positive unit (1/39) | Active phoneme unit activation (target = 1) |
| `avg_eval_n_active_{input,output_pos,output_neg}` | — | — | Count of units outside the dead zone |

All values are prefixed `avg_eval_` to distinguish them from the online training
`avg_loss` — they are computed in a separate eval-mode forward pass after each epoch
with the final epoch weights and do not numerically match the online loss. Enabled by
default; disable with `--no-log-loss-decomp` for long runs.

### 15.2 Validation

- Projection-specific tests (`tests/test_sound_projection.py`): **17 passed**.
- Loss decomposition tests (`tests/test_loss_decomposition.py`): **8 passed**.
- Full test suite: **376 passed**.
- Smoke runs for both baseline and dense20 completed with finite losses.

### 15.3 Controlled runs (200 words, 50 epochs, seed=0)

Two controlled runs with identical settings except the sound projection:

| Run | Output directory |
|-----|-----------------|
| Baseline (no projection) | `outputs/repetition_only_baseline_small_decomp/20260622_015455` |
| Dense 39→20 (`--sound-proj-size 20`) | `outputs/repetition_only_dense20_small_decomp/20260622_015640` |

**Shared settings:**
```
source=words  max_items=200  epochs=50  lr=0.01  seed=0
zero_error_radius=0.1  eval_radius=0.1  loss_reduction=sum  device=cpu
```

### 15.4 Results after 50 epochs

| Metric | Baseline | Dense 39→20 | Notes |
|--------|----------|-------------|-------|
| Final avg BCE | 20.3011 | 19.8860 | Dense slightly lower |
| Phoneme argmax accuracy | 0.0476 | 0.0999 | Dense improves phoneme-level ranking |
| Mean positive unit output | 0.1021 | 0.1197 | Dense slightly stronger |
| Mean max motor per output tick | 0.1757 | 0.2031 | Dense slightly stronger motor peaks |
| Output positive threshold acc (> 0.9) | 0.0000 | 0.0000 | No near-saturation phoneme production in either |
| Exact match | 0/200 | 0/200 | No word-level success |
| Paper-like word accuracy (output phase) | 0/200 | 0/200 | No strict repetition success |

### 15.5 Loss decomposition at final epoch (eval pass, zero_error_radius=0.1)

| BCE component | Baseline | Dense 39→20 | Notes |
|---------------|----------|-------------|-------|
| Input-phase BCE | 0.0206 | 0.0668 | Input silence largely learned in both |
| Output-negative BCE | 7.3957 | 7.8313 | Negative units still contribute |
| Output-positive BCE | 14.1906 | 13.6311 | Dominant residual error in both |
| Active neg / pos units (per trial, after dead-zone) | 50 / 6 | 50 / 6 | Same active unit count |

**Reading the decomposition:** input-phase BCE is near-zero, confirming that the
model has learned to suppress motor activation during the listening phase. The main
residual error is in the output-positive component — the single target-1 unit per
output tick is not being driven toward 1. Output-negative BCE is non-trivial but
secondary. The active counts (50 negative, 6 positive units per trial on average
after dead-zone at radius=0.1) tell the same story: many negative units are still
outside the dead zone, but the dominant unresolved error is on the positive units.

### 15.6 Interpretation

The dense 39→20 projection is technically stable (no NaN, finite losses, all tests
passing) and produces modest improvements in phoneme argmax accuracy and positive unit
activation relative to baseline. However, it does not solve repetition. Both baseline
and dense20 fail at word-level repetition after 50 epochs.

The loss decomposition confirms that the BCE decrease observed in earlier runs
(§13) is driven primarily by negative-unit suppression and input silence, not by
positive-unit activation. At the final epoch:
- Input-phase BCE ≈ 0 — silence is learned.
- Output-negative BCE has decreased substantially from its initial value.
- **Output-positive BCE remains by far the largest component.**

The current failure mode appears to be a **low-activation motor-output regime**: the
model learns to suppress many units, but it does not sufficiently amplify the correct
positive phoneme unit during the output phase. This is the mechanistic complement of
the gradient-imbalance analysis in §13: with 38× more zero-target units per tick,
negative-unit suppression dominates the early gradient signal; by the time positive
units are the main residual, the learning rate may be too small relative to the
remaining loss landscape.

The dense projection does not change this regime — the failure mode is the same for
both variants at 50 epochs.

**Do not report the dense-projection result as an improvement over the paper
baseline.** The baseline is itself already an adapted run (English one-hot, 50 epochs,
lr=0.01). The dense-projection run is a further adaptation on top of that. Both are
diagnostic variants for discussion with Yair, not replication claims.

### 15.7 Next diagnostic: dorsal-only sanity check

A reasonable next diagnostic step is an **explicit dorsal-only variant** implemented
as an opt-in experimental flag — not as a default model change or a permanent
replacement for the full architecture. The idea is to test whether the dorsal
repetition route (sound → iSMG → motor, with motor copy-back) can learn under the
same repetition loss when the ventral pathway's contribution to motor output
(`triangularis_to_motor`) is disabled or zeroed.

The motivation: in the full model (Option A), the ventral pathway adds a second input
to the motor layer that is not driven by a direct phoneme-to-phoneme signal during
repetition. Removing or zeroing `triangularis_to_motor` would isolate the dorsal
phonological loop and clarify whether the current failure is specific to the full
dual-route setup or common to the dorsal pathway alone.

**This should be treated as a diagnostic-only exploration.** It is not a silent model
correction and must not be presented as a replication of Ueno et al. 2011 (which uses
the full dual-route architecture). Any dorsal-only run should be discussed with Yair
before drawing comparative conclusions. See also `[Open #1]` (§12) for the original
formulation of this option.

---

## 16. Diagnostic: dorsal-motor-only motor readout (Phase 3j)

### 16.1 What this is and what it is not

**This is not the Lichtheim 2 model.** It is a debugging sanity check implemented as
an opt-in flag. The full dual-route architecture remains the reference model. Results
from dorsal-motor-only runs must be interpreted as diagnostic only and discussed with
Yair before drawing scientific conclusions.

The flag addresses one specific question: does the `triangularis_to_motor` connection
help or hinder repetition learning? Removing it from the motor readout isolates whether
the dorsal iSMG→motor route can, on its own, learn to map phoneme input to motor
output under the repetition loss.

### 16.2 Technical description

Enable with `--dorsal-motor-only`. Effect on `Lichtheim2Model.forward_tick`:

**Full model (default, `dorsal_motor_only=False`):**
```
motor_net = iSMG_to_motor(new_iSMG) + triangularis_to_motor(new_triangularis)
new_motor = sigmoid(motor_net)
```

**Dorsal-motor-only diagnostic (`dorsal_motor_only=True`):**
```
motor_net = iSMG_to_motor(new_iSMG)
new_motor = sigmoid(motor_net)
```

**What is still computed in dorsal-motor-only mode:**
- The full ventral pathway: `new_mSTG`, `new_aSTG`, `new_vATL_out`, `new_triangularis`
- All copy-back mechanisms (iSMG Elman, motor copy-back to iSMG, vATL context)
- The `triangularis_to_motor` weight matrix (still initialized; simply not added to `motor_net`)

**What changes:**
- `triangularis_to_motor` is excluded from `motor_net` → excluded from the loss → receives
  no gradient during dorsal-motor-only training. Ventral pathway weights (`aSTG_to_triangularis`,
  `mSTG_to_aSTG`, `sound_to_mSTG`) also receive no gradient through the motor path, since
  the connection to the loss is severed at `triangularis_to_motor`.

When `dorsal_motor_only=False` (the default), the model is numerically/behaviorally
equivalent to the previous full model — the new conditional adds no overhead to the
default code path.

### 16.3 How to enable

CLI flag (boolean, off by default):
```bash
--dorsal-motor-only
```

Can be combined with `--sound-proj-size N` (dense projection) for a 2×2 experimental
matrix. Example:
```bash
PYTHONPATH=src python scripts/train_repetition_only.py \
  --data-dir data/raw/nwr_swp --config configs/english_nwr.yaml \
  --source words --max-items 200 --epochs 50 --lr 0.01 \
  --device cpu --seed 0 --zero-error-radius 0.1 --eval-radius 0.1 \
  --loss-reduction sum --dorsal-motor-only \
  --output-dir outputs/repetition_only_dorsal_motor_200
```

The `run_config.json` for each run records:
```json
"dorsal_motor_only": true,
"motor_readout_mode": "dorsal_only"
```
(or `false` / `"full"` for the default).

The terminal output in `[1/5]` and `[5/5]` shows:
```
motor_readout:    dorsal only (diagnostic; triangularis_to_motor disabled)
```

### 16.4 Interpretation guide

| Outcome | Interpretation |
|---------|---------------|
| Dorsal-only learns faster / achieves higher phoneme acc | The `triangularis_to_motor` contribution in the full model interferes with repetition; the dorsal route is sufficient for this task |
| Dorsal-only learns similarly to full model | The ventral motor contribution is neutral for repetition; bottleneck is elsewhere (e.g. learning rate, BPTT dynamics, loss imbalance) |
| Dorsal-only fails similarly or worse | The dorsal route alone is not sufficient; the ventral contribution (even if suboptimal) provides a useful signal |

**Do not over-interpret these comparisons.** The dorsal-motor-only variant is not
architecturally faithful to Ueno et al. 2011, and neither is the English one-hot
encoding. All comparisons are within-adaptation-space and should be framed as
diagnostic observations, not model validation results.

---

## 17. Diagnostic: output-phase positive-unit loss weighting (Phase 3k)

### 17.1 What this is and what it is not

**This is not the Lichtheim 2 model.** It is a training-objective diagnostic
implemented as an opt-in scalar weight (default 1.0 = unweighted BCE, identical to
the baseline code path). All comparison runs using `output_positive_weight > 1` are
explicitly labelled diagnostic variants and must not be reported as replications of
Ueno et al. 2011.

The flag addresses one specific question: can amplifying the gradient on the single
correct-phoneme unit per output tick break the low-activation motor-output regime
identified in §15.6?

**Motivation.** In the current failure mode, 38 zero-target units compete with 1
positive-target unit per output tick. With `zero_error_radius=0.1`, negative units
enter the dead zone quickly (their gradient disappears once output < 0.1), but the
positive unit — initially around 0.5 — has a large residual error (`|output−1|≈0.5`)
that never enters the dead zone. However, its *gradient contribution* to the total
loss is diluted by the 77× more numerous zero-target units across the full 2T ticks.

The `output_positive_weight` parameter multiplies the BCE loss for the one positive
unit per output tick by a factor `w > 1`, increasing its gradient relative to the
negative units — without touching the negative-unit loss, the dead zone, or any
architectural parameters.

### 17.2 Technical description

Enable with `--output-positive-weight W` (float, default 1.0). Scoping rules:

- **Task**: applies only when `trial.task == Task.REPETITION`.
- **Phase**: applies only to output-phase ticks (indices T..2T-1), never input-phase.
- **Unit**: applies only to units with `motor_target >= 0.5` (the one-hot active unit).
- **Dead zone**: applied first; the weight multiplier never resuscitates dead-zoned units.

When `output_positive_weight=1.0` (default), the `if` branch is never entered and
the code path is byte-for-byte identical to the previous baseline.

Implementation inside `compute_trial_loss_breakdown` (after building `alive_motor`):

```python
if output_positive_weight != 1.0 and trial.task == Task.REPETITION:
    T = trial.phon_tensor.shape[0]
    output_phase_mask = torch.zeros_like(trial.motor_targets, dtype=torch.bool)
    output_phase_mask[T:, :] = True
    positive_mask = trial.motor_targets >= 0.5
    weighted_mask = output_phase_mask & positive_mask
    weight_motor = torch.ones_like(trial.motor_targets)
    weight_motor = torch.where(
        weighted_mask,
        motor_outputs.new_full(trial.motor_targets.shape, output_positive_weight),
        weight_motor,
    )
    motor_loss = (raw_motor * alive_motor.float() * weight_motor).sum()
else:
    motor_loss = (raw_motor * alive_motor.float()).sum()
```

**What does NOT change:**
- Negative-unit BCE (target < 0.5) — unchanged in all phases.
- Input-phase BCE — target = 0 everywhere, so no positive units exist there.
- Post-epoch eval decomposition (`_compute_epoch_loss_decomp`) — uses a separate
  code path (`_compute_trial_loss_decomposition`, which always uses unweighted BCE).
  Eval metrics (`avg_eval_output_pos_bce`, etc.) are always comparable across runs
  regardless of training weight.
- Model architecture, copy-back, sigmoid, motor targets, trial construction.
- SPEAKING trials — `trial.task != Task.REPETITION` → weight branch never taken.

### 17.3 How to enable

CLI flag (float, default 1.0):
```bash
--output-positive-weight 5.0
```

Can be combined with `--sound-proj-size N` and/or `--dorsal-motor-only`. Example:
```bash
PYTHONPATH=src python scripts/train_repetition_only.py \
  --data-dir data/raw/nwr_swp --config configs/english_nwr.yaml \
  --source words --max-items 200 --epochs 50 --lr 0.01 \
  --device cpu --seed 0 --zero-error-radius 0.1 --eval-radius 0.1 \
  --loss-reduction sum --output-positive-weight 5.0 \
  --output-dir outputs/repetition_only_posw5_200
```

The terminal `[5/5]` header shows:
```
output_positive_weight: 5.0  [diagnostic; output-phase target=1 units upweighted]
```
(or `1.0  (baseline standard BCE)` at default).

The `run_config.json` for each run records:
```json
"output_positive_weight": 5.0,
"loss_variant": "output_positive_weighted"
```
(or `1.0` / `"standard_bce"` at default).

### 17.4 Risks and scientific notes

- **Loss scale changes**: with `output_positive_weight=W`, total loss is larger by
  up to `W × T × BCE_per_positive_unit`. This may interact with learning rate;
  smoke runs should confirm no NaN/divergence.
- **Gradient imbalance shifts**: upweighting the positive unit shifts the effective
  gradient ratio from ~77:1 (zero:positive targets) toward a lower ratio. Whether
  this crosses a threshold to cause reliable positive-unit activation vs. instability
  is empirical.
- **Not paper behavior**: the paper uses unweighted BCE. This is explicitly a
  diagnostic. Label all results accordingly and discuss with Yair.
- **Eval metrics remain unweighted**: `avg_eval_output_pos_bce` is always computed
  with `output_positive_weight=1.0` regardless of training weight. This ensures that
  eval decomposition values are comparable across runs.

### 17.5 Interpretation guide

| Outcome | Interpretation |
|---------|---------------|
| Higher weight → mean_positive_output rises significantly | Gradient amplification on the positive unit is sufficient to break the suppression regime; consider a sweep of weight values |
| Higher weight → instability or NaN | Learning rate (0.01) too high relative to the amplified loss scale; try lower lr |
| Higher weight → no improvement in mean_positive_output | The bottleneck is not gradient imbalance per se; may be learning rate, BPTT dynamics, or architecture capacity |

**Do not report these results as model validation.** Any run with
`output_positive_weight != 1.0` is labelled `loss_variant: output_positive_weighted`
in `run_config.json` and must be discussed with Yair before comparative
interpretation.

---

## 18. Output-positive weighting and learning-rate stabilization (Phase 3k results)

### 18.1 Prior diagnostic results — the low-activation regime

Before the output-positive weighting runs, all controlled variants (baseline, dense20,
dorsal-motor-only) shared the same failure mode after 50 epochs:

- Exact match: **0/200**; paper-like word accuracy: **0/200**.
- Output-positive BCE was the dominant residual loss component in all variants.
- `mean_positive_output` stayed well below 0.5 — the model entered and remained in
  a low-activation motor-output regime.
- Dense20 improved phoneme argmax accuracy (0.05 → 0.10) and mean positive output
  slightly, but did not escape the regime.
- Dorsal-motor-only also provided only marginal improvement.

The decomposition confirmed (§15.5–15.6): input silence was largely learned, negative
units were being suppressed, but the single positive phoneme unit per output tick was
not being driven toward 1. The hypothesis was that gradient imbalance — 38 negative
zero-target units vs. 1 positive unit per tick — was suppressing the positive gradient
early enough to leave the model stuck.

### 18.2 Output-positive weighting — controlled 50-epoch runs (200 words)

**Shared settings:** `source=words  max_items=200  epochs=50  lr=0.01  seed=0`
`zero_error_radius=0.1  eval_radius=0.1  loss_reduction=sum  sound_proj_size=20  device=cpu`

| Configuration | Epochs | lr | Phoneme argmax acc | Mean pos. output | Mean max motor | Output pos > 0.9 | Exact match | Paper-like word acc | Output-pos BCE | Output-neg BCE |
|---|---|---|---|---|---|---|---|---|---|---|
| Dense20 + posw1.5 | 50 | 0.01 | 0.2204 | 0.1906 | 0.3410 | 0.0000 | 0/200 | 0/200 | 11.9678 | 9.9335 |
| Dense20 + posw2.0 | 50 | 0.01 | 0.3066 | 0.3023 | 0.4930 | 0.0039 | 0/200 | 0/200 | 9.4314 | 13.0921 |
| Dense20 + posw3.0 | 50 | 0.01 | 0.3677 | 0.4639 | 0.6747 | 0.0746 | 2/200 | 0/200 | 6.9146 | 18.7929 |

**Reading the table:**

- Increasing `output_positive_weight` monotonically improves phoneme argmax accuracy
  and mean positive unit activation: `posw3` is the first setting to show non-zero exact
  match (2/200) and non-zero strong positive activation (`output_pos > 0.9` fraction: 0.075).
- A clear selectivity trade-off is visible: as `output_positive_weight` rises, output-positive
  BCE falls while output-negative BCE rises. The model is learning to activate the positive
  unit but at the cost of partially suppressing fewer negative units — the gradient balance
  has shifted, but at the expense of global precision.
- Paper-like word accuracy (all units within `eval_radius=0.1`) remains 0/200 across all three
  settings: strict whole-word accuracy under the radius criterion is not yet achieved.

### 18.3 Learning-rate comparison — 150-epoch runs (200 words, dense20 + posw3)

With `posw3` identified as the first setting to produce meaningful positive activation, a 150-epoch
learning-rate comparison was run. All other settings were fixed.

**Shared settings:** `source=words  max_items=200  epochs=150  seed=0  zero_error_radius=0.1`
`eval_radius=0.1  loss_reduction=sum  sound_proj_size=20  output_positive_weight=3.0  device=cpu`

| Configuration | Epochs | lr | Phoneme argmax acc | Mean pos. output | Mean max motor | Output pos > 0.9 | Exact match | Paper-like word acc | Output-pos BCE | Output-neg BCE | Input BCE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dense20 + posw3 | 150 | 0.01 | 0.4193 | 0.5224 | 0.7495 | 0.1597 | 6/200 | 0/200 | 6.3210 | 15.3653 | 0.0269 |
| Dense20 + posw3 | 150 | 0.005 | **0.8191** | **0.7296** | 0.7717 | 0.2675 | **79/200** | 0/200 | **2.4654** | 8.8548 | 0.1444 |
| Dense20 + posw3 | 150 | 0.003 | 0.6480 | 0.5922 | 0.6777 | 0.1054 | 25/200 | 0/200 | 4.2621 | 11.7648 | 0.1087 |

Output directories:

| Run | Output directory |
|-----|-----------------|
| lr=0.01 | `outputs/repetition_only_dense20_posw3_150ep_200/20260622_165214` |
| lr=0.005 | `outputs/repetition_only_dense20_posw3_lr005_150ep_200/20260622_165839` |
| lr=0.003 | `outputs/repetition_only_dense20_posw3_lr003_150ep_200/20260622_172616` |

**Reading the table:**

- `lr=0.005` is substantially better than both `lr=0.01` and `lr=0.003` at 150 epochs.
  It is the first diagnostic setting to produce substantial repetition-like learning:
  phoneme argmax accuracy reaches **0.8191** and exact whole-word match reaches **79/200**.
- `lr=0.01` converges faster early but reaches a worse local attractor at 150 epochs than
  `lr=0.005`; 6/200 exact matches vs. 79/200.
- `lr=0.003` is slower and also outperformed by `lr=0.005`; the curve appears under-trained
  at 150 epochs.
- The `lr=0.005` run also shows a lower output-negative BCE (8.85 vs. 15.37 at lr=0.01),
  suggesting better overall selectivity, not just stronger positive activation.
- Input BCE is slightly higher for `lr=0.005` than `lr=0.01` (0.144 vs. 0.027), indicating
  slightly less perfect input silence — a minor trade-off.
- **Paper-like word accuracy remains 0/200 in all three runs.** The model escapes the
  low-activation regime, but does not yet satisfy the strict all-units-within-radius criterion.

### 18.4 Main conclusion

The best diagnostic setting so far is **dense20 + full motor readout + `output_positive_weight=3.0`
+ `lr=0.005` for 150 epochs**. This is the first setting that shows substantial
repetition learning: phoneme argmax accuracy reaches **0.8191** and exact whole-word
match reaches **79/200**. The model also escapes the previous low-activation regime,
with mean positive output rising to **0.7296** and output-positive BCE dropping to
**2.4654**.

However, this is still not a faithful Ueno-style result: **paper-like word accuracy
remains 0/200** under the current radius-based criterion (`eval_radius=0.1`, all motor
units within radius on all output-phase ticks). The model is learning useful output
rankings and producing stronger motor activations, but strict all-units-within-radius
repetition is still unsolved.

This result is informative because it **localizes the failure mode**: the model can
learn repetition-like behavior when the training objective gives sufficient pressure to
output-positive units, suggesting that the original low-activation failure was at least
partly optimization- and loss-imbalance-driven rather than a fundamental architectural
limitation.

**Scientific caution:** these are diagnostic training adaptations. `output_positive_weight`,
`sound_proj_size`, and the learning rate chosen here (0.005) are not settings from
Ueno et al. 2011. These results should not be presented as a faithful reproduction of
the paper's training dynamics without discussion with Yair.

### 18.5 Interpretation summary

| Intervention | Effect |
|---|---|
| Dense 39→20 projection alone | Modest improvement in phoneme argmax (~0.05→0.10) and mean positive output; did not escape low-activation regime |
| Dorsal-motor-only | Marginal improvement; did not solve the regime |
| `output_positive_weight > 1` | First intervention to clearly break the low-activation regime; positive activation rises monotonically with weight |
| Lower lr (0.01→0.005) with posw3 | Decisive: 6→79 exact matches at 150 epochs; much better overall |
| lr=0.003 vs. 0.005 | Under-trained at 150 epochs; 0.005 is the better choice |

**What remains unsolved:** strict output selectivity. Positives are stronger, but
not all words satisfy the all-units-within-radius paper-like criterion. The remaining
bottleneck appears to be output precision: some positive units are not yet above
`eval_radius` from their target of 1.0, and/or some negative units are not yet below
`eval_radius` from their target of 0.0, causing 0/200 on the paper-like criterion even
while 79/200 exact matches (argmax-based) are achieved.

### 18.6 Suggested next steps

1. **Discuss with Yair** whether `output_positive_weight` is acceptable as a diagnostic
   only, or whether we should search for a more faithful explanation of how the original
   paper's model avoided this gradient-imbalance problem (e.g. initialization, learning
   rate schedule, LENS-specific training dynamics, or mora-based phonology that avoids
   the 39-dimensional one-hot sparsity).

2. **Inspect predictions from the best run** (`lr=0.005`, 150 epochs, 79/200 exact match):
   - Which 79 words are exact matches? Are they shorter or higher-frequency?
   - Where does the paper-like radius criterion fail — are positives not above `1 − eval_radius`,
     negatives not below `eval_radius`, or both?
   - Are timing mismatches contributing (e.g. correct phoneme but off by one tick)?

3. **Consider scaling to more items** with the best diagnostic setting:
   - `max_items=1200`, `epochs=150 or 200`, `lr=0.005`, `sound_proj_size=20`,
     `output_positive_weight=3.0`.
   - Clearly label this as a diagnostic run, not a paper replication.
   - Monitor for overfitting and gradient stability at 1200 items.

---

## 19. First diagnostic full-set solution (500-epoch run, Phase 3k)

### 19.1 Run configuration

The same best diagnostic setting from §18 was extended to 500 epochs:

```
Run directory:      outputs/repetition_only_dense20_posw3_lr005_500ep_200/20260622_213921
max_items:          200
epochs:             500
lr:                 0.005
sound_proj_size:    20         [dense 39→20 projection, not in Ueno et al. 2011]
output_positive_weight: 3.0   [diagnostic loss weighting, not in Ueno et al. 2011]
loss_reduction:     sum
zero_error_radius:  0.1
eval_radius:        0.1
seed:               0
motor_readout:      full  (iSMG + triangularis)
```

### 19.2 Final metrics

| Metric | Value |
|---|---|
| Phoneme argmax accuracy | **1.0000** |
| Threshold accuracy (mixed) | **1.0000** |
| Whole-word exact match | **200/200** |
| Input silence threshold acc | **1.0000** |
| Output negative threshold acc | **1.0000** |
| Output positive threshold acc | **1.0000** |
| Mean positive output | **0.9384** |
| Mean negative output | **0.0058** |
| Mean max motor output | **0.9384** |
| Output phase all-within-radius | **200/200** |
| Input phase all-silent | **200/200** |
| Strict trial (input + output) | **200/200** |
| **Paper-like word accuracy** | **200/200** |
| Final avg loss | **0.000000** |
| Input-phase BCE | **0.0000** |
| Output-negative BCE | **0.0000** |
| Output-positive BCE | **0.0000** |
| Active neg / pos after dead-zone | **0 / 0 per trial** |

### 19.3 Interpretation

This is the first diagnostic setting that fully solves the 200-word repetition
subset under the current radius-based criterion (`eval_radius=0.1`,
all-units-within-radius on the output phase).

This should not be described as a faithful Ueno et al. 2011 reproduction, because
the setting uses diagnostic adaptations that are not in the original paper: a dense
39→20 sound input projection (`sound_proj_size=20`) and explicit output-positive
loss weighting (`output_positive_weight=3.0`). Both adaptations were introduced to
address the low-activation motor-output regime identified in earlier diagnostic runs
(§15.6, §18.1) and must be labelled as such in any presentation or report.

**Why does the final loss reach zero?**
The final zero loss does not mean that raw BCE is mathematically zero. It occurs
because `zero_error_radius=0.1` masks all supervised units once they fall within the
target radius: units with `|output − target| < 0.1` contribute zero loss (they enter
the dead zone). At epoch 500, the active neg/pos count after the dead-zone is 0/0
per trial, meaning every supervised unit in every word has converged within radius
of its target. This is consistent with the 200/200 strict-trial result and the
threshold accuracy figures of 1.0000 — it is the correct and expected outcome once
the model fully solves the radius-based criterion.

**Training dynamics.**
The 500-epoch trajectory is not uniformly smooth. Transient instability — visible
as loss spikes — was observed around the mid-training phase. The model eventually
recovers from these spikes and converges to full radius-based success by epoch 500.
This suggests the learning rate (0.005) is at or near the upper boundary of
stability for this setting; further stabilization experiments (lr warmup, lower lr)
may be warranted before scaling.

### 19.4 Comparison table: diagnostic run progression

| Configuration | Epochs | lr | Phoneme acc | Exact match | Paper-like word acc | Mean pos out | Mean neg out | Output-pos BCE | Output-neg BCE |
|---|---|---|---|---|---|---|---|---|---|
| Dense20 + posw3 | 150 | 0.005 | 0.8191 | 79/200 | 0/200 | 0.7296 | 0.0409 | 2.4654 | 8.8548 |
| Dense20 + posw3 | 300 | 0.005 | 0.9963 | 194/200 | 1/200 | 0.8940 | 0.0126 | 0.5641 | 1.7823 |
| Dense20 + posw3 | 500 | 0.005 | **1.0000** | **200/200** | **200/200** | **0.9384** | **0.0058** | **0.0000** | **0.0000** |

The progression is monotone across all three checkpoints: phoneme accuracy, exact
match, paper-like word accuracy, mean positive output, and output-positive BCE all
improve consistently from 150 → 300 → 500 epochs. The gap between 300-epoch
(194/200 exact, 1/200 paper-like) and 500-epoch (200/200 both) narrows the
bottleneck identified in the prediction error analysis (§19.5 below): the remaining
6 exact-match failures at 300 epochs, and the widespread below-threshold positive
activations, are resolved by continued training.

### 19.5 Main conclusion

The result shows that the current PyTorch implementation can learn strict repetition
on the 200-word subset when the low-activation regime is addressed by diagnostic
objective and representation changes. The remaining scientific questions are:

- whether this convergence can be made faithful to the original Ueno et al. 2011
  model (without `output_positive_weight` or dense projection);
- whether it scales to the full 1200-word set.

### 19.6 Next steps

1. **Interpret the diagnostic result with Yair.** This is successful diagnostic
   learning, not a faithful replication. The distinction matters: the original paper
   used standard BCE with no positive-unit upweighting, and Japanese mora features
   rather than English one-hot phonemes. Understanding why the paper's setting did
   not encounter the same low-activation regime — whether due to initialization,
   learning rate schedule, mora-feature structure, or LENS-specific training dynamics
   — is the key scientific question.

2. **Scale the best diagnostic setting to 1200 words:**

   ```bash
   PYTHONPATH=src python scripts/train_repetition_only.py \
     --data-dir data/raw/nwr_swp --config configs/english_nwr.yaml \
     --source words --max-items 1200 --epochs 500 --lr 0.005 \
     --device cpu --seed 0 --zero-error-radius 0.1 --eval-radius 0.1 \
     --loss-reduction sum --sound-proj-size 20 --output-positive-weight 3.0 \
     --output-dir outputs/repetition_only_dense20_posw3_lr005_500ep_1200
   ```

   Label this explicitly as a diagnostic run (not a paper replication). Monitor for
   training instability (the transient spikes observed at 200 words may worsen with
   more items) and for generalization patterns — the 200-word result is on the
   training set, so per-word accuracy at 1200 items will reveal whether the model
   generalizes or overfits within the diagnostic setting.

3. **Investigate faithful alternatives.** Determine whether a more faithful
   loss and representation choice — standard BCE (no `output_positive_weight`),
   without dense projection — could reproduce the same convergence. Candidate
   directions: phonologically-structured English phoneme features (to replace
   one-hot, closer in spirit to the paper's mora features), learning rate schedule
   (e.g. cosine annealing), or longer training at a lower base rate. Any such
   experiment should be discussed with Yair before running, as it bears directly
   on whether the implementation constitutes a replication of Ueno et al. 2011.

---

## 20. Diagnostic scaling attempt: 1200-word run and plotting utilities

### 20.1 1200-word run — not solved

Following the 200-word success in §19, the same diagnostic setting was applied to
1200 words (the full word set available from `wfe.csv`):

```
Run directory:          outputs/repetition_only_dense20_posw3_lr005_500ep_1200/20260623_104531
max_items:              1200
epochs:                 500
lr:                     0.005
sound_proj_size:        20    [diagnostic]
output_positive_weight: 3.0   [diagnostic]
loss_reduction:         sum
zero_error_radius:      0.1
eval_radius:            0.1
seed:                   0
```

**Final metrics after 500 epochs:**

| Metric | Value |
|---|---|
| Phoneme argmax accuracy | 0.4835 |
| Whole-word exact match | **91/1200** |
| Paper-like word accuracy | **0/1200** |
| Input silence threshold acc | 0.9998 |
| Output negative threshold acc | 0.8159 |
| Output positive threshold acc | 0.1182 |
| Mean positive output | 0.5003 |
| Mean negative output | 0.0640 |
| Output-positive BCE | 6.4193 |
| Output-negative BCE | 13.3695 |
| Input-phase BCE | 0.0147 |
| Active neg / pos after dead-zone | 44 / 5 per trial |

**Interpretation:** The best diagnostic setting does not scale directly from 200 to
1200 words after 500 epochs. Input silence is essentially solved (0.9998) — the same
pattern as in the 200-word progression. However, both output-positive and
output-negative BCE remain high, and the active dead-zone counts (44 / 5 per trial)
confirm that the model has not brought most units within the radius-based criterion.
This is a qualitatively different failure mode from the 200-word case: at 1200
words, the model is still in an early-to-mid convergence regime at epoch 500, with
mean positive output of only 0.5003 (compared to 0.9384 at 200 words).

**What this is not:** This run used the same diagnostic adaptations as the 200-word
run (dense projection, output-positive weighting). It is not a test of the faithful
Ueno et al. 2011 setting at 1200 words. The faithful multi-task training at full
scale remains pending.

### 20.2 Plotting improvements

Two plotting improvements were added alongside the diagnostic runs:

**`scripts/train_repetition_only.py` — updated `save_loss_curve()`:**
- `loss_curve.png` now shows **average loss only** (no min/max fill). The previous
  fill_between(min, max) was causing early `max_loss` spikes to compress the y-axis
  and render the average trajectory invisible on long runs.
- An optional rolling mean (default window=10) is overlaid as a visual smoothing.
  The rolling mean is purely cosmetic — it does not affect training.
- `loss_decomposition_curve.png` is automatically saved alongside `loss_curve.png`
  if the `avg_eval_output_pos_bce`, `avg_eval_output_neg_bce`, and `avg_eval_input_bce`
  columns are present in the epoch metrics (requires `--log-loss-decomp`, which is on
  by default). This makes it easy to see which BCE component remains the bottleneck.

**`scripts/plot_repetition_metrics.py` — new standalone replotting script:**
Regenerates plots for any completed run without rerunning training. Reads
`metrics.csv` from a run directory and saves:
- `loss_curve.png` — avg loss + rolling mean (default window=20)
- `loss_curve_range_clipped.png` — avg/min/max with y-axis clipped to the
  `clip_percentile` (default 95th percentile) of `avg_loss`, so early `max_loss`
  spikes do not compress the scale
- `loss_decomposition_curve.png` — output-positive, output-negative, input BCE

Each optional plot is silently skipped if its required columns are absent from
`metrics.csv`. Example:

```bash
python scripts/plot_repetition_metrics.py \
  --run-dir outputs/repetition_only_dense20_posw3_lr005_500ep_1200/20260623_104531 \
  --rolling-window 20
```

### 20.3 Current status summary

| Experiment | Words | Epochs | lr | Exact match | Paper-like | Status |
|---|---|---|---|---|---|---|
| Dense20 + posw3 | 200 | 500 | 0.005 | 200/200 | 200/200 | Solved (diagnostic conditions) |
| Dense20 + posw3 | 1200 | 500 | 0.005 | 91/1200 | 0/1200 | Not solved |
| Faithful Ueno replication (multitask) | — | — | — | — | — | Pending |

The 200-word result demonstrates that the current PyTorch implementation can converge
to strict radius-based repetition under diagnostic conditions. The 1200-word result
shows that scaling is non-trivial: 6× more items requires substantially more
convergence capacity (more epochs, lower lr, or a different objective). Faithful
replication at any scale remains pending.
