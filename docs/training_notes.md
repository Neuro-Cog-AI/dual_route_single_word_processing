# Training Notes

Expected training logic for the Lichtheim 2 PyTorch reimplementation.

---

## Training Paradigm: Online Item-by-Item Updates

The original model was trained with **online learning**: weights are updated after each word presentation, not after each epoch `[Supp]`.

This means:
- No mini-batching.
- No shuffled batch of gradients: each word's gradient is applied immediately.
- This is equivalent to `batch_size=1` SGD in PyTorch terms.

---

## Per-Word Presentation Schedule

Each word in the vocabulary is presented the following number of times per epoch `[Paper/Supp]`:

| Task | Presentations per word per epoch |
|------|----------------------------------|
| Repetition | 1 |
| Speaking / naming | 2 |
| Comprehension | 3 |

Total presentations per word per epoch: **6**.

The order of presentations within an epoch is `[Open]` — likely randomised across task types and words, but the exact shuffling strategy should be confirmed from the supplement.

---

## Tick Counts per Task

| Task | Ticks |
|------|-------|
| Repetition | 6 (3 input + 3 output) `[Paper]` |
| Comprehension | 3 `[Paper]` |
| Speaking / naming | 3 `[Paper]` |

---

## Loss Function

The error was estimated with the **cross-entropy method** `[Paper — Hinton, 1989]`.

PyTorch translation note `[Inferred]`: the LENS cross-entropy formulation operates unit-by-unit on sigmoid outputs. The standard PyTorch `nn.BCELoss` (binary cross-entropy) is the natural equivalent, but the exact equivalence with the LENS implementation (especially with the zero-error radius and frequency scaling applied before gradient computation) needs to be verified.

---

## Zero-Error Radius

No error is backpropagated from a unit if `|output − target| < 0.1` `[Paper]`.

This is applied before computing the gradient, not as a loss modification. In PyTorch this must be implemented explicitly: mask out the gradient contribution of any unit already within 0.1 of its target value.

---

## Learning Rate and Schedule

- **Epochs 1–150:** learning rate = 0.5 `[Paper]`
- **Epochs 151–180:** learning rate reduced by 0.1 every 10 epochs (0.5 → 0.4 → 0.3 → 0.2) `[Paper]`
- **Epochs 181–200:** learning rate fixed at 0.1 `[Paper]`
- **Momentum:** not used `[Paper]`
- **Weight decay:** same schedule as learning rate; initialised at 10⁻⁶, reduced by 10⁻⁷ every 10 epochs from epoch 150–180, then fixed at 6×10⁻⁷ until epoch 200 `[Paper]`
- PyTorch translation `[Inferred]`: implement as a custom LR scheduler; weight decay can be passed to the SGD optimizer or applied manually.

---

## Weight Initialisation

- Most weights: uniform in **[−1, 1]** `[Paper]`
- Recurrent connections (Elman): uniform in **[−0.5, 0.5]** `[Paper]`
- Bias weights to hidden units: initialised at **−1** to suppress strong hidden unit activation early in training `[Paper]`
- Copy and Elman context layers: no trainable bias `[Paper]`

---

## Evaluation Metric

- Per-task word-level accuracy: a word is "correct" if all output units are within the zero-error radius of their target values at the scored ticks `[Inferred from paper description of Figure 2]`
- Accuracy reported as proportion of words correct per epoch, separately for each task `[Paper Figure 2]`

---

## Training Loop — Phase 3c-1 Implementation

Online (item-by-item) training is implemented in `src/lichtheim2/trainer.py`:

```python
loss = train_step(model, trial, optimizer, cfg, zero_error_radius=0.0, device="cpu")
```

Loss computation (`src/lichtheim2/losses.py`):
- Binary cross-entropy on sigmoid outputs vs binary targets
- Motor loss: applied at all ticks indicated by `motor_loss_mask`
- Semantic loss: applied at all ticks indicated by `semantic_loss_mask` (comprehension)
- Zero-error radius: optional dead-zone threshold (default 0.0 = disabled; paper: 0.1)

Device handling: `init_state`, `forward_tick`, and `run_trial` all propagate the model device. `trainer.py` moves `SupervisedTrial` tensors to the target device before each step.

## Training Loop — Phase 3c-2 Implementation

Phase 3c-2 adds a real-data repetition training script (`scripts/train_repetition_real_data.py`) with four helper functions:

- `load_repetition_items(data_dir, source)` — loads `WordItem`/`PseudowordItem` from CSV and returns `(phon_tensor, label, item_id)` tuples. Labels always include a source prefix (`word:` or `pseudo:`) so word and pseudoword row indices never collide.
- `sample_items(items, max_items, rng)` — takes a random subset for smoke runs; `--max-items` is applied to the pooled list (not per-source).
- `build_repetition_trials(items, motor_size)` — calls `make_repetition_trial()` per item, preserving `item_id` and `label` for traceability.
- `train_repetition_epochs(model, trials, optimizer, cfg, epochs, zero_error_radius, device, rng)` — shuffles trials each epoch and calls `train_step()` item-by-item.

CLI arguments: `--data-dir`, `--config`, `--source words|pseudowords|mixed`, `--max-items`, `--epochs`, `--lr`, `--device`, `--seed`, `--zero-error-radius` (default 0.1, per paper; 0.0 disables dead-zone for debugging).

Device handling: if `--device cuda` or `--device mps` is requested but the backend is unavailable, the script raises an explicit error and exits 1. No silent fallback.

Scope (Phase 3c-2 only):
- Repetition task only. No comprehension, speaking, batching, EOS, or semantic vectors.
- No LR schedule, frequency weighting, accuracy logging, checkpoints, or plots.
- Full multi-task epoch loop deferred to a later sub-step.

## Training Loop — Phase 3c-3 Implementation

Phase 3c-3 adds a real-data multi-task training script (`scripts/train_multitask_real_data.py`) with three helper functions:

- `build_word_trials(word_items, sem_map, motor_size)` — builds REPETITION, COMPREHENSION, and SPEAKING trials for each real word. Semantics are assigned to **all** words before sampling so each word's vector is stable regardless of subset. Raises `ValueError` if any `WordItem.row_index is None`. Labels: `word:{row_index}:{word}`.
- `build_pseudo_trials(pseudo_items, motor_size)` — builds REPETITION-only trials for pseudowords. Labels: `pseudo:{row_index}`.
- `train_multitask_epochs(model, trials, optimizer, cfg, epochs, zero_error_radius, device, rng)` — shuffles all trials each epoch (fully random, per-item); returns `list[dict[Task, list[float]]]` keyed by task. Prints per-task avg loss per epoch.

CLI defaults: `--max-words 50`, `--max-pseudowords 50` (safe defaults to avoid ~20k+ trials/epoch before full training is intended).

Semantic vectors: first-pass random binary vectors from `assign_artificial_semantics()` (prototype-based faithful generation deferred to Phase 3c-4+).

Task semantics:
- For COMPREHENSION: `sem_tensor` is the vATL **output target** (what vATL should produce).
- For SPEAKING: `sem_tensor` is the vATL **input** (clamped at each tick).

Scope (Phase 3c-3 only):
- No presentation schedule (1× rep, 2× spk, 3× comp per word). Each trial appears once per epoch.
- No LR schedule, frequency weighting, accuracy logging, checkpoints, or plots.

## Training Loop — Phase 3c-4 Implementation

Phase 3c-4 adds a stability diagnostic script (`scripts/diagnose_small_subset_training.py`).

Key design decisions:
- New script (not an extension of 3c-3): different defaults, different purpose, richer diagnostic output. Helper functions from `train_multitask_real_data.py` are imported directly (safe since that script is guarded by `if __name__ == "__main__"`).
- `sample_items(items, max_count, rng)`: `None` = use all, `0` = use none, `n > 0` = sample up to n. Avoids truthiness bugs where `0` would be treated as "use all".
- CSV loading is mode-selective: `words-multitask` never touches `ssp.csv`; `repetition`/`mixed-multitask` load pseudowords only if `max_pseudowords != 0`.
- Zero-trial guard: if trial list is empty after building, exits with a clear error.
- Non-finite loss guard in `run_diagnostic_epochs`: raises `RuntimeError` with epoch, task, and label context rather than silently continuing or returning NaN stats.
- `DiagnosticResult` dataclass: `epoch_losses`, `epoch_avgs`, `initial_avg`, `final_avg`, `best_avg`, `best_epoch` (1-indexed). Allows tests to assert on specific statistics.
- Safer defaults than pipeline-validation scripts: `--lr 0.01`, `--zero-error-radius 0.0` (no dead-zone, good for debugging), `--epochs 20`, `--max-words 10`, `--max-pseudowords 10`.

## Training Loop — Phase 3c-5 Implementation

Phase 3c-5 adds a loss breakdown utility and a standalone audit script.

### `LossBreakdown` dataclass (`src/lichtheim2/losses.py`)

Returned by `compute_trial_loss_breakdown()`. Fields:

| Field | Type | Description |
|---|---|---|
| `task` | `Task` | Task enum value |
| `n_ticks` | `int` | Total ticks in trial |
| `total_loss` | `Tensor` | Scalar, grad-capable |
| `motor_loss` | `Tensor` | Scalar, grad-capable |
| `semantic_loss` | `Tensor` | Zero for REP/SPK; grad-capable |
| `n_active_motor` | `int` | Active (tick × unit) pairs for motor |
| `n_active_semantic` | `int` | Active (tick × unit) pairs for semantic; 0 for REP/SPK |
| `motor_loss_per_unit` | `float` | `motor_loss / n_active_motor`; `nan` if 0 |
| `semantic_loss_per_unit` | `float` | `semantic_loss / n_active_semantic`; `nan` if 0 |

"Active" means: tick's loss mask is True AND (if `zero_error_radius > 0`)
the unit is outside the dead zone. This applies to ALL output elements
regardless of target value — even zero-target units contribute to BCE unless
excluded by the dead zone.

`compute_trial_loss()` is refactored to call `compute_trial_loss_breakdown()`
and return `.total_loss`. Interface and numerical output are unchanged.
`compute_trial_loss_breakdown()` does not call `torch.no_grad()` internally —
gradient flows through for training use. The audit script wraps calls in
`torch.no_grad()` because it does not need gradients.

### Expected active unit counts for English config (motor=39, vATL=50)

| Task | Active motor units | Active semantic units | Total active |
|---|---|---|---|
| REPETITION | 2T × 39 = 78T | 0 | 78T |
| COMPREHENSION | T × 39 = 39T | T × 50 = 50T | 89T |
| SPEAKING | T × 39 = 39T | 0 | 39T |

Interpretation: if `mot/unit ≈ sem/unit`, comprehension cost is explained
by more active elements. If `sem/unit >> mot/unit`, semantic targets are
genuinely harder to learn — expected early in training when vATL output is
far from the binary semantic target.

### Audit script (`scripts/audit_loss_scaling.py`)

Runs in eval mode with `torch.no_grad()`. No optimizer, no gradient step.
Prints a per-trial breakdown table and per-task summary (mean over N words).
CLI: `--data-dir`, `--config`, `--max-words` (default 5), `--seed`, `--device`
(with same CUDA/MPS validation as other scripts), `--zero-error-radius`.

## Training Loop — Phase 3c-6 Implementation

Phase 3c-6 adds an optional `loss_reduction` parameter to the core loss functions and the diagnostic script.

### `compute_trial_loss()` — `loss_reduction` parameter

```python
compute_trial_loss(tick_results, trial, zero_error_radius=0.0, loss_reduction="sum")
```

| Value | Behavior |
|---|---|
| `"sum"` | Returns `breakdown.total_loss` — unchanged from Phase 3c-5. Default. Preserves existing behavior exactly. |
| `"mean_active"` | Returns `breakdown.total_loss / (n_active_motor + n_active_semantic)`. For diagnostics only. |

Raises `ValueError` if `loss_reduction` is any other string, or if `mean_active` is used and `n_active == 0` (all elements masked or in dead zone).

`compute_trial_loss_breakdown()` is unchanged and gradient-compatible. Division by `n_active` (a Python int) preserves gradient flow through `total_loss`.

### When to use `mean_active`

`mean_active` is intended for small-subset stability diagnostics where task imbalance (comprehension has more active elements than speaking/repetition) makes raw summed loss comparisons misleading. It is **not** yet adopted as the paper training objective.

### `train_step()` — `loss_reduction` parameter

```python
train_step(model, trial, optimizer, cfg, zero_error_radius=0.0, device="cpu", loss_reduction="sum")
```

Passes `loss_reduction` through to `compute_trial_loss()` as a keyword argument.

### `diagnose_small_subset_training.py` — `--loss-reduction` CLI option

```
--loss-reduction sum|mean_active   (default: sum)
```

`run_diagnostic_epochs()` accepts `loss_reduction: str = "sum"` and passes it to each `train_step()` call. The print header includes `loss_reduction=...` for reproducibility.

## Training Loop — Phase 3c-7 Implementation

Phase 3c-7 adds a comparison script (`scripts/compare_loss_reductions.py`) that runs
both loss reductions under strictly identical conditions.

### Fairness guarantee in `run_comparison()`

Items are loaded and sampled **once** before calling `run_comparison()`; both
reductions share the same `trials` list. For each reduction in sequence:

1. `torch.manual_seed(seed)` — identical model weight initialization
2. `Lichtheim2Model(cfg)` — fresh model
3. `random.Random(seed)` passed as `rng` — identical trial shuffle order every epoch

Same `epochs`, `lr`, `zero_error_radius`, and `device` for both runs.

### Interpreting the output

The summary table reports initial, final, and best average loss plus best epoch.
**Do not compare absolute loss values across reductions** — `sum` and `mean_active`
operate on different scales. Use `% decrease = (initial_avg − final_avg) / initial_avg × 100`
to compare training dynamics.

### `verbose=False` in `run_diagnostic_epochs()`

`run_diagnostic_epochs()` in `diagnose_small_subset_training.py` gained a
`verbose: bool = True` parameter. Default is `True` → existing behavior unchanged.
The compare script passes `verbose=False` to suppress per-epoch logs so only the
final table is printed.

## Training Loop — Phase 3c-8 Implementation

Phase 3c-8 adds a task schedule comparison script (`scripts/compare_task_schedules.py`).

### Schedule definitions

```
SCHEDULES["uniform"] = {REP: 1, COMP: 1, SPK: 1}   # 3 trials/word
SCHEDULES["paper"]   = {REP: 1, COMP: 3, SPK: 2}   # 6 trials/word [Ueno et al. 2011]
```

Pseudowords always receive 1×REP regardless of schedule.

### Within-word trial order

Trials are constructed in the order: REP×n_rep, then COMP×n_comp, then SPK×n_spk.
This order is fixed for reproducibility and testing. `run_diagnostic_epochs()` shuffles
the full trial list every epoch, so construction order does not bias training.

Example (paper schedule, one word): `[REP, COMP, COMP, COMP, SPK, SPK]`

### Fairness guarantee in `run_schedule_comparison()`

Identical to Phase 3c-7:
- Items loaded and sampled once; both schedules share the same `word_items`/`pseudo_items`.
- `torch.manual_seed(seed)` + `random.Random(seed)` per schedule run.
- Same `epochs`, `lr`, `zero_error_radius`, `device`, `loss_reduction`.
- `verbose=False`.

### Interpreting the output

The paper schedule produces 2× more trials per epoch than uniform. **Do not compare
absolute losses directly.** Use `% decrease` and per-task `initial→final` losses:
- `% decrease = (initial_avg − final_avg) / initial_avg × 100` (positive = loss dropped)
- Per-task `initial→final` shows which tasks improve most under each schedule

Default `--loss-reduction mean_active` (unlike other scripts that default to `sum`)
because normalising by active elements is more informative when comparing schedules
with different trial-per-epoch counts.

## Training Loop — Phase 3c-9 Implementation

Phase 3c-9 adds optional frequency weighting as an opt-in diagnostic. Default behavior is unchanged.

### `loss_weight` field on `SupervisedTrial`

```python
loss_weight: float = 1.0  # scalar multiplier on the loss; 1.0 = unweighted (default)
```

All existing `make_*_trial()` calls produce `loss_weight=1.0`. `move_trial_to_device` via `dc_replace` preserves the field automatically (it is a float, not a tensor).

### `train_step()` — weight multiply

```python
if trial_dev.loss_weight != 1.0:
    loss = loss * trial_dev.loss_weight
```

The conditional avoids touching the computation graph when `loss_weight=1.0`. When weight=1.0, `train_step` is numerically identical to the Phase 3c-6 behaviour.

### `compute_word_weights(word_items, frequency_source, normalization)` → `dict[int, float]`

| `frequency_source` | Transform |
|---|---|
| `"zipf"` (default) | `WordItem.zipf_frequency` as-is |
| `"frequency"` | `math.log1p(WordItem.frequency)` (handles zero and right-skewed raw counts) |

Normalisation (`normalization="mean_one"`): `weight_i = value_i / mean(positive_values)`.
- Words with `None` or non-positive transformed frequency → weight 1.0.
- If no words have usable frequency data, all weights default to 1.0.

Returns `dict[row_index → weight]`.

### `apply_weights_to_trials(trials, weight_map)` → `list[SupervisedTrial]`

Only trials whose `label.startswith("word:")` receive weights from `weight_map`.
All other trials (pseudowords, unlabelled) keep `loss_weight=1.0`.

**Why the label guard matters:** `word_items` (wfe.csv) and `PseudowordItems` (ssp.csv)
are loaded from separate files and may have overlapping `row_index` values. Guarding on
`label.startswith("word:")` prevents incorrect weight assignment when a pseudoword
happens to share a `row_index` with a word in `weight_map`.

### Fairness guarantee in `run_frequency_comparison()`

- Base trials are built once; "unweighted" uses them as-is; "weighted" applies `apply_weights_to_trials`.
- Per condition: `torch.manual_seed(seed)` + `random.Random(seed)` for identical initialisation and shuffle.
- Same `schedule`, `epochs`, `lr`, `zero_error_radius`, `device`, `loss_reduction`.
- `verbose=False`.

### Interpreting the output

The "weighted" initial avg may differ from "unweighted" because `train_step()` returns
the weighted loss value. **Compare `% decrease` within each condition, not absolute
loss values across conditions.** Weight stats (min, mean, max) are printed before the
table to aid interpretation.

Default `--loss-reduction mean_active` (same as Phase 3c-8) because normalised loss
is more informative for schedule/weight comparisons.

## Open Issues for Phase 3c+ (continuation)

1. Multi-task epoch loop: 1× repetition, 2× speaking, 3× comprehension per word per epoch `[Paper]`
2. Frequency-weighted presentation rate `[Open — D16]`
3. LR schedule: 0.5 (epochs 1–150) → stepwise decay → 0.1 (epochs 181–200) `[Paper]`
4. Accuracy metric: proportion of words correct per epoch per task `[Inferred]`
5. Phoneme coverage validation is already implemented (see `validate_phoneme_coverage()` in `encoding.py` and coverage tests). It should continue to be enforced before running full training experiments.
6. Presentation order within epoch: fully random vs. task-blocked `[Open — D8]`

---

## GPU Readiness

`Lichtheim2Model` is a standard `nn.Module`. Device support is implemented in Phase 3c-1:

- `init_state(cfg, device=device)` creates state tensors on the model device
- `forward_tick` and `run_trial` infer device from model parameters
- `run_trial` moves task-generated tensors (sound inputs, clamp_vATL) to the model device before each tick
- `move_trial_to_device(trial, device)` in `trainer.py` moves all `SupervisedTrial` tensors

To train on GPU: `model.to("cuda")` before calling `train_step(..., device="cuda")`.
