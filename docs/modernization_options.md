# Modernization Options: Roadmap from Faithful Replication to Modern Architecture

This document describes a structured sequence of optional improvements to the Lichtheim 2 PyTorch reimplementation, ordered from zero-risk documentation through increasingly modern architectural changes. Each level is independent: you can stop at any level and the model at that point is scientifically valid for the stated claims.

The central principle: **do not modify the model in a way that changes scientific claims without documenting it as a deliberate departure.**

---

## Meeting-Ready Summary

> **Where we are (Level 0):** The codebase now has full technical documentation of the training loop, forward equations, loss masking, and classification of faithful vs. diagnostic components. No code has changed.
>
> **Level 1 (safe cleanup):** Tighten naming and remove dead code. Zero risk to model behavior. Recommended before any external sharing.
>
> **Level 2 (modularization):** Refactor dorsal and ventral pathways into separate `nn.Module` objects. The forward equations do not change; only file organization changes. Makes lesioning and ablation studies easier.
>
> **Level 3 (iSMG cell):** Encapsulate the iSMG Elman + motor copy-back as a custom `nn.Module`. Enables clean recurrent cell unit tests. No change to equations.
>
> **Level 4 (padded batching):** Enable training on mini-batches of variable-length words via padding and loss masking. No EOS, no CrossEntropyLoss. Speeds up training by ~B× per epoch. Minor task change for shorter words in mixed-length batches.
>
> **Level 5 (modern EOS model):** Replace BCE on sigmoid outputs with softmax + CrossEntropyLoss, add EOS token, and treat repetition as a sequence-to-sequence problem with variable output length. A clearly different model. Should be branched and documented as a new experiment, not as a Lichtheim 2 replication.

---

## Level 0: Document before modifying

**Goal:** Understand and document the current implementation before making any changes.

**What changes:** Documentation only. No code.

**Deliverables completed:**
- [repetition_training_loop.md](repetition_training_loop.md) — full pipeline from CSV to gradient update
- [forward_tick_equations.md](forward_tick_equations.md) — exact forward_tick equations with tensor shapes
- [loss_masks_zero_radius.md](loss_masks_zero_radius.md) — BCE, mask broadcasting, dead zone, loss decomposition
- [diagnostic_vs_faithful.md](diagnostic_vs_faithful.md) — classification of every component

**What must not change:** Nothing.

**Scientific risk:** None.

**Tests required:** None.

**Yair validation needed:** No — documentation only.

---

## Level 1: Safe cleanup without behavior changes

**Goal:** Remove dead code, fix misleading names, and ensure the codebase is clean and readable before any architectural work.

**What changes:**
- Remove or archive `repetition_ticks`, `comprehension_ticks`, `speaking_ticks` from `ModelConfig` if they are confirmed to be unused at runtime (they are currently stored but never used, per `tasks.py` comment). Keep them only if they serve as documentation of the paper's original setup.
- Rename `motor_copy_to_iSMG` to `motor_context_to_iSMG` for clarity (it is a learned projection, not a pure copy).
- Add type annotations where missing.
- Consolidate the `_compute_repetition_metric_breakdown` and `_compute_repetition_word_accuracy` functions from `scripts/train_repetition_only.py` into `losses.py` or a new `metrics.py` module if they will be reused.

**What must not change:** Forward equations, loss computation, weight initialization, training dynamics.

**Scientific risk:** Zero if done carefully. Any name change should be grep'd across all call sites before applying.

**Tests required:** The full existing test suite must still pass (`python -m pytest tests/ -v`). Additionally, behavior-preserving cleanup warrants new non-regression tests covering:
- **Metrics and evaluation:** unit tests for `_compute_repetition_metric_breakdown` and `_compute_repetition_word_accuracy` with synthetic motor tensors, verifying that threshold accuracy, argmax accuracy, and radius-based word accuracy all return the expected values after any rename or move.
- **Loss decomposition:** tests that `compute_trial_loss_breakdown` returns the correct `n_active_motor`, `motor_loss_per_unit`, and the expected three-component split (input BCE, output-positive BCE, output-negative BCE) after any refactoring.
- **`train_repetition_only` smoke test:** a synthetic end-to-end run (toy config, 3 items, 2 epochs, CPU) that verifies the training loop completes without error and writes `metrics.csv` with the expected columns.

**Yair validation needed:** No.

---

## Level 2: Modularize dorsal and ventral pathways

**Goal:** Encapsulate the dorsal pathway (sound → iSMG → motor) and ventral pathway (sound → mSTG → aSTG → {vATL, triangularis} → motor) as separate `nn.Module` subclasses within `Lichtheim2Model`.

**What changes:**
- New class `DorsalPathway(nn.Module)` holding `sound_to_iSMG`, `iSMG_elman`, `motor_copy_to_iSMG`, `iSMG_to_motor`.
- New class `VentralPathway(nn.Module)` holding `sound_to_mSTG`, `mSTG_to_aSTG`, `vATL_in_to_aSTG`, `aSTG_to_vATL`, `aSTG_to_triangularis`, `triangularis_to_motor`.
- `Lichtheim2Model.__init__` creates `self.dorsal` and `self.ventral` instead of individual Linear layers.
- `forward_tick` calls `self.dorsal(...)` and `self.ventral(...)` internally.

**What must not change:** The forward equations themselves. The mathematical computation in `forward_tick` should produce bit-identical results before and after refactoring. Weight initialization, loss computation, and all external interfaces remain the same.

**Why this is useful:**
- **Lesioning:** Zeroing `self.dorsal.iSMG_to_motor.weight` is cleaner than navigating the flat model attribute namespace.
- **Ablation studies:** The `dorsal_motor_only` flag can be replaced by a cleaner pathway API.
- **Unit testing:** Each pathway can be tested independently with synthetic inputs.

**Scientific risk:** Low if the refactoring is validated by numerical equality. A hidden risk is in the weight initialization order — `_init_weights` must still initialize the same parameters in the same way.

**Tests required:**
- New unit tests for `DorsalPathway.forward` and `VentralPathway.forward` with synthetic inputs.
- A numerical equivalence test: given the same random seed and initial weights, `forward_tick` output must be exactly equal before and after refactoring.

**Yair validation needed:** No — this is a pure refactoring with no scientific content change.

---

## Level 3: Custom iSMG recurrent cell

**Goal:** Encapsulate the iSMG update rule (Elman self-recurrence + motor copy-back + sound input → new iSMG) as a custom `nn.Module` resembling an `nn.RNNCell`.

**What changes:**
- New class `iSMGCell(nn.Module)` with a `forward(sound_in, iSMG_context, motor_context)` method that returns `new_iSMG`.
- Used inside `DorsalPathway.forward` or directly in `forward_tick`.

**What must not change:** The exact equations. The cell is a reorganization, not a new computation.

**Why this is useful:**
- Enables clean unit tests for the iSMG recurrence in isolation.
- Makes it easier to replace the iSMG update rule (e.g., for future LSTM-gated experiments at Level 5) without touching the full model.
- Self-documents that iSMG is the only truly recurrent layer in the model.

**Scientific risk:** Zero if validated numerically. The main risk is a subtle change to gradient flow if the refactoring inadvertently detaches or reattaches tensors.

**Tests required:**
- Unit test for `iSMGCell.forward` with synthetic inputs.
- Numerical equivalence with current `forward_tick`.

**Yair validation needed:** No.

---

## Level 4: Padded batching without EOS

**Goal:** Enable mini-batch training on variable-length repetition trials by padding all sequences in a batch to `2 × Tmax` ticks, where `Tmax = max(T_i)` within the batch. No change to the loss function, no EOS, no CrossEntropyLoss.

**What changes:**
- New `init_state_batched(cfg, batch_size)` function in `layers.py` returning `ModelState` with fields of shape `(B, layer_size)`.
- New `BatchedRepetitionTrial` dataclass and `build_batched_trial()` function in `trials.py`.
- New `batched_run_trial()` and `train_step_batched()` functions in `trainer.py`.
- New `compute_batch_loss()` in `losses.py` with a `(B, 2*Tmax, motor_size)` loss mask.
- `--batch-size` CLI argument in `train_repetition_only.py` (default 1 = current behavior).

**What must not change:** `forward_tick()`, `run_trial()`, `train_step()`, `compute_trial_loss()`, and all existing tests. Batch-size-1 behavior must be identical to the current implementation (enforced by a compatibility test).

**Task semantics with per-word boundaries (Option 1, recommended):**
Each word in the batch starts its output phase at its own T_i tick. The batch is padded to 2*Tmax total ticks, but item i is supervised only at ticks 0..2*T_i-1. Shorter words are padded with zeros (silence) at the end; padded ticks contribute no gradient. This preserves the exact task structure for each individual word.

**Why this is useful:**
- Reduces the number of `.backward()` and optimizer steps per epoch from N to ⌈N/B⌉. Whether this translates into faster wall-clock time depends on hardware, batch size, and sequence length distribution — actual speedup must be measured rather than assumed.
- May stabilize training through gradient averaging across multiple words.
- Enables efficient GPU utilization (batched matrix multiplications), where the speedup can be substantial on CUDA hardware.

**Remaining gap from paper:** The paper trains on a fixed-size vocabulary of 1710 Japanese words, all 3 morae, so batching was not an issue. Padded batching is an engineering adaptation for English variable-length training.

**Scientific risk:** Mild. Gradient averaging across different words in a batch changes the optimization dynamics relative to online (one-trial-at-a-time) gradient descent. The paper uses online learning. Batching may converge faster or slower and may change which words are learned first. Results with `--batch-size > 1` should be reported with this caveat.

**Tests required:**
- `test_batch_size_1_matches_single_trial_loss`: verify numerical equivalence with batch_size=1.
- `test_loss_mask_excludes_padded_ticks`: verify padded positions contribute zero loss.
- `test_zero_error_radius_in_batch_mode`: verify dead zone applies correctly.
- `test_output_positive_weight_not_applied_to_padded_ticks`: verify weighting scoping.
- `test_batched_run_trial_output_shape`: shape check.
- `test_batched_training_smoke`: end-to-end smoke test, 2-3 items, 2 epochs, no crash.

**Yair validation needed:** Yes, before using batched results in publications — need to confirm that the optimization trajectory with batch training is acceptable as a stand-in for the online learning described in the paper.

---

## Level 5: Alternative modern EOS/token-ID/CrossEntropy model

**Goal:** Replace the continuous sigmoid outputs with softmax + CrossEntropyLoss, add an EOS token to the phoneme vocabulary, and treat repetition as a sequence-to-sequence problem where the model predicts a sequence of phoneme tokens ending with EOS.

**What changes:**
- Phoneme representations become integer token IDs (or one-hot with EOS dimension added): `sound_input_size = 40` (39 phonemes + 1 EOS).
- Motor output becomes logits over 40 classes (pre-sigmoid/no-sigmoid final layer); loss becomes `CrossEntropyLoss(logits, target_indices, ignore_index=PAD_ID)`.
- The output phase length is no longer fixed at T — it runs until the model predicts EOS or a maximum length is reached.
- Training requires teacher forcing during output phase.
- Evaluation requires autoregressive decoding.

**What must not change (to remain scientifically comparable to Lichtheim 2):** Ideally nothing — this is a substantial departure. If this is pursued, it should be explicitly labeled as a "modernized Lichtheim-inspired model" rather than a Lichtheim 2 replication.

**Why this might be interesting:**
- EOS enables variable-length output without knowing T in advance at test time.
- CrossEntropyLoss avoids the 38:1 imbalance problem entirely.
- Softmax forces a one-winner selection per tick, which is linguistically interpretable.
- Comparable to modern seq2seq or CTC-based ASR systems.

**Why this is a different model:**
- LENS used sigmoid + BCE, not softmax + CrossEntropy.
- The original Lichtheim 2 does not have an EOS concept.
- Softmax imposes a hard mutual exclusivity on motor units that BCE does not.
- The lesioning and aphasia-profile analyses in the paper assume sigmoid-output dynamics.

**Scientific risk:** High if results are presented as Lichtheim 2 replications. Low if presented as a new model inspired by Lichtheim 2.

**Tests required:** A full new test suite for the EOS model, entirely separate from the current tests.

**Yair validation needed:** Yes — this must be framed correctly in any manuscript or presentation. Should be a separate git branch and possibly a separate model class.

---

## Decision guide

| Question | Recommended level |
|---|---|
| Do I want to add a new feature without breaking existing results? | Level 0 first, then the relevant level in isolation |
| Should I use `--dorsal-motor-only` or `--sound-proj-size` in a paper result? | No — these are diagnostic-only |
| Can I use `--batch-size 4` and call it "faithful Lichtheim training"? | Only with explicit caveat about gradient averaging vs. online learning |
| Is the current one-hot BCE setup "wrong"? | No — it is a valid English/NWR adaptation; it is not faithful to Japanese 21-bit features |
| Should I adopt Level 5 for a faster-converging baseline? | Yes, but document it as a departure, not a replication |
