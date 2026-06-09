# Roadmap

Phased plan for the Lichtheim 2 PyTorch reimplementation.

**Rule:** No phase may begin until the success criterion of the previous phase has been reviewed and confirmed.

---

## Phase 0 — Repo Scaffold and Specification — **Complete**

**Goal:** Establish the repository structure, documentation, and specification before writing any model code.

**Deliverables:**
- `README.md`, `.gitignore`, branch structure
- `docs/replication_spec.md`: precise layer/pathway/tick specification
- `docs/open_questions.md`: open design decisions
- `docs/architecture_notes.md`, `docs/training_notes.md`, `docs/data_encoding_notes.md`
- `configs/toy.yaml`: minimal synthetic config
- `src/lichtheim2/__init__.py`: package marker

**Success criterion:** All spec questions answered or explicitly flagged `[Open]`. PR merged to `develop`. ✓

---

## Phase 1 — Toy Tick-by-Tick Forward Pass — **Complete**

**Goal:** Implement and validate the tick-by-tick activation dynamics in isolation, using the toy config. No training, no real data.

**Deliverables:**
- `src/lichtheim2/config.py`: `ModelConfig` dataclass, `load_config()`
- `src/lichtheim2/layers.py`: `ModelState`, `TickResult` dataclasses; `init_state()`
- `src/lichtheim2/model.py`: `Lichtheim2Model` with `forward_tick()` (returns `(ModelState, vATL_input_used)`) and `run_trial()` (returns `list[TickResult]`)
- `src/lichtheim2/tasks.py`: `Task` enum, `build_trial_inputs()` for repetition, comprehension, speaking
- `conftest.py`: `sys.path` shim (temporary, until `pyproject.toml`)
- `tests/test_tick_dynamics.py`: tests covering shapes, copy-back chain, task schedules

**Success criterion:** Pytest passes; forward pass runs end-to-end for all three task types; activations in [0, 1]; carry state verified to influence computation. ✓

---

## Phase 2 — Faithful Architecture — **In Progress**

Phase 2 is split into sub-steps:

### Phase 2a — Architecture Skeleton — **Complete**

**Goal:** Instantiate `Lichtheim2Model` with the real Lichtheim 2 layer sizes and validate forward dynamics on dummy tensors. No training, no faithful weight initialisation, no real data.

**Deliverables:**
- `configs/lichtheim2.yaml`: full-size layer configuration with `metadata:` block for reference values
- `tests/test_faithful_architecture.py`: tests for real-size shapes, activation range, copy-back, and task tick counts

**Success criterion:** All three task types run `forward_tick` and `run_trial` with faithful layer sizes; shapes correct; activations in [0, 1].

### Phase 2b — Faithful Weight Initialisation and Connectivity Audit — **In Progress**

**Goal:** Implement the weight initialisation scheme from the paper and audit all connection/bias conventions against Supplement Figure S1.

**Deliverables:**
- `Lichtheim2Model._init_weights()`: standard weights [−1, 1] `[Paper]`; Elman weights [−0.5, 0.5] `[Paper]`; copy-back weights [−0.5, 0.5] `[Inferred]`; additive bias = −1.0 `[Inferred — PyTorch approximation of LENS bias-link convention]`
- `tests/test_weight_initialization.py`: tests for weight ranges, bias values, connectivity completeness

**Success criterion:** All weight tensors and bias values match the implemented conventions; connectivity audit confirms all 10 connections from Supplement Figure S1 are present; existing toy and faithful tests still pass.

### Phase 2c — Unbatched Variable-Length Trial Support — **In Progress**

**Goal:** Extend `build_trial_inputs` and `run_trial` to handle phoneme sequences of arbitrary length T. No padding, masking, EOS, batching, or real data loading. No training.

Tick structure:
- Repetition: T input ticks + T output ticks (2T total)
- Comprehension: T input ticks; semantic output evaluated at tick T
- Speaking/naming: T output ticks with semantic input clamped; `phon_pattern` is still required to determine T

Note: `cfg.repetition_ticks`, `cfg.comprehension_ticks`, and `cfg.speaking_ticks` are now historical reference values for the original fixed-length setup. Runtime tick counts are derived from `phon_pattern.shape[0]`.

**Deliverables:**
- Updated `src/lichtheim2/tasks.py`: T inferred from `phon_pattern.shape[0]`; 1-D input treated as T=1
- `tests/test_variable_length_trials.py`: tests for T=1, 2, 3, 5 including 1-D fallback

**Success criterion:** Model runs unbatched forward trials for T=1, 2, 3, 5; tick counts correct (2T / T / T); activations in [0, 1]; all existing toy and faithful tests unaffected. Padding, masking, EOS, and batching remain `[Open]`.

---

## Phase 3 — English/NWR-Style Training

Phase 3 is split into sub-steps:

### Phase 3a — Data Inspection — **Complete**

**Goal:** Inspect local CSV files (`phonemes.csv`, `wfe.csv`, `ssp.csv`) and document schemas, encoding options, and open questions. No data loading or training.

**Deliverables:**
- Updated `docs/data_encoding_notes.md`: CSV schemas, No_Stress recommendation, encoding options
- Updated `docs/open_questions.md`: D10–D18 covering encoding, coverage validation, pseudowords

**Success criterion:** Encoding plan documented; coverage validation requirement identified; phoneme feature dimension open but candidate options listed.

### Phase 3b — English Phoneme Encoder and Word Item Loaders — **In Progress**

**Goal:** Implement a minimal phoneme inventory loader, one-hot encoder, and word/pseudoword item loaders. Validate that all No_Stress phonemes are covered and that encoded items can be passed through `run_trial()`. No training, no semantic vectors, no batching.

**Deliverables:**
- `src/lichtheim2/encoding.py`: `PhonemeInventory`, `load_phoneme_inventory()`, `parse_phoneme_sequence()`, `validate_phoneme_coverage()`, `encode_phoneme_sequence()`
- `src/lichtheim2/data.py`: `WordItem`, `PseudowordItem` dataclasses; `load_word_items()`; `load_pseudoword_items()`; required-field validation; Length consistency check
- `configs/english_nwr.yaml`: English/NWR config with provisional `sound_input_size: 39`
- `tests/test_english_phoneme_encoder.py`: always-run + CSV-dependent encoder tests
- `tests/test_english_word_items.py`: always-run mock-CSV tests + CSV-dependent loader tests; integration with `run_trial()` for REPETITION

**Success criterion:** All No_Stress phonemes covered; lengths validated; `WordItem` and `PseudowordItem` load with correct tensor shapes; required-field errors are explicit; encoded items run through REPETITION `run_trial()`; existing tests unaffected.

**Additional Phase 3b deliverables (supervised trial targets):**
- `src/lichtheim2/semantics.py`: `assign_artificial_semantics()` — reproducible binary semantic vectors keyed by row_index
- `src/lichtheim2/trials.py`: `SupervisedTrial` dataclass; `make_repetition_trial()`, `make_comprehension_trial()`, `make_speaking_trial()` with shape validation
- `tests/test_supervised_trials.py`: always-run tests for semantics and all three trial factories
- `docs/training_notes.md`: GPU-readiness note

### Phase 3c — Training Loop — **In Progress**

Phase 3c is split into sub-steps:

### Phase 3c-1 — Minimal Training Loop — **In Progress**

**Goal:** Implement the first minimal PyTorch training loop for supervised trials.

**Deliverables:**
- `src/lichtheim2/losses.py`: masked BCE loss over motor and semantic outputs
- `src/lichtheim2/trainer.py`: `train_step()` and device-aware trial transfer
- `scripts/smoke_train_repetition.py`: synthetic repetition smoke training
- `tests/test_training_loop.py`: loss, gradient, parameter update, finite-loss, and loss-decrease tests

**Success criterion:** A small synthetic repetition training run decreases loss; existing tests pass; no full experiment or batching yet.

#### Phase 3c-2 — Real-Data Repetition Training — **Complete**

**Goal:** Pipeline validation step. Connect real CSV data (wfe.csv, ssp.csv) to the training loop for the repetition task only. Repetition only; no comprehension, speaking, batching, EOS, checkpoints, or plots.

**Deliverables:**
- `scripts/train_repetition_real_data.py`: CLI script with helper functions (`load_repetition_items`, `sample_items`, `build_repetition_trials`, `train_repetition_epochs`)
- `tests/test_repetition_training_data.py`: always-run tests (mock CSVs, toy config) + CSV-dependent integration tests (skip gracefully if private data absent)

**Success criterion:** Script runs end-to-end on real wfe.csv / ssp.csv items with finite loss; all always-run tests pass without private CSV files; existing tests unaffected. ✓

#### Phase 3c-3 — Real-Data Multi-Task Training — **Complete**

**Goal:** Pipeline validation step. Extend real-data training to all three task types: real words receive repetition + comprehension + speaking trials; pseudowords remain repetition-only. No batching, no LR schedule, no frequency weighting, no accuracy logging, no checkpoints.

**Deliverables:**
- `scripts/train_multitask_real_data.py`: CLI script with helper functions (`build_word_trials`, `build_pseudo_trials`, `train_multitask_epochs`). Semantics assigned to all words before sampling for reproducibility.
- `tests/test_multitask_training_data.py`: always-run tests (mock `WordItem`/`PseudowordItem` dataclasses, toy config) + CSV-dependent integration tests

**Success criterion:** Script runs end-to-end with finite losses across all three task types; always-run tests pass without private CSV files; existing tests unaffected. ✓

#### Phase 3c-4 — Small-Subset Training Stability Diagnostics — **Complete**

**Goal:** Determine whether the model can learn stably on tiny controlled subsets before committing to paper-like schedules, frequency weighting, and full-scale training.

**Modes:**
- `repetition` — repetition trials for words + pseudowords
- `words-multitask` — rep+comp+spk for words only (no ssp.csv required)
- `mixed-multitask` — rep+comp+spk for words, rep for pseudowords

**Deliverables:**
- `scripts/diagnose_small_subset_training.py`: diagnostic script with `DiagnosticResult` dataclass, `sample_items()`, `build_word_rep_trials()`, `build_trials_for_mode()`, `run_diagnostic_epochs()`. Defaults: `--lr 0.01`, `--zero-error-radius 0.0`, `--epochs 20`, `--max-words 10`, `--max-pseudowords 10`. Non-finite loss raises `RuntimeError` with epoch/task/label context. Loads CSVs only as needed by mode.
- `tests/test_small_subset_training_diagnostics.py`: always-run tests (mock dataclasses, toy config) + CSV-dependent integration tests

**Success criterion:** Diagnostic script runs with finite losses for all modes; `DiagnosticResult` fields are correct; always-run tests pass without private CSV files; existing tests unaffected. ✓

#### Phase 3c-5 — Loss Normalization and Task-Balance Audit — **Complete**

**Goal:** Expose per-component loss breakdown (motor vs semantic, total vs per-active-unit) before committing to paper schedules. Determine whether comprehension loss is large due to more active output elements, higher vATL dimensionality, or genuinely higher per-unit BCE.

**Deliverables:**
- `src/lichtheim2/losses.py`: `LossBreakdown` dataclass and `compute_trial_loss_breakdown()`; `compute_trial_loss()` refactored to delegate to it (interface and output unchanged)
- `scripts/audit_loss_scaling.py`: CLI audit tool (no training; eval mode, `torch.no_grad()`) printing per-trial and per-task-summary breakdown tables
- `tests/test_loss_scaling_audit.py`: parity tests, active-count tests, normalized-loss finiteness tests, dead-zone count-reduction tests

**Success criterion:** `compute_trial_loss_breakdown().total_loss` numerically identical to `compute_trial_loss()`; active unit counts correct; normalized per-unit losses finite; always-run tests pass without private CSV files; existing tests unaffected. ✓

#### Phase 3c-6 — Optional Loss Reduction Modes — **Complete**

**Goal:** Add a `mean_active` loss reduction option for small-subset diagnostics. Keeps `sum` as the default (paper behavior); `mean_active` normalizes by the number of active output elements to diagnose task imbalance.

**Deliverables:**
- `src/lichtheim2/losses.py`: `loss_reduction` parameter added to `compute_trial_loss()` (`"sum"` default, `"mean_active"` option); raises `ValueError` for unknown values or zero active elements
- `src/lichtheim2/trainer.py`: `loss_reduction` parameter added to `train_step()`; passed as keyword arg to `compute_trial_loss()`
- `scripts/diagnose_small_subset_training.py`: `--loss-reduction sum|mean_active` CLI argument (default `sum`); threaded through `run_diagnostic_epochs()`
- `tests/test_training_loop.py`: parity test (`sum` == default), `mean_active` value/finiteness tests, dead-zone interaction test, invalid-string error test, `train_step` with `mean_active`
- `tests/test_small_subset_training_diagnostics.py`: `run_diagnostic_epochs` with `loss_reduction="mean_active"` stays finite

**Success criterion:** `loss_reduction="sum"` is bit-for-bit identical to previous default; `mean_active` = `total_loss / n_active`; dead-zone interaction correct; invalid string raises `ValueError`; always-run tests pass without CSV files; existing tests unaffected. ✓

#### Phase 3c-7 — Loss Reduction Comparison — **In Progress**

**Goal:** Lightweight comparison script that runs both `sum` and `mean_active` reductions under identical conditions and prints a compact table, so training dynamics can be directly compared before committing to a default.

**Deliverables:**
- `scripts/diagnose_small_subset_training.py`: `verbose: bool = True` added to `run_diagnostic_epochs()` (default unchanged; compare script calls with `verbose=False`)
- `scripts/compare_loss_reductions.py`: `ComparisonResult` dataclass, `run_comparison()` (same seed/data/epochs/lr for both reductions), `print_comparison_table()` (summary + per-task final losses)
- `tests/test_loss_reduction_comparison.py`: always-run tests for `run_comparison` return shape, epoch counts, field presence, loss finiteness; `validate_args` tests; CSV-dependent integration test
- `tests/test_small_subset_training_diagnostics.py`: `test_run_diagnostic_epochs_verbose_false_returns_finite`

**Success criterion:** `run_comparison()` returns `DiagnosticResult` for both reductions with identical epoch counts and finite losses; `verbose=False` path tested; always-run tests pass without CSV files; existing tests unaffected. ✓

#### Phase 3c-8 — Task Schedule Diagnostics — **In Progress**

**Goal:** Compare the uniform (1×REP+1×COMP+1×SPK) and paper-like (1×REP+3×COMP+2×SPK) task presentation schedules on small controlled subsets to understand how schedule affects learning dynamics before committing to the paper schedule.

**Deliverables:**
- `scripts/compare_task_schedules.py`: `SCHEDULES` dict, `ScheduleComparisonResult`, `build_word_trials_with_schedule()`, `build_trials_for_schedule()`, `run_schedule_comparison()` (same fairness guarantee as Phase 3c-7), `print_schedule_comparison_table()` (summary + per-task initial→final losses). Default `--loss-reduction mean_active`.
- `tests/test_task_schedule_diagnostics.py`: trial count/order tests, mode-aware builder tests, comparison result tests, `validate_args` tests, CSV-dependent integration test

**Success criterion:** `build_word_trials_with_schedule` produces correct counts and order for both schedules; `run_schedule_comparison` returns finite losses for both; `trials_per_epoch["paper"] > trials_per_epoch["uniform"]`; always-run tests pass without CSV files; existing tests unaffected.

---

## Phase 4 — Lesioning and Recovery

**Goal:** Implement selective weight lesioning and recovery training. Reproduce the aphasic profiles from the paper.

**Deliverables:**
- `src/lichtheim2/lesioning.py`: pathway-specific lesion hooks
- Recovery training loop (retrain after lesion)
- Replication of key lesion figures / tables from Ueno et al.

**Success criterion:** Selective dorsal/ventral lesions produce the expected aphasic profiles (e.g., phonological repetition spared with ventral lesion; semantic comprehension impaired with ventral lesion). To be reviewed before moving to Phase 5.

---

## Phase 5 — Representational Similarity Analyses

**Goal:** Analyse the internal representations learned by the model.

**Deliverables:**
- Extraction of hidden-layer activations across stimuli
- RSA / MDS / t-SNE visualisations
- Comparison to human neuroimaging data where available

**Success criterion:** TBD.

---

## Phase 6 — Further Extensions

**Goal:** Extensions beyond the English/NWR training setup, to be scoped after Phase 3 is confirmed. Possible directions include:

- Richer semantic representations (distributional embeddings, curated norms)
- Cross-linguistic comparisons or alternative phoneme encodings
- Larger-scale representational similarity analyses and links to SWP behavioural/neural data
- Lesioning and recovery simulations
- Alternative architectures or ablation studies

**Success criterion:** TBD.
