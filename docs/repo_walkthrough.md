# Repository Walkthrough

A guided tour of the Lichtheim 2 PyTorch reimplementation, written to support an
oral walkthrough. It assumes the reader knows the Ueno et al. (2011) paper but
not this codebase.

**Citation tags used throughout (inherited from the rest of the repo):**
- `[Paper §X]` — stated in Ueno et al. 2011 main text
- `[Supp §X]` — stated in supplementary materials
- `[Inferred]` — best-effort PyTorch translation, not directly stated in the paper
- `[Open]` — unresolved design decision (tracked in `docs/open_questions.md`, IDs D1–D18)

---

## 1. Big picture

### What this repo is trying to replicate

This is a PyTorch reimplementation of **Lichtheim 2** (Ueno, Saito, Rogers, &
Lambon Ralph, 2011, *Neuron*), a neurocomputational model of single-word
processing built around two pathways:

- A **dorsal pathway** (sound → iSMG → motor) supporting **repetition**.
- A **ventral pathway** (sound → mSTG → aSTG → vATL / triangularis → motor)
  supporting **comprehension** (sound → semantics) and **speaking/naming**
  (semantics → sound).

The original model was implemented in **LENS** (a connectionist simulator) as
a hand-rolled, tick-by-tick recurrent network with explicit "copy-back" context
units realizing bidirectional connections via Plaut-style unfolding. This repo
reproduces that **tick-by-tick dynamic**, layer by layer, in plain PyTorch — it
is explicitly **not** a modern seq2seq/Transformer/`nn.RNN`/`nn.LSTM` model
(see `docs/architecture_notes.md`, "Why Not Use `nn.RNN` or `nn.LSTM`").

### What is already implemented

Based on the current code (cross-checked against `docs/roadmap.md` and
`CLAUDE.md`):

- The full 7-layer architecture at faithful Lichtheim-2 sizes
  (`configs/lichtheim2.yaml`), with a working tick-by-tick `forward_tick()` /
  `run_trial()` (`src/lichtheim2/model.py`).
- Paper-aligned weight initialisation (`_init_weights()`): uniform `[-1, 1]` for
  standard feedforward connections, uniform `[-0.5, 0.5]` for the Elman and
  copy-back connections, bias `= -1.0` where biases exist `[Paper]`/`[Inferred]`.
- Variable-length (T-phoneme) trials for all three tasks
  (`src/lichtheim2/tasks.py`, `src/lichtheim2/trials.py`).
- An English phoneme encoder and real-data loaders for `wfe.csv` (real words)
  and `ssp.csv` (pseudowords) (`src/lichtheim2/encoding.py`,
  `src/lichtheim2/data.py`).
- A first-pass artificial semantics assignment (`src/lichtheim2/semantics.py`).
- A masked binary-cross-entropy loss and a minimal online (item-by-item)
  training step (`src/lichtheim2/losses.py`, `src/lichtheim2/trainer.py`).
- A series of small-subset **training diagnostics** (Phases 3c-2 through
  3c-11) that exercise this pipeline on real CSV data and compare specific
  training-loop choices (loss reduction, task schedule, frequency weighting,
  LR schedule, and integrated "recipes" combining all four).
- **Level 1B safe cleanup** (Phase 3e): pure metric computation, repetition
  evaluation, and diagnostic loss decomposition extracted to dedicated library
  modules (`src/lichtheim2/metrics.py`, `src/lichtheim2/repetition_evaluation.py`,
  `src/lichtheim2/loss_decomposition.py`). The primary repetition-only training
  script (`scripts/train_repetition_only.py`) now serves as a clean orchestrator.
  **Test suite: 429 passed. Smoke run passed.**

**Documentation-status note `[Open]`:** `docs/roadmap.md` still marks Phase
2b ("Faithful Weight Initialisation"), Phase 2c ("Variable-Length Trials"),
and Phase 3c-1 ("Minimal Training Loop") as **"In Progress"** and their
success criteria lack the trailing `✓` used elsewhere. By inspection, the
corresponding code (`_init_weights()`, the T-derived tick logic in
`tasks.py`, and `train_step()`/`compute_trial_loss()`) exists, is tested, and
is used pervasively by every later "Complete" phase. `CLAUDE.md`'s "Current
Development Status" section already lists these as complete. This looks like
a roadmap bookkeeping lag rather than missing functionality, but it's worth
flagging rather than silently asserting "done."

### What is still not implemented

- **Lesioning and recovery training** (Phase 4) — no `lesioning.py` yet. The
  architecture (one `nn.Linear` per named connection) is designed to make this
  easy, but nothing is wired up.
- **Representational similarity analyses** (Phase 5) — not started.
- **A full, paper-scale integrated training run** (e.g. 200 epochs, full
  vocabulary, the paper's 1×REP/3×COMP/2×SPK schedule as the *actual* training
  loop, not just a diagnostic) — README lists this as "Pending."
- **Faithful prototype-based semantic vectors** (50 prototypes × 40 exemplars,
  20 on-bits, ≥4-bit Hamming distance, per the supplement) — currently
  `assign_artificial_semantics()` produces independent random binary vectors
  per word `[Inferred]` (D15 `Provisional`).
- **Padding / batching / EOS** for variable-length sequences (D12 `Open`) —
  everything currently runs unbatched, one trial at a time.
- **Weight decay schedule** from the paper (1e-6 → 6e-7 stepped) — not
  implemented.
- A **zero-error radius of 0.1** as the *default* everywhere — the paper value
  is 0.1 `[Paper]`, but `compute_trial_loss()` / `train_step()` default to
  `0.0`, and most diagnostic scripts also default to `0.0`.

### Diagnostic vs. model code

It's worth being explicit about this distinction when presenting:

| Category | Files | Role |
|---|---|---|
| **Model code** | `src/lichtheim2/{layers,model,tasks}.py` | The actual Lichtheim 2 architecture and tick dynamics — this *is* the replication. |
| **Data / encoding code** | `src/lichtheim2/{config,encoding,data,semantics}.py` | Loading config, phonemes, word items, and assigning semantic targets. |
| **Training-mechanics code** | `src/lichtheim2/{trials,losses,trainer}.py` | Turning a `WordItem`/`PseudowordItem` into a supervised trial, computing loss, and taking one optimizer step. Shared infrastructure for both diagnostics and any future full training run. |
| **Evaluation / metrics code** | `src/lichtheim2/{metrics,repetition_evaluation,loss_decomposition}.py` | Post-training and post-epoch evaluation: argmax accuracy, threshold metrics, word-level accuracy, and diagnostic BCE decomposition. Extracted in Level 1B. No model changes. |
| **Diagnostic / comparison scripts** | `scripts/compare_*.py`, `scripts/diagnose_*.py`, `scripts/audit_*.py` | Not part of the model. Each varies *one or a few* training-loop choices on a small subset (≈10 words). |
| **Pipeline-validation scripts** | `scripts/train_repetition_real_data.py`, `scripts/train_multitask_real_data.py`, `scripts/smoke_train_repetition.py` | Earlier "does this even run end-to-end" scripts. |
| **Repetition-only training script** | `scripts/train_repetition_only.py` | Primary repetition-only training run on English NWR data (Phase 3e). Orchestrates data loading, trial construction, training, evaluation, and output saving. |
| **Documentation** | `README.md`, `CLAUDE.md`, `docs/*.md` | Specification, design rationale, roadmap, open questions. |

---

## 2. File-by-file map

### Core package (`src/lichtheim2/`)

#### `config.py` — *model/config code*
- **Role:** Defines the model's size parameters and loads them from YAML.
- **Key items:** `ModelConfig` dataclass (12 fields: `sound_input_size`,
  `motor_output_size`, `iSMG_hidden_size`, `mSTG_hidden_size`,
  `aSTG_hidden_size`, `triangularis_hidden_size`, `vATL_size`,
  `repetition_ticks=6`, `comprehension_ticks=3`, `speaking_ticks=3`,
  `sound_proj_size=None`, `dorsal_motor_only=False`);
  `load_config(path)` reads the `model:`/`tasks:` YAML sections.
- **Connections:** Every other module that needs layer sizes takes a
  `ModelConfig`. The three tick-count fields are now **historical/reference
  values only** — see `tasks.py`.

#### `layers.py` — *model code*
- **Role:** Defines the state that is threaded between ticks, and the
  per-tick record.
- **Key items:**
  - `ModelState` — 9 tensor fields: `iSMG`, `iSMG_context`, `motor`,
    `motor_context`, `mSTG`, `aSTG`, `vATL_out`, `vATL_context`,
    `triangularis`. This is the **between-tick carry state**.
  - `TickResult` — `tick_index`, `task`, `sound_input`, `vATL_input_used`,
    `state` (the `ModelState` *after* this tick). One per tick, returned by
    `run_trial()`.
  - `init_state(cfg, device)` — hidden layers (incl. `vATL_out`/context) → 0.5,
    `motor`/`motor_context` → 0, per `[Paper]`'s stated trial-initial
    conditions.
- **Connections:** `model.py` consumes/produces `ModelState`; `losses.py`
  reads `TickResult.state.motor` / `.vATL_out`.

#### `model.py` — *model code*
- **Role:** The architecture itself — `Lichtheim2Model(nn.Module)`.
- **Key items:**
  - `__init__(cfg)` — instantiates 10–11 `nn.Linear` layers (one per named
    connection, plus an optional `sound_proj`) and calls `_init_weights()`.
  - `_init_weights()` — applies the paper's initialisation scheme (see
    Section 4 below for the full table).
  - `forward_tick(state, sound=None, clamp_vATL_in=None) -> (new_state, vATL_input_used)`
    — one tick of the dynamics; fully differentiable, no `.detach()` /
    `no_grad()`.
  - `run_trial(task, phon_pattern, sem_pattern, cfg) -> list[TickResult]` —
    initialises state via `init_state()`, calls `build_trial_inputs()` to get
    per-tick inputs, then loops `forward_tick()`.
- **Connections:** Calls into `tasks.py` (`build_trial_inputs`) and
  `layers.py` (`init_state`). Called by `trainer.train_step()` and every
  diagnostic script.
- **Diagnostic flags** (not in paper): `dorsal_motor_only` excludes
  `triangularis_to_motor` from the motor sum. `sound_proj_size` enables a
  dense projection from the raw phoneme vector before both pathways.

#### `tasks.py` — *model code*
- **Role:** Defines the three tasks and converts a phoneme/semantic pattern
  into the literal per-tick `(sound_input, clamp_vATL_in)` sequence.
- **Key items:**
  - `Task` enum — `REPETITION`, `COMPREHENSION`, `SPEAKING`.
  - `build_trial_inputs(task, phon_pattern, sem_pattern, cfg) -> list[(sound, clamp_vATL_in | None)]`
    — trial length `T` is **inferred from `phon_pattern.shape[0]`**
    (1-D input ⇒ `T=1`); `cfg.repetition_ticks` etc. are *not* used at
    runtime.
- **Connections:** Called by `model.run_trial()`. The exact tick structure
  per task is detailed in Section 5.

#### `trials.py` — *training-mechanics / data code*
- **Role:** Builds the **supervised targets and loss masks** for one trial —
  this is where "what should the network output, and when" is encoded.
- **Key items:**
  - `SupervisedTrial` dataclass — `task`, `n_ticks`, `phon_tensor`,
    `sem_input` (speaking only), `motor_targets`, `motor_loss_mask`,
    `semantic_targets`, `semantic_loss_mask`, plus `item_id`, `label`, and
    `loss_weight: float = 1.0` (added Phase 3c-9 for frequency weighting).
  - `make_repetition_trial(phon_tensor, motor_size, ...)` — `2T` ticks;
    motor target = zeros for the first `T` ticks, then `phon_tensor` for the
    last `T`; loss mask is **all `True`**.
  - `make_comprehension_trial(phon_tensor, sem_tensor, motor_size, ...)` —
    `T` ticks; semantic target = `sem_tensor` repeated at every tick, motor
    target = zeros, **both masks all `True`**.
  - `make_speaking_trial(phon_tensor, sem_tensor, motor_size, ...)` — `T`
    ticks; `sem_input = sem_tensor` (clamped every tick); motor target =
    `phon_tensor`; mask all `True`; no semantic target.
- **Connections:** Consumed by `trainer.train_step()` and every script that
  builds trials.

#### `losses.py` — *training-mechanics code*
- **Role:** Computes the masked BCE loss, with an optional per-unit
  breakdown.
- **Key items:**
  - `LossBreakdown` dataclass — `task`, `n_ticks`, `total_loss`,
    `motor_loss`, `semantic_loss`, `n_active_motor`, `n_active_semantic`,
    `motor_loss_per_unit`, `semantic_loss_per_unit`.
  - `compute_trial_loss_breakdown(tick_results, trial, zero_error_radius=0.0, output_positive_weight=1.0)`
    — stacks `motor`/`vATL_out` across ticks, computes
    `F.binary_cross_entropy(..., reduction="none")`, masks by
    `motor_loss_mask`/`semantic_loss_mask` (and optionally by a zero-error
    "dead zone"), and sums. Optionally upweights the output-phase positive unit
    via `output_positive_weight` (diagnostic; default 1.0 = no effect).
  - `compute_trial_loss(tick_results, trial, zero_error_radius=0.0, loss_reduction="sum", output_positive_weight=1.0)`
    — returns `breakdown.total_loss` (`"sum"`, default, paper-faithful) or
    `breakdown.total_loss / n_active` (`"mean_active"`, diagnostic only).
- **Connections:** Called by `trainer.train_step()`. Also called directly by
  `loss_decomposition.py::compute_trial_loss_decomposition` for decomp
  consistency tests.

#### `trainer.py` — *training-mechanics code*
- **Role:** The actual online training step.
- **Key items:**
  - `move_trial_to_device(trial, device)` — `dataclasses.replace` with all
    tensor fields `.to(device)`.
  - `train_step(model, trial, optimizer, cfg, zero_error_radius=0.0, device="cpu", loss_reduction="sum", output_positive_weight=1.0) -> float`
    — `model.train()` → `optimizer.zero_grad()` → move trial to device →
    `model.run_trial(...)` → `compute_trial_loss(...)` → multiply by
    `trial.loss_weight` if `!= 1.0` → `loss.backward()` → `optimizer.step()`
    → return `loss.item()`.
- **Connections:** This is the single function every training/diagnostic
  script calls per item. Also imported by `loss_decomposition.py` and
  `repetition_evaluation.py` for `move_trial_to_device`.

#### `metrics.py` — *evaluation / metrics code* *(Level 1B)*
- **Role:** Pure, stateless metric computation helpers. No model calls, no
  file I/O. All functions work on raw tensors.
- **Key items:**
  - `compute_repetition_metric_breakdown(motor_input, motor_output, motor_targets_out) -> dict`
    — 6-key split breakdown separating input-phase silence from output-phase
    positive/negative unit accuracy (see Section 8 for key descriptions).
  - `compute_repetition_word_accuracy(motor_input, motor_output, motor_targets_out, radius) -> dict`
    — 3 boolean keys: `output_all_units_within_radius`,
    `input_all_silent_within_radius`, `trial_all_supervised_units_within_radius`.
  - `prediction_summary(preds: list[dict]) -> dict` — 15-key aggregate over
    a list of per-trial prediction dicts (NaN floats when preds is empty).
  - `rolling_mean(values, window) -> list[float]` — trailing rolling average;
    used by the plot script.
- **Connections:** Imported by `repetition_evaluation.py`. Imported by
  `train_repetition_only.py` under original private-name aliases (behavior-
  preserving). Imported by `plot_repetition_metrics.py` for `rolling_mean`.
  No model imports.

#### `repetition_evaluation.py` — *evaluation / metrics code* *(Level 1B)*
- **Role:** Runs the model in eval mode (`model.eval()` + `torch.no_grad()`)
  and computes per-trial prediction metrics for repetition.
- **Key items:**
  - `evaluate_one_trial(model, trial, cfg, inventory_symbols, device, eval_radius=0.1) -> dict`
    — runs `model.run_trial()`, stacks input-phase and output-phase motor
    outputs, computes argmax accuracy, threshold accuracy, exact match,
    metric breakdown, and word accuracy; returns a flat 18-key dict.
  - `evaluate_predictions(model, trials, cfg, inventory_symbols, device, eval_radius=0.1) -> list[dict]`
    — maps `evaluate_one_trial` over all trials.
- **Note:** `eval_radius` can differ from the training `zero_error_radius`.
  The train script evaluates with `eval_radius=0.1` (paper value) even when
  training with a different radius.
- **Connections:** Called by `train_repetition_only.py` before and after
  training. Imports `metrics.py` and `trainer.move_trial_to_device`.

#### `loss_decomposition.py` — *evaluation / metrics code* *(Level 1B)*
- **Role:** Diagnostic post-epoch eval pass that decomposes BCE by phase and
  unit polarity. Intentionally separate from `losses.py` to avoid confusion
  with the training objective.
- **Key items:**
  - `compute_trial_loss_decomposition(tick_results, trial, zero_error_radius) -> dict`
    — 8-key dict: `input_bce`, `output_bce`, `motor_bce`, `output_pos_bce`,
    `output_neg_bce`, `n_active_input`, `n_active_output_pos`,
    `n_active_output_neg`. Called in eval mode under `torch.no_grad()`.
  - `compute_epoch_loss_decomp(model, trials, cfg, zero_error_radius, device) -> dict`
    — 11-key dict with `avg_eval_` prefix. Runs a second forward pass over all
    trials after each training epoch. Sets `model.eval()` internally; caller
    must restore `model.train()` afterward (the train script does this).
- **Critical distinction:** Keys are prefixed `avg_eval_` because values
  reflect model weights at *epoch-end*, not at the time of each online update.
  These are diagnostic, not the training loss.
- **Connections:** Called by `run_training()` in `train_repetition_only.py`
  when `log_loss_decomp=True`. Imported by the same script under original
  private-name aliases.

#### `encoding.py` — *data code*
- **Role:** English phoneme inventory and one-hot encoder.
- **Key items:**
  - `PhonemeInventory` — ordered `symbols` + `symbol_to_index`.
  - `load_phoneme_inventory(path)` — reads `phonemes.csv`'s `Phoneme` column.
  - `parse_phoneme_sequence(value)` — parses the Python-list-style
    `No_Stress` column via `ast.literal_eval`.
  - `validate_phoneme_coverage(sequences, inventory) -> set[str]` — returns
    any out-of-vocabulary symbols (D18).
  - `encode_phoneme_sequence(sequence, inventory) -> Tensor (T, N)` — one-hot.
- **Connections:** Used by `data.py` loaders; `sound_input_size`/
  `motor_output_size` in `configs/english_nwr.yaml` (`= 39`) is provisional
  pending this validation (D14, D18).

#### `data.py` — *data code*
- **Role:** Loads `wfe.csv` (real words) and `ssp.csv` (pseudowords) into
  typed items with encoded phoneme tensors.
- **Key items:**
  - `WordItem` — `word`, `phonemes` (No_Stress list), `phon_tensor (T,N)`,
    `lexicality`, `length`, `frequency`, `zipf_frequency`, `part_of_speech`,
    `condition`, `morphology`, `source="wfe"`, `row_index`.
  - `PseudowordItem` — `phonemes`, `phon_tensor`, `length`, `sonority`,
    `syllable_type`, `lexicality="pseudoword"`, `source="ssp"`, `row_index`.
  - `load_word_items(wfe_path, inventory) -> list[WordItem]` — validates
    required fields (`Word`, `Lexicality`, `Length`, `No_Stress`), checks
    `len(phonemes) == Length`, encodes via `encode_phoneme_sequence`.
  - `load_pseudoword_items(ssp_path, inventory) -> list[PseudowordItem]` —
    analogous, required fields `No_Stress`, `Length`.
- **Connections:** `row_index` is the join key used by `semantics.py` and by
  the frequency-weighting diagnostics.

#### `semantics.py` — *data code*
- **Role:** Assigns a semantic target vector to each real word.
- **Key items:** `assign_artificial_semantics(word_items, vATL_size=50, seed=42) -> dict[row_index, Tensor(vATL_size,)]`
  — for each `WordItem` (must have a unique `row_index`), draws an
  independent random 0/1 vector via a seeded `torch.Generator`. **This is a
  first-pass placeholder** — the supplement's prototype-based generation
  scheme (50 prototypes × 40 exemplars, 20 on-bits, ≥4-bit Hamming distance)
  is not yet implemented (D15 `Open`/`Provisional`).
- **Connections:** Output dict is passed to `make_comprehension_trial` (as
  the semantic *target*) and `make_speaking_trial` (as the semantic *input*).
  Not used in the repetition-only training script.

### Configs (`configs/`)

| File | sound/motor | iSMG | mSTG | aSTG | vATL | triangularis | Notes |
|---|---|---|---|---|---|---|---|
| `toy.yaml` | 5 | 8 | 10 | 10 | 6 | 10 | Synthetic only; fast tests. |
| `lichtheim2.yaml` | 21 | 50 | 200 | 650 | 50 | 200 | Faithful Japanese sizes `[Paper]`. Includes a `metadata:` block (vocab_size=1710, mora_count=3, etc.) for documentation only. |
| `english_nwr.yaml` | 39 | 50 | 200 | 650 | 50 | 200 | `sound_input_size`/`motor_output_size = 39` is **provisional**, derived from the local `phonemes.csv` inventory size, pending coverage validation (D14/D18). |

### Documentation (`docs/`, `README.md`, `CLAUDE.md`)

- `README.md` — top-level overview, status table, diagnostic-script table,
  appendix with the equation-level architecture summary.
- `CLAUDE.md` — instructions for Claude Code; also a compact "Current
  Development Status" summary.
- `docs/roadmap.md` — the canonical phase-by-phase plan with deliverables and
  success criteria.
- `docs/replication_spec.md` — precise layer/pathway/tick/loss specification
  with citation tags; the most "ground truth" doc for architecture questions.
- `docs/architecture_notes.md` — *why* the architecture is hand-rolled
  instead of `nn.RNN`/`nn.LSTM`, plus the weight-init table.
- `docs/data_encoding_notes.md` — phonological/semantic encoding details,
  Japanese vs. English, CSV schemas.
- `docs/training_notes.md` — training paradigm, schedules, and one
  "Phase 3c-N Implementation" section per training-loop diagnostic.
- `docs/open_questions.md` — D1–D18, the canonical open-questions ledger.
- `docs/repetition_training_loop.md` — full pipeline from CSV to gradient
  update, including BPTT scope and the 38:1 imbalance problem.
- `docs/forward_tick_equations.md` — exact `forward_tick()` equations with
  tensor shapes, learned vs. activation distinction, and diagnostic flags.
- `docs/loss_masks_zero_radius.md` — BCE formula, mask broadcasting, dead
  zone, loss decomposition, and `output_positive_weight` behavior.
- `docs/diagnostic_vs_faithful.md` — classification of every implementation
  component as Faithful / Inferred / English/NWR adaptation / Diagnostic-only.
- `docs/modernization_options.md` — structured roadmap of optional changes
  from zero-risk cleanup (Level 1) to alternative EOS/CrossEntropy model
  (Level 5), with scientific risk assessment for each.

### Diagnostic / training scripts (`scripts/`)

| Script | Phase | Role |
|---|---|---|
| `smoke_train_repetition.py` | 3c-1 | Synthetic repetition smoke test (referenced directly in `CLAUDE.md`'s Commands section). |
| `train_repetition_real_data.py` | 3c-2 | Pipeline validation: real CSV → repetition-only training loop. |
| `train_multitask_real_data.py` | 3c-3 | Pipeline validation: real CSV → REP+COMP+SPK for words, REP-only for pseudowords. |
| `diagnose_small_subset_training.py` | 3c-4 | Baseline stability diagnostic; defines `DiagnosticResult`, `sample_items()`, `run_diagnostic_epochs()` reused by all later comparison scripts. |
| `audit_loss_scaling.py` | 3c-5 | No-training audit of `LossBreakdown` (motor vs. semantic, per-unit) on real data. |
| `compare_loss_reductions.py` | 3c-7 | `sum` vs. `mean_active` loss reduction, identical conditions. |
| `compare_task_schedules.py` | 3c-8 | `uniform` (1/1/1) vs. `paper` (1×REP/3×COMP/2×SPK) task schedule. |
| `compare_frequency_weighting.py` | 3c-9 | Unweighted vs. frequency-weighted (`zipf`/`frequency`) training. |
| `compare_lr_schedules.py` | 3c-10 | Constant vs. paper-proportional LR schedule. |
| `compare_training_recipes.py` | 3c-11 | Combines all four dimensions above into named "recipes" and compares them. |
| `train_repetition_only.py` | 3e | **Primary repetition-only script.** Orchestrates: load config → load English NWR data → build repetition trials → pre-training eval → online SGD training with per-epoch loss decomp → post-training eval → save outputs (metrics.csv, predictions_before/after.json, loss_curve.png). Uses `metrics.py`, `repetition_evaluation.py`, `loss_decomposition.py`. |
| `plot_repetition_metrics.py` | 3e | Standalone plotting utility: reads a completed run's `metrics.csv` and regenerates loss-curve PNGs. No model imports. |

---

## 3. End-to-end pipeline: one word, start to finish

Walking through what happens for a single real word (say, row 0 of `wfe.csv`)
under the `mixed-multitask` mode used by the diagnostic scripts:

1. **Load CSV row.** `load_word_items(wfe_path, inventory)` (`data.py`) reads
   one row of `wfe.csv`, validates `Word`/`Lexicality`/`Length`/`No_Stress`
   are present, and parses `No_Stress` (e.g. `"['AH', 'T', 'EH', 'N', 'D']"`)
   via `parse_phoneme_sequence()`.

2. **Encode phonemes.** Still inside `load_word_items`,
   `encode_phoneme_sequence(phonemes, inventory)` (`encoding.py`) turns the
   `T`-symbol phoneme list into a one-hot tensor of shape `(T, 39)` (English
   config), using `inventory.symbol_to_index`.

3. **Create `WordItem`.** A `WordItem` is constructed with `word`,
   `phonemes`, `phon_tensor (T,39)`, `lexicality="real"`, `length=T`,
   `frequency`, `zipf_frequency`, `part_of_speech`, `condition`, `morphology`,
   `source="wfe"`, `row_index=0`.

4. **Assign artificial semantics.** Once, for *all* loaded `WordItem`s (not
   per-item — done before any subsetting so each word's vector is stable),
   `assign_artificial_semantics(word_items, vATL_size=50, seed=42)`
   (`semantics.py`) produces `sem_map: dict[row_index -> Tensor(50,)]`. For
   our word, `sem_map[0]` is a random 0/1 vector.

5. **Create REP/COMP/SPK trials.** For this word (`build_word_trials` in
   `train_multitask_real_data.py`, or the schedule-aware equivalent in
   `compare_task_schedules.py`):
   - `make_repetition_trial(phon_tensor, motor_size=39, item_id=0, label="word:0:<word>")`
     → `SupervisedTrial(task=REPETITION, n_ticks=2T, motor_targets=[zeros(T,39); phon_tensor], motor_loss_mask=all True, semantic_targets=None)`.
   - `make_comprehension_trial(phon_tensor, sem_map[0], motor_size=39, ...)`
     → `SupervisedTrial(task=COMPREHENSION, n_ticks=T, motor_targets=zeros(T,39), semantic_targets=sem_map[0]` repeated `T` times, both masks all `True`)`.
   - `make_speaking_trial(phon_tensor, sem_map[0], motor_size=39, ...)`
     → `SupervisedTrial(task=SPEAKING, n_ticks=T, sem_input=sem_map[0], motor_targets=phon_tensor, motor_loss_mask=all True, semantic_targets=None)`.

   Pseudowords (from `ssp.csv` / `PseudowordItem`) get only the repetition
   trial (`build_pseudo_trials`).

6. **Initialise `ModelState`.** Inside `model.run_trial(task, phon_tensor, sem_pattern, cfg)`,
   `init_state(cfg, device)` sets `iSMG`, `iSMG_context`, `mSTG`, `aSTG`,
   `vATL_out`, `vATL_context`, `triangularis` to `0.5`, and `motor`/
   `motor_context` to `0` `[Paper]`.

7. **Run `forward_tick` for every tick.** `build_trial_inputs(task, phon_tensor, sem_pattern, cfg)`
   (`tasks.py`) produces the `(sound, clamp_vATL_in)` sequence for this task
   and `T` (see Section 5 for exact contents per task). `run_trial` then
   loops: `new_state, vATL_used = model.forward_tick(state, sound, clamp_vATL_in)`,
   appends a `TickResult(tick_index, task, sound_input, vATL_input_used, state=new_state)`,
   and sets `state = new_state` for the next tick. The whole loop is one
   differentiable computation graph (no `.detach()`/`no_grad()` anywhere).

8. **Compute masked loss.** `compute_trial_loss(tick_results, trial, zero_error_radius, loss_reduction)`
   (`losses.py`) stacks `state.motor` (and `state.vATL_out` for
   comprehension) across all ticks, computes element-wise BCE against
   `trial.motor_targets` (and `semantic_targets`), zeroes out elements where
   the tick's loss mask is `False` (and, if `zero_error_radius > 0`, where
   `|prediction - target| < zero_error_radius`), and sums. Depending on
   `loss_reduction`, this sum is returned as-is (`"sum"`) or divided by the
   number of active elements (`"mean_active"`).

9. **Apply optional loss reduction / frequency weight / LR schedule.**
   - **Loss reduction** is the `loss_reduction` argument from step 8 —
     `"sum"` (paper-faithful default) or `"mean_active"` (diagnostic).
   - **Frequency weight**: if `trial.loss_weight != 1.0` (set by
     `apply_weights_to_trials()` in `compare_frequency_weighting.py`, based on
     `compute_word_weights()` over `zipf_frequency` or `log1p(frequency)`),
     `train_step()` multiplies the loss by `trial.loss_weight` *before*
     `.backward()`.
   - **LR schedule**: at the *start of each epoch* (not per-trial),
     `run_diagnostic_epochs()` can call an `lr_schedule_fn(epoch)` (from
     `compare_lr_schedules.lr_for_epoch`) and overwrite
     `optimizer.param_groups[*]["lr"]`.

10. **Update weights.** `train_step()` (`trainer.py`) finishes with
    `loss.backward()` then `optimizer.step()` — a single SGD step for this one
    trial (online, item-by-item learning, `[Supp]`). The function returns
    `loss.item()` for logging.

This sequence repeats once per `(word/pseudoword, task)` trial, in whatever
order/multiplicity the active task schedule dictates, for every epoch.

### Repetition-only training call graph

For the primary training script (`scripts/train_repetition_only.py`), the
call graph is more specific. See also `docs/repetition_training_loop.md` for
the detailed tick-by-tick breakdown.

```
main()
├── parse_args() / validate_args()
├── load_config(english_nwr.yaml)          → ModelConfig
├── Lichtheim2Model(cfg).to(device)        → model (~350k params)
├── load_phoneme_inventory(phonemes.csv)   → PhonemeInventory (39 symbols)
├── load_word_items(wfe.csv, inventory)    → list[WordItem]
├── build_repetition_trials(items, 39)     → list[SupervisedTrial]
│   └── make_repetition_trial per item
│       motor_targets = [zeros(T,39); phon_tensor]  shape (2T, 39)
│       motor_loss_mask = ones(2T, bool)  — ALL ticks supervised
│
├── evaluate_predictions(model, trials, ...) [pre-training]
│   └── evaluate_one_trial per trial
│       ├── model.eval() + torch.no_grad()
│       ├── model.run_trial() → list[TickResult] (2T entries)
│       └── compute argmax acc, threshold acc, breakdown, word acc
│
├── SGD optimizer
└── run_training(model, trials, optimizer, cfg, epochs, ...)
    └── for each epoch:
        ├── model.train() + shuffle
        └── for each trial:
            └── train_step(model, trial, optimizer, cfg, ...)
                ├── model.train() + zero_grad()
                ├── model.run_trial() → list[TickResult]
                │   └── for t in 0..2T-1: forward_tick(state, sound_t, None)
                ├── compute_trial_loss(tick_results, trial, radius)
                │   ├── stack motor outputs → (2T, 39)
                │   ├── BCE(outputs, targets, reduction="none") → (2T, 39)
                │   ├── apply mask + dead zone → alive (2T, 39)
                │   └── (raw_motor * alive.float()).sum() → scalar
                ├── loss.backward()   ← full BPTT through all 2T ticks
                └── optimizer.step()
        └── [if log_loss_decomp]: compute_epoch_loss_decomp()
            — eval-mode pass; input/output-pos/output-neg BCE breakdown
```

---

## 4. Architecture / forward pass

### `ModelState` fields (the "between-tick memory")

`ModelState` (`layers.py`) has 9 fields. Three of them are **copy/context
buffers** — values computed at tick `t` that become inputs at tick `t+1`:

| Field | What it is | Copy-back? |
|---|---|---|
| `iSMG` | Current dorsal hidden activation | — |
| `iSMG_context` | = `iSMG` from the previous tick | Yes — Elman self-recurrence |
| `motor` | Current insular-motor output activation | — |
| `motor_context` | = `motor` from the previous tick | Yes — motor→iSMG copy-back |
| `mSTG` | Current ventral first-hidden activation | — |
| `aSTG` | Current ventral second-hidden activation | — |
| `vATL_out` | Current semantic *output* activation | — |
| `vATL_context` | = `vATL_out` from the previous tick | Yes — vATL→aSTG copy-back (overridden by an external clamp during speaking) |
| `triangularis` | Current ventral output-side hidden activation | — |

### `forward_tick()` computation

For one tick, given `state` (previous `ModelState`), `sound` (the current
phonological input, or zeros), and an optional `clamp_vATL_in` (used only by
speaking):

```text
vATL_input_used = clamp_vATL_in   if provided
                = state.vATL_context   otherwise

[Optional sound projection — not in paper; bias=False so zeros → zeros]
sound_in = sound_proj(sound)  if sound_proj_size is set
         = sound               otherwise

iSMG_net   = sound_to_iSMG(sound_in)
           + iSMG_elman(state.iSMG_context)
           + motor_copy_to_iSMG(state.motor_context)
new_iSMG   = sigmoid(iSMG_net)

new_mSTG   = sigmoid(sound_to_mSTG(sound_in))

aSTG_net   = mSTG_to_aSTG(new_mSTG)
           + vATL_in_to_aSTG(vATL_input_used)
new_aSTG   = sigmoid(aSTG_net)

new_vATL_out      = sigmoid(aSTG_to_vATL(new_aSTG))
new_triangularis  = sigmoid(aSTG_to_triangularis(new_aSTG))

# Standard (paper-faithful):
motor_net  = iSMG_to_motor(new_iSMG)
           + triangularis_to_motor(new_triangularis)
# Diagnostic (dorsal_motor_only=True): motor_net = iSMG_to_motor(new_iSMG)

new_motor  = sigmoid(motor_net)
```

Then the copy-back update (end of tick):

```text
iSMG_context(t+1)  = new_iSMG     # Elman self-recurrence
motor_context(t+1) = new_motor    # insular-motor → iSMG copy-back
vATL_context(t+1)  = new_vATL_out # vATL_out → vATL_in copy-back
```

`forward_tick` returns `(new_state, vATL_input_used)` — the latter is recorded
in `TickResult` purely for inspection.

Every `*_net` is a sum of `nn.Linear` outputs followed by `sigmoid`, so **all
activations live in `[0, 1]`** `[Paper]`. Clamped inputs (`sound`,
`clamp_vATL_in`) are fed in as raw tensors — they are *not* passed through any
activation function `[Inferred]`.

For the full equation-level detail with shapes and learned-vs-activation
distinctions, see `docs/forward_tick_equations.md`.

### Important note on `motor_copy_to_iSMG`

The name says "copy" — following the paper's vocabulary of "copy-back
connections" — but the implementation is a learned `nn.Linear(motor_size,
iSMG_size, bias=False)` with weights initialized to `uniform(−0.5, 0.5)`. It
transforms the motor output through a learnable projection matrix before adding
it to the iSMG net input. The term "copy connection" in the paper means
"feedback pathway from one region to another," not an identity map.

### Comparison with vanilla RNN / LSTM

- **Vanilla RNN (`nn.RNN`):** `h_t = φ(W_x x_t + W_h h_{t-1} + b)` — one hidden
  vector, one input matrix, one recurrent matrix, applied uniformly at every
  step.
- **LSTM (`nn.LSTM`):** adds gates and a cell state `c_t` to control what's
  retained/discarded. **Nothing like this exists here** — there are no learned
  gates and no cell state.
- **This model:** seven named layers, each updated by its *own* equation, from
  (a) the current task input, (b) feedforward inputs from layers already
  computed earlier in the *same* tick, and (c) the three copy/context buffers
  above carried from the *previous* tick. Recurrence is **local** — only
  `iSMG` has true Elman self-recurrence; the other "recurrent" connections
  (`motor_copy_to_iSMG`, `vATL_in_to_aSTG`) are one-tick-delay copy-backs
  realizing specific bidirectional arrows from Supplementary Figure S1.

The key practical consequence: **recurrence here is manually implemented via
explicit `ModelState` fields that the caller threads from tick to tick** —
there is no internal hidden state managed by a PyTorch recurrent module, and
every intermediate activation at every tick is inspectable.

### Connection table (10–11 `nn.Linear` modules) and initialisation

| Connection | Shape | Bias | Init range | Source |
|---|---|---|---|---|
| `sound_proj` (optional) | `(sound, proj)` | none | `[-1, 1]` | `[Adapted]` not in paper; `bias=False` so zero sound → zero projection |
| `sound_to_iSMG` | `(sound, iSMG)` | yes, `=-1.0` | `[-1, 1]` | `[Paper]` weights / `[Inferred]` bias |
| `iSMG_elman` | `(iSMG, iSMG)` | none | `[-0.5, 0.5]` | `[Paper]` |
| `motor_copy_to_iSMG` | `(motor, iSMG)` | none | `[-0.5, 0.5]` | `[Inferred]` (treated as "recurrent") |
| `iSMG_to_motor` | `(iSMG, motor)` | yes, `=-1.0` | `[-1, 1]` | `[Paper]` weights / `[Inferred]` bias; **carries the motor bias** |
| `sound_to_mSTG` | `(sound, mSTG)` | yes, `=-1.0` | `[-1, 1]` | `[Paper]`/`[Inferred]` |
| `mSTG_to_aSTG` | `(mSTG, aSTG)` | yes, `=-1.0` | `[-1, 1]` | `[Paper]`/`[Inferred]` |
| `vATL_in_to_aSTG` | `(vATL, aSTG)` | none | `[-0.5, 0.5]` | `[Inferred]` (treated as "recurrent") |
| `aSTG_to_vATL` | `(aSTG, vATL)` | yes, `=-1.0` | `[-1, 1]` | `[Paper]`/`[Inferred]` |
| `aSTG_to_triangularis` | `(aSTG, triangularis)` | yes, `=-1.0` | `[-1, 1]` | `[Paper]`/`[Inferred]` |
| `triangularis_to_motor` | `(triangularis, motor)` | none | `[-1, 1]` | `[Paper]` weights; **no bias — motor bias carried by `iSMG_to_motor` only**, `[Inferred]` |

---

## 5. Task logic

All three tasks share the same `Lichtheim2Model`; only the per-tick inputs
(`build_trial_inputs`, `tasks.py`) and the supervised targets/masks
(`make_*_trial`, `trials.py`) differ.

### 5.1 Repetition

- **Inputs:** Sound input clamped to the word's phoneme sequence for ticks
  `0..T-1`; zero sound for ticks `T..2T-1`. `clamp_vATL_in = None` throughout
  (vATL gets its input from `vATL_context`, i.e. its own previous output).
- **Targets:** `motor_targets = [zeros(T, motor_size); phon_tensor]` —
  silence during the input phase, then reproduce the phoneme sequence
  during the output phase.
- **Ticks:** `2T` (`T` input + `T` output) `[Paper, generalised to variable T]`.
- **Active loss:** `motor_loss_mask` is **all `True`** — motor loss is
  computed at *every* tick, including the silent input phase (the model is
  trained to actually output zeros/silence there) `[Paper: motor "required to
  be silent" during input phase]`. No semantic loss.

### 5.2 Comprehension

- **Inputs:** Sound input clamped to the phoneme sequence for all `T` ticks.
  `clamp_vATL_in = None` (vATL input still comes from `vATL_context`).
- **Targets:** `semantic_targets` = the word's semantic vector, **repeated at
  every tick**; `motor_targets = zeros(T, motor_size)`.
- **Ticks:** `T` `[Paper]`.
- **Active loss:** Both `semantic_loss_mask` and `motor_loss_mask` are **all
  `True`** — semantic loss and motor silence loss at every tick.

### 5.3 Speaking / naming

- **Inputs:** Zero sound for all `T` ticks; `clamp_vATL_in = sem_tensor`,
  hard-clamped at *every* tick.
- **Targets:** `motor_targets = phon_tensor`; no semantic targets.
- **Ticks:** `T` `[Paper]`. `phon_tensor` is required even though sound input
  is zero, because its length determines `T`.
- **Active loss:** `motor_loss_mask` is **all `True`**.

### Summary table

| Task | Ticks | Sound input | vATL input | Motor target | Semantic target | Active losses |
|---|---|---|---|---|---|---|
| REPETITION | `2T` | phonemes (T), then zero (T) | `vATL_context` (own previous output) | zeros (T) then phonemes (T) | none | motor, all `2T` ticks |
| COMPREHENSION | `T` | phonemes (T) | `vATL_context` | zeros (T) | semantic vector, all `T` ticks | motor + semantic, all `T` ticks |
| SPEAKING | `T` | zero (T) | clamped semantic vector, all `T` ticks | phonemes (T) | none | motor, all `T` ticks |

---

## 6. Training diagnostics already completed (Phases 3c-7 to 3c-11)

These five diagnostics each isolate **one (or, in 3c-11, several combined)
training-loop choice(s)** and run small (~10-word) comparisons under
identical seeds/data/epochs. All use
`run_diagnostic_epochs()`/`DiagnosticResult` from
`diagnose_small_subset_training.py` (3c-4) as their shared engine, and all
follow the same **fairness pattern**: load/sample items once, then for each
condition do `torch.manual_seed(seed)` → fresh `Lichtheim2Model` → fresh
`optim.SGD` → `random.Random(seed)` for shuffling, with `verbose=False`.

> **Caveat that applies across 3c-8 through 3c-11:** because conditions can
> differ in `loss_reduction`, `trials_per_epoch`, or `frequency_source`,
> **absolute loss values are often not comparable across conditions**. The
> scripts consistently recommend comparing `% decrease = (initial_avg -
> final_avg) / initial_avg × 100` and per-task `initial→final` losses instead.

### Phase 3c-7 — Loss reduction comparison (`compare_loss_reductions.py`)

- **Question:** Does training look different/better under `"sum"` vs. `"mean_active"`?
- **Outcome:** Both reductions produce finite losses with matching epoch
  counts; `docs/training_notes.md` does not record specific smoke-run numbers.

### Phase 3c-8 — Task schedule comparison (`compare_task_schedules.py`)

- **Question:** How does the paper's schedule (`1×REP/3×COMP/2×SPK`) compare
  to a uniform schedule (`1/1/1`)?
- **Outcome:** Marked Complete `✓`; comparison methodology documented.

### Phase 3c-9 — Frequency-weighting diagnostics (`compare_frequency_weighting.py`)

- **Question:** Does per-word frequency weighting (`SupervisedTrial.loss_weight`)
  change training dynamics?
- **Outcome:** Marked Complete `✓`. Unit tests confirm `loss_weight=1.0` is
  numerically identical to no weighting; `loss_weight=2.0` gives ≈2× loss.

### Phase 3c-10 — LR schedule diagnostics (`compare_lr_schedules.py`)

- **Question:** Does the paper's 5-phase LR decay (`0.5` for epochs 1–150,
  stepping down to `0.1` by epoch 200) — proportionally rescaled — change
  small-subset training dynamics?
- **Outcome:** Marked Complete `✓`; schedule-shape tests pass.

### Phase 3c-11 — Integrated training recipe diagnostics (`compare_training_recipes.py`)

- **Question:** Do the four dimensions combine sensibly into named "recipes"?
- **Recipes:** `baseline_constant`, `frequency_constant`, `frequency_paper_lr`,
  `zipf_constant`.
- **Validated smoke run outcome:** `baseline_constant` and `frequency_constant`
  behaved similarly over 20 epochs; `frequency_paper_lr` was slightly slower to
  decrease loss, consistent with LR decay reducing effective step size in later
  epochs.

---

## 7. Level 1B safe cleanup

### What was extracted and where it went

Phase 3e (Level 1B) extracted pure, reusable functions from
`scripts/train_repetition_only.py` into three new library modules in
`src/lichtheim2/`. The train script re-imports them under their original private
names (e.g. `from lichtheim2.metrics import rolling_mean as _rolling_mean`) so
no call sites changed and no behavior changed.

| Function group | Extracted to | Reason |
|---|---|---|
| `_compute_repetition_metric_breakdown`, `_compute_repetition_word_accuracy`, `_prediction_summary`, `_rolling_mean` | `src/lichtheim2/metrics.py` | Pure tensor functions; no model dependencies; independently testable |
| `_evaluate_one_trial`, `evaluate_predictions` | `src/lichtheim2/repetition_evaluation.py` | Eval-mode model runner; can be used from notebooks or future scripts |
| `_compute_trial_loss_decomposition`, `_compute_epoch_loss_decomp` | `src/lichtheim2/loss_decomposition.py` | Diagnostic-only; intentionally separate from `losses.py` to avoid confusion with the training objective |
| `_rolling_mean` (duplicate) | Removed from `plot_repetition_metrics.py`; now imports from `metrics.py` | De-duplicates identical function |

### What files were NOT modified

- `model.py` — unchanged. `forward_tick` equations unmodified.
- `trainer.py` — unchanged.
- `losses.py` — unchanged.
- `tasks.py` — unchanged.
- `trials.py` — unchanged.
- `layers.py` — unchanged.
- `config.py` — unchanged.
- `encoding.py` — unchanged.
- CLI flags, defaults, help text — unchanged.
- Output file formats (`metrics.csv` columns, `run_config.json` keys, `predictions*.json` keys) — unchanged.

### How `train_repetition_only.py` is now an orchestrator

After Level 1B, `train_repetition_only.py` contains only:
1. CLI argument parsing (`parse_args`, `validate_args`)
2. Data loading and trial construction (`build_repetition_trials`)
3. The training epoch loop (`run_training`)
4. Output saving (`save_run_config`, `save_metrics_csv`, `save_predictions`, `save_loss_curve`)
5. `main()`

All computational logic lives in library modules. This enables future scripts
(e.g. `train_multitask.py`, notebook-based analysis) to import and call
`evaluate_predictions` or `compute_epoch_loss_decomp` directly.

### Why the import-alias pattern

```python
# In train_repetition_only.py, after Level 1B:
from lichtheim2.metrics import (
    compute_repetition_metric_breakdown as _compute_repetition_metric_breakdown,
    ...
)
```

Existing tests (e.g. `test_loss_decomposition.py`) load the train script via
`importlib.util.exec_module` at module import time and access functions by
their private-underscore names. The alias preserves those names in the script's
namespace, so all existing tests continue to pass without modification. This is
a behavior-preserving refactor by construction.

### Test status after Level 1B

**429 tests passed. Smoke run passed.**

New test files added:
- `tests/test_repetition_metrics.py` — 11 tests for `metrics.py`
- `tests/test_repetition_evaluation.py` — 5 tests for `repetition_evaluation.py`
- `tests/test_loss_decomposition.py` — 8 pre-existing tests, not modified

---

## 8. Metrics and diagnostic evaluation

### Where metrics come from

The training script (`train_repetition_only.py`) runs `evaluate_predictions`
before and after training and prints a table of metrics. These metrics come
from `metrics.py` via `repetition_evaluation.py::evaluate_one_trial`.

### Per-trial metrics returned by `evaluate_one_trial`

All metrics are computed on the **output phase only** (ticks T..2T-1) unless
noted.

| Metric | Key | What it measures |
|---|---|---|
| Argmax accuracy | `phoneme_accuracy` | Fraction of output ticks where `argmax(motor_output) == argmax(motor_target)`. Main measure of phoneme production. |
| Threshold accuracy (mixed) | `threshold_accuracy` | Fraction of all (output-tick, unit) pairs where `|output − target| < 0.1`. **Dominated by the 38 negative units.** See below. |
| Exact match | `exact_match` | True iff argmax matches target at every output tick. Very strict; near zero at start. |
| Input silence threshold | `input_silence_threshold_acc` | Fraction of (input-tick, unit) pairs with activation < 0.1. Measures whether the model stays silent while listening. |
| Output negative threshold | `output_negative_threshold_acc` | Fraction of output-phase negative-target (target=0) units with activation < 0.1. Rises early in training as negative suppression is learned. |
| Output positive threshold | `output_positive_threshold_acc` | Fraction of output-phase positive-target (target=1) units with activation > 0.9. Rises slowly; measures actual phoneme production. |
| Mean positive activation | `mean_positive_output` | Mean activation of the one target unit per output tick. Rising above 0.5 is the earliest reliable signal of phoneme learning. |
| Mean negative activation | `mean_negative_output` | Mean activation of the 38 non-target units per output tick. Should stay near 0. |
| Mean max motor | `mean_max_motor_output` | Mean per-tick max across all motor units. Increases when the model starts producing any phoneme. |
| Output word accuracy | `output_all_units_within_radius` | True iff every (output-tick, unit) pair is within `eval_radius=0.1` of its target. Candidate paper-like word accuracy `[Open #9]`. |
| Input silence accuracy | `input_all_silent_within_radius` | True iff every motor unit during the input phase is below `eval_radius`. |
| Strict trial accuracy | `trial_all_supervised_units_within_radius` | AND of the two above. Strictest criterion. |

### The 38:1 imbalance problem

With 39-dimensional one-hot encoding, each output tick has:
- **1 positive-target unit** (`target = 1.0`)
- **38 negative-target units** (`target = 0.0`)

The gradient direction in weight space initially favors **suppression** of all
motor output (reducing all 38 negative-unit BCE values) over selective
activation of the single positive unit. A model can reduce total BCE loss
substantially while failing to produce any correct phonemes. Signs of this
pathology:
- `threshold_accuracy` (mixed) rises while `output_positive_threshold_acc` stays near 0
- `output_negative_threshold_acc` rises while `mean_positive_output` stays below 0.5
- `avg_loss` decreases while `phoneme_accuracy` stays near chance

For a detailed analysis of why this happens and strategies to address it
(`output_positive_weight`, `zero_error_radius`, `loss_reduction`), see
`docs/loss_masks_zero_radius.md` sections 5–8.

### Why `threshold_accuracy` is misleading as a primary metric

Starting from sigmoid(bias = −1) ≈ 0.27, all motor units are initially above
zero. The model can quickly learn to push all outputs toward zero, scoring
~97% threshold accuracy (38 out of 39 units have target=0) while producing no
correct phonemes. **Always report argmax accuracy and `mean_positive_output`
alongside threshold accuracy.**

### Diagnostic BCE decomposition

After each training epoch, `compute_epoch_loss_decomp` runs a second eval-mode
forward pass and reports:

| Key | Meaning |
|---|---|
| `avg_eval_input_bce` | Mean BCE on input-phase ticks (motor must be silent) |
| `avg_eval_output_pos_bce` | Mean BCE on the 1 positive unit per output tick |
| `avg_eval_output_neg_bce` | Mean BCE on the 38 negative units per output tick |
| `avg_eval_motor_bce` | Sum of the above three |
| `avg_eval_*_per_active` | Per-active-unit BCE (after dead zone) |

**Typical learning pattern:**
1. `output_neg_bce` drops first (easy: suppress all units)
2. `input_bce` drops moderately (motor silent during listening)
3. `output_pos_bce` drops last (hard: activate only the correct phoneme)

A model that only achieves step 1 has learned to suppress but not to produce.

---

## 9. Paper / supplement alignment table

| Component | Paper / supplement claim | Current implementation | Confidence |
|---|---|---|---|
| Layer sizes (sound/motor 21, iSMG 50, mSTG 200, aSTG 650, vATL 50, triangularis 200) | `[Paper]` Supplementary Fig S1 | `configs/lichtheim2.yaml` matches exactly | Paper |
| English `sound_input_size`/`motor_output_size` (= 39) | n/a (English setup is not in the paper) | `configs/english_nwr.yaml`, derived from local `phonemes.csv` inventory size | Inferred / Open (D14, D18 — coverage validation pending) |
| Activation function (sigmoid, outputs in `[0,1]`) | `[Paper]` | `torch.sigmoid` on every `*_net` in `forward_tick` | Paper |
| Standard feedforward weight init `[-1,1]` | `[Paper]` | `nn.init.uniform_(-1,1)` on 7 connections | Paper |
| Elman weight init `[-0.5,0.5]` (`iSMG_elman`) | `[Paper]` | `nn.init.uniform_(-0.5,0.5)` | Paper |
| Copy-back weight init `[-0.5,0.5]` (`motor_copy_to_iSMG`, `vATL_in_to_aSTG`) | Paper says "recurrent connections" use `[-0.5,0.5]`; doesn't explicitly say copy-back qualifies | Same `[-0.5,0.5]` range applied | Inferred |
| Bias = −1.0 on hidden/output layers | `[Paper]` describes LENS bias-link convention suppressing early activation; exact `nn.Linear.bias` mapping unspecified | `nn.init.constant_(bias, -1.0)` on the 7 modules with `bias=True` | Inferred |
| iSMG Elman self-recurrence | `[Paper]` | `iSMG_context` copy + `iSMG_elman` | Paper |
| Insular-motor → iSMG copy-back | `[Supp Fig S1]` | `motor_context` copy + `motor_copy_to_iSMG` | Supplement |
| vATL_out → vATL_in copy-back into aSTG | `[Supp Fig S1]` | `vATL_context` copy + `vATL_in_to_aSTG`; overridden by `clamp_vATL_in` during speaking | Supplement |
| aSTG/STS copy-back to mSTG | Unclear whether a dedicated copy layer exists (D4) | **Not implemented** | Open (D4) |
| Trial-initial state (hidden incl. vATL_out = 0.5, motor = 0) | `[Paper]` | `init_state()` | Paper |
| Repetition: `2T` ticks, motor silent then reproduces phonemes | `[Paper]` (fixed `T=3`); generalisation to variable `T` | `build_trial_inputs` + `make_repetition_trial` | Paper (structure) / Inferred (variable `T`) |
| Comprehension: evaluated at "tick 3" (last tick of 3) | `[Paper]` | Current code applies semantic **and** motor loss at **every** tick | Paper (last-tick claim) / Inferred (every-tick implementation) |
| Speaking: semantic input clamped, motor evaluated | `[Paper]` clamping; eval tick(s) `[Open]` (D5) | Motor loss applied at every tick | Paper (clamping) / Open (eval ticks, D5) |
| Loss function: cross-entropy `[Paper — Hinton 1989]` | `[Paper]` | `F.binary_cross_entropy` on sigmoid outputs | Inferred (BCE as PyTorch analogue of LENS cross-entropy) |
| Zero-error radius = 0.1 | `[Paper]` | `zero_error_radius` parameter; **default `0.0`** in core code | Paper (value) / Inferred (default usage) |
| Presentation schedule: 1×REP, 3×COMP, 2×SPK per word per epoch | `[Paper/Supp]` | `SCHEDULES["paper"]` in `compare_task_schedules.py` — **opt-in diagnostic** | Paper (numbers) / Open (not yet the default training loop) |
| LR schedule: 0.5 → stepped decay → 0.1 over 200 epochs | `[Paper]` | `lr_for_epoch()` in `compare_lr_schedules.py` — **opt-in diagnostic only** | Paper (schedule) / Inferred (proportional rescaling) |
| Weight decay schedule (1e-6 → 6e-7 stepped) | `[Paper]` | Not implemented | Open / not implemented |
| Online (item-by-item, batch_size=1) updates | `[Supp]` | `train_step()` processes one `SupervisedTrial` per call | Supplement |
| Phonological representation: 21-bit Japanese mora features | `[Supp]` | English config: 39-dim one-hot over ARPAbet phonemes | Supplement (Japanese) / Inferred (English one-hot encoding choice, D14 Open) |
| Semantic representation: prototype-based (50-bit, 50 prototypes × 40 exemplars) | `[Supp]` | `assign_artificial_semantics()` — independent random 50-bit binary vectors | Inferred (placeholder; D15 Open/Provisional) |
| Backpropagation scope (full vs. truncated BPTT within a trial) | `[Open]` per paper | `run_trial` builds one graph per trial; `loss.backward()` called once → full BPTT within the trial | Inferred / Open (D6) |
| Lesioning (pathway zeroing + recovery training) | `[Paper]` | Not implemented; per-connection `nn.Linear` design intended to support it (Phase 4) | Open / Pending |

---

## 10. Open questions before Phase 4

These are the items worth resolving (or at least explicitly deferring) before
moving to larger controlled training and lesioning:

- **Full integrated training recipe.** Phases 3c-7 through 3c-11 are all
  small-subset (≈10 word) diagnostics. A decision is needed on: which recipe
  (if any) becomes the default in `trainer.py`/a new full-training script,
  what vocabulary subset/size to use, and how many epochs.
- **Vocabulary scale.** Preliminary results suggest the model learns well on
  ~200 words but struggles with 1200. Whether this is a capacity issue
  (one-hot 39D encoding, no shared phonological features across phoneme classes)
  or an optimization issue (learning rate, number of epochs, `output_positive_weight`)
  is unresolved. The `--sound-proj-size` option provides a dense learned
  embedding that may help, at the cost of departing from the paper.
- **Weighted training vs. unweighted evaluation.** As flagged in 3c-9 and
  3c-11, frequency-weighted recipes currently report *weighted* training
  losses. If frequency weighting is adopted, an unweighted evaluation pass
  will be needed.
- **Exact semantic target design (D15).** Random independent binary vectors
  vs. the supplement's prototype-based scheme. This affects both how
  "learnable" the comprehension/speaking tasks are and whether semantic
  similarity structure exists at all.
- **Lesioning protocol (Phase 4).** Needs: which pathways correspond to which
  aphasia profiles, how lesions are applied (zeroing vs. noise), and what
  "recovery training" looks like operationally.
- **Comparison with paper results.** Even informally, no run has yet been
  compared against the paper's Figure 2 learning curves or Figure 3 lesion
  profiles.
- Related still-open items: **D4** (aSTG↔mSTG copy-back), **D5** (speaking
  evaluation tick(s)), **D6** (BPTT scope), **D8** (presentation order),
  **D12** (padding/batching/EOS), **D16/D17** (frequency/lexicality effects).

---

## 11. Oral explanation and questions to be ready for

### How to explain this in 5 minutes

> "This is a PyTorch reimplementation of the Lichtheim 2 model from Ueno et
> al. 2011 — a neurocomputational model with two pathways: a dorsal pathway
> for repetition, and a ventral pathway for comprehension and naming, all
> sharing one recurrent network.
>
> The key thing to understand is that this is **not** a standard RNN or
> Transformer. The original model was built in LENS as a hand-rolled,
> tick-by-tick network with seven named layers — iSMG, mSTG, aSTG, vATL,
> triangularis, plus sound input and motor output — each with its own update
> equation, all using sigmoid activations. Three of those layers also have
> 'copy-back' connections: iSMG has Elman self-recurrence, the motor output
> feeds back into iSMG one tick later, and the semantic layer's output feeds
> back into its own input one tick later — except during naming, where that
> semantic input is externally clamped to the target meaning instead.
>
> We've implemented a paper-aligned version of that tick-by-tick structure: same layer sizes,
> same sigmoid activations, same weight initialisation ranges from the paper
> — uniform [-1,1] for most connections, [-0.5,0.5] for the recurrent ones,
> and a -1 bias to suppress early activation. All of that lives in
> `model.py`'s `forward_tick`, and `ModelState` is literally the set of
> tensors that gets threaded from one tick to the next.
>
> On top of that architecture, we've built the data pipeline: an English
> phoneme encoder and loaders for real words and pseudowords, a first-pass
> artificial semantics assignment, and code that turns one word into three
> supervised trials — repetition, comprehension, and naming — each with its
> own tick count, target tensors, and loss mask.
>
> Then there's a training loop: one word at a time, forward through all
> ticks, masked binary cross-entropy loss, backward, one SGD step — that's
> the paper's 'online learning.'
>
> What's *not* done yet: lesioning, faithful prototype-based semantic vectors
> from the supplement, and any full-scale training run. Several architectural
> details — like whether aSTG feeds back to mSTG, and exactly which ticks are
> scored for naming — are still open questions flagged in the docs."

### Questions to be ready for

**"Why not nn.RNN?"**
`nn.RNN` manages the recurrence internally and bundles all timesteps in a
single forward call. This model needs to clamp different inputs at each tick
depending on the task (phonemes in input phase, silence in output phase,
semantic clamp in speaking). It also needs to write different targets for
each tick. An `nn.RNN` would require hacking around these requirements. The
explicit tick loop enables per-tick introspection via `TickResult`, which is
required for future lesioning work. The cost is speed; the benefit is
transparency and faithfulness to the LENS formulation.

**"Why sigmoid?"**
The paper uses sigmoid. Sigmoid outputs in [0,1] model bounded continuous
activation and are compatible with BCE loss. ReLU produces unbounded outputs,
making BCE undefined. Changing to ReLU + alternative loss would be a
different scientific formulation. LENS (the original simulator) used sigmoid
units throughout.

**"What is copy-back?"**
At the end of each tick, the activation of iSMG is saved and fed back as
input to iSMG at the next tick (Elman self-recurrence). The motor output is
saved and fed back into iSMG (motor copy-back — the phonological feedback
loop). And vATL output is saved and fed back into aSTG (semantic recurrence).
These implement temporal integration — the model can "remember" what it heard
and what it produced. In `ModelState`, the `*_context` fields carry this
memory. In `forward_tick`, these context fields appear on the right-hand side
of the equations.

**"Is motor_copy_to_iSMG a real copy?"**
No. The name follows the paper's vocabulary of "copy connections" (meaning
"feedback pathway"), but the implementation is a learned `nn.Linear(39, 50,
bias=False)` with weights initialized to `uniform(−0.5, 0.5)`. It transforms
the motor output through a learnable projection before adding it to the iSMG
net input.

**"When does backward happen?"**
Once per word, after all 2T ticks have been forward-passed. `run_trial()`
builds the full computational graph for all ticks. `compute_trial_loss()`
sums BCE values across all ticks. Then `loss.backward()` propagates
gradients backwards through all 2T ticks in one call. This is full BPTT
within the trial. There is no `.detach()` between ticks.

**"What does the mask do?"**
`motor_loss_mask` is a boolean vector of length 2T, all `True` for
repetition. It is broadcast to `(2T, 39)` to create an element-wise alive
mask, then multiplied by the raw BCE tensor before summing. All ticks
(including the input/silence phase) contribute to the loss.

**"Why did output_positive_weight help?"**
With one-hot 39D encoding, there is 1 positive unit and 38 negative units per
output tick. The 38 negative units are easy to suppress and generate 38 times
as many gradient signals as the 1 positive unit. By multiplying the positive
unit loss by a weight > 1, the gradient pressure on the correct phoneme
is increased relative to the suppression signals. This counteracts the early
learning pathology where the model reduces loss by suppressing everything
rather than by activating the correct phoneme.

**"Why does 200 words work but 1200 not yet?"**
With 200 words, the model can memorize item-specific motor patterns. With
1200 words, it needs to generalize — to learn phonological features shared
across words. The 39D one-hot encoding has no shared structure between
phonetically similar phonemes (e.g. /p/ and /b/ share no active bits). The
model must learn phonological similarity purely from the input/output co-
occurrence statistics. The `--sound-proj-size` option provides a dense
learned embedding that may help, at the cost of departing from the paper.

**"What is the safest batching strategy?"**
Zero-pad all sequences in a batch to the length of the longest word.
Use a boolean padding mask that is False at padded positions. Apply the
padding mask in `compute_trial_loss()` instead of the per-trial
`motor_loss_mask`. The BCE computation remains identical; only the
vectorization changes. This should produce near-identical results to online
training. Batch-size-1 behavior must be verified numerically equivalent to
the current single-trial implementation. Results with batch_size > 1 should
be reported with the caveat that gradient averaging differs from the paper's
online learning. See `docs/modernization_options.md` Level 4.

**"What would EOS change?"**
EOS would add a stop token to the phoneme vocabulary. The model would need
to predict both what phoneme to produce and when to stop. The loss would need
to become CrossEntropyLoss on a softmax distribution over phonemes+EOS.
This fundamentally changes the task from "produce a specific continuous
activation level at each predetermined tick" to "predict the next phoneme
as a categorical distribution until stop." BCE + sigmoid is the paper
formulation. CrossEntropy + softmax is a different model. See
`docs/modernization_options.md` Level 5.

**"What is faithful vs. diagnostic?"**
Faithful means the implementation directly reflects Ueno et al. 2011:
dual-pathway architecture, sigmoid units, online SGD, BCE loss,
`zero_error_radius=0.1`. Diagnostic means additions made to understand or
debug the model, not described in the paper: `dorsal_motor_only`,
`output_positive_weight`, loss decomposition by phase and unit polarity,
`sound_proj_size`, `eval_radius` separate from training radius. Diagnostic
features are always flagged in comments and CLI help text. A full reference
table is in `docs/diagnostic_vs_faithful.md`.

**"What is the current repetition-only result on 200 words?"**
This depends on the specific experimental conditions (epochs, LR, radius).
Point to the most recent `outputs/` run directory and the `predictions_after.json`
and `metrics.csv` files for exact numbers. The loss decomposition curves
(if `--log-loss-decomp` was on) show input_bce, output_neg_bce, and
output_pos_bce separately and are the clearest summary of what the model has
actually learned.

---

## 12. Meeting-ready summary

A condensed reference for preparing to walk someone through the codebase.

### The code path in one paragraph

A word is loaded from `wfe.csv` as a `WordItem` with `phon_tensor` of shape
`(T, 39)`. It becomes a `SupervisedTrial` with `motor_targets = [zeros(T,39);
phon_tensor]` and `motor_loss_mask = ones(2T, bool)`. The model runs
`forward_tick` for each of 2T ticks: ticks 0..T-1 hear the phonemes; ticks
T..2T-1 are silent. All 9 `ModelState` fields update each tick via sigmoid
on linear combinations. After all 2T ticks, the single loss
`sum(BCE(outputs, targets) × alive_mask)` is computed and `loss.backward()`
propagates gradients through the full 2T-tick graph. `optimizer.step()` (SGD)
updates all ~350k parameters.

### The loss in one paragraph

Motor outputs are stacked to shape `(2T, 39)`. BCE is computed element-wise
with `reduction="none"`. A boolean alive mask (broadcast from `motor_loss_mask
(2T,)` to `(2T, 39)`) removes ticks outside the loss window — for repetition,
all ticks are masked True. If `zero_error_radius > 0`, elements where
`|output − target| < radius` are additionally excluded (hard-gated, no
gradient). The surviving BCEs are summed to a scalar. `loss.backward()` is
called once for the whole trial — this is full BPTT through all 2T ticks.

### The cleanup in one paragraph

Level 1B extracted pure metric and evaluation functions from the 1149-line
train script into three new library modules in `src/lichtheim2/`: `metrics.py`
(pure tensor functions), `repetition_evaluation.py` (eval-mode model runner),
`loss_decomposition.py` (diagnostic BCE decomposition). The train script re-
imports them under their original private names, so no call sites changed and
all 429 tests continued to pass. `model.py`, `losses.py`, `trainer.py`, and
`tasks.py` were not touched.

### Quick reference

| Symbol | Meaning | Size (english_nwr config) |
|---|---|---|
| T | Phonemes in one word | variable (e.g. 5 for "attend") |
| 2T | Total ticks for repetition | variable |
| N | Phoneme inventory size | 39 |
| iSMG | Inferior supramarginal gyrus (dorsal hidden) | 50 |
| mSTG | Middle superior temporal gyrus (ventral first hidden) | 200 |
| aSTG | Anterior superior temporal gyrus (ventral second hidden) | 650 |
| vATL | Ventral anterior temporal lobe (semantic) | 50 |
| tri | Triangularis-opercularis | 200 |
| Motor | Motor output | 39 |

| Term | File | Meaning |
|---|---|---|
| `ModelState` | `layers.py` | 9-tensor carry state between ticks |
| `TickResult` | `layers.py` | Full record of one tick (inputs + state) |
| `SupervisedTrial` | `trials.py` | Targets + masks for one word |
| `forward_tick` | `model.py` | One tick of computation |
| `run_trial` | `model.py` | Full word (2T ticks) |
| `train_step` | `trainer.py` | Forward + backward + optimizer step |
| `motor_loss_mask` | `trials.py` | Which ticks contribute to loss (all True for repetition) |
| `zero_error_radius` | `losses.py` | Dead zone threshold; paper value 0.1 |
| `output_positive_weight` | `losses.py` | Diagnostic: upweight positive phoneme unit loss |
| `iSMG_context` | `layers.py` | Elman copy: iSMG state from previous tick |
| `motor_context` | `layers.py` | Motor copy-back: motor from previous tick |
| `vATL_context` | `layers.py` | vATL feedback: vATL_out from previous tick |
| `dorsal_motor_only` | `model.py` | Diagnostic: exclude ventral from motor sum |
| `avg_eval_*` | `loss_decomposition.py` | Post-epoch eval pass (not the training loss) |
| `evaluate_predictions` | `repetition_evaluation.py` | Eval-mode model runner (before/after training) |
| `prediction_summary` | `metrics.py` | 15-key aggregate over a list of trial predictions |

---

*Document scope: this walkthrough describes the state of the repository after
Phase 3e / Level 1B safe cleanup. No Python source files were modified in the
production of this document.*
