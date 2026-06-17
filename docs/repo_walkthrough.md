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
- **An accuracy / evaluation metric** (proportion of words correct per epoch
  per task, as in paper Figure 2) — not implemented anywhere in
  `src/lichtheim2/` or `scripts/` as far as this walkthrough's source review
  found. `[Open]`
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
  `0.0`, and most diagnostic scripts also default to `0.0` (one real-data
  script defaults to `0.1`). This is a per-call argument, not a global
  constant.

### Diagnostic vs. model code

It's worth being explicit about this distinction when presenting:

| Category | Files | Role |
|---|---|---|
| **Model code** | `src/lichtheim2/{layers,model,tasks}.py` | The actual Lichtheim 2 architecture and tick dynamics — this *is* the replication. |
| **Data / encoding code** | `src/lichtheim2/{config,encoding,data,semantics}.py` | Loading config, phonemes, word items, and assigning semantic targets. |
| **Training-mechanics code** | `src/lichtheim2/{trials,losses,trainer}.py` | Turning a `WordItem`/`PseudowordItem` into a supervised trial, computing loss, and taking one optimizer step. This is shared infrastructure, used by both diagnostics and any future full training run. |
| **Diagnostic / comparison scripts** | `scripts/compare_*.py`, `scripts/diagnose_*.py`, `scripts/audit_*.py` | Not part of the model. Each one varies *one or a few* training-loop choices (loss reduction, task schedule, frequency weighting, LR schedule, or combinations) on a small subset (≈10 words) and prints a comparison table. They exist to build intuition and de-risk decisions *before* committing to a full training recipe. |
| **Pipeline-validation scripts** | `scripts/train_repetition_real_data.py`, `scripts/train_multitask_real_data.py`, `scripts/smoke_train_repetition.py` | Earlier "does this even run end-to-end on real/synthetic data" scripts. Not the final training run either. |
| **Documentation** | `README.md`, `CLAUDE.md`, `docs/*.md` | Specification, design rationale, roadmap, open questions. |

---

## 2. File-by-file map

### Core package (`src/lichtheim2/`)

#### `config.py` — *model/config code*
- **Role:** Defines the model's size parameters and loads them from YAML.
- **Key items:** `ModelConfig` dataclass (10 fields: `sound_input_size`,
  `motor_output_size`, `iSMG_hidden_size`, `mSTG_hidden_size`,
  `aSTG_hidden_size`, `triangularis_hidden_size`, `vATL_size`,
  `repetition_ticks=6`, `comprehension_ticks=3`, `speaking_ticks=3`);
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
  - `__init__(cfg)` — instantiates 10 `nn.Linear` layers (one per named
    connection) and calls `_init_weights()`.
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
  builds trials (`train_multitask_real_data.py`,
  `diagnose_small_subset_training.py`, the `compare_*` scripts).

#### `losses.py` — *training-mechanics code*
- **Role:** Computes the masked BCE loss, with an optional per-unit
  breakdown.
- **Key items:**
  - `LossBreakdown` dataclass — `task`, `n_ticks`, `total_loss`,
    `motor_loss`, `semantic_loss`, `n_active_motor`, `n_active_semantic`,
    `motor_loss_per_unit`, `semantic_loss_per_unit`.
  - `compute_trial_loss_breakdown(tick_results, trial, zero_error_radius=0.0)`
    — stacks `motor`/`vATL_out` across ticks, computes
    `F.binary_cross_entropy(..., reduction="none")`, masks by
    `motor_loss_mask`/`semantic_loss_mask` (and optionally by a zero-error
    "dead zone"), and sums.
  - `compute_trial_loss(tick_results, trial, zero_error_radius=0.0, loss_reduction="sum")`
    — returns `breakdown.total_loss` (`"sum"`, default, paper-faithful) or
    `breakdown.total_loss / n_active` (`"mean_active"`, diagnostic only).
- **Connections:** Called by `trainer.train_step()` and `audit_loss_scaling.py`.

#### `trainer.py` — *training-mechanics code*
- **Role:** The actual online training step.
- **Key items:**
  - `move_trial_to_device(trial, device)` — `dataclasses.replace` with all
    tensor fields `.to(device)`.
  - `train_step(model, trial, optimizer, cfg, zero_error_radius=0.0, device="cpu", loss_reduction="sum") -> float`
    — `model.train()` → `optimizer.zero_grad()` → move trial to device →
    `model.run_trial(...)` → `compute_trial_loss(...)` → multiply by
    `trial.loss_weight` if `!= 1.0` → `loss.backward()` → `optimizer.step()`
    → return `loss.item()`.
- **Connections:** This is the single function every training/diagnostic
  script calls per item.

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
  the frequency-weighting diagnostics (`compare_frequency_weighting.py`).

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

### Configs (`configs/`)

| File | sound/motor | iSMG | mSTG | aSTG | vATL | triangularis | Notes |
|---|---|---|---|---|---|---|---|
| `toy.yaml` | 5 | 8 | 10 | 10 | 6 | 10 | Synthetic only; fast tests. |
| `lichtheim2.yaml` | 21 | 50 | 200 | 650 | 50 | 200 | Faithful Japanese sizes `[Paper]`. Includes a `metadata:` block (vocab_size=1710, mora_count=3, etc.) for documentation only. |
| `english_nwr.yaml` | 39 | 50 | 200 | 650 | 50 | 200 | `sound_input_size`/`motor_output_size = 39` is **provisional**, derived from the local `phonemes.csv` inventory size, pending coverage validation (D14/D18). |

### Documentation (`docs/`, `README.md`, `CLAUDE.md`)

- `README.md` — top-level overview, status table, diagnostic-script table,
  appendix with the equation-level architecture summary (Section 4 below
  draws on this).
- `CLAUDE.md` — instructions for Claude Code; also a compact "Current
  Development Status" summary.
- `docs/roadmap.md` — the canonical phase-by-phase plan with deliverables and
  success criteria (Phases 0 through 3c-11 detailed; 4–6 sketched).
- `docs/replication_spec.md` — precise layer/pathway/tick/loss specification
  with citation tags; the most "ground truth" doc for architecture questions.
- `docs/architecture_notes.md` — *why* the architecture is hand-rolled
  instead of `nn.RNN`/`nn.LSTM`, plus the weight-init table.
- `docs/data_encoding_notes.md` — phonological/semantic encoding details,
  Japanese vs. English, CSV schemas.
- `docs/training_notes.md` — training paradigm, schedules, and one
  "Phase 3c-N Implementation" section per training-loop diagnostic.
- `docs/open_questions.md` — D1–D18, the canonical open-questions ledger.

### Diagnostic / training scripts (`scripts/`)

| Script | Phase | Role |
|---|---|---|
| `smoke_train_repetition.py` | 3c-1 | Synthetic repetition smoke test (referenced directly in `CLAUDE.md`'s Commands section). |
| `train_repetition_real_data.py` | 3c-2 | Pipeline validation: real CSV → repetition-only training loop. |
| `train_multitask_real_data.py` | 3c-3 | Pipeline validation: real CSV → REP+COMP+SPK for words, REP-only for pseudowords. |
| `diagnose_small_subset_training.py` | 3c-4 | Baseline stability diagnostic; defines `DiagnosticResult`, `sample_items()`, `run_diagnostic_epochs()` reused by all later comparison scripts. |
| `audit_loss_scaling.py` | 3c-5 | No-training audit of `LossBreakdown` (motor vs. semantic, per-unit) on real data. |
| `compare_loss_reductions.py` | 3c-7 | `sum` vs. `mean_active` loss reduction, identical conditions. |
| `compare_task_schedules.py` | 3c-8 | `uniform` (1/1/1) vs. `paper` (1×REP/3×COMP/2×SPK) task schedule. Defines `SCHEDULES`. |
| `compare_frequency_weighting.py` | 3c-9 | Unweighted vs. frequency-weighted (`zipf`/`frequency`) training. |
| `compare_lr_schedules.py` | 3c-10 | Constant vs. paper-proportional LR schedule. Defines `lr_for_epoch()`. |
| `compare_training_recipes.py` | 3c-11 | Combines all four dimensions above into named "recipes" (`baseline_constant`, `frequency_constant`, `frequency_paper_lr`, `zipf_constant`) and compares them. |

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

iSMG_net   = sound_to_iSMG(sound)
           + iSMG_elman(state.iSMG_context)
           + motor_copy_to_iSMG(state.motor_context)
new_iSMG   = sigmoid(iSMG_net)

new_mSTG   = sigmoid(sound_to_mSTG(sound))

aSTG_net   = mSTG_to_aSTG(new_mSTG)
           + vATL_in_to_aSTG(vATL_input_used)
new_aSTG   = sigmoid(aSTG_net)

new_vATL_out      = sigmoid(aSTG_to_vATL(new_aSTG))
new_triangularis  = sigmoid(aSTG_to_triangularis(new_aSTG))

motor_net  = iSMG_to_motor(new_iSMG)
           + triangularis_to_motor(new_triangularis)
new_motor  = sigmoid(motor_net)
```

Then the copy-back update (end of tick):

```text
iSMG_context(t+1)  = new_iSMG     # Elman self-recurrence
motor_context(t+1) = new_motor    # insular-motor → iSMG copy-back
vATL_context(t+1)  = new_vATL_out # vATL_out → vATL_in copy-back
```

`forward_tick` returns `(new_state, vATL_input_used)` — the latter is recorded
in `TickResult` purely for inspection (it tells you whether vATL's input came
from the previous tick's own output, or from an external clamp).

Every `*_net` is a sum of `nn.Linear` outputs followed by `sigmoid`, so **all
activations live in `[0, 1]`** `[Paper]`. Clamped inputs (`sound`,
`clamp_vATL_in`) are fed in as raw tensors — they are *not* passed through any
activation function `[Inferred]`.

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
  realizing specific bidirectional arrows from Supplementary Figure S1, not a
  generic hidden state.

The key practical consequence: **recurrence here is manually implemented via
explicit `ModelState` fields that the caller threads from tick to tick** —
there is no internal hidden state managed by a PyTorch recurrent module, and
every intermediate activation at every tick is inspectable (used for the
per-component loss breakdown in Section 6, and intended for future
lesioning / RSA work).

### Connection table (10 `nn.Linear` modules) and initialisation

| Connection | Shape | Bias | Init range | Source |
|---|---|---|---|---|
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
  i.e. silence during the input phase, then reproduce the phoneme sequence
  during the output phase.
- **Ticks:** `2T` (`T` input + `T` output) `[Paper, generalised to variable T]`.
  The original paper's fixed-length setup is `2×3=6` ticks.
- **Active loss:** `motor_loss_mask` is **all `True`** — motor loss is
  computed at *every* tick, including the silent input phase (the model is
  trained to actually output zeros/silence there) `[Paper: motor "required to
  be silent" during input phase]`. No semantic loss.
- **Paper vs. inferred:** The 2T structure and "motor silent during input"
  are `[Paper]`; generalising from the fixed 3-mora case to arbitrary `T` is
  `[Inferred]`/`[Paper, generalised]`.

### 5.2 Comprehension

- **Inputs:** Sound input clamped to the phoneme sequence for all `T` ticks.
  `clamp_vATL_in = None` (vATL input still comes from `vATL_context`, which at
  tick 0 is the initial 0.5 state).
- **Targets:** `semantic_targets` = the word's semantic vector, **repeated at
  every tick** (`sem_tensor.unsqueeze(0).expand(T, -1)`); `motor_targets =
  zeros(T, motor_size)`.
- **Ticks:** `T` `[Paper]`.
- **Active loss:** Both `semantic_loss_mask` and `motor_loss_mask` are **all
  `True`** — semantic loss (vATL output vs. target) and motor loss (motor
  required silent) are computed at every tick.
- **Paper vs. inferred:** `[Paper]` confirms comprehension is evaluated "at
  tick 3" (i.e., the *last* tick of a 3-tick trial) — `docs/replication_spec.md`
  and `docs/open_questions.md` both note the evaluation tick(s) for variable
  `T` and for *every* tick (vs. only the last) is not fully pinned down. The
  current code applies the semantic loss at **every** tick, which is a
  generalisation/simplification `[Inferred]`, not a literal reading of "tick
  3 only."

### 5.3 Speaking / naming

- **Inputs:** Zero sound for all `T` ticks; `clamp_vATL_in = sem_tensor`,
  hard-clamped at *every* tick (so `vATL_input_used` is always the external
  semantic pattern, never `vATL_context`).
- **Targets:** `motor_targets = phon_tensor` (the word's own phoneme
  sequence); no semantic targets.
- **Ticks:** `T` `[Paper]`. `phon_tensor` is required even though sound input
  is zero, because its length determines `T`.
- **Active loss:** `motor_loss_mask` is **all `True`** — motor loss at every
  tick.
- **Paper vs. inferred:** Clamping the semantic pattern onto vATL input and
  evaluating motor output is `[Paper]`; the *exact* tick(s) at which motor
  output is scored is `[Open]` (D5) — the current code scores every tick.

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
identical seeds/data/epochs, so the choices can be made with some empirical
grounding before a full run. All use
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

- **Question:** Does training look different/better under `"sum"` (paper
  default — total summed BCE) vs. `"mean_active"` (normalised by the number
  of active output elements, intended to control for task imbalance — e.g.
  comprehension has more active elements than speaking/repetition)?
- **What it does:** Runs both reductions on the same trials/seed/epochs and
  reports initial/final/best average loss and best epoch for each.
- **How to interpret:** Don't compare `sum` vs. `mean_active` absolute values
  (different scales) — compare `% decrease` for each. `verbose=False` was
  added to `run_diagnostic_epochs()` here so only the summary table prints.
- **Outcome recorded in docs:** The implementation and always-run tests pass
  (success criterion has `✓`); `docs/training_notes.md` does not record
  specific smoke-run numbers for this phase beyond confirming the script
  produces finite losses for both reductions with matching epoch counts.

### Phase 3c-8 — Task schedule comparison (`compare_task_schedules.py`)

- **Question:** How does the **paper's presentation schedule**
  (`SCHEDULES["paper"] = {REP: 1, COMP: 3, SPK: 2}`, 6 trials/word) compare to
  a **uniform schedule** (`{REP: 1, COMP: 1, SPK: 1}`, 3 trials/word)?
  Pseudowords always get `1×REP` regardless of schedule.
- **What it does:** Builds trials in a fixed within-word order
  (`REP×n_rep, COMP×n_comp, SPK×n_spk`), shuffles the full list every epoch,
  and compares the two schedules with `--loss-reduction mean_active` (the
  default for this script, since `mean_active` is more informative when
  `trials_per_epoch` differs between conditions — `paper` produces 2× the
  trials of `uniform`).
- **How to interpret:** `% decrease` and per-task `initial→final` losses;
  `trials_per_epoch["paper"] > trials_per_epoch["uniform"]` is asserted by the
  tests.
- **Outcome recorded in docs:** Marked Complete `✓`; `docs/training_notes.md`
  documents the comparison methodology but does not record specific smoke-run
  loss numbers for this phase.

### Phase 3c-9 — Frequency-weighting diagnostics (`compare_frequency_weighting.py`)

- **Question:** Does multiplying each word's loss by a frequency-derived
  weight (`compute_word_weights`, source `"zipf"` or `"frequency"`,
  normalised so the mean weight is 1) change training dynamics, and is the
  `loss_weight` plumbing (`SupervisedTrial.loss_weight`,
  `train_step`'s conditional multiply) correct?
- **What it does:** Builds trials once; "unweighted" uses them as-is,
  "weighted" applies `apply_weights_to_trials()` (which only weights trials
  whose `label` starts with `"word:"`, guarding against `row_index` collisions
  between `wfe.csv` and `ssp.csv`). Same seed/schedule/epochs for both.
- **How to interpret:** The "weighted" initial average loss may differ from
  "unweighted" simply because `train_step` returns the *weighted* loss value
  — compare `% decrease` **within** each condition, not absolute values
  across conditions. Weight stats (min/mean/max) are printed first.
- **Outcome recorded in docs:** Marked Complete `✓`, including unit tests that
  `loss_weight=1.0` is numerically identical to no weighting and
  `loss_weight=2.0` gives ≈2× loss. No specific smoke-run loss numbers
  recorded for the frequency comparison itself.

### Phase 3c-10 — LR schedule diagnostics (`compare_lr_schedules.py`)

- **Question:** Does the paper's 5-phase LR decay (`0.5` for epochs 1–150,
  stepping down to `0.1` by epoch 200, `[Paper]`) — proportionally rescaled to
  whatever `--epochs` is requested — change small-subset training dynamics
  vs. a constant LR?
- **What it does:** `lr_for_epoch(base_lr, epoch, lr_schedule, total_epochs)`
  maps the paper's epoch-fraction boundaries (75/80/85/90% →
  ×1.0/0.8/0.6/0.4/0.2) onto the requested epoch count; `run_diagnostic_epochs()`
  gained an `lr_schedule_fn` hook that overwrites `optimizer.param_groups[*]["lr"]`
  at the start of each epoch. Both conditions share `--task-schedule` and
  `--frequency-source`.
- **How to interpret:** `lr_schedule_used["constant"]` should be flat;
  `lr_schedule_used["paper"]` should decay over the run. Compare `%decrease`
  and final losses as elsewhere.
- **Outcome recorded in docs:** Marked Complete `✓`; no specific smoke-run
  numeric outcome recorded beyond the schedule-shape tests passing.

### Phase 3c-11 — Integrated training recipe diagnostics (`compare_training_recipes.py`)

- **Question:** Do the four dimensions above **combine sensibly** when fixed
  into named "recipes," rather than varied one at a time?
- **What it does:** `RecipeConfig` bundles `task_schedule` / `loss_reduction` /
  `frequency_source` / `lr_schedule`. `RECIPES` defines four recipes — all
  using `task_schedule="paper"` and `loss_reduction="mean_active"`:

  | Recipe | frequency_source | lr_schedule |
  |---|---|---|
  | `baseline_constant` | none | constant |
  | `frequency_constant` | frequency | constant |
  | `frequency_paper_lr` | frequency | paper |
  | `zipf_constant` (optional control) | zipf | constant |

  `run_recipe_comparison()` validates every recipe (`validate_recipe_config`),
  caches base trials by `task_schedule` and frequency-weight maps by
  `frequency_source`, then runs each recipe with a fresh
  model/optimizer/seed. `print_recipe_comparison_table()` prints frequency
  weight stats, a summary table (incl. `% decrease`), and final per-task
  training losses.
- **How to interpret:** Same `% decrease` caveat as 3c-9; **additionally**,
  per-task losses for `frequency_*` recipes are **weighted training losses**
  from `train_step()`, *not* a separate unweighted evaluation pass.
- **Outcome — validated smoke run** (`--mode mixed-multitask --max-words 10
  --max-pseudowords 10 --epochs 20 --lr 0.01 --device cpu --seed 0`, default
  recipes `baseline_constant frequency_constant frequency_paper_lr`):
  - `baseline_constant` and `frequency_constant` behaved **similarly** over 20
    epochs.
  - `frequency_paper_lr` was **slightly slower** to decrease loss on this
    short diagnostic — consistent with its LR decay reducing effective step
    size in later epochs.
  - Reported losses for the frequency-weighted recipes remain **weighted
    training losses**, not unweighted evaluation metrics — a caveat to keep in
    mind before drawing conclusions about whether frequency weighting "helps."

---

## 7. Paper / supplement alignment table

| Component | Paper / supplement claim | Current implementation | Confidence |
|---|---|---|---|
| Layer sizes (sound/motor 21, iSMG 50, mSTG 200, aSTG 650, vATL 50, triangularis 200) | `[Paper]` Supplementary Fig S1 | `configs/lichtheim2.yaml` matches exactly | Paper |
| English `sound_input_size`/`motor_output_size` (= 39) | n/a (English setup is not in the paper) | `configs/english_nwr.yaml`, derived from local `phonemes.csv` inventory size | Inferred / Open (D14, D18 — coverage validation pending) |
| Activation function (sigmoid, outputs in `[0,1]`) | `[Paper]` | `torch.sigmoid` on every `*_net` in `forward_tick` | Paper |
| Standard feedforward weight init `[-1,1]` | `[Paper]` | `nn.init.uniform_(-1,1)` on 7 connections | Paper |
| Elman weight init `[-0.5,0.5]` (`iSMG_elman`) | `[Paper]` | `nn.init.uniform_(-0.5,0.5)` | Paper |
| Copy-back weight init `[-0.5,0.5]` (`motor_copy_to_iSMG`, `vATL_in_to_aSTG`) | Paper says "recurrent connections" use `[-0.5,0.5]`; doesn't explicitly say copy-back qualifies | Same `[-0.5,0.5]` range applied | Inferred |
| Bias = −1.0 on hidden/output layers | `[Paper]` describes LENS bias-link convention suppressing early activation; exact `nn.Linear.bias` mapping unspecified | `nn.init.constant_(bias, -1.0)` on the 7 modules with `bias=True`; `triangularis_to_motor` and the 3 copy/Elman layers have `bias=False` | Inferred |
| iSMG Elman self-recurrence | `[Paper]` | `iSMG_context` copy + `iSMG_elman` | Paper |
| Insular-motor → iSMG copy-back | `[Supp Fig S1]` | `motor_context` copy + `motor_copy_to_iSMG` | Supplement |
| vATL_out → vATL_in copy-back into aSTG | `[Supp Fig S1]` | `vATL_context` copy + `vATL_in_to_aSTG`; overridden by `clamp_vATL_in` during speaking | Supplement |
| aSTG/STS copy-back to mSTG | Unclear whether a dedicated copy layer exists (D4) | **Not implemented** — no feedback connection from aSTG to mSTG | Open (D4) |
| Trial-initial state (hidden incl. vATL_out = 0.5, motor = 0) | `[Paper]` | `init_state()` | Paper |
| Repetition: `2T` ticks, motor silent then reproduces phonemes | `[Paper]` (fixed `T=3`); generalisation to variable `T` | `build_trial_inputs` + `make_repetition_trial`, `T` from `phon_pattern.shape[0]` | Paper (structure) / Inferred (variable `T`) |
| Comprehension: evaluated at "tick 3" (last tick of 3) | `[Paper]` | Current code applies semantic **and** motor loss at **every** tick (`T` of them), not only the last | Paper (last-tick claim) / Inferred (every-tick implementation is a generalisation, not confirmed equivalent) |
| Speaking: semantic input clamped, motor evaluated; exact eval tick(s) | `[Paper]` clamping; eval tick(s) `[Open]` (D5) | Motor loss applied at every tick | Paper (clamping) / Open (eval ticks, D5) |
| Loss function: cross-entropy `[Paper — Hinton 1989]` | `[Paper]` | `F.binary_cross_entropy` on sigmoid outputs | Inferred (BCE as the PyTorch analogue of LENS cross-entropy; exact equivalence unverified) |
| Zero-error radius = 0.1 | `[Paper]` | `zero_error_radius` parameter on `compute_trial_loss`/`train_step`; **default `0.0`** in core code and most diagnostics (one real-data script defaults to `0.1`) | Paper (value) / Inferred (default usage) |
| Presentation schedule: 1×REP, 3×COMP, 2×SPK per word per epoch | `[Paper/Supp]` | `SCHEDULES["paper"]` in `compare_task_schedules.py` — **opt-in diagnostic**, not the default in `trainer.py`/core training | Paper (numbers) / Open (not yet the default training loop) |
| LR schedule: 0.5 → stepped decay → 0.1 over 200 epochs | `[Paper]` | `lr_for_epoch()` in `compare_lr_schedules.py`, proportionally rescaled to `--epochs` — **opt-in diagnostic only** | Paper (schedule) / Inferred (proportional rescaling for non-200 epoch counts) |
| Weight decay schedule (1e-6 → 6e-7 stepped) | `[Paper]` | Not implemented anywhere | Open / not implemented |
| Online (item-by-item, batch_size=1) updates | `[Supp]` | `train_step()` processes one `SupervisedTrial` per call | Supplement |
| Phonological representation: 21-bit Japanese mora distinctive features, 3 morae/word | `[Supp]` | English config: 39-dim one-hot over ARPAbet phonemes (`No_Stress`), variable length `T` | Supplement (Japanese) / Inferred (English one-hot encoding choice, D14 Open) |
| Semantic representation: 50-bit, 50 prototypes × 40 exemplars, 20 on-bits, ≥4-bit Hamming distance | `[Supp]` | `assign_artificial_semantics()` — independent random 50-bit binary vectors, no prototype structure | Inferred (placeholder; D15 Open/Provisional) |
| Backpropagation scope (full vs. truncated BPTT within a trial) | `[Open]` per paper | `run_trial` builds one graph per trial; `loss.backward()` called once per trial → full BPTT within the trial | Inferred / Open (D6) |
| Lesioning (pathway zeroing + recovery training) | `[Paper]` | Not implemented; per-connection `nn.Linear` design intended to support it (Phase 4) | Open / Pending |

---

## 8. Open questions before Phase 4

These are the items worth resolving (or at least explicitly deferring) before
moving to larger controlled training and lesioning:

- **Evaluation metrics.** No accuracy/word-correct metric exists yet. The
  paper reports "proportion of words correct per epoch per task" (Figure 2),
  defined via the zero-error radius `[Inferred from paper description of
  Figure 2]`. We need to decide and implement this before any run can be
  compared to Figure 2.
- **Full integrated training recipe.** Phases 3c-7 through 3c-11 are all
  small-subset (≈10 word) diagnostics. None of them constitutes "the" training
  run — `compare_training_recipes.py`'s recipes are diagnostic comparisons,
  not a committed default. A decision is needed on: which recipe (if any)
  becomes the default in `trainer.py`/a new full-training script, what
  vocabulary subset/size to use, and how many epochs.
- **Weighted training vs. unweighted evaluation.** As flagged in 3c-9 and
  3c-11, frequency-weighted recipes currently report *weighted* training
  losses. If frequency weighting is adopted, an unweighted evaluation pass
  (separate from `train_step`) will be needed so that "did it learn" isn't
  confounded with "how is the loss scaled."
- **Exact semantic target design (D15).** Random independent binary vectors
  vs. the supplement's prototype-based scheme (50 prototypes × 40 exemplars,
  20 on-bits, ≥4-bit Hamming distance). This affects both how "learnable" the
  comprehension/speaking tasks are and whether semantic similarity structure
  (relevant for later RSA, Phase 5) exists at all.
- **Lesioning protocol (Phase 4).** Nothing implemented yet. Needs: which
  pathways/connections correspond to which aphasia profiles in the paper, how
  lesions are applied (zeroing vs. noise, per `docs/replication_spec.md`
  §9), and what "recovery training" looks like operationally.
- **Comparison with paper results.** Even informally, no run has yet been
  compared against the paper's Figure 2 learning curves or Figure 3 lesion
  profiles — this depends on the evaluation metric above and on settling D2
  (what counts as a "successful" replication: curve-shape only, ~10% accuracy
  match, or maximal numerical fidelity).
- Related still-open items worth keeping visible: **D4** (aSTG↔mSTG
  copy-back), **D5** (speaking evaluation tick(s)), **D6** (BPTT scope), **D8**
  (presentation order — fully random vs. task-blocked), **D12**
  (padding/batching/EOS — currently everything is unbatched), **D16/D17**
  (frequency/lexicality effects and whether pseudowords get
  comprehension/speaking trials).

---

## 9. Oral explanation

### How I would explain this in 5 minutes

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
> Most of the recent work has been **diagnostics**, not the model itself: five
> small scripts that each test one training-loop choice — sum vs. normalised
> loss, the paper's 1/3/2 task presentation schedule vs. a uniform one,
> frequency-weighted training, the paper's learning-rate decay schedule, and
> finally combinations of all four — each on a tiny ~10-word subset, just to
> make sure these choices behave sensibly before committing to a real run.
>
> What's *not* done yet: lesioning, any accuracy metric to compare against the
> paper's Figure 2, the faithful prototype-based semantic vectors from the
> supplement, and any full-scale training run. Several architectural details
> — like whether aSTG feeds back to mSTG, and exactly which ticks are scored
> for naming — are still open questions flagged in the docs rather than
> resolved guesses."

---

*Document scope: this walkthrough describes the state of the repository as of
the `docs/repo-walkthrough` branch. It is documentation only — no source files
were modified.*
