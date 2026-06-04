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

---

## Phase 3 — Training and Figure 2-Like Learning Curves

**Goal:** Implement the online training loop and reproduce the learning curve from Figure 2 of Ueno et al. (2011).

**Deliverables:**
- `src/lichtheim2/training.py`: online item-by-item update loop
- `src/lichtheim2/data.py`: synthetic phonological/semantic vocabulary generator
- Training schedule: 1× repetition, 2× speaking, 3× comprehension per word per epoch `[Paper/Supp]`
- Logging of per-task accuracy over epochs
- `scripts/train_faithful.py`: training entry point
- Figure 2-like learning curve plot

**Success criterion:** Model reproduces the qualitative learning profile reported in Figure 2: repetition develops first, followed by comprehension, then speaking/naming, with appropriate frequency effects. To be reviewed before moving to Phase 4.

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

## Phase 6 — English Adaptation / SWP-Inspired Extension

**Goal:** Adapt the model to English phonology and/or a richer semantic representation. Begins only after Phase 3 is confirmed faithful.

**Open questions before starting:**
- Fixed-length vs. variable-length phonological sequences
- Articulatory vs. phonemic feature representation
- Distributional vs. artificial semantic vectors
- Whether to use SWP word norms or a curated English vocabulary

See [docs/open_questions.md](open_questions.md) for the full list.

**Success criterion:** TBD after Phase 3 is confirmed.
