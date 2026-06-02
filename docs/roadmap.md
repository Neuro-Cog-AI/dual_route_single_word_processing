# Roadmap

Phased plan for the Lichtheim 2 PyTorch reimplementation.

**Rule:** No phase may begin until the success criterion of the previous phase is confirmed in discussion with Yair.

---

## Phase 0 — Repo Scaffold and Specification (current)

**Goal:** Establish the repository structure, documentation, and specification before writing any model code.

**Deliverables:**
- `README.md`, `.gitignore`, branch structure
- `docs/replication_spec.md`: precise layer/pathway/tick specification
- `docs/open_questions.md`: questions for Yair
- `docs/architecture_notes.md`, `docs/training_notes.md`, `docs/data_encoding_notes.md`
- `configs/toy.yaml`: minimal synthetic config
- `src/lichtheim2/__init__.py`: package marker
- `tests/test_placeholder.py`: placeholder test

**Success criterion:** All spec questions answered or explicitly flagged `[Open]`. PR merged to `develop`.

---

## Phase 1 — Toy Tick-by-Tick Forward Pass

**Goal:** Implement and validate the tick-by-tick activation dynamics in isolation, using the toy config. No training, no real data.

**Deliverables:**
- `src/lichtheim2/layers.py`: `LayerState` dataclass and sigmoid activation
- `src/lichtheim2/model.py`: `Lichtheim2Model` with explicit `forward_tick()` and `run_trial()` methods
- `src/lichtheim2/tasks.py`: trial construction for repetition, comprehension, speaking
- `tests/test_tick_dynamics.py`: unit tests for forward pass shape, activation range, copy-back

**Success criterion:** Pytest passes; forward pass runs end-to-end for all three task types with toy-config layer sizes; activations remain in [0, 1].

---

## Phase 2 — Faithful Architecture

**Goal:** Scale to the full layer sizes from the paper and implement all pathways and copy-back mechanisms faithfully.

**Deliverables:**
- Full-size weight matrices matching `[Paper]` / `[Supp]` layer sizes
- Dorsal path: sound input → iSMG (Elman recurrence) → insular-motor output
- Ventral path: sound input → mSTG/STS → aSTG/STS → vATL (input/output split) → triangularis-opercularis → insular-motor output
- `configs/faithful.yaml`: full-size configuration
- Tests verifying connectivity and copy-back state updates

**Success criterion:** All three task types run forward with full-size layers; copy-back states updated correctly at each tick; all `[Open]` items from Phase 0 resolved.

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

**Success criterion:** Model reaches near-perfect performance on all three tasks in a similar number of epochs as reported in the paper. Curve shape qualitatively matches Figure 2. Confirmed with Yair.

---

## Phase 4 — Lesioning and Recovery

**Goal:** Implement selective weight lesioning and recovery training. Reproduce the aphasic profiles from the paper.

**Deliverables:**
- `src/lichtheim2/lesioning.py`: pathway-specific lesion hooks
- Recovery training loop (retrain after lesion)
- Replication of key lesion figures / tables from Ueno et al.

**Success criterion:** Selective dorsal/ventral lesions produce the expected aphasic profiles (e.g., phonological repetition spared with ventral lesion; semantic comprehension impaired with ventral lesion). Confirmed with Yair.

---

## Phase 5 — Representational Similarity Analyses

**Goal:** Analyse the internal representations learned by the model.

**Deliverables:**
- Extraction of hidden-layer activations across stimuli
- RSA / MDS / t-SNE visualisations
- Comparison to human neuroimaging data where available

**Success criterion:** TBD with Yair.

---

## Phase 6 — English Adaptation / SWP-Inspired Extension

**Goal:** Adapt the model to English phonology and/or a richer semantic representation. Begins only after Phase 3 is confirmed faithful.

**Open questions before starting:**
- Fixed-length vs. variable-length phonological sequences
- Articulatory vs. phonemic feature representation
- Distributional vs. artificial semantic vectors
- Whether to use SWP word norms or a curated English vocabulary

See [docs/open_questions.md](open_questions.md) for the full list.

**Success criterion:** TBD with Yair after Phase 3.
