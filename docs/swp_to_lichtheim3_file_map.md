# SWP → Lichtheim 3 Porting Map

> **Phase 0A — Planning only. No porting has occurred.**
> Created: 2026-06-29 | Branch: `feat/lichtheim3-swp-baseline`

This document maps SWP (Single Word Processing / NWR) components by Daniel Dager and Robin Sobczyk
to their proposed Lichtheim 3 destinations. It records the porting decision, cognitive interpretation,
implementation risk, and target phase for each component.

**Legend for Action column:**
- `copy` — copy with minimal changes (mainly imports/paths)
- `adapt` — port but rework substantially for Lichtheim 3 conventions
- `inspect only` — read and understand but do not port code
- `do not port` — explicitly excluded

---

## Section 1: SWP Training and Entry Points

| SWP file | Role in SWP | Action | Lichtheim 3 destination | Cognitive interpretation | Risk | Phase | Notes |
|----------|------------|--------|-------------------------|--------------------------|------|-------|-------|
| `scripts/train_repetition.py` | Main training entry point for repetition task | adapt | `scripts/train_swp_baseline.py` → `scripts/train_lichtheim3.py` | Orchestrates phonological loop: sound in → sound out | Low | Phase 1 | Adapt for Lichtheim 3 config and package structure; do not blindly copy SWP paths |
| `scripts/test_repetition.py` | Main evaluation entry point for repetition | adapt | `scripts/evaluate_swp_baseline.py` | Evaluates phonological repetition accuracy | Low | Phase 2 | Adapt metrics to include cognitive effect evaluation |

---

## Section 2: Data and Vocabulary

| SWP file | Role in SWP | Action | Lichtheim 3 destination | Cognitive interpretation | Risk | Phase | Notes |
|----------|------------|--------|-------------------------|--------------------------|------|-------|-------|
| `swp/datasets/phonemes.py` | Phoneme dataset: maps word phoneme sequences to token IDs | adapt | `src/lichtheim3/data/phoneme_dataset.py` | Input/output representation of phonological form | Low | Phase 1 | Confirm PAD/SOS/EOS IDs with Daniel/Robin before porting [Pending Daniel/Robin] |
| `swp/utils/datasets.py` | Dataset utilities: collation, batching, splits | adapt | `src/lichtheim3/data/collate.py` + `loaders.py` + `folds.py` | Data pipeline (not cognitively meaningful) | Low | Phase 1 | Separate collation and fold logic cleanly |

---

## Section 3: Models

| SWP file | Role in SWP | Action | Lichtheim 3 destination | Cognitive interpretation | Risk | Phase | Notes |
|----------|------------|--------|-------------------------|--------------------------|------|-------|-------|
| `swp/models/encoders.py` | LSTM encoder over phoneme sequence | adapt | `src/lichtheim3/models/phoneme_encoder.py` | Auditory processing / speech perception pathway (mSTG-like) | Medium | Phase 1 | Candidate A: no route labelling. Candidate B: may be split into dorsal input vs. ventral input [Pending Yair] |
| `swp/models/decoders.py` | LSTM decoder generating phoneme sequence | adapt | `src/lichtheim3/models/phoneme_decoder.py` | Motor speech output (motor cortex / articulatory) | Medium | Phase 1 | Candidate A: receives encoder final state directly. Candidate B+: may receive route-processed state instead [Pending Yair] |
| (does not exist in SWP) | — | create | `src/lichtheim3/models/swp_seq2seq_baseline.py` | Candidate A: pure seq2seq, no explicit routes | Low | Phase 1 | Thin wrapper combining encoder + decoder for Candidate A |
| (does not exist in SWP) | — | create later | `src/lichtheim3/models/dorsal_route.py` | [Future B] iSMG-like phonological bottleneck | High | Phase 3 | Architecture pending Yair's schema [Pending Yair] |
| (does not exist in SWP) | — | create later | `src/lichtheim3/models/ventral_route.py` | [Future C] vATL-like semantic bottleneck | High | Phase 5 | Requires semantic supervision target [Pending Yair] |
| (does not exist in SWP) | — | create later | `src/lichtheim3/models/dual_route_model.py` | [Future B/D] Combined dual-route model | High | Phase 3+ | Do not implement prematurely |
| (does not exist in SWP) | — | create later | `src/lichtheim3/models/route_gating.py` | [Future D] Gating / route interaction | Very high | Phase 6 | Must wait for Yair's gating design [Pending Yair] |

---

## Section 4: Training Logic

| SWP file | Role in SWP | Action | Lichtheim 3 destination | Cognitive interpretation | Risk | Phase | Notes |
|----------|------------|--------|-------------------------|--------------------------|------|-------|-------|
| `swp/train/repetition.py` | Core training loop logic (forward, loss, backward) | adapt | `src/lichtheim3/training/repetition.py` | Training procedure for phonological repetition | Low | Phase 1 | Do not copy SWP-specific logging or path management |
| `swp/models/losses.py` | CrossEntropyLoss with PAD masking | adapt | `src/lichtheim3/training/losses.py` | Loss = phoneme prediction accuracy (cognitive concession: CE not biologically stated) [Cognitive concession: CrossEntropyLoss is a modern training objective, not stated in Ueno et al.] | Low | Phase 1 | Confirm `ignore_index` convention with Daniel/Robin [Pending Daniel/Robin] |

---

## Section 5: Evaluation

| SWP file | Role in SWP | Action | Lichtheim 3 destination | Cognitive interpretation | Risk | Phase | Notes |
|----------|------------|--------|-------------------------|--------------------------|------|-------|-------|
| `swp/test/repetition.py` | Evaluation loop: accuracy, PER | adapt | `src/lichtheim3/evaluation/repetition.py` + `edit_distance.py` | Measures phonological fidelity of repetition output | Low | Phase 2 | Add cognitive effect metrics (word-length, frequency, lexicality) separately |
| (does not exist in SWP) | — | create later | `src/lichtheim3/evaluation/cognitive_effects.py` | Formal evaluation of cognitive phenomena (NWR effects, frequency effects) | Medium | Phase 3+ | Must design evaluation protocol with Yair before implementing |
| (does not exist in SWP) | — | create later | `src/lichtheim3/evaluation/ablations.py` | Route-level ablation evaluation | High | Phase 4+ | Must be redesigned for Lichtheim 3 — do not copy SWP ablation scripts |

---

## Section 6: Visualisation

| SWP file | Role in SWP | Action | Lichtheim 3 destination | Cognitive interpretation | Risk | Phase | Notes |
|----------|------------|--------|-------------------------|--------------------------|------|-------|-------|
| `swp/viz/test/` | Diagnostic plots (bar charts, confusion matrices) | inspect only → adapt | `src/lichtheim3/figures/repetition_diagnostics.py` | Not cognitively meaningful; diagnostic tooling | Low | Phase 2 | Inspect SWP plot style for consistency; rewrite cleanly rather than copy |

---

## Section 7: Utilities

| SWP file | Role in SWP | Action | Lichtheim 3 destination | Cognitive interpretation | Risk | Phase | Notes |
|----------|------------|--------|-------------------------|--------------------------|------|-------|-------|
| `swp/utils/models.py` | Model utility functions (loading, saving) | adapt | `src/lichtheim3/training/checkpoints.py` | Not cognitively meaningful | Low | Phase 1 | Simplify to Lichtheim 3 conventions; do not copy SWP-specific assumptions |
| `swp/utils/paths.py` | Path resolution for data, outputs, models | adapt | `src/lichtheim3/utils/paths.py` | Not cognitively meaningful | Low | Phase 1 | Centralise Lichtheim 3 data paths; do not hardcode SWP paths |

---

## Section 8: Files to NOT Port

The following SWP components must **not** be ported to Lichtheim 3, either now or in Phase 1.
They are excluded for scientific, architectural, or scope reasons.

| SWP component | Reason for exclusion |
|---------------|---------------------|
| AuriStream-related files | Not part of Lichtheim 3 scope; different input modality pipeline |
| CORnet / visual components | Visual route is out of scope for V0 [Future phase only] |
| Grapheme route | Visual/reading extension is out of scope for V0 |
| Generated model outputs (`.pt`, `.pth`, `.ckpt`) | Never committed; local only per repo policy |
| Raw weights and trained results | Do not copy; re-train in Lichtheim 3 |
| Logs and run artifacts | Not portable; generate fresh |
| Notebooks | Exploratory only; rewrite relevant plots as scripts |
| SWP ablation scripts | Must be redesigned for Lichtheim 3 cognitive route structure — do not copy blindly |
| SWP intervention scripts | Specific to SWP experimental design; not applicable to Lichtheim 3 |
| Neural regression / trajectory scripts | Out of scope for Lichtheim 3 V0 |
| GRU-specific changes | LSTM first; GRU deferred (DEC-010) |
| Any SWP file referencing CORnet or AuriStream by import | Risk of importing wrong dependencies |

---

## Section 9: Files to Keep Intact

The following **must not be modified** during Lichtheim 3 development (all phases through Phase 2):

| File / directory | Reason |
|-----------------|--------|
| `src/lichtheim2/` | Complete Lichtheim 2 implementation; do not touch |
| Existing Lichtheim 2 docs | Unless explicitly updated in a later phase |
| Existing Lichtheim 2 training scripts | Unless a later phase explicitly requests changes |
| Existing Lichtheim 2 tests (`tests/`) | Unless a later phase explicitly requests changes |
| `CLAUDE.md` | Do not overwrite; update carefully if needed |
| `configs/toy.yaml`, `configs/lichtheim2.yaml`, `configs/english_nwr.yaml` | Lichtheim 2 configs; leave intact |

---

## Section 10: Files to Create Later (Not Now)

The following will be created in future phases. They do not exist yet and **must not be created in Phase 0A or 0B**.

### Phase 0B (skeleton only)
- `src/lichtheim3/__init__.py` (empty)
- `src/lichtheim3/data/__init__.py` (empty)
- `src/lichtheim3/models/__init__.py` (empty)
- `src/lichtheim3/training/__init__.py` (empty)
- `src/lichtheim3/evaluation/__init__.py` (empty)
- `src/lichtheim3/figures/__init__.py` (empty)
- `src/lichtheim3/utils/__init__.py` (empty)

### Phase 1 (SWP-compatible baseline)
- `src/lichtheim3/data/vocabulary.py`
- `src/lichtheim3/data/phoneme_dataset.py`
- `src/lichtheim3/data/collate.py`
- `src/lichtheim3/data/loaders.py`
- `src/lichtheim3/data/folds.py`
- `src/lichtheim3/models/phoneme_encoder.py`
- `src/lichtheim3/models/phoneme_decoder.py`
- `src/lichtheim3/models/swp_seq2seq_baseline.py`
- `src/lichtheim3/training/losses.py`
- `src/lichtheim3/training/repetition.py`
- `src/lichtheim3/training/checkpoints.py`
- `src/lichtheim3/utils/paths.py`
- `src/lichtheim3/utils/seeds.py`
- `src/lichtheim3/utils/config.py`
- `scripts/train_swp_baseline.py`
- `configs/swp_baseline.yaml`

### Phase 2 (baseline evaluation)
- `src/lichtheim3/evaluation/repetition.py`
- `src/lichtheim3/evaluation/edit_distance.py`
- `src/lichtheim3/figures/repetition_diagnostics.py`
- `scripts/evaluate_swp_baseline.py`
- `scripts/plot_swp_results.py`

### Phase 3+ (Lichtheim 3 proper)
- `src/lichtheim3/models/dorsal_route.py`
- `src/lichtheim3/models/dual_route_model.py`
- `src/lichtheim3/evaluation/cognitive_effects.py`
- `scripts/train_lichtheim3.py`
- `configs/lichtheim3_dorsal.yaml`

### Phase 4+ (ablations, vATL, gating)
- `src/lichtheim3/evaluation/ablations.py`
- `src/lichtheim3/models/ventral_route.py`
- `src/lichtheim3/models/route_gating.py`
- `configs/lichtheim3_dual_route.yaml`

### Phase 7+ (lexicalization)
- Design pending Yair's schema

### Future (visual / reading extension)
- Not named yet; out of V0 scope entirely

---

## Section 11: Porting Principles

1. **Understand before porting**: read each SWP file fully before deciding how to adapt it.
2. **Adapt, don't copy**: prefer clean rewrites over literal copying; preserve the logic, not the code.
3. **Cognitive labels first**: before porting any model component, name its cognitive function and note it in a comment.
4. **Document concessions**: whenever a modern shortcut (CE loss, teacher forcing, EOS) departs from Ueno et al., note it with `[Cognitive concession]`.
5. **Verify with sources**: confirm any SWP convention (vocabulary, folds, PAD/SOS/EOS) with Daniel/Robin before hard-coding it.
6. **No premature modules**: do not create route modules, gating, or vATL until the corresponding phase is approved.
7. **Tests before eval**: write unit tests for each new module before running any evaluation.