# Lichtheim 3 — Phase Plan

> **Phase 0A — Planning only. No code written. No Python files modified.**
> Created: 2026-06-29 | Branch: `feat/lichtheim3-swp-baseline`

This document defines the phased roadmap for Lichtheim 3 development.
Each phase has a clear goal, allowed/forbidden changes, expected outputs, validation criteria, risks, and open questions to resolve before advancing.

---

## Phase 0A — Planning Docs Only

**Goal:** Create comprehensive planning documentation for the Lichtheim 3 / SWP-compatible baseline integration. No code is written.

**Allowed:**
- Read existing code and docs
- Create or update Markdown docs under `docs/`
- Plan package layout and porting strategy

**Forbidden:**
- Any Python code modification
- Modifying `src/lichtheim2/`
- Creating `src/lichtheim3/` or any package skeleton
- Porting any SWP module
- Running training, evaluation, or tests
- Creating git commits

**Expected outputs:**
- `docs/lichtheim3_technical_plan.md`
- `docs/swp_to_lichtheim3_file_map.md`
- `docs/lichtheim3_decision_log.md`
- `docs/lichtheim3_phase_plan.md` (this file)

**Validation criteria:**
- All four docs created and reviewed by Yair
- No Python files modified (confirm via `git status --short`)
- `src/lichtheim2/` untouched

**Risks:**
- Docs may not match Yair's intended architecture — review needed before Phase 0B
- Naming decisions (Candidate A/B/C/D) must be agreed before use in any shared material

**Dependencies:**
- None — docs-only

**Questions to resolve before Phase 0B:**
1. Does Yair agree with the Candidate A/B/C/D hierarchy?
2. Does Yair agree that Candidate B (SWP + dorsal/iSMG) is the right first true Lichtheim 3?
3. Are the proposed package layout and naming conventions acceptable?
4. Are there existing docs to merge or reconcile?
5. Does Daniel/Robin need to review the file map before Phase 1 begins?

---

## Phase 0B — Package Skeleton Only

**Goal:** Create the `src/lichtheim3/` package skeleton (empty `__init__.py` files only). No logic, no models, no data loading.

**Allowed:**
- Create `src/lichtheim3/` directory tree with empty `__init__.py` files
- Add `src/lichtheim3/` to `pyproject.toml` or `setup.cfg` as a package
- Update `CLAUDE.md` to reflect new package
- Create placeholder stub files if needed for import resolution

**Forbidden:**
- Any actual implementation in skeleton files
- Porting any SWP module
- Creating model, data, or training code
- Running training or evaluation

**Expected outputs:**
- `src/lichtheim3/__init__.py` (empty)
- `src/lichtheim3/data/__init__.py` (empty)
- `src/lichtheim3/models/__init__.py` (empty)
- `src/lichtheim3/training/__init__.py` (empty)
- `src/lichtheim3/evaluation/__init__.py` (empty)
- `src/lichtheim3/figures/__init__.py` (empty)
- `src/lichtheim3/utils/__init__.py` (empty)
- Updated `pyproject.toml` or `setup.cfg` if needed

**Validation criteria:**
- `python -c "import lichtheim3"` succeeds without error
- `src/lichtheim2/` remains fully intact
- No SWP code present

**Risks:**
- Package naming conflicts with existing Lichtheim 2 setup
- Premature directory creation locking in a layout that Yair wants to change

**Dependencies:**
- Phase 0A docs reviewed and approved by Yair

**Questions to resolve before Phase 1:**
1. Confirm Python package name (`lichtheim3` vs. `lichtheim_3` vs. other)
2. Confirm `pyproject.toml` build system and install mode
3. Confirm that tests for Lichtheim 3 will live under `tests/lichtheim3/` (separate from Lichtheim 2 tests)

---

## Phase 1 — SWP-Compatible Baseline (Candidate A)

**Goal:** Implement a working SWP-compatible seq2seq LSTM baseline (Candidate A) for the phonological repetition task. This is the technical reference, **not yet Lichtheim 3**.

**Allowed:**
- Implement `src/lichtheim3/data/` modules (vocabulary, dataset, collation, loaders, folds)
- Implement `src/lichtheim3/models/phoneme_encoder.py` (LSTM encoder)
- Implement `src/lichtheim3/models/phoneme_decoder.py` (LSTM decoder)
- Implement `src/lichtheim3/models/swp_seq2seq_baseline.py` (Candidate A wrapper)
- Implement `src/lichtheim3/training/losses.py` (CrossEntropyLoss + PAD masking)
- Implement `src/lichtheim3/training/repetition.py` (training loop)
- Implement `src/lichtheim3/training/checkpoints.py` (save/load)
- Implement `src/lichtheim3/utils/` (paths, seeds, config)
- Create `scripts/train_swp_baseline.py`
- Create `configs/swp_baseline.yaml`
- Write unit tests for each new module under `tests/lichtheim3/`

**Forbidden:**
- Implementing `dorsal_route.py`, `ventral_route.py`, `route_gating.py`, `dual_route_model.py`
- Adding GRU
- Adding vATL
- Adding gating
- Calling Candidate A "Lichtheim 3"
- Modifying `src/lichtheim2/`

**Expected outputs:**
- Working Candidate A training script (`scripts/train_swp_baseline.py`)
- Unit tests passing for all new modules
- Candidate A trains without error on local data

**Validation criteria:**
- `python -m pytest tests/lichtheim3/ -v` passes
- Training script runs to completion on local phoneme data
- Loss decreases during training
- All PAD/SOS/EOS conventions confirmed with Daniel/Robin

**Risks:**
- Phoneme vocabulary mismatch with SWP if token IDs are not confirmed first
- Teacher forcing conventions may differ from SWP
- Collation / padding bugs with variable-length sequences
- Overfitting without careful train/val/test fold design

**Dependencies:**
- Phase 0B skeleton approved
- PAD/SOS/EOS token IDs confirmed with Daniel/Robin [Pending Daniel/Robin]
- Phoneme vocabulary confirmed with Daniel/Robin [Pending Daniel/Robin]
- Train/val/test fold specification confirmed [Pending Daniel/Robin]

**Questions to resolve before Phase 2:**
1. Does Candidate A training loss curve match expected SWP behavior?
2. Does PER at convergence match Daniel/Robin's published baseline?
3. Are there unexpected failure modes in the data pipeline?

---

## Phase 2 — Baseline Reproduction and Comparison

**Goal:** Evaluate Candidate A formally and compare against Daniel/Robin's published SWP results. Establish diagnostic figures. Confirm the baseline before adding any Lichtheim 3 cognitive structure.

**Allowed:**
- Implement `src/lichtheim3/evaluation/repetition.py` (PER, accuracy)
- Implement `src/lichtheim3/evaluation/edit_distance.py`
- Implement `src/lichtheim3/figures/repetition_diagnostics.py`
- Create `scripts/evaluate_swp_baseline.py`
- Create `scripts/plot_swp_results.py`
- Run evaluation and produce diagnostic figures

**Forbidden:**
- Implementing any route module
- Adding GRU, vATL, gating
- Calling Candidate A results "Lichtheim 3 results"
- Modifying `src/lichtheim2/`

**Expected outputs:**
- PER and accuracy metrics for Candidate A
- Diagnostic bar plots and loss curves
- Comparison table: Candidate A vs. Daniel/Robin SWP results
- Written comparison report (Markdown, not a paper)

**Validation criteria:**
- Candidate A PER within acceptable range of SWP published results (exact threshold to agree with Yair/Daniel)
- Diagnostic figures reviewed by Yair
- No systematic discrepancy in training behaviour

**Risks:**
- If Candidate A diverges significantly from SWP results, Phase 3 is blocked
- Overclaiming effects from baseline runs before they are validated

**Dependencies:**
- Phase 1 complete and passing tests
- Daniel/Robin's published results available for comparison [Pending Daniel/Robin]

**Questions to resolve before Phase 3:**
1. Is Candidate A sufficiently close to SWP results to serve as a Lichtheim 3 foundation?
2. Which cognitive effects will be measured and what threshold constitutes "replication"?
3. Are there data or vocabulary issues that need fixing before adding cognitive structure?

---

## Phase 3 — Add Dorsal / iSMG Route (Candidate B — First True Lichtheim 3)

**Goal:** Extend Candidate A with an explicit dorsal phonological route containing an iSMG-like bottleneck. This produces **Candidate B**, the first model that deserves the name "Lichtheim 3".

**Allowed:**
- Implement `src/lichtheim3/models/dorsal_route.py` (iSMG bottleneck; architecture to be specified by Yair)
- Implement `src/lichtheim3/models/dual_route_model.py` (Candidate B: encoder → dorsal route → decoder)
- Implement `src/lichtheim3/evaluation/cognitive_effects.py` (word-length effect, frequency effect, lexicality)
- Create `scripts/train_lichtheim3.py`
- Create `configs/lichtheim3_dorsal.yaml`

**Forbidden:**
- Implementing vATL, gating, lexicalization
- Implementing `ventral_route.py` or `route_gating.py`
- Modifying `src/lichtheim2/`

**Expected outputs:**
- Working Candidate B training script
- Candidate B evaluation: PER + at least one cognitive effect
- Comparison: Candidate A vs. Candidate B

**Validation criteria:**
- Candidate B trains without error
- At least one cognitive effect replicated qualitatively (to be specified with Yair)
- Dorsal route architecture validated by Yair

**Risks:**
- iSMG bottleneck architecture may not be well-specified (depends on Yair)
- Adding a route may degrade PER without cognitive gain — need to understand tradeoff
- Cognitive effects may not emerge from route addition alone

**Dependencies:**
- Phase 2 complete and validated
- Yair's iSMG/dorsal route architecture specification [Pending Yair]
- Cognitive effect evaluation protocol agreed with Yair [Pending Yair]

**Questions to resolve before Phase 4:**
1. Does Candidate B show the expected cognitive effects?
2. Is the iSMG bottleneck in the right position (post-encoder? separate layer? within encoder?)?
3. What lesioning protocol is appropriate for Candidate B?

---

## Phase 4 — Route-Level Ablations

**Goal:** Design and implement route-level ablations for Candidate B (and later Candidate D). Ablations must be cognitively motivated, not just technically possible.

**Allowed:**
- Implement `src/lichtheim3/evaluation/ablations.py`
- Design ablation protocol with Yair (lesion dorsal route, partial lesioning, graded damage)
- Run ablation experiments on Candidate B

**Forbidden:**
- Copying SWP ablation scripts directly — must redesign for Lichtheim 3
- Implementing vATL, gating, lexicalization
- Modifying `src/lichtheim2/`

**Expected outputs:**
- Ablation runner supporting at minimum: full route knock-out, graded lesioning
- Ablation results for Candidate B
- Comparison of ablation profiles with Ueno et al. 2011 findings

**Validation criteria:**
- Ablation results are cognitively interpretable (not just numerically interesting)
- Results reviewed by Yair and Mandonnet
- Ablation protocol documented with cognitive motivation

**Risks:**
- Ablations that are technically possible but cognitively meaningless
- Ablation results that contradict Ueno et al. without explanation

**Dependencies:**
- Phase 3 complete
- Ablation protocol approved by Yair [Pending Yair]
- Clinical interpretation guidance from Mandonnet (optional)

**Questions to resolve before Phase 5:**
1. Do Candidate B ablation profiles match expected clinical deficits?
2. Is the lesioning granularity (full knock-out vs. graded) clinically motivated?

---

## Phase 5 — Add vATL / Semantic Bottleneck (Candidate C)

**Goal:** Extend Candidate B or dual-route model with an explicit ventral route containing a vATL-like semantic bottleneck. Produces **Candidate C**.

**Allowed:**
- Implement `src/lichtheim3/models/ventral_route.py` (vATL; architecture to be specified by Yair)
- Extend `dual_route_model.py` for Candidate C
- Add semantic evaluation metrics if a semantic supervision target is available

**Forbidden:**
- Implementing gating / route interaction (Phase 6)
- Implementing lexicalization (Phase 7)
- Modifying `src/lichtheim2/`

**Expected outputs:**
- Working Candidate C with ventral semantic route
- Comparison: Candidate B vs. Candidate C
- Semantic comprehension evaluation (if supervision available)

**Validation criteria:**
- Candidate C trains without error
- Ventral route shows cognitive differentiation from dorsal route
- Architecture validated by Yair

**Risks:**
- No clear semantic supervision target for vATL without additional data
- Semantic representation choice (word vectors? concept space?) may be contentious
- Without semantic data, vATL cannot be trained meaningfully

**Dependencies:**
- Phase 4 complete
- Yair's vATL architecture specification [Pending Yair]
- Semantic representation or supervision signal identified [Pending Yair]

**Questions to resolve before Phase 6:**
1. What semantic representation is used to supervise vATL?
2. Does Candidate C show dissociation between repetition and comprehension?

---

## Phase 6 — Add Gating / Route Interaction (Candidate D)

**Goal:** Add a gating or route interaction mechanism to the dual-route model, producing **Candidate D** — the fuller scientific target closest in spirit to Ueno et al. 2011.

**Allowed:**
- Implement `src/lichtheim3/models/route_gating.py`
- Extend `dual_route_model.py` for Candidate D
- Create `configs/lichtheim3_dual_route.yaml`
- Run full Candidate A/B/C/D comparison

**Forbidden:**
- Implementing lexicalization (Phase 7)
- Modifying `src/lichtheim2/`

**Expected outputs:**
- Working Candidate D with gating
- Full model comparison across all four candidates
- Cognitive effect replication report

**Validation criteria:**
- Candidate D shows expected cognitive dissociations
- Gating behaviour is interpretable
- Architecture validated by Yair
- Comparison reviewed by full team

**Risks:**
- Gating may not improve cognitive effects if route structure alone is sufficient
- Gating type (sigmoid, soft, hard, attention) may strongly affect results
- Risk of optimising gating for performance rather than cognitive plausibility

**Dependencies:**
- Phase 5 complete
- Yair's gating design specification [Pending Yair]

**Questions to resolve before Phase 7:**
1. Does Candidate D outperform Candidate B on the agreed cognitive effect metrics?
2. Is the gating mechanism interpretable and cognitively motivated?

---

## Phase 7 — Lexicalization Experiment

**Goal:** Investigate how repeated pseudowords or new words are lexicalized — i.e., how they enter long-term memory after repeated exposure.

**Allowed:**
- Implement lexicalization mechanism as specified by Yair
- Design lexicalization experiment (items, trials, metrics)
- Evaluate whether lexicalized items show word-like vs. nonword-like processing

**Forbidden:**
- Implementing lexicalization as mere memorization (fine-tuning on new items) without cognitive motivation
- Modifying `src/lichtheim2/`

**Expected outputs:**
- Lexicalization mechanism implemented and documented
- Experiment showing lexicalization effects
- Comparison with human NWR lexicalization data (if available)

**Validation criteria:**
- Lexicalization effects are cognitively meaningful
- Mechanism described and validated by Yair

**Risks:**
- Implementing lexicalization as memorization is scientifically misleading
- Lexicalization may require architectural changes that conflict with earlier phases

**Dependencies:**
- Phase 6 complete (or may proceed in parallel with Phase 5/6 if independent)
- Yair's explicit design for lexicalization mechanism [Pending Yair]

---

## Future — Visual / Reading Extension

**Goal:** Extend Lichtheim 3 to include a visual/grapheme route for reading and written word processing. This is **out of scope for V0** and all phases above.

**Possible future content:**
- Grapheme encoder (reading pathway)
- Visual-to-phonological conversion route
- Multi-modal input (auditory + visual)
- CORnet or similar visual processing front-end (if warranted)
- AuriStream integration (if warranted)

**When to revisit:**
- Only if Yair, Mandonnet, or a clinical collaborator identifies a specific research question requiring visual input
- Only after Candidate D is validated for auditory phonological processing

**Risks:**
- Adding visual input prematurely would greatly complicate the architecture
- Visual route may require separate data and significantly different training objectives

---

## Summary Table

| Phase | Label | Goal | Adds | Key blocker |
|-------|-------|------|------|-------------|
| 0A | Planning | Docs only | Four planning Markdown docs | None |
| 0B | Skeleton | Empty package | `src/lichtheim3/__init__.py` hierarchy | Phase 0A approved by Yair |
| 1 | Candidate A | SWP-compatible LSTM baseline | Data pipeline, encoder, decoder, training | PAD/SOS/EOS confirmed with Daniel/Robin |
| 2 | Baseline eval | Reproduce SWP results | Evaluation, diagnostics, comparison | Candidate A training stable |
| 3 | Candidate B | First true Lichtheim 3 | Dorsal route, iSMG bottleneck | Yair's iSMG architecture spec |
| 4 | Ablations | Route-level lesioning | Ablation runner | Ablation protocol approved by Yair |
| 5 | Candidate C | Semantic route | vATL bottleneck | Yair's vATL spec + semantic supervision |
| 6 | Candidate D | Full dual-route + gating | Route gating | Yair's gating spec |
| 7 | Lexicalization | New word memory | Lexicalization mechanism | Yair's explicit lexicalization design |
| Future | Visual ext. | Reading pathway | Visual/grapheme route | Clinical/scientific motivation |