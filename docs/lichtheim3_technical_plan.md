# Lichtheim 3 — Technical Plan

> **Phase 0A — Planning only. No code written. No Python files modified.**
> Created: 2026-06-29 | Branch: `feat/lichtheim3-swp-baseline`

---

## 1. Project Context

This repository hosts work done in collaboration with **Yair Lakretz** (ENS/LSCP) and **Emmanuel Mandonnet** on neurocomputational modelling of language processing pathways.

**Team:**
- Yair Lakretz — scientific lead, architecture design, cognitive validity
- Emmanuel Mandonnet — clinical/neurosurgical perspective
- Louis Hayot — implementation
- Daniel Dager & Robin Sobczyk — SWP/NWR baseline reference (external repo)

---

## 2. Project Pivot Summary

### Old objective (Lichtheim 2 replication)

Faithful PyTorch reimplementation of the Ueno et al. (2011) Lichtheim 2 neurocomputational model:
- Hand-rolled Elman-style RNN
- Explicit tick-by-tick dynamics with copy-back feedback
- Three tasks: REPETITION, COMPREHENSION, SPEAKING
- Paper-faithful weight initialisation
- Housed in `src/lichtheim2/`

This implementation is **complete through Phase 3c-1** (minimal training loop) and **must remain intact**.

### New objective (Lichtheim 3)

Build **Lichtheim 3**, a modernised successor that:
- Uses the SWP/NWR repository by Daniel Dager and Robin Sobczyk as a **technical base** (not a blueprint to copy literally)
- Adopts modern seq2seq conventions (LSTM encoder–decoder, token IDs, CrossEntropyLoss, PAD/SOS/EOS)
- Replicates cognitive effects **qualitatively**, not tick-by-tick dynamics
- Preserves neurocognitive interpretability with explicit route structure
- Documents all cognitive concessions explicitly
- Allows future return to more biologically faithful variants

**Core guidance from Yair (2026-06):**
- "On doit recopier SWP."
- "On fait Lichtheim 3, pas Lichtheim 2."
- "Répliquer = répliquer les effets."
- "Si EOS ou CrossEntropy sont moins cognitivement plausibles, ce n'est pas grave, mais il faut le noter clairement."
- "Plus tard, on pourra revenir et tester d'autres variantes."

---

## 3. Candidate Hierarchy

This is the agreed scientific/architectural hierarchy. Use this terminology throughout all docs and conversations.

```
Candidate A — SWP baseline only
= Technical reference, NOT Lichtheim 3.
= Modern seq2seq LSTM, no cognitive route structure.
= Purpose: establish a working baseline, sanity-check data pipeline,
  compare against Daniel/Robin's published results.

Candidate B — SWP baseline + explicit dorsal/iSMG route
= First likely minimal true Lichtheim 3 candidate.
= Adds a dorsal phonological route with an explicit iSMG-like bottleneck.
= Pending Yair's architecture specification.

Candidate C — SWP baseline + vATL / semantic bottleneck
= Future semantic/lexical extension.
= Requires a semantic supervision signal or lexical target.
= Pending Yair's schema and a suitable semantic representation.

Candidate D — SWP baseline + dorsal + vATL + gating / route interaction
= Fuller scientific target, closer in spirit to Ueno et al. 2011.
= Pending Yair's complete architecture details.
= Long-term goal, do not implement yet.
```

**Important:**
- Do **not** call Candidate A "Lichtheim 3".
- Lichtheim 3 proper begins at **Candidate B** when explicit cognitive route structure is added.
- Candidate D is the long-term target but must not be prematurely frozen or implemented.

---

## 4. Current Status (Phase 0A)

| Item | Status |
|------|--------|
| `src/lichtheim2/` | Complete through Phase 3c-1, must remain intact |
| SWP baseline exploration | Sanity check + short LSTM baseline + diagnostic figures (incomplete) |
| Full SWP reproduction | Not done — no parity with Daniel/Robin's final results |
| Cognitive effects established | Not yet — no formal effect validation |
| Code port (SWP → Lichtheim 3) | Not started |
| `src/lichtheim3/` package | Does not exist yet |
| Phase 0A docs | **In progress (this document)** |

---

## 5. Phase 0A Strict Scope

Phase 0A is **documentation only**.

### Allowed in Phase 0A

- Read existing code and docs
- Create or update Markdown docs under `docs/`
- Plan future package layout and porting strategy

### Forbidden in Phase 0A

- Any Python code modification
- Modifying `src/lichtheim2/`
- Creating `src/lichtheim3/` or any package skeleton
- Porting any SWP module
- Testing GRU, vATL, gating, CORnet, AuriStream
- Running training or evaluation
- Adding configs for real experiments (only future placeholders in docs)
- Creating git commits

---

## 6. Non-Goals

The following are explicitly out of scope, now and for Lichtheim 3 V0:

- Faithful tick-by-tick dynamics (that is Lichtheim 2)
- Copy-back context mechanisms
- Elman-style recurrency
- Visual / grapheme / reading route (AuriStream, CORnet)
- GRU as primary cell (LSTM first)
- vATL implementation before Yair's schema
- Gating / route interaction before Yair's schema
- Lexical memory / lexicalization before explicit design
- Route-level ablations copied blindly from SWP
- Neural regression / trajectory analysis
- Any component not yet requested by Yair

---

## 7. Proposed Future Package Layout

> **This layout is NOT implemented yet. It is planning only.**
> Do not create these files until Phase 0B is approved.

```
src/
  lichtheim2/           ← KEEP INTACT. Do not touch.
    ...

  lichtheim3/           ← Future. Does not exist yet.
    __init__.py

    data/
      __init__.py
      vocabulary.py         ← PAD/SOS/EOS tokens, phoneme vocab
      phoneme_dataset.py    ← Dataset wrapping phoneme sequences
      collate.py            ← Padding/batching collate_fn
      loaders.py            ← train/val/test DataLoader builders
      folds.py              ← k-fold or fixed-fold splitting

    models/
      __init__.py
      phoneme_encoder.py    ← LSTM encoder (SWP-compatible)
      phoneme_decoder.py    ← LSTM decoder (SWP-compatible)
      swp_seq2seq_baseline.py  ← Candidate A: pure seq2seq, no routes
      dorsal_route.py       ← [Future B] iSMG-like bottleneck module
      ventral_route.py      ← [Future C] vATL-like semantic module
      dual_route_model.py   ← [Future B/D] combined route model
      route_gating.py       ← [Future D] gating / route interaction

    training/
      __init__.py
      losses.py             ← CrossEntropyLoss with PAD masking
      repetition.py         ← Training loop for repetition task
      checkpoints.py        ← Save/load model state

    evaluation/
      __init__.py
      repetition.py         ← Accuracy, PER metrics
      edit_distance.py      ← Phoneme edit distance utilities
      cognitive_effects.py  ← Word-length, frequency, NWR effects
      ablations.py          ← [Future] route-level ablation runner

    figures/
      __init__.py
      repetition_diagnostics.py  ← Diagnostic bar plots, loss curves

    utils/
      __init__.py
      paths.py              ← Centralised data/output path resolution
      seeds.py              ← Reproducibility seeding
      config.py             ← Config loading (YAML → dataclass)

scripts/
  train_swp_baseline.py     ← [Future Phase 1] Train Candidate A
  evaluate_swp_baseline.py  ← [Future Phase 2] Evaluate Candidate A
  plot_swp_results.py       ← [Future Phase 2] Plot diagnostics
  train_lichtheim3.py       ← [Future Phase 3+] Train Candidate B+

configs/
  swp_baseline.yaml         ← [Future Phase 1] Candidate A config
  lichtheim3_dorsal.yaml    ← [Future Phase 3] Candidate B config
  lichtheim3_dual_route.yaml  ← [Future Phase 4+] Candidate D config
```

### Coexistence plan

`src/lichtheim2/` and `src/lichtheim3/` are **separate Python packages** in the same repo.
- They share no code at the import level.
- Common utilities (paths, seeds) are replicated per-package to avoid coupling.
- Each has its own tests under `tests/lichtheim2/` and `tests/lichtheim3/`.
- Future optional: extract a `src/shared/` package if duplication becomes burdensome — but not prematurely.

---

## 8. Naming Conventions

| Concept | Convention |
|---------|-----------|
| Package | `lichtheim3` (snake_case, short) |
| Model candidates | Candidate A / B / C / D (never "Lichtheim 3" for Candidate A) |
| Route modules | `dorsal_route.py`, `ventral_route.py` (not "pathway") |
| Decision log IDs | `DEC-001`, `DEC-002`, … |
| Phase labels | `Phase 0A`, `Phase 0B`, `Phase 1`, … |
| Documentation markers | `[Paper §X]`, `[SWP]`, `[Inferred]`, `[Open]`, `[Pending Yair]`, `[Pending Daniel/Robin]` |
| Cognitive concessions | Always marked with `[Cognitive concession: reason]` inline |

---

## 9. Documentation Markers

Extend existing markers from Lichtheim 2:

| Marker | Meaning |
|--------|---------|
| `[Paper §X]` | Stated in Ueno et al. 2011 |
| `[Supp §X]` | Stated in supplementary materials |
| `[SWP]` | Follows Daniel/Robin's SWP implementation |
| `[Inferred]` | Best-effort inference, not stated in any source |
| `[Open]` | Unresolved design decision |
| `[Pending Yair]` | Awaiting Yair's architecture specification |
| `[Pending Daniel/Robin]` | Awaiting clarification from SWP authors |
| `[Cognitive concession]` | Modern shortcut that departs from biological plausibility — must be noted |

---

## 10. Future Placeholders

The following architectural components are **reserved for future phases** and must not be implemented prematurely. They are listed here so that Phase 1 design decisions do not inadvertently foreclose them.

| Component | Placeholder name | Earliest phase |
|-----------|-----------------|----------------|
| Dorsal phonological route | `DorsalRoute` | Phase 3 |
| Supramarginal gyrus bottleneck | `iSMG` | Phase 3 |
| Ventral semantic route | `VentralRoute` | Phase 5 |
| Anterior temporal lobe (semantic) | `vATL` | Phase 5 |
| Route gating mechanism | `RouteGating` | Phase 6 |
| Dorsal–ventral route interaction | part of `dual_route_model.py` | Phase 6 |
| Lexical memory / long-term store | `LexicalMemory` | Phase 7 |
| Lexicalization mechanism | `Lexicalization` | Phase 7 |
| Route-level ablations | `ablations.py` | Phase 4+ |
| Visual / reading extension | (not named yet) | Future (post-V1) |

---

## 11. Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Calling Candidate A "Lichtheim 3" in papers or presentations | High | Enforce naming convention strictly; document in decision log |
| Freezing architecture before Yair's full schema | High | Keep all route modules as placeholders; no frozen design |
| Overclaiming cognitive effects from short baseline runs | High | Label all results as "baseline diagnostic only" until formal validation |
| Mixing Lichtheim 2 and Lichtheim 3 code | Medium | Strict package separation; no cross-imports |
| Copying SWP too literally, losing cognitive interpretability | Medium | Review each porting decision against cognitive function in file map |
| Rewriting too much too early (over-engineering) | Medium | Implement only what is needed per phase |
| Importing CORnet/visual dependencies accidentally | Low–Medium | Explicit "do not port" list in file map |
| Implementing GRU prematurely | Low | DEC-010 explicitly blocks GRU |
| Adding vATL without a clear semantic target | Medium | DEC-011 blocks vATL until Phase 5 |
| Adding gating without a cognitive interpretation | Medium | DEC-012 blocks gating until Phase 6 |
| Implementing lexicalization as mere memorization | Medium | Defer until DEC-013 is resolved with Yair |
| Producing ablations that are technically possible but cognitively uninformative | Medium | DEC-017: ablations must be redesigned for Lichtheim 3, not blindly copied |

---

## 12. Open Questions

### For Yair

1. Is **Candidate B** (SWP + dorsal/iSMG) the right first minimal Lichtheim 3 candidate, or should the first candidate already include vATL?
2. Is vATL required in V0 / first publication, or can it be deferred to a later phase?
3. Where exactly should **iSMG / supramarginal** sit in the encoder–decoder architecture? (Transform encoder final state? Full sequence? Separate bottleneck layer?)
4. Should the decoder receive encoder state **directly**, or only route-processed state (i.e., must all information pass through a route)?
5. What kind of **gating / route interaction** do you expect? (Sigmoid gate? Soft mixture? Hard switch? Attention-based?)
6. How should **repeated pseudowords** enter long-term memory in the model? (Gradient update? External key-value store? Dedicated module?)
7. Which **lesion / ablation profiles** should be prioritised for first validation? (Single route knock-out? Partial lesioning? Graded?)
8. Which **cognitive effects** are mandatory for a first validation of Lichtheim 3? (Word-length effect? Frequency effect? Lexicality? NWR deficit after lesion?)
9. Should the visual / reading extension remain purely future, or is it a near-term priority?
10. Is there a specific list of cognitive phenomena from Ueno et al. 2011 that Lichtheim 3 must replicate to be publishable?

### For Daniel / Robin

1. What is the **exact SWP baseline configuration** used in your final published results? (hidden size, num layers, dropout, etc.)
2. Which **train/validation/test folds** were used, and are they fixed across runs?
3. What are the **phoneme vocabulary conventions**? (phoneme set size, index mapping, special tokens)
4. What are the exact **PAD/SOS/EOS** token IDs and how are they used during teacher forcing and evaluation?
5. What is the exact **CrossEntropyLoss** implementation — is PAD ignored with `ignore_index`? Is label smoothing used?
6. Which **evaluation scripts and metrics** are canonical? (PER? Token accuracy? Sequence accuracy?)
7. How should **Word Feature Evaluation (WFE)** and **Sonority Sequencing Principle (SSP)** results be interpreted from the baseline?
8. Is the **short LSTM baseline** behavior (poor performance at early steps) expected, or a sign of a data issue?
9. Are there **known bugs or assumptions** in the plotting/enrichment scripts we should be aware of?
10. Which SWP files should be **adapted vs. avoided** in a new repo? Are there any known pitfalls in porting?

---

## 13. Related Documents

- [docs/swp_to_lichtheim3_file_map.md](swp_to_lichtheim3_file_map.md) — detailed SWP module porting decisions
- [docs/lichtheim3_decision_log.md](lichtheim3_decision_log.md) — tabular decision log
- [docs/lichtheim3_phase_plan.md](lichtheim3_phase_plan.md) — phased roadmap
- [CLAUDE.md](../CLAUDE.md) — Lichtheim 2 implementation notes (not Lichtheim 3)