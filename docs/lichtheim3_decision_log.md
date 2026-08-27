# Lichtheim 3 — Decision Log

> **Phase 0A — Planning only. No code written. No Python files modified.**
> Created: 2026-06-29 | Branch: `feat/lichtheim3-swp-baseline`

This log records all significant design and architectural decisions for Lichtheim 3.
Each decision is assigned a stable ID. Decisions may be updated but IDs are never recycled.

**Status vocabulary:**
- `accepted` — agreed and applied
- `proposed` — proposed but not yet validated
- `pending Yair` — awaiting Yair's architecture specification or approval
- `pending Daniel/Robin` — awaiting clarification from SWP authors
- `future` — acknowledged but explicitly deferred to a later phase
- `blocked` — cannot proceed without external input

---

## Decision Table

| ID | Decision | Chosen option | Alternatives considered | Reason | Cognitive concession | Implementation consequence | Validation needed | Status |
|----|----------|--------------|------------------------|--------|----------------------|---------------------------|------------------|--------|
| DEC-001 | Use SWP as technical reference baseline | Use Daniel/Robin's SWP/NWR repo as the baseline to build on | Build Lichtheim 3 from scratch; extend Lichtheim 2 directly | Yair: "On doit recopier SWP"; SWP provides proven data pipeline, vocabulary, and training logic | None at baseline level — cognitive meaning is added in later candidates | All new code must be SWP-compatible at the data and training interface level | Verify SWP baseline reproduces Daniel/Robin's results before calling it a reference | accepted |
| DEC-002 | Candidate A is NOT Lichtheim 3 | Candidate A = SWP-compatible seq2seq baseline only; "Lichtheim 3" begins at Candidate B | Call Candidate A "Lichtheim 3 baseline" | Naming discipline: Lichtheim 3 requires explicit cognitive route structure, which Candidate A lacks | N/A — Candidate A is a technical reference | All docs, presentations, and code comments must use "Candidate A / SWP baseline" not "Lichtheim 3" | Confirm naming with Yair before any publication | accepted |
| DEC-003 | First true Lichtheim 3 candidate = Candidate B (SWP + dorsal/iSMG) | Candidate B: add explicit iSMG-like dorsal phonological bottleneck to Candidate A | Start directly from Candidate C (semantic); stay at Candidate A | Adding dorsal route is the minimal step toward Lichtheim spirit without requiring semantic supervision | Adding an explicit route is a modelling choice — the route does not perfectly mirror neuroanatomy | Candidate B implementation requires `dorsal_route.py` and updated `dual_route_model.py` | Confirm with Yair whether iSMG is a separate module or part of encoder | pending Yair |
| DEC-004 | Candidate D as fuller scientific target | Candidate D (dorsal + vATL + gating) is the long-term design goal | Stop at Candidate B or C | This is the design most in the spirit of Ueno et al. with both routes and interactions | Gating and route interaction are not stated in SWP or necessarily in Ueno et al. | Candidate D requires architecture details from Yair before any implementation | Awaiting Yair's full schema | pending Yair |
| DEC-005 | Keep `src/lichtheim2/` intact | Do not modify any Lichtheim 2 code during Lichtheim 3 development | Merge Lichtheim 2 and Lichtheim 3 codebases; deprecate Lichtheim 2 | Lichtheim 2 is a working reference implementation and potential fallback; must not be broken | None | `src/lichtheim2/` and `src/lichtheim3/` are separate packages with no shared imports | Verify no accidental imports of lichtheim2 from lichtheim3 code | accepted |
| DEC-006 | Use token IDs + learned embeddings for phoneme representation | Integer token IDs mapped to embeddings via `nn.Embedding` | One-hot vectors; fixed acoustic features; articulatory feature vectors | SWP convention; compatible with modern seq2seq training; avoids fixed phoneme geometry assumptions | Token IDs + embeddings do not directly encode phonological features — this is a cognitive concession from more phonologically grounded input representations [Cognitive concession] | Vocabulary module must define PAD/SOS/EOS IDs and phoneme-to-index mapping | Confirm phoneme set and embedding size with Daniel/Robin [Pending Daniel/Robin] | accepted |
| DEC-007 | Use PAD / SOS / EOS for seq2seq conventions | PAD for collation masking, SOS as decoder seed, EOS as sequence terminator | No special tokens; fixed-length sequences; Lichtheim 2-style tick-count | SWP convention; standard seq2seq; needed for variable-length sequences | EOS and SOS are engineering devices, not biologically stated; noted as cognitive concessions [Cognitive concession: SOS, EOS tokens are not stated in Ueno et al. 2011] | All dataset, collation, and loss code must handle PAD with `ignore_index`; EOS must be stripped at evaluation | Confirm exact token IDs with Daniel/Robin | accepted |
| DEC-008 | Use CrossEntropyLoss with PAD ignored | `nn.CrossEntropyLoss(ignore_index=PAD_ID)` | Binary cross-entropy (Lichtheim 2 approach); CTC loss; other sequence losses | SWP convention; standard for seq2seq over discrete tokens; efficient and well-understood | CrossEntropyLoss is a training objective, not stated in Ueno et al. 2011; BCE was used in Lichtheim 2 [Cognitive concession: CrossEntropyLoss replaces BCE; neither is neurobiologically grounded] | Loss module must mask PAD positions; evaluation must be separate from training loss | Confirm `ignore_index` convention with Daniel/Robin [Pending Daniel/Robin] | accepted |
| DEC-009 | Use LSTM as first baseline cell | LSTM for encoder and decoder | GRU; vanilla RNN; Transformer; Elman (Lichtheim 2 approach) | SWP uses LSTM; LSTM is well-established; avoids premature architecture choices | LSTM is not stated in Ueno et al.; it introduces gating not present in Lichtheim 2 [Cognitive concession: LSTM gating is not neurobiologically motivated in this context] | All `phoneme_encoder.py` and `phoneme_decoder.py` use `nn.LSTM` | Compare LSTM vs GRU only after LSTM baseline is established | accepted |
| DEC-010 | No GRU in Lichtheim 3 V0 | LSTM only for now | GRU | Avoid premature architecture proliferation; establish LSTM baseline first | None — GRU vs. LSTM is a technical choice with minor cognitive implications | No GRU code in Phase 0–2; may revisit in Phase 4+ as ablation | Revisit if LSTM shows systematic weaknesses vs. Daniel/Robin GRU results | accepted |
| DEC-011 | No vATL implementation until Phase 5 | Defer vATL to Phase 5 | Add vATL in Phase 1; add vATL as early as Candidate B | vATL requires a semantic supervision target; no clear semantic representation is available for V0; architecture pending Yair | None — this is a deferral, not a concession | No semantic route module in Phases 0–3; `ventral_route.py` is a placeholder name only | Requires Yair's schema and a semantic representation (word vectors, concept space, etc.) [Pending Yair] | future |
| DEC-012 | No gating / route interaction until Phase 6 | Defer gating to Phase 6 | Add gating in Phase 3 alongside dorsal route | Gating requires understanding of both routes; premature gating risks producing uninterpretable dynamics | None — deferral | `route_gating.py` is a placeholder name only; no implementation before Phase 6 | Requires Yair's full dual-route schema [Pending Yair] | future |
| DEC-013 | No lexical memory / lexicalization until Phase 7 | Defer to Phase 7 | Add lexicalization in Phase 5 | Lexicalization is a complex research question; requires explicit cognitive model from Yair | None — deferral | No lexical store module; no in-context word memorization mechanism before Phase 7 | Requires Yair's design for how repeated pseudowords enter long-term memory [Pending Yair] | future |
| DEC-014 | No lexicalization experiment until explicit design | Defer entirely | Include as Phase 6 experiment | Implementing lexicalization as mere memorization (e.g. fine-tuning on new items) would be scientifically misleading | Lexicalization is cognitively meaningful; implementing it incorrectly would misrepresent the model's capabilities | No experiment scripts for lexicalization until Phase 7 or later | Requires Yair's explicit description of the desired mechanism [Pending Yair] | future |
| DEC-015 | No visual / CORnet / grapheme route in Lichtheim 3 V0 | Exclude visual route from all phases through V1 | Include as future extension in Candidate D or later | Out of scope for current research questions; adds substantial complexity | Reading/visual processing is a separate cognitive function from phonological repetition | No CORnet imports; no grapheme dataset; no visual encoder | May be revisited post-V1 if clinical or behavioural data require it | future |
| DEC-016 | No AuriStream | Exclude AuriStream from all phases | Use AuriStream for auditory input | Different technical paradigm; not compatible with SWP baseline conventions | None | No AuriStream imports anywhere in Lichtheim 3 | N/A | accepted |
| DEC-017 | Route-level ablations must be redesigned for Lichtheim 3, not copied from SWP | Design ablation protocol specifically for Lichtheim 3 route structure | Copy SWP ablation scripts directly | SWP ablations are designed for SWP's architecture; Lichtheim 3 has explicit routes, which allow (and require) more cognitively meaningful ablations | Copying SWP ablations could produce technically valid but cognitively uninformative results | `ablations.py` will be written from scratch in Phase 4+; no SWP ablation code ported | Ablation protocol must be reviewed with Yair before implementation [Pending Yair] | proposed |
| DEC-018 | Visual / reading extension stays out of V0; remains possible future direction | Note as future direction in docs; do not design now | Include as Candidate E | Premature visual route design risks constraining the phonological architecture unnecessarily | None — deferral | No visual-route code, no grapheme vocabulary, no visual encoder in any V0 phase | Revisit only if Yair or Mandonnet requests it for a specific research question | future |

---

## Pending Decisions (to be resolved before Phase 1)

| ID | Question | Blocked by | Target |
|----|----------|-----------|--------|
| — | Exact iSMG architecture (separate module vs. encoder layer) | Pending Yair | Before Phase 3 |
| — | Whether encoder final state or full sequence feeds the dorsal route | Pending Yair | Before Phase 3 |
| — | Exact PAD/SOS/EOS token IDs | Pending Daniel/Robin | Before Phase 1 |
| — | Phoneme vocabulary size and index mapping | Pending Daniel/Robin | Before Phase 1 |
| — | Train/val/test fold specification | Pending Daniel/Robin | Before Phase 1 |
| — | CrossEntropyLoss exact configuration (ignore_index, smoothing?) | Pending Daniel/Robin | Before Phase 1 |
| — | Which cognitive effects are mandatory for first Lichtheim 3 validation | Pending Yair | Before Phase 3 |
| — | Gating type and interaction mechanism | Pending Yair | Before Phase 6 |

---

## Notes on Cognitive Concessions

All cognitive concessions accepted so far are **technical engineering choices** that depart from the Ueno et al. 2011 framework or from biological plausibility. They are:
1. Acceptable for Lichtheim 3 V0
2. Must be documented clearly in papers and presentations
3. May be revisited in future phases

Currently accepted concessions:
- Token IDs + embeddings (vs. one-hot or articulatory features)
- SOS / EOS tokens (vs. explicit tick-count control)
- CrossEntropyLoss (vs. BCE or other objectives)
- LSTM gating (vs. Elman RNN)
- Teacher forcing during training (vs. free running)

Not yet decided but likely to be concessions:
- How to represent semantic knowledge (if vATL is added)
- Whether gating is sigmoid, soft, hard, or attention-based