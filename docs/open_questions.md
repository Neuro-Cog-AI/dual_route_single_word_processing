# Open Questions and Design Decisions

Unresolved design decisions for the Lichtheim 2 PyTorch reimplementation, ordered roughly by the phase in which they become blocking. Items confirmed from the paper or supplement are listed in the Resolved section at the bottom.

**Status labels used:**
- `Open` — no clear answer yet; must be decided before the relevant phase begins
- `Provisional` — a working assumption has been adopted; should be revisited
- `Resolved` — confirmed from the paper, supplement, or prior decision

---

## Scope (Phase 0–1)

### D1. Japanese vs English for the first replication

**Question:** Should the first replication use a Japanese mora-based phonological setup or English phonological representations?

**Options:**
- (a) Japanese-like: 21-bit mora vectors, NTT tri-mora vocabulary — closer to original but NTT data is not publicly available
- (b) English: ARPAbet phoneme sequences, NWR-style data, variable-length

**Status:** `Resolved` — **English/NWR direction** (project guidance). The original Japanese tri-mora setup remains a historical and architectural reference; it is not the primary training target. Local CSV data (phonemes.csv, wfe.csv, ssp.csv) is available but not committed to the public repo.

---

### D2. Success criterion for Phase 3

**Question:** What is the minimum result that constitutes a successful faithful replication?

**Why it matters:** Determines the scope of Phase 3 and whether lesioning infrastructure should be built during Phase 3 or deferred to Phase 4.

**Options:**
- (a) Figure 2 learning curves only (repetition, comprehension, speaking/naming)
- (b) Figure 2 plus at least two key aphasic profiles from Figure 3
- (c) Full lesion synthesis (all panels of Figure 3)

**Status:** `Open`

---

### D3. Original LENS files

**Question:** Should the original LENS model files be requested from the authors (Ueno / Lambon Ralph group)?

**Why it matters:** The LENS files would allow direct comparison of internal activations, weight initialisation, and training dynamics, making it much easier to verify faithfulness. They may not be publicly available or actively maintained.

**Options:**
- (a) Request files from authors (cite the paper's "available on request" statement)
- (b) Proceed from the paper and supplement alone; treat LENS files as a stretch goal

**Status:** `Open`

---

## Architecture (Phase 1–2)

### D4. aSTG/STS copy-back connections

**Question:** Does aSTG/STS have a dedicated copy-back context layer feeding back to mSTG, in addition to the vATL and insular-motor copy mechanisms described in Supplement Figure S1?

**Why it matters:** Figure S1 shows all bidirectional connections are realised via copy layers (Plaut unfolding method). It is not explicit whether every intermediate layer has a copy layer or only the ones labelled in the figure.

**Options:**
- (a) Only vATL-output → vATL-input and insular-motor → iSMG copy-backs; other bidirectional arrows are implicit in feedforward weight direction
- (b) Every layer shown with a bidirectional arrow in Figure S1 has a corresponding copy layer

**Status:** `Open` — blocking for Phase 2

---

### D5. Output evaluation ticks for speaking

**Question:** At which tick(s) is the motor output evaluated during speaking/naming trials?

**Why it matters:** The paper states that all three 21-bit mora vectors are generated over three ticks; it is not explicit whether error is computed at every tick or only specific ones.

**Options:**
- (a) Error computed at every tick (ticks 1, 2, 3)
- (b) Error computed only at the last tick

**Status:** `Open` — comprehension is confirmed at tick 3 `[Paper]`; speaking evaluation tick(s) need confirmation

---

### D6. Backpropagation scope within a trial

**Question:** Is backpropagation applied through all ticks of a trial as a single unrolled graph (full BPTT within trial), or is it truncated?

**Why it matters:** The tick-by-tick implementation in PyTorch must decide whether `.backward()` is called once per trial (through all ticks) or once per tick. This affects gradient flow through copy-back connections.

**Options:**
- (a) Full BPTT unrolled through all ticks of the trial (the natural reading of "online learning with backpropagation")
- (b) Truncated BPTT — backpropagation stopped at copy-back boundaries, following the Plaut unfolding convention

**Status:** `Provisional` — working assumption is full BPTT within a trial; LENS behaviour needs verification

---

## Training (Phase 2–3)

### D7. Required numerical faithfulness

**Question:** How closely must the PyTorch implementation match the original LENS outputs quantitatively?

**Why it matters:** LENS and PyTorch may differ in floating-point behaviour, sigmoid implementation, and gradient computation order. Achieving bit-exact equivalence is likely impossible; the question is how close is "close enough."

**Options:**
- (a) Qualitatively faithful: correct ordering of acquisition curves (repetition first, then comprehension, then speaking), correct aphasic profiles
- (b) Numerically close: epoch counts and accuracy levels within ~10% of Figure 2
- (c) Maximally faithful: minimise any deviation from LENS; justify every deviation explicitly

**Status:** `Open`

---

### D8. Presentation order within an epoch

**Question:** Is the presentation order within an epoch fully randomised across all task × word combinations, or is it blocked by task?

**Why it matters:** Online learning is sensitive to presentation order. The paper says "random order" but does not specify whether task type and word are jointly shuffled or tasks are blocked.

**Options:**
- (a) Fully random: shuffle all (word, task) pairs together each epoch
- (b) Task-blocked: present all repetition trials, then all speaking, then all comprehension (or some fixed block order), with random word order within each block

**Status:** `Provisional` — working assumption is fully random based on the paper's phrasing; confirm from supplement or LENS files

---

## Data Encoding (Phase 2–3)

### D9. Semantic vectors: artificial vs real

**Question:** Should semantic vectors remain entirely artificial (prototype-based 50-bit patterns as in the original) in the first faithful replication, or should a real-world semantic representation be used?

**Why it matters:** Using real semantic representations would depart from the original and make faithfulness harder to assess. Artificial semantics are clearly specified in the supplement.

**Options:**
- (a) Artificial prototype-based vectors exactly as in the supplement (20 on-bits per prototype, 40 exemplars per prototype, minimum 4-bit difference between any two patterns)
- (b) Real distributional semantics (word2vec, GloVe, etc.) — deferred to Phase 6

**Status:** `Provisional` — artificial vectors for Phases 1–5; real semantics deferred to Phase 6

---

## English / NWR-Style Training (Current Direction)

### D10. Fixed vs variable-length phonological sequences

**Question:** Should the model use fixed-length or variable-length phoneme sequences?

**Status:** `Resolved` — variable-length sequences, current direction. Phase 2c implemented unbatched variable-length trial support. Padding, masking, and EOS remain open for batching/training (see D12).

---

### D11. English vocabulary

**Question:** Which English vocabulary source should be used for training?

**Options:**
- Local wfe.csv — real words with frequency, POS, lexicality metadata (not committed)
- Local ssp.csv — pseudowords/nonwords with syllable structure and sonority metadata (not committed)
- CMU Pronouncing Dictionary or other public sources

**Status:** `Provisional` — local wfe.csv (real words) and ssp.csv (pseudowords) are the primary candidates. Both are available locally but not committed to the public repository. Full distribution checks needed before use.

---

### D12. Padding/masking/EOS for batching

**Question:** For training with variable-length sequences across batches, how should sequences of different lengths be handled?

**Options:** zero-padding with loss masking; EOS token; unbatched training only.

**Status:** `Open` — Phase 2c uses unbatched trials only; this question is deferred until training begins.

---

### D13. Repetition tick structure for variable-length sequences

**Question:** For a word of length T phonemes, is repetition always T input + T output = 2T ticks?

**Status:** `Provisional` — treating 2T as the tick structure; implemented in Phase 2c.

---

### D14. English phoneme feature representation

**Question:** What encoding should be used for English phonemes? What is `sound_input_size` in the English/NWR config?

**Options:**
- (A) Phoneme one-hot — one dimension per phoneme; `sound_input_size` = inventory size (initial inspection suggests ~40, pending coverage validation)
- (B) Binary feature vectors from phonemes.csv — Type + vowel features (Height, Backness, Diphthong) + consonant features (Place, Manner, Voiced); produces sparse vectors (~17 bits) but linguistically richer
- (C) One-hot-expanded categorical features — similar to B but with NA category for inapplicable features

**Recommendation:** Option A (one-hot) for the first implementation — simple, auditable, and directly derived from the inventory. Differs from the 21-bit Japanese encoding; the English config will have a different `sound_input_size`.

**Coverage validation required before implementation:** every phoneme in wfe.csv::No_Stress and ssp.csv::No_Stress must appear in phonemes.csv. Any out-of-vocabulary symbol must be resolved first. Do not assume `sound_input_size = 40` is final until this check passes.

**Status:** `Open` — Option A recommended; pending coverage validation and final decision.

---

### D15. Semantic representation for English words

**Question:** Should English words use artificial prototype-based semantic vectors (as in the original) or a real distributional representation?

**Status:** `Provisional` — artificial vectors for Phase 3; revisit later.

---

### D16. Lexicality and frequency in training

**Question:** How should real words and pseudowords be handled differently during training? Should frequency affect presentation rate?

**Why it matters:** The original model used log-frequency scaling. The English/NWR setup should decide whether to replicate this or use a different schedule.

**Status:** `Open`

---

### D17. Pseudowords in comprehension and speaking

**Question:** Should pseudowords be used only for repetition/generalization testing (NWR spirit), or should they also receive artificial semantic vectors for comprehension and speaking/naming?

**Why it matters:** If pseudowords are repetition-only, they require no semantic representation. If used in comprehension/speaking, they need an assigned semantic vector, which changes what the model learns.

**Status:** `Open` — for now, treat pseudowords as repetition/generalization items only, unless a later decision assigns artificial semantic vectors to them.

---

### D18. Phoneme coverage validation (pre-implementation check)

**Question:** Are all phoneme symbols in wfe.csv::No_Stress and ssp.csv::No_Stress covered by the phonemes.csv inventory?

**Why it matters:** If any symbol is missing, the one-hot encoder cannot be built without resolving the gap. Stress-marked forms (AH0, EH1, …) must not be used as inventory keys.

**Status:** `Open` — must be completed before Phase 3b encoding implementation begins.

---

## Resolved Items

These items were open at project start but are now confirmed from the paper or supplement.

| Item | Decision | Source |
|------|----------|--------|
| Training/data direction | English/NWR-style, variable-length phoneme sequences | Project guidance |
| Loss function | Cross-entropy | `[Paper]` — in this codebase, "cross-entropy" means element-wise BCE (`F.binary_cross_entropy`) applied to independent sigmoid outputs, one per unit. This is **not** `torch.nn.CrossEntropyLoss`, which applies softmax and treats all units as a probability distribution over classes. See `docs/diagnostic_vs_faithful.md §5` for the full reasoning and the provisional caveat. |
| Zero-error radius | 0.1 (no gradient if \|output − target\| < 0.1) | `[Paper]` |
| Learning rate schedule | 0.5 until epoch 150; −0.1 per 10 epochs until epoch 180; fixed 0.1 until epoch 200 | `[Paper]` |
| Weight initialisation | Uniform [−1, 1] most; [−0.5, 0.5] recurrent; bias to hidden = −1 | `[Paper]` |
| vATL split | Explicit: vATL-output and vATL-input in Supplement Figure S1 | `[Supp]` |
| 21-bit phonological structure | 15 consonant features + 5 vowel features + 1 pitch-accent bit | `[Supp]` |
| Semantic representation | 50-bit prototype-based; 20 on-bits per prototype; 40 exemplars each | `[Supp]` |
| Comprehension evaluation tick | Tick 3 (last tick, after full word presented) | `[Paper]` |
| Presentations per word per epoch | 1× repetition, 2× speaking, 3× comprehension | `[Paper]` |
| Training length | 200 epochs (≈ 2.05 million word presentations) | `[Paper]` |
