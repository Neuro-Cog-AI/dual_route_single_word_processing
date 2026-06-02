# Open Questions and Design Decisions

Unresolved design decisions for the Lichtheim 2 PyTorch reimplementation, ordered roughly by the phase in which they become blocking. Items confirmed from the paper or supplement are listed in the Resolved section at the bottom.

**Status labels used:**
- `Open` — no clear answer yet; must be decided before the relevant phase begins
- `Provisional` — a working assumption has been adopted; should be revisited
- `Resolved` — confirmed from the paper, supplement, or prior decision

---

## Scope (Phase 0–1)

### D1. Japanese vs English for the first replication

**Question:** Should the first faithful replication use a Japanese mora-based phonological setup (21-bit vectors, fixed 3-unit sequences) or move directly to an English phonological representation?

**Why it matters:** Choosing English from the start requires additional encoding decisions (variable-length sequences, phoneme inventory, feature system) that are not specified in the paper. Choosing Japanese is safer for faithfulness and closer to the original, but the NTT vocabulary database is not publicly available.

**Options:**
- (a) Japanese-like: random 21-bit vectors following the supplement's feature structure, synthetic vocabulary — no real NTT data required
- (b) English from the start: controlled CVC vocabulary with an articulatory feature set, fixed 3-segment sequences

**Status:** `Open`

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

## English Adaptation (Phase 6 only)

The following questions are deferred until Phase 3 is confirmed faithful. They are listed here to avoid losing them.

### D10. Fixed vs variable-length phonological sequences for English

**Question:** Should the English adaptation preserve fixed-length 3-segment sequences, or use variable-length representations?

**Status:** `Open` — Phase 6

---

### D11. English vocabulary selection

**Question:** What English vocabulary should be used for Phase 6?

**Options:** CVC-only controlled set; SWP word norms; CMU Pronouncing Dictionary subset matched to the original Japanese vocabulary size and frequency distribution.

**Status:** `Open` — Phase 6

---

## Resolved Items

These items were open at project start but are now confirmed from the paper or supplement.

| Item | Decision | Source |
|------|----------|--------|
| Loss function | Cross-entropy | `[Paper]` |
| Zero-error radius | 0.1 (no gradient if \|output − target\| < 0.1) | `[Paper]` |
| Learning rate schedule | 0.5 until epoch 150; −0.1 per 10 epochs until epoch 180; fixed 0.1 until epoch 200 | `[Paper]` |
| Weight initialisation | Uniform [−1, 1] most; [−0.5, 0.5] recurrent; bias to hidden = −1 | `[Paper]` |
| vATL split | Explicit: vATL-output and vATL-input in Supplement Figure S1 | `[Supp]` |
| 21-bit phonological structure | 15 consonant features + 5 vowel features + 1 pitch-accent bit | `[Supp]` |
| Semantic representation | 50-bit prototype-based; 20 on-bits per prototype; 40 exemplars each | `[Supp]` |
| Comprehension evaluation tick | Tick 3 (last tick, after full word presented) | `[Paper]` |
| Presentations per word per epoch | 1× repetition, 2× speaking, 3× comprehension | `[Paper]` |
| Training length | 200 epochs (≈ 2.05 million word presentations) | `[Paper]` |
