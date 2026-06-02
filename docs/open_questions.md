# Open Questions for Yair Lakretz

Questions to resolve before or during Phase 1–2 implementation. Ordered roughly by priority / blocking order.

---

## Scope and Goals

**Q1.** Should the first faithful replication use a Japanese-like phonological setup (21-bit mora-based vectors, fixed 3-unit sequences) or should we move directly to an English phonological representation?

*Why it matters:* This determines the vocabulary generator, the phonological encoding, and the evaluation metric from the start. Starting Japanese-like is safer for faithfulness; starting English requires additional encoding decisions.

**Q2.** What is the first success criterion for calling Phase 3 "done"?
- (a) Figure 2 only (learning curves for all three tasks)?
- (b) Figure 2 plus basic lesioning (one or two key aphasic profiles)?
- (c) Something else?

*Why it matters:* Determines scope of Phase 3 and whether lesioning infrastructure should be built in Phase 3 or deferred to Phase 4.

**Q3.** Which parts of the replication are essential for the lab's paper/project goals, and which can wait or be simplified?

---

## Faithfulness and Numerics

**Q4.** How numerically faithful does the first PyTorch version need to be?
- (a) Qualitatively faithful (correct curve shape, correct lesion profiles)?
- (b) Numerically close (similar epoch counts, similar accuracy levels)?
- (c) As close as possible to the original LENS outputs?

*Why it matters:* Affects whether we need to match the exact learning rate, weight init, and zero-error radius from the supplement.

**Q5.** Should we attempt to obtain the original LENS files from the Ueno/Lambon Ralph group?

*Why it matters:* Original LENS files would allow direct comparison of activations and help resolve `[Open]` items, but may not be publicly available.

---

## Architecture Specifics

**Q6.** What are the exact copy-back connections?
- iSMG Elman self-recurrence is clear from the paper.
- Does aSTG/STS also have a copy-back context unit?
- Is the vATL split into input and output parts explicitly described in the supplement, or is this an implementation inference?

**Q7.** At which tick(s) is the output evaluated for comprehension and for speaking?
- Comprehension: is the semantic output evaluated at every tick, only the last tick, or only during the input period?
- Speaking: is the motor output evaluated at every tick, or only at specific ticks?

**Q8.** What is the exact zero-error radius value from the supplement?

**Q9.** What is the loss function: sum-of-squared-error (SSE), cross-entropy, or something else?

**Q10.** What is the learning rate and learning rate schedule? Is it constant, decayed, or adaptive?

**Q11.** What is the weight initialisation distribution (uniform, Gaussian, small random)?

---

## Data Encoding

**Q12.** Should semantic vectors remain entirely artificial (random orthogonal 50-bit patterns) in the first version, or should they encode any real semantic structure?

**Q13.** For an eventual English adaptation: should the phonological representation preserve the fixed-length 3-unit (mora/syllable) sequences, or move to a variable-length representation?

**Q14.** What English vocabulary would be appropriate? A small controlled set (CVC words)? SWP word norms? Something matched to the original Japanese vocabulary structure?

---

## Questions to Answer Later (Phase 4+)

- Which aphasic profiles are most important to replicate first?
- Should recovery training use the same learning rate as initial training?
- For RSA (Phase 5): which neuroimaging datasets should representations be compared against?
