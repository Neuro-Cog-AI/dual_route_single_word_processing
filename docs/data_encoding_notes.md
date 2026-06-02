# Data Encoding Notes

Phonological and semantic representations in the Lichtheim 2 model.

---

## Phonological / Motor Representation

- **Dimensionality:** 21 bits `[Paper]`
- **Type:** Binary distinctive-feature vectors `[Supp]`
- **Basis:** Japanese mora-based phonology — each mora (CV syllable unit) is one symbol `[Paper]`
- **Sequence length:** Fixed 3 morae per word `[Paper]`
- **Representation:** A word's phonological form is 3 successive 21-bit patterns, presented over 3 input ticks (one mora per tick) `[Paper]`

**Breakdown of the 21 bits `[Supp]`:**
- **15 bits** — distinctive phonetic features of the consonant (consonantal, approximant, sonorant, continuant, strident, nasal, voiced, aspirated, glottalized, coronal-anterior, coronal-distributed, labial, dorsal-high, dorsal-low, dorsal-back)
- **5 bits** — distinctive features of the vowel (voiced, labial-round, dorsal-high, dorsal-low, dorsal-back)
- **1 bit** — pitch accent (on/off for each mora position, varies by accent type: flat, type-1, type-2)

The exact binary vectors for all Japanese morae are provided in Table 1 of the Supplemental Experimental Procedures `[Supp]`. These must be reproduced exactly for the faithful replication. Random binary vectors are acceptable only for the toy Phase 1 config and must never be used for the faithful architecture.

---

## Semantic Representation

- **Dimensionality:** 50 bits `[Paper]`
- **Type:** Artificial binary feature vectors; values represent abstract semantic properties `[Paper]`
- **Structure:** Generated from 50 prototype patterns `[Supp]`:
  1. Each prototype has 20 of 50 units set to 1 (randomly)
  2. 40 exemplars per prototype: randomly turn off 10 of the 20 "on" units
  3. Resulting 2000 exemplars checked so every pair differs in at least 4 bits
  4. 1710 exemplars randomly assigned to the 1710 auditory-motor patterns (arbitrary mapping)
- **Interpretation:** No real-world semantic content; the patterns are arbitrary and consistent across training
- **Purpose:** Allows the model to learn form-meaning associations without requiring a real lexical-semantic knowledge base

**Note:** The original model uses artificial semantics deliberately, to test whether the architecture and training dynamics alone can account for the observed aphasic profiles. Real semantic representations are explicitly out of scope for the faithful replication `[Paper rationale]`.

---

## Japanese Setup (Original)

The original paper uses Japanese because:
- Japanese mora structure provides clean, fixed-length phonological units
- A controlled vocabulary of CVC-like patterns can be constructed systematically
- The 3-mora fixed-length constraint maps cleanly onto the 3-tick input window

The vocabulary construction procedure (how many words, how phonological overlap is managed) is `[Open]` — to be extracted from the supplement.

---

## English Adaptation (Phase 6 — Future Work)

**Do not implement until Phase 3 is confirmed faithful.**

Candidate options for English phonological encoding:

| Option | Pros | Cons |
|--------|------|------|
| CVC words with articulatory features | Clean, controlled; close to original structure | Artificial; limited vocabulary |
| IPA phoneme features (e.g., Chomsky-Halle) | Linguistically grounded | Variable-length words; requires alignment |
| Fixed-length 3-segment sequences | Preserves tick structure | Restricts vocabulary to monosyllables |
| CMU Pronouncing Dictionary | Real English words | Variable length; needs padding strategy |

All options require choosing:
- A feature set for each phoneme/segment
- A strategy for handling variable-length words (padding, truncation, or length normalisation)
- A vocabulary size and sampling strategy

These decisions are deferred to Phase 6 and should be made with Yair. See [docs/open_questions.md](open_questions.md) Q13–Q14.

---

## Synthetic Data for Phase 1 (Toy Config Only)

For Phase 1 (toy tick-dynamics validation) only, all phonological and semantic vectors will be **randomly generated**:

- Sample random binary vectors of the appropriate dimensionality
- Assign each word a unique phonological pattern and a unique semantic pattern
- No linguistic structure required — purely for testing forward-pass mechanics

**Phase 2 onward:** the faithful 21-bit phonological representation from the Supplement Table must be used. Random vectors are not acceptable for the faithful replication.
