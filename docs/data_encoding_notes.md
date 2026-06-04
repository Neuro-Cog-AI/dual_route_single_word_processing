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

## English / NWR-Style Data — Current Direction

The current training and data direction targets English phoneme sequences, NWR-style, with variable-length words. The Lichtheim 2 task structure (repetition, comprehension, speaking/naming) is preserved.

### Available Local Data (not committed to this repository)

Three CSV files are available locally and inform encoding design decisions. They are **not committed** to this public repository. Raw data belongs in `data/raw/` (gitignored).

| File | Contents |
|------|----------|
| `phonemes.csv` | Phoneme inventory with categorical phonetic features (ARPAbet-like entries) |
| `ssp.csv` | Pseudoword/nonword phoneme sequences with sonority profile, sequence type, and length metadata |
| `wfe.csv` | Word-level items with lexicality, morphology, frequency, length, and phoneme sequence metadata |

### Phoneme Feature Representation (`[Open]` — D14)

The features in `phonemes.csv` are categorical. The encoding choice determines `sound_input_size` in the English config. **Do not assume a feature dimension until this decision is made.** Candidate approaches:

| Option | Notes |
|--------|-------|
| Phoneme one-hot | One dimension per phoneme; size = phoneme inventory |
| Hand-designed binary features | Place, manner, voicing, etc.; linguistically motivated |
| One-hot-expanded categorical features from phonemes.csv | Preserves categorical structure; size depends on feature count |
| Other ARPAbet feature encoding | TBD |

See D14 in [docs/open_questions.md](open_questions.md).

### Variable-Length Encoding (Phase 2c)

Unlike the original 3-mora Japanese setup, English words vary in phoneme count. Phase 2c implements **unbatched variable-length trial support**:

- Repetition: T input ticks + T output ticks (2T total)
- Comprehension: T input ticks; semantic output evaluated at tick T
- Speaking/naming: T output ticks with semantic input clamped

Padding, masking, EOS, and batching are design decisions deferred to a later phase. See D12 in [docs/open_questions.md](open_questions.md).

---

## Synthetic Data for Phase 1 (Toy Config Only)

For Phase 1 (toy tick-dynamics validation) only, all phonological and semantic vectors will be **randomly generated**:

- Sample random binary vectors of the appropriate dimensionality
- Assign each word a unique phonological pattern and a unique semantic pattern
- No linguistic structure required — purely for testing forward-pass mechanics

**Phase 2 onward (architecture testing):** the faithful 21-bit phonological representation from the Supplement Table is used for architecture validation with the faithful config. For English/NWR training (Phase 3+), the phoneme feature encoding from the English config will be used instead — see the English/NWR section above.
