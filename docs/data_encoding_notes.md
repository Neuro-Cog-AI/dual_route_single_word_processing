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

The current training and data direction targets English phoneme sequences with variable-length words, in the spirit of the NWR literature. The Lichtheim 2 task structure (repetition, comprehension, speaking/naming) is preserved.

### Available Local Data (not committed to this repository)

Three CSV files exist locally at `data/raw/nwr_swp/`. They are **not committed** to this public repository (path is gitignored) and must not be committed until an explicit decision is made.

#### phonemes.csv — Phoneme inventory

Initial inspection: 40 phonemes (15 vowels, 25 consonants). Each row is one ARPAbet phoneme symbol with categorical phonetic features.

| Feature | Vowels (V) | Consonants (C) |
|---------|-----------|----------------|
| Height | High / Mid / Low | — |
| Backness | Front / Central / Back | — |
| Diphthong | True / False | — |
| Place | — | Labial / Coronal / Dorsal / Glottal |
| Manner | — | Stop / Fricative / Nasal / Approximant |
| Voiced | — | True / False |

Diphthongs (OY, AY, EY, OW, AW) appear as single phoneme entries. The `No_Stress` column stores the lookup key (e.g. `['AH']`) matching the stress-free forms used in wfe.csv and ssp.csv.

#### wfe.csv — Real-word items

Columns: Word, Condition, Lexicality, Size, Morphology, Frequency, Length, Zipf_Frequency, Phonemes, No_Stress, Part of Speech, Vowel Count, Consonant Count.

Initial inspection of a few rows suggests words are multisyllabic (7–9 phonemes in the sample). Lexicality, Size, and Morphology may not be uniform across the full file — **distribution checks are needed before assuming the file is homogeneous.**

The `No_Stress` column contains stress-stripped ARPAbet sequences (e.g. `['AH', 'T', 'EH', 'N', 'D']`) that map directly to phonemes.csv keys. `Zipf_Frequency` is likely more useful than raw `Frequency` for modelling frequency effects.

#### ssp.csv — Pseudoword / nonword items

Columns: Phonemes, No_Stress, Sonority, Type, Length.

`Type` encodes syllable structure (CCV, VCC, …); `Sonority` is an integer profile metric that may be useful for stratifying pseudoword difficulty. Initial inspection shows shorter sequences (length 3 in the sample) — full distribution should be checked.

---

### No_Stress vs Stress-Marked Phonemes

| Column | Example | Matches phonemes.csv? |
|--------|---------|----------------------|
| Phonemes | `['AH0', 'EH1', 'N']` | No — stress digits not in inventory keys |
| No_Stress | `['AH', 'EH', 'N']` | Yes |

**Use `No_Stress` for phoneme lookup.** Stress information can be incorporated as an additional feature later if needed.

**Coverage validation required before implementation (see D14):** confirm that every phoneme symbol in `wfe.csv::No_Stress` and `ssp.csv::No_Stress` is present as a key in `phonemes.csv`. Any out-of-vocabulary symbol must be resolved before encoding.

---

### English Phoneme Encoding Options

The encoding choice determines `sound_input_size` in the English/NWR config. This differs from the 21-bit Japanese representation — the English config will have a different `sound_input_size` depending on the option chosen.

**Option A — Phoneme one-hot (recommended first implementation)**

One dimension per phoneme in the inventory. If the inventory has N phonemes (initial inspection suggests ~40), then `sound_input_size = N`. Recommended because it is simple, auditable, and requires no feature engineering. The exact size is subject to coverage validation.

**Option B — Binary feature vectors from phonemes.csv**

Encode each phoneme as a binary vector from the categorical features (Type, Height/Place, Backness/Manner, Diphthong/Voiced). Features not applicable to a phoneme type (e.g. Place for vowels) are set to zero. This is linguistically richer but produces sparse, heterogeneous vectors and requires decisions about the exact feature decomposition.

**Option C — Other ARPAbet encodings**

One-hot-expanded categorical features or other schemes. TBD.

**Note on sound_input_size:** Do not assume a fixed value until coverage validation is complete and an encoding option is chosen. The English/NWR config will carry a different `sound_input_size` than the Japanese 21-bit faithful config. Both can coexist since the model is parameterised by `ModelConfig`.

See D14 in [docs/open_questions.md](open_questions.md) for the open design decision.

---

### Variable-Length Encoding (Phase 2c — Implemented)

Phase 2c extended `build_trial_inputs` to handle arbitrary-length sequences T:
- Repetition: T input ticks + T output ticks
- Comprehension: T input ticks
- Speaking/naming: T output ticks with semantic input clamped

Padding, masking, EOS, and batching remain open — see D12 in [docs/open_questions.md](open_questions.md).

---

## Synthetic Data for Phase 1 (Toy Config Only)

For Phase 1 (toy tick-dynamics validation) only, all phonological and semantic vectors will be **randomly generated**:

- Sample random binary vectors of the appropriate dimensionality
- Assign each word a unique phonological pattern and a unique semantic pattern
- No linguistic structure required — purely for testing forward-pass mechanics

**Phase 2 onward (architecture validation):** the 21-bit Japanese representation from the Supplement Table is used for testing the faithful Lichtheim 2 config. For English/NWR training (Phase 3+), a separate English/NWR config with the appropriate `sound_input_size` will be used — encoding option to be confirmed in Phase 3b.
