# Replication Specification

Precise specification for the Lichtheim 2 PyTorch reimplementation.

**Source tags used throughout:**
- `[Paper §X]` — stated in Ueno et al. 2011 main text, section X
- `[Supp §X]` — stated in supplementary materials, section X
- `[Inferred]` — our best-effort inference for a PyTorch implementation; must be confirmed
- `[Open]` — unclear; to be confirmed with Yair Lakretz before implementing

---

## 1. Layer Inventory

| Layer | Role | Size | Source |
|-------|------|------|--------|
| Sound input (pSTG) | Phonological/motor input | 21 | [Paper] |
| Insular-motor output | Phonological/motor output | 21 | [Paper] |
| iSMG hidden | Dorsal pathway hidden layer; Elman self-recurrent | 50 | [Paper] |
| mSTG/STS hidden | Ventral pathway, first hidden layer | 200 | [Paper] |
| aSTG/STS hidden | Ventral pathway, second hidden layer | 650 | [Paper] |
| vATL semantic | Semantic representation; split into input & output for copy-back | 50 | [Paper] |
| Triangularis-opercularis hidden | Ventral output-side hidden layer | 200 | [Paper] |

**vATL split `[Supp Figure S1]`:** The vATL layer is explicitly divided into two components — `vATL [semantics output]` and `vATL [semantics input]` — in order to realise the copy-back feedback to aSTG. At every tick, the output pattern from `vATL_out` is copied and hard-clamped to `vATL_in` on the next tick, which then propagates to aSTG simultaneously with the incoming sound activation. The PyTorch implementation pattern (two `nn.Linear` layers vs. a single weight matrix with copy) is `[Inferred]`.

---

## 2. Dorsal Pathway

```
Sound input (21)
    ↓
iSMG (50)   ← Elman self-recurrence: previous-tick iSMG activation fed back in
    ↓
Insular-motor output (21)
```

- The dorsal pathway supports **repetition** (sound → motor) `[Paper]`.
- iSMG receives sound input and its own previous activation (copy-back / Elman context) `[Paper]`.
- Weights: sound→iSMG, iSMG_context→iSMG (Elman recurrent), iSMG→motor `[Inferred from LENS pattern]`.

---

## 3. Ventral Pathway

**Input (auditory) side:**
```
Sound input (21)
    ↓
mSTG/STS (200)
    ↓
aSTG/STS (650)
    ↓
vATL_in (50)   ← semantic input / comprehension
```

**Output (production) side:**
```
vATL_out (50)   ← semantic output / speaking; copy-back of vATL from previous tick
    ↓
Triangularis-opercularis (200)
    ↓
Insular-motor output (21)
```

- The ventral pathway supports **comprehension** (sound → semantics) and **speaking/naming** (semantics → motor) `[Paper]`.
- Copy-back: previous-tick activations of aSTG/STS and vATL are fed back to earlier layers `[Inferred]`. Exact copy-back connections need verification from supplement `[Open]`.

---

## 4. Copy-Back / Context Mechanisms

The original model uses LENS-style context units (Elman recurrence generalised across layers):

- **iSMG Elman recurrence:** iSMG_context ← iSMG activation at end of previous tick. Fed as additional input to iSMG at the next tick `[Paper]`.
- **vATL copy-back:** `vATL_in` at tick `t` receives the hard-clamped output of `vATL_out` from tick `t−1`. This propagates to aSTG simultaneously with the incoming sound activation `[Supp Figure S1]`.
- **Insular-motor copy-back:** the insular-motor output activation at tick `t` is duplicated to an invisible copy layer and fed back to iSMG at tick `t+1` `[Supp Figure S1]`.
- **aSTG/STS copy-back:** `[Open]` — Figure S1 shows all bidirectional arrows are realised via copy layers; confirm whether aSTG has a dedicated copy layer feeding back to mSTG, or whether the arrows in Figure S1 are all covered by the vATL and insular-motor copy mechanisms described above.

---

## 5. Task Definitions

### 5.1 Repetition
- **Duration:** 6 ticks total `[Paper]`
- **Ticks 1–3:** Sound input is clamped to the target phonological pattern; motor output is not evaluated
- **Ticks 4–6:** Sound input is clamped to zero; motor output is compared to target
- **Pathway used:** Primarily dorsal (sound → iSMG → motor) `[Paper]`
- **Copy-back note:** iSMG context carries information across the input/output boundary

### 5.2 Comprehension
- **Duration:** 3 ticks `[Paper]`
- **Ticks 1–3:** Sound input is clamped to the target phonological pattern
- **Output evaluation:** vATL (semantic) output is compared to the target semantic pattern at each (or the final) tick `[Open — exact evaluation tick(s)]`
- **Pathway used:** Primarily ventral (sound → mSTG → aSTG → vATL) `[Paper]`

### 5.3 Speaking / Naming
- **Duration:** 3 ticks `[Paper]`
- **Ticks 1–3:** vATL semantic input is clamped to the target semantic pattern; motor output is evaluated sequentially
- **Pathway used:** Primarily ventral output side (vATL → triangularis → motor) `[Paper]`
- **Note:** Whether a start-token or empty-state initialisation is used is `[Open]`

---

## 6. Activation Function

- All hidden and output units use **sigmoid** activation: σ(x) = 1 / (1 + exp(−x)) `[Paper]`
- All activations are in **[0, 1]** `[Paper]`
- Input units are clamped directly (no activation function applied to clamped inputs) `[Inferred]`

---

## 7. Output Scoring / Error Signal

- Error is computed as the discrepancy between the output unit activations and the target (0/1) values `[Inferred]`
- A **zero-error radius** (also called error criterion or dead zone) is used: units within a small neighbourhood of the target contribute zero error `[Supp — exact value Open]`
- The exact loss function (cross-entropy vs. SSE) needs verification from the supplement `[Open]`

---

## 8. Training Schedule

- **Online learning:** weights are updated after each item (word), not after each epoch `[Supp]`
- **Presentations per word per epoch:**
  - 1× repetition `[Paper/Supp]`
  - 2× speaking/naming `[Paper/Supp]`
  - 3× comprehension `[Paper/Supp]`
- **Vocabulary size:** Original paper uses a Japanese mora-based vocabulary of fixed-length 3-mora words `[Paper]`
- **Learning rate schedule:** `[Open]` — need to extract the exact schedule from the supplement
- **Weight initialisation:** `[Open]` — small random weights; exact distribution unspecified

---

## 9. Lesioning (Phase 4)

Lesioning is not implemented in Phase 1–3. Placeholder for the specification:

- **Pathway lesioning:** zero out (or add noise to) all weights on a specified pathway `[Paper]`
- **Recovery training:** retrain on the same schedule after lesion; track per-task accuracy recovery `[Paper]`
- Exact lesion magnitudes and recovery epochs to be extracted from the paper

---

## 10. Open Items Summary

See [docs/open_questions.md](open_questions.md) for the full list. Items resolved from the paper/supplement:

- Zero-error radius: 0.1 `[Paper]`
- Loss function: cross-entropy `[Paper]`
- Learning rate schedule: 0.5 → decay → 0.1 over epochs 1–200 `[Paper]`
- Weight initialisation: uniform [−1, 1]; recurrent [−0.5, 0.5]; bias to hidden = −1 `[Paper]`
- vATL split and copy-back: explicit in Figure S1 `[Supp]`
- 21-bit phonological representation: fully specified in Supplemental Table `[Supp]`

Critical open items for Phase 2:

1. Exact aSTG/STS copy-back connections — does aSTG feed back to mSTG? `[Open]`
2. Evaluation tick(s) for comprehension: confirmed as last tick (tick 3) `[Paper]` — confirm for speaking `[Open]`
3. LENS-to-PyTorch translation of cross-entropy + zero-error radius + frequency scaling `[Inferred]`
4. Whether backpropagation is full BPTT through all ticks or truncated within a trial `[Open]`
