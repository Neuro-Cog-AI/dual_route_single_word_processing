# Diagnostic vs. Faithful: Implementation Classification

This document classifies every significant implementation component by its relationship to the Ueno et al. (2011) paper. The classification is intended to make clear which aspects of the current codebase are direct replications of the paper, which are engineering assumptions, which are adaptations for English/NWR data, and which are purely diagnostic additions that should be removed or isolated before making scientific claims about the Lichtheim 2 architecture.

---

## Classification labels

| Label | Meaning |
|---|---|
| **Faithful** | Directly specified in Ueno et al. (2011) main text or supplement |
| **Inferred** | Consistent with the paper but required an assumption not explicitly stated |
| **English/NWR adaptation** | A deliberate departure from the Japanese setup to support English/NWR data |
| **Diagnostic-only** | Added for debugging / training experiments; not in the paper; should not be presented as a faithful replication |
| **Pure engineering** | Implementation convenience with no scientific relevance |

---

## 1. Dense 39→N sound projection (`sound_proj_size`)

**Classification: Diagnostic-only**

`sound_proj` is an `nn.Linear(39, N, bias=False)` applied to the raw phoneme vector before both dorsal and ventral input paths. This is not present in Ueno et al. (2011), which feeds the phonological representation directly into `sound_to_iSMG` and `sound_to_mSTG`.

**Purpose:** The one-hot 39D representation has exactly one active unit per tick. The matrix multiplications `W_{sound→iSMG}` and `W_{sound→mSTG}` then only ever see one column of each weight matrix at a time, which can slow learning or create dead-unit pathologies. The projection learns a dense embedding that distributes information across multiple iSMG/mSTG input units simultaneously.

**Scientific risk:** The projection adds `39 × N` additional parameters shared between the dorsal and ventral pathways, which is not part of the original model. Results with the projection active cannot be directly compared to paper figures or used as evidence of model faithfulness.

**Config field:** `sound_proj_size` in `ModelConfig`; controlled by `--sound-proj-size N` CLI flag. Default `None` = paper pathway.

---

## 2. Dorsal-motor-only mode (`dorsal_motor_only`)

**Classification: Diagnostic-only**

When `dorsal_motor_only = True`, `triangularis_to_motor` is excluded from the motor net:

```
motor_net = W_{iSMG→motor} · iSMG(t) + b_{motor}   [only dorsal contribution]
```

The full ventral pathway (mSTG → aSTG → vATL, triangularis) is still computed every tick, but `triangularis_to_motor` receives no gradient because it does not reach the motor output.

**Purpose:** To isolate whether the dorsal pathway alone can learn phoneme repetition, as a sanity check before interpreting ventral pathway contributions.

**Scientific risk:** Disconnecting `triangularis_to_motor` removes a key convergence connection from Ueno et al.'s Figure 1. Experiments with this flag active do not replicate the full Lichtheim 2 model.

**Config field:** `dorsal_motor_only` in `ModelConfig`; controlled by `--dorsal-motor-only` CLI flag.

---

## 3. Output-positive weight (`output_positive_weight`)

**Classification: Diagnostic-only**

A scalar multiplier applied to the BCE loss for output-phase positive-target units (target ≥ 0.5) during repetition training. Default 1.0 (no effect).

**Purpose:** Compensates for the 38:1 imbalance between negative and positive target units in the one-hot encoding. Without it, the gradient is dominated by negative-target suppression, and the model can reduce loss by suppressing all motor outputs without learning to selectively produce phonemes.

**Scientific risk:** This is a training trick not present in the original paper. With Japanese 21-bit phonological features, the same imbalance may not occur (21 bits are not one-hot; multiple bits can be active per phoneme). In English with one-hot encoding, the imbalance is more severe and this correction may be necessary — but it changes the optimization landscape relative to the paper.

**Code:** Applied in `compute_trial_loss_breakdown` (`losses.py`); controlled by `--output-positive-weight` CLI flag.

---

## 4. 39D one-hot phoneme encoding

**Classification: English/NWR adaptation**

The paper uses 21-bit phonological feature vectors for Japanese: 15 consonant features + 5 vowel features + 1 pitch-accent bit per mora `[Supp]`. These are dense binary vectors (multiple bits active per phoneme), not one-hot.

The current English/NWR implementation uses 39D one-hot vectors: one unit per phoneme symbol in the inventory, exactly one unit active per tick.

**Key differences from Japanese 21-bit:**
- Much larger input/output dimension (39 vs 21)
- Sparse: only 1 active bit per tick (vs multiple active bits)
- No linguistic feature structure (no shared features between phoneme classes)
- Creates the 38:1 positive/negative imbalance in the loss (see §3)
- Does not allow the model to generalize across phonetically similar phonemes without learning the similarity from data

**Config field:** `sound_input_size = 39`, `motor_output_size = 39` in `english_nwr.yaml`.

**Note:** Phoneme feature encoding (binary features from `phonemes.csv`) is listed as an alternative in `docs/open_questions.md D14` but has not been implemented.

---

## 5. BCE loss vs. potential CrossEntropy formulation

**Classification:** Motor BCE is the appropriate PyTorch translation of cross-entropy for sigmoid outputs, consistent with the paper. Semantic BCE follows the same convention. `torch.nn.CrossEntropyLoss` (softmax-based) is not used.

**Reasoning:** The paper states "cross-entropy" as the loss function `[Paper]`. LENS (the original simulator) applied this to independent sigmoid outputs — one per unit — not to a softmax distribution over units. `F.binary_cross_entropy` in PyTorch is the correct equivalent: it computes the binary cross-entropy per element and treats each output unit independently.

**Caveat — provisional:** The exact LENS training details (whether gradient was applied per-unit, per-tick, or in some other form) have not been independently verified from LENS source files or a direct correspondence with the authors. The current BCE-on-sigmoid interpretation is the most natural reading of the paper and supplement, but it should be treated as `[Inferred]` until confirmed.

**Not used: softmax + CrossEntropyLoss.** `torch.nn.CrossEntropyLoss` in modern PyTorch applies softmax over logits and computes the negative log probability of the correct class. This forces a single-winner selection at every tick. The Lichtheim model does NOT use softmax — motor units are independent sigmoids. Using `nn.CrossEntropyLoss` here would be a substantive model change, not a faithful translation.

**Key implication:** BCE allows multiple motor units to be simultaneously active (which can happen early in training), while CrossEntropyLoss would force a single-winner selection at every tick. BCE is the correct loss for this architecture under the current interpretation.

---

## 6. No EOS in the current formulation

**Classification: Faithful / not applicable (in the current fixed-length setting)**

There is no end-of-sequence token, no EOS unit, and no variable-length output mechanism in the current implementation. The model produces exactly T output ticks, where T is determined by the length of the input phoneme sequence — it is fixed per trial, not learned.

This matches the original Lichtheim 2 design: trials have a fixed length determined by the word (or mora sequence), and the model is expected to produce output at each of T output ticks. There is no learned "stop" signal.

**What "faithful / not applicable" means here:** The absence of EOS is not a claim that EOS is wrong or unnecessary in all contexts. It simply means EOS is not part of the Lichtheim 2 formulation as described in Ueno et al. (2011). A model with EOS and variable-length output would be a different (possibly better-performing) model for English, but it would depart from the original architecture. That departure is described at Level 5 in [modernization_options.md](modernization_options.md).

**Relevance for future work:** If padded batching (Level 4) is introduced, EOS is still not needed — variable-length sequences are handled via padding and loss masking. EOS becomes relevant only if the goal is variable-length autoregressive output at test time (Level 5).

---

## 7. zero_error_radius (dead zone)

**Classification: Faithful**

Zero-error radius = 0.1 `[Paper]`. Units where `|prediction − target| < 0.1` contribute zero loss and zero gradient.

The implementation in `compute_trial_loss_breakdown` (`losses.py`) uses `.detach()` on the dead-zone computation to prevent gradients from flowing through the mask. The mask is a hard gate.

**Note:** The paper states `zero_error_radius = 0.1` but does not specify exactly whether it applies to both motor and semantic outputs. The current implementation applies it uniformly to both, which is a reasonable assumption.

---

## 8. Motor silence supervision during input phase

**Classification: Faithful** (paper statement) + **Inferred** (mechanism)

The paper states that the motor layer is "required to be silent" during the listening/input phase. This is implemented by:
- `motor_targets[0:T] = zeros(T, motor_size)`
- `motor_loss_mask = ones(2T, dtype=bool)` — all ticks supervised

The paper does not say explicitly that gradient flows back through the input-phase ticks with respect to the silence requirement — it only says the motor "is required to be silent." The current implementation does provide gradient for silence (BCE against zero target at input ticks), which is the natural reading. Whether LENS computed loss at input-phase ticks or simply clamped motor output during input phase is unconfirmed.

**Open question `[D6]`:** If LENS clamped motor output to zero during input phase rather than supervising it with gradient, the training dynamics would differ significantly.

---

## 9. Variable-length English words vs. fixed Japanese mora setup

**Classification: English/NWR adaptation**

The paper uses Japanese words with a fixed length of 3 morae `[Paper]`. Every trial has exactly T=3 input ticks + 3 output ticks = 6 total ticks. This means the model only encounters one trial length during training, and there are no variable-length issues.

In the English/NWR setup:
- Words vary in length (typically 2–11 phonemes for English monosyllabic and polysyllabic words)
- Each word produces a trial of `2T` ticks where T = word length
- Trials are processed one at a time in the current implementation (no batching)
- The model encounters different T values within a single epoch

**Consequences:**
- The tick counts in `configs/english_nwr.yaml` (`repetition_ticks: 6`, etc.) are legacy reference values from the Japanese setup; they are not used at runtime (see comment in `tasks.py`)
- BPTT depth varies per trial (6 ticks for a 3-phoneme word, 22 ticks for an 11-phoneme word)
- The model must generalize across different sequence lengths, which is a harder problem than the fixed-length Japanese setup

---

## Summary table

| Component | Classification | Notes |
|---|---|---|
| Dual dorsal-ventral architecture | Faithful | iSMG, mSTG, aSTG, vATL, triangularis, motor |
| Layer sizes (iSMG=50, etc.) | Faithful | From Table S1 |
| Weight init uniform [-1,1] feedforward | Faithful | Paper |
| Weight init uniform [-0.5,0.5] Elman | Faithful | Paper |
| Bias = -1.0 | Inferred | Paper says LENS bias links; PyTorch mapping is assumed |
| Elman self-recurrence on iSMG | Faithful | Paper |
| Motor copy-back into iSMG | Faithful | Supplement Fig S1 |
| vATL copy-back into aSTG | Faithful | Supplement Fig S1 |
| Full BPTT within trial | Inferred/Provisional | LENS may differ; see open_questions D6 |
| BCE loss (element-wise) | Faithful | Standard LENS cross-entropy for sigmoid outputs |
| zero_error_radius = 0.1 | Faithful | Paper |
| Motor silence during input phase | Faithful / Inferred | Paper states requirement; gradient approach assumed |
| 2T tick structure for repetition | Faithful / Provisional | Paper confirms structure; T-variable generalization is assumed |
| 39D one-hot English phonemes | English/NWR adaptation | Paper uses 21-bit Japanese mora features |
| Variable-length English words | English/NWR adaptation | Paper uses fixed 3-mora Japanese words |
| sound_proj_size | Diagnostic-only | Not in paper |
| dorsal_motor_only | Diagnostic-only | Not in paper |
| output_positive_weight | Diagnostic-only | Not in paper; compensates for one-hot imbalance |
| No EOS | Faithful / not applicable | Paper has fixed trial lengths; EOS not needed |
| loss_decomposition logging | Pure engineering | Diagnostic tooling; no scientific content |
| plot_repetition_metrics.py | Pure engineering | Visualization utility |
| analyze_repetition_predictions.py | Pure engineering | Post-hoc analysis tool |
