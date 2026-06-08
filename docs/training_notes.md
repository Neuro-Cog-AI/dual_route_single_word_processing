# Training Notes

Expected training logic for the Lichtheim 2 PyTorch reimplementation.

---

## Training Paradigm: Online Item-by-Item Updates

The original model was trained with **online learning**: weights are updated after each word presentation, not after each epoch `[Supp]`.

This means:
- No mini-batching.
- No shuffled batch of gradients: each word's gradient is applied immediately.
- This is equivalent to `batch_size=1` SGD in PyTorch terms.

---

## Per-Word Presentation Schedule

Each word in the vocabulary is presented the following number of times per epoch `[Paper/Supp]`:

| Task | Presentations per word per epoch |
|------|----------------------------------|
| Repetition | 1 |
| Speaking / naming | 2 |
| Comprehension | 3 |

Total presentations per word per epoch: **6**.

The order of presentations within an epoch is `[Open]` — likely randomised across task types and words, but the exact shuffling strategy should be confirmed from the supplement.

---

## Tick Counts per Task

| Task | Ticks |
|------|-------|
| Repetition | 6 (3 input + 3 output) `[Paper]` |
| Comprehension | 3 `[Paper]` |
| Speaking / naming | 3 `[Paper]` |

---

## Loss Function

The error was estimated with the **cross-entropy method** `[Paper — Hinton, 1989]`.

PyTorch translation note `[Inferred]`: the LENS cross-entropy formulation operates unit-by-unit on sigmoid outputs. The standard PyTorch `nn.BCELoss` (binary cross-entropy) is the natural equivalent, but the exact equivalence with the LENS implementation (especially with the zero-error radius and frequency scaling applied before gradient computation) needs to be verified.

---

## Zero-Error Radius

No error is backpropagated from a unit if `|output − target| < 0.1` `[Paper]`.

This is applied before computing the gradient, not as a loss modification. In PyTorch this must be implemented explicitly: mask out the gradient contribution of any unit already within 0.1 of its target value.

---

## Learning Rate and Schedule

- **Epochs 1–150:** learning rate = 0.5 `[Paper]`
- **Epochs 151–180:** learning rate reduced by 0.1 every 10 epochs (0.5 → 0.4 → 0.3 → 0.2) `[Paper]`
- **Epochs 181–200:** learning rate fixed at 0.1 `[Paper]`
- **Momentum:** not used `[Paper]`
- **Weight decay:** same schedule as learning rate; initialised at 10⁻⁶, reduced by 10⁻⁷ every 10 epochs from epoch 150–180, then fixed at 6×10⁻⁷ until epoch 200 `[Paper]`
- PyTorch translation `[Inferred]`: implement as a custom LR scheduler; weight decay can be passed to the SGD optimizer or applied manually.

---

## Weight Initialisation

- Most weights: uniform in **[−1, 1]** `[Paper]`
- Recurrent connections (Elman): uniform in **[−0.5, 0.5]** `[Paper]`
- Bias weights to hidden units: initialised at **−1** to suppress strong hidden unit activation early in training `[Paper]`
- Copy and Elman context layers: no trainable bias `[Paper]`

---

## Evaluation Metric

- Per-task word-level accuracy: a word is "correct" if all output units are within the zero-error radius of their target values at the scored ticks `[Inferred from paper description of Figure 2]`
- Accuracy reported as proportion of words correct per epoch, separately for each task `[Paper Figure 2]`

---

## Training Loop — Phase 3c-1 Implementation

Online (item-by-item) training is implemented in `src/lichtheim2/trainer.py`:

```python
loss = train_step(model, trial, optimizer, cfg, zero_error_radius=0.0, device="cpu")
```

Loss computation (`src/lichtheim2/losses.py`):
- Binary cross-entropy on sigmoid outputs vs binary targets
- Motor loss: applied at all ticks indicated by `motor_loss_mask`
- Semantic loss: applied at all ticks indicated by `semantic_loss_mask` (comprehension)
- Zero-error radius: optional dead-zone threshold (default 0.0 = disabled; paper: 0.1)

Device handling: `init_state`, `forward_tick`, and `run_trial` all propagate the model device. `trainer.py` moves `SupervisedTrial` tensors to the target device before each step.

## Training Loop — Full Epoch (Phase 3c-2 — Pending)

```
for epoch in range(n_epochs):
    shuffle word order
    for word in vocabulary:
        present word 1× for repetition   → update weights
        present word 2× for speaking     → update weights each time
        present word 3× for comprehension → update weights each time
    log per-task accuracy on full vocabulary
```

---

## Open Issues for Phase 3

1. Exact LENS-to-PyTorch translation of the cross-entropy loss with zero-error radius and frequency scaling `[Inferred]`
2. Whether presentation order within epoch is fully random across all task × word combinations, or partially blocked `[Open]`
3. Whether backpropagation covers all 6 ticks of a repetition trial as a single unrolled graph, or is truncated `[Open]`
4. Weight initialisation: uniform in [−1, 1] for most connections, [−0.5, 0.5] for recurrent connections `[Paper]`; PyTorch additive bias = −1.0 is an implementation assumption approximating the LENS bias-link convention `[Inferred]`

---

## GPU Readiness

`Lichtheim2Model` is a standard `nn.Module`. Device support is implemented in Phase 3c-1:

- `init_state(cfg, device=device)` creates state tensors on the model device
- `forward_tick` and `run_trial` infer device from model parameters
- `run_trial` moves task-generated tensors (sound inputs, clamp_vATL) to the model device before each tick
- `move_trial_to_device(trial, device)` in `trainer.py` moves all `SupervisedTrial` tensors

To train on GPU: `model.to("cuda")` before calling `train_step(..., device="cuda")`.
