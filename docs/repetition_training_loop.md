# Repetition Training Loop

This document traces exactly how a word in a CSV file becomes a gradient update in the repetition-only training setup.

---

## 1. From CSV word to phon_tensor

A word such as "attend" is stored in `wfe.csv` with a `No_Stress` column containing a Python-style list string, e.g. `"['AH', 'T', 'EH', 'N', 'D']"`.

The loading pipeline (`data.py`, `encoding.py`):

1. `load_phoneme_inventory(phonemes.csv)` builds a `PhonemeInventory`: an ordered list of 39 symbols and a `symbol_to_index` dict. Row order in `phonemes.csv` is the canonical ordering.
2. `load_word_items(wfe.csv, inventory)` reads each row:
   - `parse_phoneme_sequence(row["No_Stress"])` converts the string to `['AH', 'T', 'EH', 'N', 'D']` using `ast.literal_eval`.
   - `encode_phoneme_sequence(phonemes, inventory)` builds a `(T, 39)` float32 tensor where row `t` is a one-hot vector with a single `1.0` at `inventory.symbol_to_index[phonemes[t]]` and zeros elsewhere.
3. The result is stored in `WordItem.phon_tensor`.

For "attend" with T=5 phonemes, `phon_tensor` has shape `(5, 39)`.

---

## 2. From phon_tensor to SupervisedTrial

`make_repetition_trial(phon_tensor, motor_size)` in `trials.py` constructs the full trial:

```
n_ticks = 2 * T                                   # 10 for a 5-phoneme word
motor_targets = cat([zeros(T, 39), phon_tensor])   # (10, 39)
motor_loss_mask = ones(10, dtype=bool)             # all ticks supervised
```

`SupervisedTrial` holds:
- `task = Task.REPETITION`
- `n_ticks = 2 * T`
- `phon_tensor` — the raw phoneme sequence, shape `(T, 39)`
- `motor_targets` — shape `(2T, 39)`: first T rows are all-zero (silence target), last T rows are the phoneme one-hots
- `motor_loss_mask` — shape `(2T,)`, all True

The trial has no semantic targets; `semantic_targets` and `semantic_loss_mask` are `None`.

---

## 3. Why repetition has 2T ticks

The paper structure for repetition is:
- **Input / listening phase** (T ticks): the model hears one phoneme per tick; it is not required to produce output yet.
- **Output / repetition phase** (T ticks): the model receives no auditory input (sound = zero); it must produce the phoneme sequence from memory.

This matches the Ueno et al. (2011) design where phonological input drives iSMG (dorsal pathway) during listening, and the model must then reproduce the sequence through the motor pathway without further acoustic input.

T is the number of phonemes in the word. For a 3-mora Japanese word this was always T=3 (giving 6 ticks total). For English words with variable length, T varies per item — hence 2T varies too.

---

## 4. What happens during input / listening ticks (ticks 0..T-1)

`tasks.py` `build_trial_inputs` for REPETITION:

```
tick 0: (morae[0], None)   — phoneme 0 presented; no semantic clamp
tick 1: (morae[1], None)
...
tick T-1: (morae[T-1], None)
```

Each tick calls `model.forward_tick(state, sound=morae[t], clamp_vATL_in=None)`.

- Sound is the raw one-hot phoneme vector, shape `(39,)`. It is **not** passed through a sigmoid — it is a clamped input. If `sound_proj` is enabled, it is linearly projected first.
- All 9 state fields are updated (see [forward_tick_equations.md](forward_tick_equations.md)).
- Motor output is produced but supervised against a **zero target** — the model is required to be silent.
- The Elman copy and motor copy-back update `iSMG_context` and `motor_context` for use at the next tick.

Motor silence supervision during input phase is explicit in the paper: the motor layer is required to be inactive while listening.

---

## 5. What happens during output / repetition ticks (ticks T..2T-1)

`build_trial_inputs` for the output phase:

```
tick T:   (zero_sound, None)
tick T+1: (zero_sound, None)
...
tick 2T-1: (zero_sound, None)
```

Each tick calls `model.forward_tick(state, sound=zeros(39), clamp_vATL_in=None)`.

- Sound is all-zeros — no acoustic input.
- The model must reproduce the phoneme sequence from the state accumulated during the input phase, via the iSMG Elman recurrence and motor copy-back.
- Motor output at tick `T+t` is supervised against `phon_tensor[t]` — the one-hot for the t-th phoneme.
- The copy-back connections (`iSMG_context`, `motor_context`, `vATL_context`) allow the stored representation to unfold over the output ticks.

---

## 6. When run_trial is called

`train_step` in `trainer.py`:

```python
tick_results = model.run_trial(trial_dev.task, trial_dev.phon_tensor, sem_zeros, cfg)
```

`run_trial` iterates over all `2T` ticks in a single Python for-loop, threading `state` from one `forward_tick` call to the next, and returns a `list[TickResult]` of length `2T`.

Each `TickResult` records:
- `tick_index` (int)
- `task` (Task.REPETITION)
- `sound_input` — the actual sound tensor fed in
- `vATL_input_used` — the actual vATL input used this tick
- `state` — the full `ModelState` after this tick (all 9 fields, all gradients attached)

---

## 7. When compute_trial_loss is called

Immediately after `run_trial`:

```python
loss = compute_trial_loss(tick_results, trial_dev, zero_error_radius=..., ...)
```

`compute_trial_loss` calls `compute_trial_loss_breakdown` which:

1. Stacks motor outputs from all `2T` tick results: `motor_outputs = stack([r.state.motor for r in tick_results])` — shape `(2T, 39)`.
2. Computes per-element BCE: `raw_motor = binary_cross_entropy(motor_outputs, motor_targets, reduction="none")` — shape `(2T, 39)`.
3. Builds the alive mask: `motor_loss_mask.unsqueeze(-1).expand_as(raw_motor)` — broadcasts the `(2T,)` tick mask to `(2T, 39)`.
4. Applies zero_error_radius (dead zone) if nonzero.
5. Applies `output_positive_weight` if != 1.0.
6. Returns `(raw_motor * alive.float()).sum()`.

---

## 8. When loss.backward is called

After loss computation, still in `train_step`:

```python
loss.backward()
optimizer.step()
```

`loss.backward()` is called **once** at the end of the full `2T` forward pass.

---

## 9. Why this is full BPTT within a trial, not backward after every tick

`run_trial` builds a single unrolled computational graph spanning all `2T` ticks. The PyTorch autograd graph connects each `forward_tick` output to the next via the `state` object — `state.iSMG_context`, `state.motor_context`, and `state.vATL_context` are PyTorch tensors that carry gradients through the tick boundary.

Calling `loss.backward()` once unrolls through this entire graph, propagating gradients from the output ticks backward through the output phase, then through the copy-back connections, and into the input phase. This is **backpropagation through time (BPTT)** applied to the entire `2T`-tick trial.

**This is not backward-after-each-tick (online BPTT)**:
- No `.detach()` is called on the state between ticks.
- No intermediate `.backward()` is called mid-trial.
- No `torch.no_grad()` wraps any sub-sequence of ticks during training.

The practical implication: gradients from the motor supervision on output ticks flow back through the zero-sound output ticks and through the copy-back connections to influence the weights that were active during the input ticks. The entire `2T`-tick trial constitutes one gradient update.

**Open question [D6]:** Whether this matches LENS's implementation of online learning remains unverified. LENS may have used truncated BPTT at copy-back boundaries (the Plaut unfolding convention). This is flagged as `[Provisional]` in `docs/open_questions.md`.
