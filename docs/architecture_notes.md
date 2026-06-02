# Architecture Notes

Design rationale and implementation strategy for the Lichtheim 2 PyTorch reimplementation.

---

## Why Not Use `nn.RNN` or `nn.LSTM`

PyTorch's built-in recurrent modules (`nn.RNN`, `nn.LSTM`, `nn.GRU`) are inappropriate for this model for several reasons:

1. **Multi-layer heterogeneous dynamics.** The Lichtheim 2 model has seven distinct layers with different sizes, asymmetric connectivity, and layer-specific copy-back rules. `nn.RNN` assumes a uniform stack of identical recurrent layers.

2. **Manual tick-by-tick control.** Each tick requires:
   - Selectively clamping input units (depending on task and tick index)
   - Computing activations in a specific order along the pathway
   - Updating copy-back / context states at the end of each tick
   - Optionally zeroing out (lesioning) specific weight matrices mid-forward-pass
   None of this is naturally supported by `nn.RNN`.

3. **Elman recurrence is local to iSMG only.** Only the iSMG layer has Elman self-recurrence; other layers do not. `nn.RNN` applies recurrence uniformly.

4. **Copy-back is not the same as RNN hidden state.** The vATL split and other copy-back connections carry activation from one tick to the next in a way that is specific to the Lichtheim 2 architecture, not a generic RNN formulation.

5. **Lesioning hooks.** Phases 4+ require zeroing or scaling specific weight matrices at inference time. This is much easier with explicit layer objects than with `nn.RNN`'s packed parameter tensors.

---

## Proposed Implementation Pattern

### `LayerState`

A simple dataclass holding the current activation of a layer:

```python
@dataclass
class LayerState:
    activation: torch.Tensor   # shape: (layer_size,)
    context: torch.Tensor      # copy-back / Elman context, same shape
```

### `Lichtheim2Model`

A `nn.Module` holding all weight matrices as `nn.Parameter` objects and implementing:

```python
def forward_tick(self, state: ModelState, tick: int, task: Task) -> ModelState:
    """Advance all layers by one tick given current state, tick index, and task."""
    ...
```

```python
def run_trial(self, task: Task, inputs: TrialInputs) -> TrialOutputs:
    """Run a full trial (e.g. 6 ticks for repetition) and return outputs at scored ticks."""
    state = self.init_state()
    for t in range(task.n_ticks):
        state = self.forward_tick(state, t, task)
    return self.extract_outputs(state, task)
```

### Why explicit ticks?

- Every internal activation is accessible at every tick — essential for copy-back, lesioning, and representational analyses.
- The tick loop is transparent and easy to debug.
- Copy-back is implemented as a simple assignment at the end of each tick, not as a hidden BPTT graph.

---

## Weight Matrix Convention

Each connection between layer A and layer B is a single `nn.Linear` (no bias by default, unless the supplement specifies otherwise `[Open]`):

```python
self.sound_to_iSMG     = nn.Linear(21, 50, bias=True)   # [Inferred: bias present]
self.iSMG_ctx_to_iSMG  = nn.Linear(50, 50, bias=False)  # Elman context
self.iSMG_to_motor     = nn.Linear(50, 21, bias=True)
# ... etc.
```

This makes it trivial to zero out a specific pathway for lesioning:

```python
model.iSMG_to_motor.weight.data.zero_()  # lesion dorsal output
```

---

## Lesioning Hooks (Phase 4)

Each weight matrix will be registered so that it can be scaled or zeroed by a `LesionConfig` object passed at inference time. Implementation deferred to Phase 4, but the architecture must support it from Phase 2 onwards.

---

## Backpropagation Note

The model uses online updates (item-by-item). Backpropagation proceeds through the tick loop for a single trial, not across trials. BPTT is limited to the ticks within one trial `[Inferred]`. Whether truncated BPTT or full BPTT within the trial is used needs confirmation `[Open]`.

---

## What This Is Not

- Not a seq2seq encoder-decoder.
- Not a Transformer.
- Not a modern speech model.
- Not a generic language model.

The implementation should look like a neurocomputational simulation, not like a standard deep learning training script.
