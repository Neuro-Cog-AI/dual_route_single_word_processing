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

## Implemented Pattern (Phase 1)

### `ModelState` and `TickResult` (`src/lichtheim2/layers.py`)

`ModelState` is a flat dataclass holding **all** layer activations and copy-back context fields as 1-D tensors. It is the **between-tick carry state** — it holds what the next tick needs as context, not a record of the inputs used during this tick.

```python
@dataclass
class ModelState:
    iSMG: Tensor;  iSMG_context: Tensor   # Elman carry
    motor: Tensor; motor_context: Tensor   # motor copy-back
    mSTG: Tensor;  aSTG: Tensor
    vATL_out: Tensor                       # computed this tick
    vATL_context: Tensor                   # = vATL_out; used as vATL input next tick
    triangularis: Tensor
```

`TickResult` is a per-tick record returned by `run_trial`, capturing everything that happened:

```python
@dataclass
class TickResult:
    tick_index: int
    task: Task
    sound_input: Tensor        # actual sound fed in (zeros if silent tick)
    vATL_input_used: Tensor    # actual vATL fed into aSTG (context or external clamp)
    state: ModelState          # all activations after this tick
```

### `Lichtheim2Model` (`src/lichtheim2/model.py`)

A `nn.Module` holding all weight matrices as `nn.Linear` layers. The two key methods:

```python
def forward_tick(
    self,
    state: ModelState,
    sound: Tensor | None = None,
    clamp_vATL_in: Tensor | None = None,
) -> tuple[ModelState, Tensor]:
    """One tick: dorsal + ventral forward pass + copy-back update.

    Returns (new_state, vATL_input_used).
    vATL_input_used is state.vATL_context or clamp_vATL_in (speaking task).
    Fully differentiable — no detach/no_grad inside.
    """
```

```python
def run_trial(
    self,
    task: Task,
    phon_pattern: Tensor,   # (n_morae, sound_size)
    sem_pattern: Tensor,    # (vATL_size,)
    cfg: ModelConfig,
) -> list[TickResult]:
    """Run all ticks; return one TickResult per tick for full inspection."""
```

### Why explicit ticks?

- Every internal activation is accessible at every tick — essential for copy-back, lesioning, and representational analyses.
- The tick loop is transparent and easy to debug.
- Copy-back is implemented as a simple assignment at the end of each tick, not as a hidden BPTT graph.

---

## Weight Matrix Convention

Each connection between layer A and layer B is a single `nn.Linear`. Sizes below are the faithful (Phase 2) values; Phase 1 uses toy sizes from `configs/toy.yaml` via `ModelConfig`:

```python
self.sound_to_iSMG      = nn.Linear(21, 50, bias=True)   # carries iSMG bias [Inferred]
self.iSMG_elman         = nn.Linear(50, 50, bias=False)   # Elman; no bias [Paper]
self.motor_copy_to_iSMG = nn.Linear(21, 50, bias=False)   # copy;  no bias [Paper]
self.iSMG_to_motor      = nn.Linear(50, 21, bias=True)    # carries motor bias [Inferred]
# ... etc.
```

This makes it trivial to zero out a specific pathway for lesioning:

```python
model.iSMG_to_motor.weight.data.zero_()  # lesion dorsal output
```

### Weight Initialisation (Phase 2b)

Implemented in `Lichtheim2Model._init_weights()`, called at the end of `__init__`:

| Connection group | Range | Source |
|-----------------|-------|--------|
| Standard feedforward weights | uniform [−1, 1] | `[Paper]` |
| Elman weights (`iSMG_elman`) | uniform [−0.5, 0.5] | `[Paper]` |
| Copy-back/context weights (`motor_copy_to_iSMG`, `vATL_in_to_aSTG`) | uniform [−0.5, 0.5] | `[Inferred]` — paper says "recurrent connections" but does not explicitly define whether copy-back connections qualify |
| Additive bias (`nn.Linear.bias`) | constant −1.0 | `[Inferred]` — PyTorch approximation of the LENS bias-link convention; the paper states LENS bias-link weights suppress early activation, but the mapping to `nn.Linear.bias` is an implementation assumption |
| Copy and Elman layers | no bias (`bias=False`) | `[Paper]` |

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
