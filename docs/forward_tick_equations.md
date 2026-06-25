# forward_tick Equations

This document describes the exact computation performed by `Lichtheim2Model.forward_tick()` in `src/lichtheim2/model.py`, expressed as equations and accompanied by notes on shapes, learned vs. activation tensors, and the effect of diagnostic flags.

---

## Notation

- `σ(x)` — sigmoid function `1 / (1 + exp(-x))`
- `W_{A→B}` — the learned weight matrix for the connection from layer A to layer B
- `b_B` — the learned bias for layer B (where applicable)
- `x(t)` — the value of tensor x after tick t; `x(t-1)` = the context / copy value from the previous tick
- Shapes in the English/NWR config: `sound_size = 39`, `iSMG = 50`, `motor = 39`, `mSTG = 200`, `aSTG = 650`, `vATL = 50`, `tri = 200`

---

## Step 0: Resolve sound input

The raw phoneme vector (shape `(sound_size,)`) is provided by the caller. If `None` is passed (safety fallback), it is replaced by `zeros(sound_size)`.

```
sound_raw ∈ ℝ^{sound_size}
```

This is a **clamped input** — it is not passed through a sigmoid. It is the raw one-hot (or projected) representation of the phoneme at this tick.

---

## Step 0b: Optional dense sound projection (diagnostic)

If `sound_proj_size` is set in the config (e.g., `sound_proj_size = 20`), a learned linear projection is applied before both pathways:

```
sound_in = W_{proj} · sound_raw        # W_{proj} ∈ ℝ^{sound_proj_size × sound_size}, no bias
```

`sound_in ∈ ℝ^{sound_proj_size}` — a dense vector shared by both dorsal and ventral input connections.

If `sound_proj_size` is `None` (default, paper pathway):

```
sound_in = sound_raw ∈ ℝ^{sound_size}
```

**Learned parameters:** `W_{proj}` (shape `(sound_proj_size, sound_size)`). Initialized uniform `[-1, 1]`. `bias=False` — no bias on the projection.

**Why no bias on the projection:** The downstream layers `sound_to_iSMG` and `sound_to_mSTG` carry their own biases. Adding a bias on the projection would be redundant (it would be absorbed into the downstream bias). With `bias=False`, each row of `W_{proj}` is the learned embedding for one phoneme symbol.

**Zero-sound invariant:** Because `bias=False`, a zero sound vector maps to a zero projection output:
```
W_{proj} · zeros(sound_size) = zeros(sound_proj_size)
```
This is the correct behavior: during the output phase (ticks T..2T-1) the model receives no auditory input (sound = zeros), and the projection correctly passes zeros to both `sound_to_iSMG` and `sound_to_mSTG`. A projection with a nonzero bias would inject a learned constant signal into both pathways during the silent output ticks, which is not the intended behavior and would break the input/output phase separation.

---

## Step 1: iSMG (inferior supramarginal gyrus — dorsal pathway)

iSMG is the dorsal hidden layer. It receives three inputs:

```
iSMG_net(t) = W_{sound→iSMG} · sound_in + b_{iSMG}
            + W_{iSMG_elman} · iSMG(t-1)
            + W_{motor→iSMG} · motor(t-1)

iSMG(t) = σ(iSMG_net(t))
```

**Learned parameters:**
- `W_{sound→iSMG}` — shape `(iSMG, s_eff)`, bias `b_{iSMG}` (the only bias in the sum). `s_eff = sound_proj_size` if projection active, else `sound_size`. Init: uniform `[-1, 1]`, bias `= -1.0`.
- `W_{iSMG_elman}` — shape `(iSMG, iSMG)`, no bias. Elman self-recurrence `[Paper]`. Init: uniform `[-0.5, 0.5]`.
- `W_{motor→iSMG}` — shape `(iSMG, motor)`, no bias. Motor copy-back `[Inferred recurrent]`. Init: uniform `[-0.5, 0.5]`.

**Activation tensors:** `iSMG(t-1)` = `state.iSMG_context` from previous tick. `motor(t-1)` = `state.motor_context` from previous tick.

---

## Step 2: mSTG (middle STG — ventral pathway, first layer)

```
mSTG(t) = σ(W_{sound→mSTG} · sound_in + b_{mSTG})
```

**Learned parameters:** `W_{sound→mSTG}` — shape `(mSTG, s_eff)`, bias `b_{mSTG}`. Init: uniform `[-1, 1]`, bias `= -1.0`.

mSTG has no recurrent input — it is a purely feedforward transformation of the current sound input.

---

## Step 3: aSTG (anterior STG — ventral pathway, second layer)

aSTG receives mSTG output and the vATL copy-back (or semantic clamp during speaking):

```
vATL_in(t) = clamp_vATL     if external clamp is provided (speaking task)
           = vATL_out(t-1)  otherwise (state.vATL_context from previous tick)

aSTG_net(t) = W_{mSTG→aSTG} · mSTG(t) + b_{aSTG}
            + W_{vATL→aSTG} · vATL_in(t)

aSTG(t) = σ(aSTG_net(t))
```

**Learned parameters:**
- `W_{mSTG→aSTG}` — shape `(aSTG, mSTG)`, bias `b_{aSTG}`. Init: uniform `[-1, 1]`, bias `= -1.0`.
- `W_{vATL→aSTG}` — shape `(aSTG, vATL)`, no bias. Copy-back / semantic input `[Inferred recurrent]`. Init: uniform `[-0.5, 0.5]`.

**Activation tensors:** `vATL_in(t)` is `state.vATL_context` for repetition/comprehension (the vATL output from the previous tick). For speaking, it is the hard-clamped external semantic pattern.

---

## Step 4: vATL (ventrolateral anterior temporal lobe — semantic output)

```
vATL_out(t) = σ(W_{aSTG→vATL} · aSTG(t) + b_{vATL})
```

**Learned parameters:** `W_{aSTG→vATL}` — shape `(vATL, aSTG)`, bias `b_{vATL}`. Init: uniform `[-1, 1]`, bias `= -1.0`.

---

## Step 5: triangularis-opercularis (ventral pathway, output side)

```
triangularis(t) = σ(W_{aSTG→tri} · aSTG(t) + b_{tri})
```

**Learned parameters:** `W_{aSTG→tri}` — shape `(tri, aSTG)`, bias `b_{tri}`. Init: uniform `[-1, 1]`, bias `= -1.0`.

---

## Step 6: motor (insular-motor cortex — convergence of both pathways)

Both dorsal and ventral pathways converge on motor output.

### Standard mode (paper-faithful):

```
motor_net(t) = W_{iSMG→motor} · iSMG(t) + b_{motor}
             + W_{tri→motor} · triangularis(t)

motor(t) = σ(motor_net(t))
```

**Learned parameters:**
- `W_{iSMG→motor}` — shape `(motor, iSMG)`, bias `b_{motor}`. Init: uniform `[-1, 1]`, bias `= -1.0`.
- `W_{tri→motor}` — shape `(motor, tri)`, no bias. The motor bias is carried only by `iSMG_to_motor` to avoid double-counting `[Inferred]`. Init: uniform `[-1, 1]`.

### Diagnostic mode: `dorsal_motor_only = True`

```
motor_net(t) = W_{iSMG→motor} · iSMG(t) + b_{motor}

motor(t) = σ(motor_net(t))
```

`W_{tri→motor}` is not included in the motor sum. The full ventral pathway (mSTG → aSTG → vATL, triangularis) is still computed every tick, but `triangularis_to_motor` receives no gradient from the motor loss because it does not contribute to `motor_net`. This is a diagnostic flag introduced in Phase 3j; it is not in Ueno et al. (2011).

---

## Step 7: Context / copy-back update

At the end of each tick, three context fields are set to carry state to the next tick:

```
iSMG_context(t) = iSMG(t)          # Elman self-recurrence on iSMG
motor_context(t) = motor(t)         # motor copy-back into iSMG
vATL_context(t) = vATL_out(t)       # vATL copy-back into aSTG
```

These are stored in the returned `ModelState.iSMG_context`, `motor_context`, `vATL_context` and become the inputs `iSMG(t-1)`, `motor(t-1)`, `vATL_in(t)` at the next tick.

**Note on speaking:** `vATL_context` is always set to `vATL_out(t)`, even during speaking where `vATL_in(t)` was the external clamp. In speaking, every tick is clamped externally anyway, so `vATL_context` is populated but never read (the clamp always overrides it).

---

## Summary: learned parameters vs. activation tensors

| Symbol | Type | Shape (English/NWR) | Notes |
|---|---|---|---|
| `W_{proj}`, `b=None` | Learned | `(20, 39)` if proj active | Optional; not in paper |
| `W_{sound→iSMG}`, `b_{iSMG}` | Learned | `(50, 39)` or `(50, 20)` | Feedforward; bias = -1.0 |
| `W_{iSMG_elman}` | Learned | `(50, 50)` | Elman recurrent; no bias |
| `W_{motor→iSMG}` | Learned | `(50, 39)` | Copy-back; no bias |
| `W_{iSMG→motor}`, `b_{motor}` | Learned | `(39, 50)` | Carries motor bias; bias = -1.0 |
| `W_{sound→mSTG}`, `b_{mSTG}` | Learned | `(200, 39)` or `(200, 20)` | Feedforward; bias = -1.0 |
| `W_{mSTG→aSTG}`, `b_{aSTG}` | Learned | `(650, 200)` | Feedforward; bias = -1.0 |
| `W_{vATL→aSTG}` | Learned | `(650, 50)` | Copy-back/clamp; no bias |
| `W_{aSTG→vATL}`, `b_{vATL}` | Learned | `(50, 650)` | Feedforward; bias = -1.0 |
| `W_{aSTG→tri}`, `b_{tri}` | Learned | `(200, 650)` | Feedforward; bias = -1.0 |
| `W_{tri→motor}` | Learned | `(39, 200)` | No bias (motor bias in iSMG→motor) |
| `sound_in` | Activation (input) | `(39,)` or `(20,)` | Clamped; not learned |
| `iSMG(t)`, `iSMG_context(t)` | Activation | `(50,)` | Output of sigmoid |
| `motor(t)`, `motor_context(t)` | Activation | `(39,)` | Output of sigmoid |
| `mSTG(t)` | Activation | `(200,)` | Output of sigmoid |
| `aSTG(t)` | Activation | `(650,)` | Output of sigmoid |
| `vATL_out(t)`, `vATL_context(t)` | Activation | `(50,)` | Output of sigmoid |
| `triangularis(t)` | Activation | `(200,)` | Output of sigmoid |

---

## Full computation order within one tick

```
sound_raw → [optional projection] → sound_in
sound_in → iSMG_net(t) → iSMG(t)          [uses iSMG_context(t-1), motor_context(t-1)]
sound_in → mSTG(t)
mSTG(t)  → aSTG_net(t) → aSTG(t)          [uses vATL_context(t-1) or clamp]
aSTG(t)  → vATL_out(t)
aSTG(t)  → triangularis(t)
iSMG(t) [+ triangularis(t)] → motor(t)
→ copy-back: iSMG_context ← iSMG(t), motor_context ← motor(t), vATL_context ← vATL_out(t)
```

All activations are in `[0, 1]` (sigmoid outputs). All learned parameters are `nn.Linear` weight matrices and biases.
