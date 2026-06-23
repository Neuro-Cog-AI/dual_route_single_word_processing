# dual_route_single_word_processing

PyTorch replication and extension of the **Lichtheim 2** neurocomputational model of dual dorsal-ventral language pathways for single-word processing, described in:

> Ueno, T., Saito, S., Rogers, T. T., & Lambon Ralph, M. A. (2011).
> *Lichtheim 2: Synthesizing Aphasia and the Neural Basis of Language in a Neurocomputational Model of the Dual Dorsal-Ventral Language Pathways.*
> Neuron, 72(2), 385–396.

The repository is currently focused on **faithful training diagnostics** — validating learning stability and schedule choices on small controlled subsets — before moving to larger-scale training, evaluation, lesioning, and aphasia-profile analyses.

This is **not** a modern seq2seq model or Transformer. The architecture is a hand-rolled Elman-style recurrent network with explicit tick-by-tick dynamics and copy-back feedback, closely following the original LENS implementation.

---

## Current Implementation Status

| Component | Status |
|---|---|
| Architecture skeleton (all 9 state fields, all pathways) | Complete |
| Paper-aligned weight initialisation / connectivity audit | Complete |
| Variable-length tick-by-tick forward pass | Complete |
| English phoneme encoder (one-hot, No_Stress inventory) | Complete |
| Real-word and pseudoword loaders (`wfe.csv`, `ssp.csv`) | Complete |
| Supervised REP / COMP / SPK trial generation | Complete |
| Minimal online training loop (`train_step`) | Complete |
| Small-subset stability diagnostics | Complete |
| Loss-reduction comparison (`sum` vs `mean_active`) | Complete |
| Task-schedule comparison (uniform vs paper) | Complete |
| Frequency-weighting diagnostics | Complete |
| LR-schedule diagnostics (constant vs paper-proportional) | Complete |
| Repetition-only diagnostic training (200-word subset) | Complete under diagnostic conditions (dense projection + output-positive weighting; not a faithful Ueno replication) |
| Repetition-only diagnostic scaling (1200-word set) | Attempted; not solved (91/1200 exact match at 500 epochs under same diagnostic setting) |
| Full-scale faithful training (paper multi-task schedule / paper settings) | Pending |
| Lesioning and recovery | Pending |
| Representational similarity analyses | Pending |

See [docs/roadmap.md](docs/roadmap.md) for the full phased plan, [docs/training_notes.md](docs/training_notes.md) for implementation details, and [docs/repetition_only_training_note.md](docs/repetition_only_training_note.md) for the repetition-only diagnostic experiment record.

---

## Private Data

Raw CSV files are **not committed** (they are in `.gitignore`). To run CSV-dependent scripts and tests, place the files locally at:

```
data/raw/nwr_swp/
    phonemes.csv
    wfe.csv
    ssp.csv
```

All CSV-dependent tests skip gracefully with a clear message if these files are absent. Always-run tests use synthetic tensors only.

---

## Tests

```bash
# From dual_route_single_word_processing/
PYTHONPATH=src python -m pytest tests/ -v
```

Always-run tests pass without private CSV files. Targeted test files are available for each diagnostic module (e.g. `tests/test_lr_schedule_diagnostics.py`).

---

## Diagnostic Scripts

All scripts are run from `dual_route_single_word_processing/` with `PYTHONPATH=src`. Each accepts `--help` for full option details. Default settings use small subsets (10 words, 10 pseudowords) and are safe to run without GPU.

| Script | Purpose |
|---|---|
| `scripts/diagnose_small_subset_training.py` | Baseline: run diagnostic training on a small subset; report per-epoch per-task losses |
| `scripts/compare_loss_reductions.py` | Compare `sum` vs `mean_active` loss normalisation under identical conditions |
| `scripts/compare_task_schedules.py` | Compare uniform (1×REP+1×COMP+1×SPK) vs paper (1×REP+3×COMP+2×SPK) task schedules |
| `scripts/compare_frequency_weighting.py` | Compare unweighted vs frequency-weighted training (`--frequency-source zipf\|frequency`) |
| `scripts/compare_lr_schedules.py` | Compare constant vs paper-proportional LR schedule (`--task-schedule paper\|uniform`) |

**Repetition-only training and analysis utilities** (require private CSV data):

| Script | Purpose |
|---|---|
| `scripts/train_repetition_only.py` | End-to-end repetition-only training on real English words; supports `--sound-proj-size`, `--output-positive-weight`, and `--dorsal-motor-only` diagnostic flags; saves metrics, predictions, and loss-curve plots per run |
| `scripts/analyze_repetition_predictions.py` | Analysis utility: reads `predictions_after.json` from a completed run directory and produces a per-word failure-mode breakdown as CSV + Markdown report |
| `scripts/plot_repetition_metrics.py` | Analysis utility: regenerates `loss_curve.png`, `loss_decomposition_curve.png`, and `loss_curve_range_clipped.png` for any completed run without retraining; accepts `--rolling-window` and `--clip-percentile` |

Example (requires private CSVs):

```bash
PYTHONPATH=src python scripts/compare_lr_schedules.py \
    --data-dir data/raw/nwr_swp --config configs/english_nwr.yaml \
    --mode mixed-multitask --task-schedule paper \
    --epochs 20 --lr 0.01 --seed 0
```

---

## Important Constraints

- **No raw CSV commits.** `wfe.csv`, `ssp.csv`, `phonemes.csv` are local only.
- **No copyrighted PDF commits.** Ueno et al. 2011 and its supplement must not be committed.
- **No model checkpoint commits.** `*.pt`, `*.pth`, `*.ckpt` are in `.gitignore`.
- **Diagnostics are small-subset only by default.** Increase `--max-words` and `--max-pseudowords` deliberately when running larger checks.
- **No full training, checkpoints, plots, or output files** by default — all diagnostic scripts exit after printing a table to stdout.

---

## Development Setup

```bash
conda create -n lichtheim2 python=3.11
conda activate lichtheim2
pip install torch pytest pyyaml ruff black
```

---

## Architecture Overview

The model implements three tasks on a shared dual-pathway recurrent network:

| Task | Ticks | Input → Output |
|---|---|---|
| REPETITION | 2T | Sound → Motor |
| COMPREHENSION | T | Sound → Semantic (vATL) |
| SPEAKING | T | Semantic (clamped) → Motor |

Decisions are tagged throughout the documentation:

- `[Paper §X]` — stated in Ueno et al. 2011
- `[Supp §X]` — stated in the supplementary materials
- `[Inferred]` — best-effort inference for PyTorch
- `[Open]` — unresolved (see [docs/open_questions.md](docs/open_questions.md))

---

## Repository Structure

```
dual_route_single_word_processing/
├── configs/                   # YAML model configurations
│   ├── toy.yaml               # Minimal synthetic config for fast testing
│   ├── lichtheim2.yaml        # Faithful paper layer sizes
│   └── english_nwr.yaml       # English / NWR config (sound_input=39)
├── data/raw/nwr_swp/          # Private CSV data (not committed)
├── docs/                      # Scientific documentation
│   ├── roadmap.md
│   ├── training_notes.md
│   ├── replication_spec.md
│   ├── open_questions.md
│   ├── architecture_notes.md
│   ├── data_encoding_notes.md
│   └── repetition_only_training_note.md   # Repetition-only diagnostic experiment record (§1–§20)
├── scripts/                   # Diagnostic and training scripts
├── src/lichtheim2/            # Python package
│   ├── config.py              # ModelConfig, load_config()
│   ├── layers.py              # ModelState, TickResult, init_state()
│   ├── model.py               # Lichtheim2Model: forward_tick, run_trial
│   ├── tasks.py               # Task enum, build_trial_inputs()
│   ├── trainer.py             # train_step(), move_trial_to_device()
│   ├── losses.py              # compute_trial_loss() with masked BCE
│   ├── trials.py              # SupervisedTrial, make_*_trial()
│   ├── encoding.py            # Phoneme inventory and encoder
│   ├── data.py                # WordItem, PseudowordItem, loaders
│   └── semantics.py           # assign_artificial_semantics()
└── tests/                     # Pytest test suite
```

---

## Branch / Git Workflow

- `main` — stable; receives only milestone PRs
- `develop` — integration branch
- Feature work on short branches from `develop`
- Do not commit copyrighted material, raw data, or model weights

---

## Appendix: Model Equations and PyTorch Mapping

This appendix gives a compact, equation-level view of how the **Lichtheim 2-style**
architecture is implemented here, for readers familiar with standard recurrent
networks who want to see how this **manual PyTorch implementation** differs from
`torch.nn.RNN` / `torch.nn.LSTM`.

### 1. Vanilla RNN (for reference)

A standard Elman/vanilla RNN updates a single hidden state with one shared
recurrent rule:

```text
h_t = φ(W_x x_t + W_h h_{t-1} + b)
```

One hidden vector `h_t`, one input-to-hidden matrix `W_x`, one hidden-to-hidden
matrix `W_h`, and one nonlinearity `φ` applied at every step. `torch.nn.RNN`
implements exactly this.

### 2. LSTM (for reference)

An LSTM extends the vanilla RNN with input/forget/output gates and a separate cell
state `c_t`, so the network can learn what to retain or discard over time. **The
current diagnostic implementation does not use LSTM-style gates or a cell state**
— there is no `torch.nn.LSTM` and no learned gating mechanism anywhere in this
codebase.

### 3. Lichtheim 2-style update rule

Instead of one hidden state, this repository implements a hand-written
multi-region recurrent system: at each tick `t`, every named layer (`iSMG`,
`mSTG`, `aSTG`, `vATL_out`, `triangularis`, `motor`) is updated explicitly from
(a) the current task input, (b) feedforward projections from layers already
computed earlier in the same tick, and (c) delayed copy/context buffers carried
over from the previous tick. In compact form, for a layer `l`:

```text
h_l(t) = σ(b_l + Σ_k W_{k→l} h_k(t) + Σ_c W_{c→l} c_c(t))
```

- `h_l(t)` — activation of layer `l` at tick `t` (in `[0, 1]`, since `σ` is the
  sigmoid).
- `Σ_k W_{k→l} h_k(t)` — the **feedforward** contribution from layers already
  computed this tick; this implements the dorsal (`sound → iSMG → motor`) and
  ventral (`sound → mSTG → aSTG → {vATL, triangularis} → motor`) pathways.
- `Σ_c W_{c→l} c_c(t)` — the **delayed copy/context** contribution; this
  implements the Elman recurrence on `iSMG`, the motor→iSMG copy-back, and the
  `vATL_out → vATL_in` copy-back into `aSTG`.

Concretely, `forward_tick()` in
[src/lichtheim2/model.py](src/lichtheim2/model.py) computes (sigmoid applied to
each `*_net`):

```text
iSMG_net  = sound_to_iSMG(sound) + iSMG_elman(iSMG_context) + motor_copy_to_iSMG(motor_context)
mSTG_net  = sound_to_mSTG(sound)
aSTG_net  = mSTG_to_aSTG(mSTG) + vATL_in_to_aSTG(vATL_in_used)
vATL_net  = aSTG_to_vATL(aSTG)
tri_net   = aSTG_to_triangularis(aSTG)
motor_net = iSMG_to_motor(iSMG) + triangularis_to_motor(triangularis)
```

The copy/context buffers (`iSMG_context`, `motor_context`, `vATL_context`) are
**explicit fields of `ModelState`** (see
[src/lichtheim2/layers.py](src/lichtheim2/layers.py)) — plain tensors carried
between ticks by the caller — not hidden states managed internally by
`torch.nn.RNN` or `torch.nn.LSTM`.

### 4. Copy / one-tick delay rule

At the end of every tick, selected activations are copied into context buffers and
become inputs at the *next* tick:

```text
c_l(t+1) = h_l(t)
```

Three such copies happen at the end of `forward_tick()`:

- `iSMG_context(t+1) = iSMG(t)` — Elman self-recurrence on iSMG.
- `motor_context(t+1) = motor(t)` — insular-motor copy-back into iSMG.
- `vATL_context(t+1) = vATL_out(t)` — semantic copy-back into aSTG (overridden by
  an external clamp during the speaking task).

This one-tick delay/unfolding pattern is how the model realises the bidirectional
copy-back connections from Supplementary Figure S1 within a per-tick feedforward
computation.

### 5. Task-specific paths

| Task | Ticks | Path |
|---|---|---|
| **Repetition** | `2T` | Sound input → `iSMG` (dorsal path) → `motor` output |
| **Comprehension** | `T` | Sound input → `mSTG` → `aSTG` → `vATL_out` (semantic output) |
| **Speaking** | `T` | Semantic pattern clamped onto vATL input → `aSTG` / `triangularis` → `motor` output |

See [src/lichtheim2/tasks.py](src/lichtheim2/tasks.py) for the exact tick-by-tick
input construction.

### 6. PyTorch mapping

This model does **not** use `torch.nn.RNN`, `torch.nn.LSTM`, or a Transformer.
Recurrence is implemented manually:

- `Lichtheim2Model.forward_tick(state, sound, clamp_vATL_in)` — consumes a
  `ModelState` (the previous tick's activations and copy/context buffers) plus the
  current task inputs, and returns `(new_state, vATL_input_used)`.
- `Lichtheim2Model.run_trial(task, phon_pattern, sem_pattern, cfg)` — repeatedly
  calls `forward_tick()` for `T` or `2T` ticks (depending on the task), threading
  `state` from one call to the next, and returns one `TickResult` per tick.

This manual PyTorch implementation keeps every intermediate activation and context
buffer inspectable, which the current diagnostic implementation relies on for
per-tick output losses and future lesioning / representational analyses.

### 7. Relationship to Supplementary Figure S1

Supplementary Figure S1 of Ueno et al. (2011) is the authoritative diagram of the
full layer/connection layout, but it is visually dense. This appendix gives a
simplified, paper-aligned, equation-level view of the same connectivity; for the
detailed layer-by-layer mapping (sizes, bias conventions, which connections are
copy vs. feedforward), see
[docs/architecture_notes.md](docs/architecture_notes.md) and
[docs/replication_spec.md](docs/replication_spec.md). The figure itself is not
reproduced here.

### Optional schematic

A simplified, original sketch of the two pathways (not a reproduction of Figure S1):

```text
                       sound input
                      /            \
            iSMG (dorsal)        mSTG -> aSTG (ventral)
              |   ^                       |        \
              |   | motor_context      vATL_in   triangularis
              v   |  (copy)               ^            |
            motor <------------------ vATL_out --------+
           (insular-motor output)   (copy: vATL_out -> vATL_in)
```
