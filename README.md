# dual_route_single_word_processing

This repository contains work toward a faithful PyTorch reimplementation of the **Lichtheim 2** neurocomputational model of dual dorsal-ventral language pathways, as described in:

> Ueno, T., Saito, S., Rogers, T. T., & Lambon Ralph, M. A. (2011).
> *Lichtheim 2: Synthesizing Aphasia and the Neural Basis of Language in a Neurocomputational Model of the Dual Dorsal-Ventral Language Pathways.*
> Neuron, 72(2), 385–396.

---

## Scientific Goal

The Lichtheim 2 model is a neurocomputational account of the dual dorsal-ventral language pathways and their roles in auditory word processing (repetition, comprehension, and speaking/naming). It reproduces the major aphasic syndromes through selective lesioning and recovery patterns consistent with neuroimaging and clinical data.

The goal of this repository is:

1. **Phase 1–3:** A faithful, tick-by-tick PyTorch reimplementation of the original model, validated against Figure 2 of Ueno et al. (2011).
2. **Phase 4–5:** Lesioning, recovery, and representational similarity analyses.
3. **Phase 6 (future):** English adaptation and SWP-inspired extension — *only after the faithful replication is confirmed*.

This is **not** a modern seq2seq model, not a Transformer, and not an audio-processing system. The architecture is a hand-rolled Elman-style recurrent network with explicit tick-by-tick dynamics and copy-back feedback, closely following the original LENS implementation.

---

## Faithful Replication First

The immediate priority is correctness with respect to the original paper and its supplementary materials, not generality or modern conventions.

Decisions are tagged throughout the documentation:

- `[Paper §X]` — stated in the main text of Ueno et al. 2011
- `[Supp §X]` — stated in the supplementary materials
- `[Inferred]` — our best-effort inference for a PyTorch reimplementation
- `[Open]` — unclear; unresolved design decision (see [docs/open_questions.md](docs/open_questions.md))

No claim of faithfulness will be made until the architecture and training dynamics have been validated against the paper.

---

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the full phased plan.

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Repo scaffold and specification | Complete |
| 1 | Toy tick-by-tick forward pass | Complete |
| 2 | Faithful architecture (all layers, all paths) | Next |
| 3 | Training and Figure 2-like learning curves | Pending |
| 4 | Lesioning and recovery | Pending |
| 5 | Representational similarity analyses | Pending |
| 6 | English adaptation / SWP extension | Pending — after Phase 3 confirmed |

---

## Public Repository Policy

This is a **public** repository. The following must never be committed:

- Copyrighted PDFs (including Ueno et al. 2011 and its supplement)
- Raw linguistic data or corpora
- Model checkpoints, weights, or large binary outputs
- Private notes, internal lab documents, or anything under NDA

The `.gitignore` enforces these exclusions. Private materials belong in `references/private_papers/` (ignored) or a separate private repository.

---

## Repository Structure

```
dual_route_single_word_processing/
├── configs/              # YAML experiment configurations
│   └── toy.yaml          # Minimal synthetic config for Phase 1 validation
├── docs/                 # Scientific documentation and specification
│   ├── roadmap.md
│   ├── replication_spec.md
│   ├── open_questions.md
│   ├── architecture_notes.md
│   ├── training_notes.md
│   └── data_encoding_notes.md
├── src/
│   └── lichtheim2/       # Python package
│       ├── __init__.py
│       ├── config.py     # ModelConfig dataclass, load_config()
│       ├── layers.py     # ModelState, TickResult dataclasses, init_state()
│       ├── model.py      # Lichtheim2Model: forward_tick, run_trial
│       └── tasks.py      # Task enum, build_trial_inputs()
├── tests/                # Pytest test suite
│   └── test_tick_dynamics.py
├── conftest.py           # sys.path shim (temporary, until pyproject.toml)
├── .gitignore
└── README.md
```

---

## Development Setup

> Phase 1 (toy tick-by-tick dynamics) is implemented. No installable package yet — `pyproject.toml` is deferred.

A dedicated Python environment is recommended:

```bash
conda create -n lichtheim2 python=3.11
conda activate lichtheim2
pip install pytest pyyaml ruff black
```

For Phase 1 toy dynamics, PyTorch is required (dependency list is not final):

```bash
pip install torch pytest pyyaml
```

To run the test suite:

```bash
python -m pytest tests/ -v
```

---

## Branch / Git Workflow

- `main` — stable; receives only milestone PRs
- `develop` — integration branch
- Feature work on short branches from `develop` (e.g., `feat/phase-2-faithful-arch`, `fix/copy-back-order`)
- Do not commit copyrighted material, raw data, or model weights