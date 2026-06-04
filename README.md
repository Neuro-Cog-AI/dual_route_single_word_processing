# dual_route_single_word_processing

This repository contains work toward a faithful PyTorch reimplementation of the **Lichtheim 2** neurocomputational model of dual dorsal-ventral language pathways, as described in:

> Ueno, T., Saito, S., Rogers, T. T., & Lambon Ralph, M. A. (2011).
> *Lichtheim 2: Synthesizing Aphasia and the Neural Basis of Language in a Neurocomputational Model of the Dual Dorsal-Ventral Language Pathways.*
> Neuron, 72(2), 385–396.

---

## Scientific Goal

The Lichtheim 2 model is a neurocomputational account of the dual dorsal-ventral language pathways and their roles in auditory word processing (repetition, comprehension, and speaking/naming). It reproduces the major aphasic syndromes through selective lesioning and recovery patterns consistent with neuroimaging and clinical data.

The goal of this repository is:

1. **Phase 1–3:** A PyTorch implementation of the Lichtheim 2 dual-route architecture trained on English/NWR-style data with variable-length phoneme sequences.
2. **Phase 4–5:** Lesioning, recovery, and representational similarity analyses.
3. **Phase 6 (future):** Further extensions — cross-linguistic comparisons and additional analyses.

This is **not** a modern seq2seq model, not a Transformer, and not an audio-processing system. The architecture is a hand-rolled Elman-style recurrent network with explicit tick-by-tick dynamics and copy-back feedback, closely following the original LENS implementation.

---

## Architectural Reference

**Lichtheim 2 (Ueno et al. 2011) is the architectural and methodological reference for this project.** The dual-route structure, explicit tick-by-tick dynamics, copy-back connections, and task definitions (repetition, comprehension, speaking/naming) are all taken directly from the paper and its supplement.

The current training and data direction targets **English/NWR-style data** with variable-length phoneme sequences, not a reconstruction of the original Japanese tri-mora setup. The original Japanese setup remains a historical reference and may serve as a future comparison point.

Decisions are tagged throughout the documentation:

- `[Paper §X]` — stated in the main text of Ueno et al. 2011
- `[Supp §X]` — stated in the supplementary materials
- `[Inferred]` — our best-effort inference for a PyTorch reimplementation
- `[Open]` — unclear; unresolved design decision (see [docs/open_questions.md](docs/open_questions.md))

---

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the full phased plan.

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Repo scaffold and specification | Complete |
| 1 | Toy tick-by-tick forward pass | Complete |
| 2 | Faithful architecture (all layers, all paths) | In progress (Phase 2a) |
| 3 | English/NWR-style training | Pending |
| 4 | Lesioning and recovery | Pending |
| 5 | Representational similarity analyses | Pending |
| 6 | Further extensions (cross-linguistic comparisons, deeper analyses) | Pending |

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
│   ├── toy.yaml          # Minimal synthetic config for Phase 1 validation
│   └── lichtheim2.yaml   # Faithful layer sizes from Ueno et al. (2011)
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
│   ├── test_tick_dynamics.py
│   └── test_faithful_architecture.py
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