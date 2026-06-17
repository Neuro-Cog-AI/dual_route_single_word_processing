# Jean Zay — Repetition-only training runs

Cluster: [Jean Zay (IDRIS)](http://www.idris.fr/jean-zay/), French national HPC.
Target: `scripts/train_repetition_only.py`, two experimental conditions.

---

## 1. Prerequisites

- Jean Zay account active; `$IDRPROJ` set in your login shell (`echo $IDRPROJ`).
- Private data CSVs (`phonemes.csv`, `wfe.csv`, `ssp.csv`) transferred to the cluster
  (they are not in the git repository — see §2 below).
- Git repository cloned or pulled to `$WORK`.

---

## 2. One-time cluster setup

### 2a. Clone the repo to `$WORK`

Run on Jean Zay (from a login node, never from a compute node):

```bash
cd $WORK
git clone <repo-url> lichtheim2
cd lichtheim2/dual_route_single_word_processing
```

If you already have a clone, update it:

```bash
cd $WORK/lichtheim2/dual_route_single_word_processing
git pull
```

### 2b. Transfer the private data CSVs

From your local machine:

```bash
# Replace <login> with your Jean Zay username
scp data/raw/nwr_swp/phonemes.csv  <login>@jean-zay.idris.fr:$WORK/lichtheim2/dual_route_single_word_processing/data/raw/nwr_swp/
scp data/raw/nwr_swp/wfe.csv       <login>@jean-zay.idris.fr:$WORK/lichtheim2/dual_route_single_word_processing/data/raw/nwr_swp/
scp data/raw/nwr_swp/ssp.csv       <login>@jean-zay.idris.fr:$WORK/lichtheim2/dual_route_single_word_processing/data/raw/nwr_swp/
```

These files are intentionally excluded from git (see CLAUDE.md §Public Repository Policy).

### 2c. Create a conda environment

On a Jean Zay login node:

```bash
module load anaconda-py3/2023.09   # check: module avail anaconda
conda create -n lichtheim2 python=3.11 -y
conda activate lichtheim2
pip install torch pyyaml matplotlib pytest
```

To verify:

```bash
python -c "import torch, yaml, matplotlib; print('OK', torch.__version__)"
```

### 2d. Configure PYTORCH_MODULE (if using Jean Zay modules instead of conda)

Add to your `~/.bashrc` on Jean Zay:

```bash
export PYTORCH_MODULE=pytorch-gpu/py3/2.3.0   # adjust to available version
```

Check available versions:

```bash
module avail pytorch-gpu
```

If using conda (recommended), edit `scripts/slurm/repetition_only.slurm` lines:

```bash
# Replace:
module load ${PYTORCH_MODULE:-pytorch-gpu/py3/2.3.0}

# With:
module load anaconda-py3/2023.09
conda activate lichtheim2
```

---

## 3. Submitting the two experimental conditions

Both conditions are submitted from the project root:

```bash
cd $WORK/lichtheim2/dual_route_single_word_processing
bash scripts/slurm/submit_conditions.sh
```

This submits two independent SLURM jobs:

| Condition | `zero_error_radius` | `eval_radius` | Meaning |
|-----------|---------------------|---------------|---------|
| A         | 0.0                 | 0.1           | Full gradient — no training dead-zone; paper-like evaluation |
| B         | 0.1                 | 0.1           | Paper dead-zone during training and evaluation |

Both use: `--epochs 200`, `--lr 0.5`, `--seed 0`, `--source words`, `--loss-reduction sum`.

`--max-items 99999` effectively uses all available words (the script samples min(N_available, max_items)).

### Submitting a single condition

`#SBATCH` directives do not expand shell variables, so `--account` must be passed
on the `sbatch` command line (the submit script does this automatically):

```bash
# Condition A only
ZERO_ERROR_RADIUS=0.0 sbatch --export=ALL --account=$IDRPROJ@cpu \
  scripts/slurm/repetition_only.slurm

# Condition B only
ZERO_ERROR_RADIUS=0.1 sbatch --export=ALL --account=$IDRPROJ@cpu \
  scripts/slurm/repetition_only.slurm
```

### Overriding other parameters at submission

```bash
# Shorter run for debugging
ZERO_ERROR_RADIUS=0.0 EPOCHS=20 MAX_ITEMS=50 LR=0.1 \
  sbatch --export=ALL --account=$IDRPROJ@cpu \
  scripts/slurm/repetition_only.slurm
```

---

## 4. Monitoring jobs

```bash
# Live queue
squeue -u $USER

# Detailed status after completion
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS

# Stream stdout log while running
tail -f outputs/slurm_logs/slurm_<JOB_ID>_lichtheim_rep.out
```

---

## 5. Output directory structure

Each run creates a timestamped subdirectory under its condition folder:

```
outputs/repetition_only/
├── zer_0.0/
│   └── 20260617_143022/        ← Condition A
│       ├── run_config.json     ← all CLI args + model config
│       ├── metrics.csv         ← per-epoch avg/min/max loss
│       ├── loss_curve.png      ← loss curve plot
│       ├── predictions_before.json
│       └── predictions_after.json
└── zer_0.1/
    └── 20260617_143025/        ← Condition B
        ├── run_config.json
        ├── metrics.csv
        ├── loss_curve.png
        ├── predictions_before.json
        └── predictions_after.json
```

All of `outputs/` is gitignored — outputs will not be committed.

---

## 6. Retrieving results to your local machine

From your local machine, after jobs complete:

```bash
# Retrieve all outputs for both conditions
rsync -avz --progress \
  <login>@jean-zay.idris.fr:$WORK/lichtheim2/dual_route_single_word_processing/outputs/repetition_only/ \
  outputs/repetition_only/

# Retrieve SLURM logs
rsync -avz --progress \
  <login>@jean-zay.idris.fr:$WORK/lichtheim2/dual_route_single_word_processing/outputs/slurm_logs/ \
  outputs/slurm_logs/
```

---

## 7. Interpreting metrics.csv

```bash
# Final epoch loss for each condition
python - <<'EOF'
import csv, pathlib
for cond in ["zer_0.0", "zer_0.1"]:
    for run_dir in sorted(pathlib.Path(f"outputs/repetition_only/{cond}").glob("*")):
        rows = list(csv.DictReader(open(run_dir / "metrics.csv")))
        if rows:
            first, last = rows[0], rows[-1]
            print(f"{cond}/{run_dir.name}: "
                  f"epoch1 avg={float(first['avg_loss']):.4f}  "
                  f"epoch{last['epoch']} avg={float(last['avg_loss']):.4f}")
EOF
```

## 8. Interpreting predictions_after.json

Key metrics per word (in `predictions_after.json`):

| Key | What it measures |
|-----|-----------------|
| `phoneme_accuracy` | Fraction of output ticks where argmax is correct |
| `exact_match` | Boolean: all ticks argmax-correct |
| `output_all_units_within_radius` | Boolean: all 39 motor units within eval_radius of target — candidate paper word accuracy [Open #9] |
| `input_all_silent_within_radius` | Boolean: all input-phase motor units below eval_radius |
| `mean_positive_output` | Mean activation of the correct phoneme unit (earliest learning signal) |
| `mean_negative_output` | Mean activation of zero-target units (suppression) |

```bash
# Quick summary: paper-like word accuracy after training
python - <<'EOF'
import json, pathlib
for cond in ["zer_0.0", "zer_0.1"]:
    for run_dir in sorted(pathlib.Path(f"outputs/repetition_only/{cond}").glob("*")):
        preds = json.loads((run_dir / "predictions_after.json").read_text())
        n = len(preds)
        n_word = sum(1 for p in preds if p["output_all_units_within_radius"])
        n_exact = sum(1 for p in preds if p["exact_match"])
        mean_pos = sum(p["mean_positive_output"] for p in preds) / n
        print(f"{cond}/{run_dir.name}:")
        print(f"  word accuracy (radius=0.1): {n_word}/{n}  ({n_word/n:.4f})  [Open #9]")
        print(f"  exact match (argmax):       {n_exact}/{n}  ({n_exact/n:.4f})")
        print(f"  mean positive output:       {mean_pos:.4f}")
EOF
```

---

## 9. Typical expected trajectory (200 epochs, lr=0.5)

Early training (epochs 1–20):
- BCE loss drops sharply (zero-target suppression dominates).
- `mean_negative_output` falls toward 0.
- `mean_positive_output` stays near 0 — phoneme production not yet visible.

Mid training (epochs 20–100):
- `mean_positive_output` begins rising.
- `phoneme_accuracy` improves.

Late training (epochs 100–200):
- `output_all_units_within_radius` and `exact_match` words accumulate.
- Whether `zero_error_radius=0.0` or `0.1` matters most here [Open #5].

See `docs/repetition_only_training_note.md` §13 for the Phase 3e diagnostic
framework and metric interpretations.

---

## 10. Troubleshooting

**Job immediately fails (OOM, not found):** Check that the data CSVs exist at `data/raw/nwr_swp/` on the cluster. The script exits with a KeyError if a phoneme is missing from the inventory.

**SLURM log missing:** Ensure `outputs/slurm_logs/` existed before submission — SLURM will not create it.

**Wrong `$PYTORCH_MODULE`:** Run `module avail pytorch-gpu` on a Jean Zay login node and set the correct name.

**Account rejected:** Check `echo $IDRPROJ` and verify you have cpu allocation with `idr_compuse`.
