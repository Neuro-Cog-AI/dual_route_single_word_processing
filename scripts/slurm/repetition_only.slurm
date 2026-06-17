#!/bin/bash
# SLURM job script — Lichtheim 2 repetition-only training
#
# Submit from dual_route_single_word_processing/:
#   sbatch scripts/slurm/repetition_only.slurm
#
# Override any variable with --export at submission:
#   ZERO_ERROR_RADIUS=0.1 sbatch --export=ALL scripts/slurm/repetition_only.slurm
#
# See docs/jeanzay_runs.md for full setup and retrieval instructions.

#SBATCH --job-name=lichtheim_rep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_p1
# --account is passed on the sbatch command line by submit_conditions.sh
# using --account=$IDRPROJ@cpu  (SBATCH directives don't expand shell variables)
#SBATCH --output=outputs/slurm_logs/slurm_%j_%x.out
#SBATCH --error=outputs/slurm_logs/slurm_%j_%x.err

set -euo pipefail

# ------------------------------------------------------------------ #
# Configurable parameters — override at submission with --export=ALL  #
# ------------------------------------------------------------------ #
# Training condition
ZERO_ERROR_RADIUS=${ZERO_ERROR_RADIUS:-0.0}   # 0.0=full gradient  0.1=paper dead-zone
EVAL_RADIUS=${EVAL_RADIUS:-0.1}               # paper scoring radius (independent of above)
LR=${LR:-0.5}                                 # paper SGD learning rate
EPOCHS=${EPOCHS:-200}
SEED=${SEED:-0}
SOURCE=${SOURCE:-words}                        # words | pseudowords | mixed
MAX_ITEMS=${MAX_ITEMS:-99999}                  # 99999 = use all available items

# Paths (relative to project root; script is submitted from there)
DATA_DIR=${DATA_DIR:-data/raw/nwr_swp}
CONFIG=${CONFIG:-configs/english_nwr.yaml}

# Output parent directory — timestamped subdir is created automatically
OUTPUT_DIR="outputs/repetition_only/zer_${ZERO_ERROR_RADIUS}"

# ------------------------------------------------------------------ #
# Environment                                                          #
# ------------------------------------------------------------------ #
module purge
# Jean Zay: check available PyTorch modules with `module avail pytorch-gpu`
# and set PYTORCH_MODULE in your ~/.bashrc or before submitting.
#   Example:  module load pytorch-gpu/py3/2.3.0
# If using a conda env, replace the two lines below with:
#   module load anaconda-py3/2023.09
#   conda activate lichtheim2
module load ${PYTORCH_MODULE:-pytorch-gpu/py3/2.3.0}

# ------------------------------------------------------------------ #
# Diagnostics                                                          #
# ------------------------------------------------------------------ #
echo "=== Lichtheim 2 — Repetition-only training ==="
echo "Job ID:             $SLURM_JOB_ID"
echo "Node:               $SLURM_NODELIST"
echo "Submit dir:         $SLURM_SUBMIT_DIR"
echo "ZERO_ERROR_RADIUS:  $ZERO_ERROR_RADIUS"
echo "EVAL_RADIUS:        $EVAL_RADIUS"
echo "LR:                 $LR"
echo "EPOCHS:             $EPOCHS"
echo "SEED:               $SEED"
echo "SOURCE:             $SOURCE"
echo "MAX_ITEMS:          $MAX_ITEMS"
echo "OUTPUT_DIR:         $OUTPUT_DIR"
echo "Python:             $(which python)"
echo "PyTorch:            $(python -c 'import torch; print(torch.__version__)')"
echo "================================================"

# ------------------------------------------------------------------ #
# Ensure output dirs exist (SLURM will not create them)               #
# ------------------------------------------------------------------ #
cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/slurm_logs "$OUTPUT_DIR"

# ------------------------------------------------------------------ #
# Run                                                                  #
# ------------------------------------------------------------------ #
PYTHONPATH=src python scripts/train_repetition_only.py \
  --data-dir      "$DATA_DIR"           \
  --config        "$CONFIG"             \
  --source        "$SOURCE"             \
  --max-items     "$MAX_ITEMS"          \
  --epochs        "$EPOCHS"             \
  --lr            "$LR"                 \
  --seed          "$SEED"               \
  --device        cpu                   \
  --zero-error-radius "$ZERO_ERROR_RADIUS" \
  --eval-radius   "$EVAL_RADIUS"        \
  --loss-reduction sum                  \
  --output-dir    "$OUTPUT_DIR"

echo "=== Job finished: $SLURM_JOB_ID ==="
