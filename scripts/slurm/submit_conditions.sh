#!/bin/bash
# Submit both experimental conditions for repetition-only training on Jean Zay.
#
# Usage (from dual_route_single_word_processing/):
#   bash scripts/slurm/submit_conditions.sh
#
# Both jobs run with the same seed, lr, epochs, and eval_radius.
# They differ only in zero_error_radius (training dead-zone).
#
# See docs/jeanzay_runs.md for setup and retrieval instructions.

set -euo pipefail

# Must be run from the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Ensure log directory exists before submitting
mkdir -p outputs/slurm_logs

SLURM_SCRIPT=scripts/slurm/repetition_only.slurm

# Jean Zay SLURM account — $IDRPROJ is set by Jean Zay at login
# #SBATCH directives don't expand shell variables, so pass --account on the command line.
ACCOUNT="${IDRPROJ}@cpu"

echo "Submitting Condition A: zero_error_radius=0.0  (full gradient, no dead-zone)"
JOB_A=$(ZERO_ERROR_RADIUS=0.0 sbatch --export=ALL --account="$ACCOUNT" --parsable "$SLURM_SCRIPT")
echo "  Job ID: $JOB_A"

echo "Submitting Condition B: zero_error_radius=0.1  (paper dead-zone)"
JOB_B=$(ZERO_ERROR_RADIUS=0.1 sbatch --export=ALL --account="$ACCOUNT" --parsable "$SLURM_SCRIPT")
echo "  Job ID: $JOB_B"

echo ""
echo "Both jobs submitted."
echo "Monitor:  squeue -u \$USER"
echo "Logs:     outputs/slurm_logs/slurm_${JOB_A}_lichtheim_rep.out"
echo "          outputs/slurm_logs/slurm_${JOB_B}_lichtheim_rep.out"
echo ""
echo "After completion, results will be in:"
echo "  outputs/repetition_only/zer_0.0/<timestamp>/"
echo "  outputs/repetition_only/zer_0.1/<timestamp>/"
