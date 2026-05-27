#!/bin/bash
# SLURM job array: one task per model in the marin-community/delphi
# collection, each running the full GDsuite suite.
#
# Submit via submit_delphi_evals.sh. That wrapper splits the collection by
# model size and can pass cluster-specific partition/account/qos/constraint,
# memory, time, and GPU options on the sbatch command line.
#
# Direct use:
#   sbatch --array=0-87 --chdir=/path/to/GDsuite eval_delphi.sbatch
#
# Every model is run single-GPU by default (TP=1). Set TP_SIZE if you
# request multiple GPUs for each array task.

#SBATCH --job-name=delphi_eval
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/delphi_%A_%a.out
#SBATCH --error=logs/delphi_%A_%a.err

set -uo pipefail

GDSUITE_DIR="${GDSUITE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MODELS_FILE="${MODELS_FILE:-$GDSUITE_DIR/delphi_models.txt}"

# Pick this task's model: the (index+1)-th non-comment, non-blank line.
MODEL=$(grep -vE '^[[:space:]]*(#|$)' "$MODELS_FILE" \
        | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [[ -z "$MODEL" ]]; then
    echo "no model for array index ${SLURM_ARRAY_TASK_ID} in $MODELS_FILE" >&2
    exit 1
fi

echo "array task ${SLURM_ARRAY_TASK_ID} (job ${SLURM_JOB_ID}) -> ${MODEL}"
export GDSUITE_DIR
exec bash "$GDSUITE_DIR/run_delphi_model.sh" "$MODEL"
