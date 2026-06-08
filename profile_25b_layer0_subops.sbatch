#!/bin/bash
# Layer-0 sub-op profile of Delphi 25B via HF (fp32), to diff against
# the Levanter v4-32 layer-0 sub-op dump.
#
# fp32 25B = ~100 GB weights → 141 GB card (H200).

#SBATCH --job-name=delphi_25b_layer0
#SBATCH --partition=sc-loprio
#SBATCH --account=nlp
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=141G
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/delphi_25b_layer0_%j.out
#SBATCH --error=logs/delphi_25b_layer0_%j.err

set -uo pipefail

GDSUITE_DIR="${GDSUITE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DEFAULT_CKPT_ROOT="${SCRATCH:-$HOME/.cache}/gdsuite-delphi"
CKPT_ROOT="${CKPT_ROOT:-$DEFAULT_CKPT_ROOT}"
VENV_DIR="${VENV_DIR:-$CKPT_ROOT/gdsuite-venv}"

source "$GDSUITE_DIR/cluster_env.sh"

echo "=== Delphi 25B HF layer-0 sub-op profile (fp32) ==="
echo "host:   $(hostname)"
echo "gpu:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo '?')"
echo "venv:   $VENV_DIR"
echo "hf:     $HF_HUB_CACHE"
echo

source "$VENV_DIR/bin/activate"
cd "$GDSUITE_DIR"
exec python profile_delphi_25b_layer0_subops.py
