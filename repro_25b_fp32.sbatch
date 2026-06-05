#!/bin/bash
# Run the Delphi 25B bf16-vs-fp32 regression repro on sc-loprio.
#
# fp32 25B = ~100 GB of weights and ~100 GB of download. Needs a 141 GB
# card (H200), so this script pins --constraint=141G.
#
# Submit:
#   sbatch --chdir=/path/to/GDsuite repro_25b_fp32.sbatch
# or via submit_repro_25b_fp32.sh, which exports CKPT_ROOT/VENV_DIR/etc.

#SBATCH --job-name=delphi_25b_fp32
#SBATCH --partition=sc-loprio
#SBATCH --account=nlp
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=141G
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/delphi_25b_fp32_%j.out
#SBATCH --error=logs/delphi_25b_fp32_%j.err

set -uo pipefail

GDSUITE_DIR="${GDSUITE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DEFAULT_CKPT_ROOT="${SCRATCH:-$HOME/.cache}/gdsuite-delphi"
CKPT_ROOT="${CKPT_ROOT:-$DEFAULT_CKPT_ROOT}"
VENV_DIR="${VENV_DIR:-$CKPT_ROOT/gdsuite-venv}"

# Pin HF / Triton / torch.compile caches under CKPT_ROOT so the ~100 GB
# fp32 25B download doesn't land on the slow home NFS.
source "$GDSUITE_DIR/cluster_env.sh"

echo "=== Delphi 25B bf16 vs fp32 regression repro ==="
echo "host:   $(hostname)"
echo "gpu:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo '?')"
echo "venv:   $VENV_DIR"
echo "hf:     $HF_HUB_CACHE"
echo

source "$VENV_DIR/bin/activate"
cd "$GDSUITE_DIR"
exec python reproduce_delphi_25b_regression_fp32.py
