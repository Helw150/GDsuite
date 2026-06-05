#!/bin/bash
# Run the Delphi 25B regression repro via vLLM (bf16 by default) on
# sc-loprio. Companion to repro_25b_fp32.sbatch — that one uses HF; this
# one uses vLLM, so the two together isolate whether bf16 is the issue or
# HF specifically is.
#
# vLLM bf16 25B = ~50 GB of weights + KV cache; an 80 GB card is enough.
# If you flip the fp32 line on inside the .py script, switch the
# constraint below to =141G.
#
# Submit via submit_repro_25b_vllm.sh, which mirrors the eval env.

#SBATCH --job-name=delphi_25b_vllm
#SBATCH --partition=sc-loprio
#SBATCH --account=nlp
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=[80G|141G]
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/delphi_25b_vllm_%j.out
#SBATCH --error=logs/delphi_25b_vllm_%j.err

set -uo pipefail

GDSUITE_DIR="${GDSUITE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DEFAULT_CKPT_ROOT="${SCRATCH:-$HOME/.cache}/gdsuite-delphi"
CKPT_ROOT="${CKPT_ROOT:-$DEFAULT_CKPT_ROOT}"
VENV_DIR="${VENV_DIR:-$CKPT_ROOT/gdsuite-venv}"

source "$GDSUITE_DIR/cluster_env.sh"

echo "=== Delphi 25B vLLM bf16 regression repro ==="
echo "host:   $(hostname)"
echo "gpu:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo '?')"
echo "venv:   $VENV_DIR"
echo "hf:     $HF_HUB_CACHE"
echo

source "$VENV_DIR/bin/activate"
cd "$GDSUITE_DIR"
exec python reproduce_delphi_25b_regression_vllm.py
