#!/bin/bash
# Evaluate ONE Delphi model on the full GDsuite suite, then delete the
# downloaded weights from the HF cache.
#
# If SLURM preempts the job and sends SIGTERM, keep the downloaded weights
# so a requeued run resumes instead of re-downloading. On a real finish
# (success or hard failure), reclaim the disk.
#
# Resume is free: run_eval.py skips task JSONs that already exist in the
# output dir, so a requeued run only redoes the task it was cut off on.
#
# Usage:
#   bash run_delphi_model.sh <hf_model_id>
#
# Env vars (have sane defaults):
#   CKPT_ROOT    scratch root            (default: ${SCRATCH:-$HOME/.cache}/gdsuite-delphi)
#   VENV_DIR     uv venv from setup_env  (default: $CKPT_ROOT/gdsuite-venv)
#   GDSUITE_DIR  this repo               (default: dir of this script)
#   OUTPUT_ROOT  eval result JSON root   (default: $CKPT_ROOT/delphi-outputs)
#   TP_SIZE      tensor_parallel_size    (default: 1)

set -uo pipefail   # NOT -e: we manage control flow around the eval below.

MODEL="${1:?usage: run_delphi_model.sh <hf_model_id>}"

DEFAULT_CKPT_ROOT="${SCRATCH:-$HOME/.cache}/gdsuite-delphi"
CKPT_ROOT="${CKPT_ROOT:-$DEFAULT_CKPT_ROOT}"
VENV_DIR="${VENV_DIR:-$CKPT_ROOT/gdsuite-venv}"
GDSUITE_DIR="${GDSUITE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CKPT_ROOT/delphi-outputs}"
TP_SIZE="${TP_SIZE:-1}"

# Pin HF / vLLM / Triton / compiler caches under CKPT_ROOT.
source "$GDSUITE_DIR/cluster_env.sh"

MODEL_SHORT="$(basename "$MODEL")"
MODEL_CACHE="$HF_HUB_CACHE/models--${MODEL//\//--}"

# SLURM preemption delivers SIGTERM. Record it so cleanup knows the job
# will be requeued and must NOT delete the half-used weights.
PREEMPTED=0
trap 'PREEMPTED=1; echo "[signal] SIGTERM — treating as preemption"' TERM

cleanup() {
    if [[ "$PREEMPTED" == "1" ]]; then
        echo "[cleanup] preempted — keeping $MODEL_CACHE for the requeued run"
    else
        echo "[cleanup] removing $MODEL_CACHE"
        rm -rf "$MODEL_CACHE"
    fi
}
trap cleanup EXIT

echo "=== $MODEL  (TP=$TP_SIZE) ==="
source "$VENV_DIR/bin/activate"
cd "$GDSUITE_DIR"
python run_eval.py \
    --model_name "$MODEL" \
    --tensor_parallel_size "$TP_SIZE" \
    --output_dir "$OUTPUT_ROOT/$MODEL_SHORT"
rc=$?

if [[ "$PREEMPTED" == "1" ]]; then
    echo "[exit] preempted; SLURM will requeue this task."
    exit 1
fi
exit "$rc"
