#!/bin/bash
# One-time setup for running GDsuite Delphi evals on a SLURM cluster.
# Creates a uv-managed virtualenv and pre-fetches the eval dataset so the
# parallel array jobs don't all hammer the HF hub at once.
#
# Usage:
#   bash setup_env.sh
#
# Override defaults via env vars:
#   CKPT_ROOT       scratch root for venv + model cache
#                   (default: ${SCRATCH:-$HOME/.cache}/gdsuite-delphi)
#   VENV_DIR        venv location  (default: $CKPT_ROOT/gdsuite-venv)
#   PYTHON_VERSION  python for the venv (default: 3.11)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CKPT_ROOT="${SCRATCH:-$HOME/.cache}/gdsuite-delphi"
CKPT_ROOT="${CKPT_ROOT:-$DEFAULT_CKPT_ROOT}"
VENV_DIR="${VENV_DIR:-$CKPT_ROOT/gdsuite-venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

mkdir -p "$CKPT_ROOT"

# Pin HF / uv / compiler caches onto CKPT_ROOT.
source "$HERE/cluster_env.sh"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing to ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating venv at $VENV_DIR (python $PYTHON_VERSION)"
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"

# vllm pins a compatible torch; the rest are light pure-python deps.
uv pip install --python "$VENV_DIR/bin/python" \
    vllm transformers pyyaml datasets huggingface_hub

echo "Pre-fetching the eval dataset into $HF_HOME ..."
"$VENV_DIR/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("jiaxin-wen/generalization-dynamics-evals", repo_type="dataset")
print("eval dataset cached.")
PY

echo
echo "Done."
echo "  venv:        $VENV_DIR"
echo "  HF cache:    $HF_HOME"
echo
echo "Next: bash submit_delphi_evals.sh"
