#!/bin/bash
# Submit the Delphi 25B bf16-vs-fp32 regression repro on sc-loprio.
# Mirrors the env setup of submit_delphi_evals.sh so CKPT_ROOT / VENV_DIR /
# the HF cache land on the same scratch tree (no re-download of weights
# already in the eval cache).
#
# Usage:
#   bash submit_repro_25b_fp32.sh
#   DRY_RUN=1 bash submit_repro_25b_fp32.sh
#
# Env vars (same defaults as submit_delphi_evals.sh):
#   CKPT_ROOT      scratch root      (default: ${SCRATCH:-$HOME/.cache}/gdsuite-delphi)
#   VENV_DIR       uv venv           (default: $CKPT_ROOT/gdsuite-venv)
#   LOG_DIR        SLURM log dir     (default: $CKPT_ROOT/logs)

set -euo pipefail

GDSUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CKPT_ROOT="${SCRATCH:-$HOME/.cache}/gdsuite-delphi"
export CKPT_ROOT="${CKPT_ROOT:-$DEFAULT_CKPT_ROOT}"
export VENV_DIR="${VENV_DIR:-$CKPT_ROOT/gdsuite-venv}"
export GDSUITE_DIR
LOG_DIR="${LOG_DIR:-$CKPT_ROOT/logs}"

cmd=( sbatch --chdir="$GDSUITE_DIR"
      --output="$LOG_DIR/delphi_25b_fp32_%j.out"
      --error="$LOG_DIR/delphi_25b_fp32_%j.err"
      "$GDSUITE_DIR/repro_25b_fp32.sbatch" )

echo "Scratch root: $CKPT_ROOT"
echo "Logs:         $LOG_DIR"
echo "Command:      ${cmd[*]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] not submitting."
    exit 0
fi

mkdir -p "$LOG_DIR"
"${cmd[@]}"

echo
echo "Submitted. Monitor with:  squeue --me --name=delphi_25b_fp32"
echo "Cancel with:              scancel --name=delphi_25b_fp32"
