#!/bin/bash
# Submit the HF layer-0 sub-op profile of Delphi 25B on sc-loprio.
#
# Usage:
#   CKPT_ROOT=/sphinx/u/salt-checkpoints bash submit_profile_25b_layer0_subops.sh
#   DRY_RUN=1 bash submit_profile_25b_layer0_subops.sh

set -euo pipefail

GDSUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CKPT_ROOT="${SCRATCH:-$HOME/.cache}/gdsuite-delphi"
export CKPT_ROOT="${CKPT_ROOT:-$DEFAULT_CKPT_ROOT}"
export VENV_DIR="${VENV_DIR:-$CKPT_ROOT/gdsuite-venv}"
export GDSUITE_DIR
LOG_DIR="${LOG_DIR:-$CKPT_ROOT/logs}"

cmd=( sbatch --chdir="$GDSUITE_DIR"
      --output="$LOG_DIR/delphi_25b_layer0_%j.out"
      --error="$LOG_DIR/delphi_25b_layer0_%j.err"
      "$GDSUITE_DIR/profile_25b_layer0_subops.sbatch" )

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
echo "Submitted. Monitor with:  squeue --me --name=delphi_25b_layer0"
echo "Cancel with:              scancel --name=delphi_25b_layer0"
