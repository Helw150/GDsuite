#!/bin/bash
# Submit the Delphi-collection eval as SLURM job arrays.
# Run setup_env.sh once first.
#
# Models are split into two arrays by size so you can request different
# resources for smaller and larger checkpoints. Both arrays index into the
# same delphi_models.txt, so eval_delphi.sbatch resolves the model the same
# way.
#
# By default only UNFINISHED models are submitted: a model with a full
# set of result JSONs ($DONE_JSON_COUNT) is dropped from the arrays.
# Partially-done models are still submitted — run_eval.py resumes them
# from the task JSONs already on disk.
#
# Usage:
#   bash submit_delphi_evals.sh
#   MAX_PARALLEL=40 bash submit_delphi_evals.sh   # cap concurrent small-GPU tasks
#   RESUBMIT_ALL=1 bash submit_delphi_evals.sh    # submit all 88, even finished
#   DRY_RUN=1 bash submit_delphi_evals.sh         # print the sbatch calls only
#
# Env vars:
#   CKPT_ROOT         scratch root      (default: ${SCRATCH:-$HOME/.cache}/gdsuite-delphi)
#   VENV_DIR          uv venv           (default: $CKPT_ROOT/gdsuite-venv)
#   OUTPUT_ROOT       result JSON root  (default: $CKPT_ROOT/delphi-outputs)
#   LOG_DIR           SLURM log dir     (default: $CKPT_ROOT/logs)
#   MODELS_FILE       model id list     (default: $GDSUITE_DIR/delphi_models.txt)
#   MAX_PARALLEL      %-throttle for the small-GPU array (default: unlimited)
#   SLURM_PARTITION   partition to pass to sbatch (default: unset)
#   SLURM_ACCOUNT     account to pass to sbatch   (default: unset)
#   SLURM_QOS         qos to pass to sbatch       (default: unset)
#   SMALL_GRES        --gres for <=14B models     (default: gpu:1)
#   BIG_GRES          --gres for >14B models      (default: gpu:1)
#   SMALL_MEM         --mem for <=14B models      (default: 96G)
#   BIG_MEM           --mem for >14B models       (default: 160G)
#   SMALL_CONSTRAINT  constraint for <=14B models (default: unset)
#   BIG_CONSTRAINT    constraint for >14B models  (default: unset)
#   SMALL_TIME        --time for the small-GPU array    (default: 06:00:00)
#   BIG_TIME          --time for the big-GPU 25B job    (default: 12:00:00)
#   DONE_JSON_COUNT   result JSONs that mark a model finished (default: 34)
#   RESUBMIT_ALL      set to 1 to submit finished models too  (default: 0)
#   BIG_THRESHOLD_B   params (B) above which a model needs a big GPU (default: 14)
#
# CKPT_ROOT / VENV_DIR / OUTPUT_ROOT are exported so the array tasks pick
# them up (sbatch propagates the submit environment).

set -euo pipefail

GDSUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CKPT_ROOT="${SCRATCH:-$HOME/.cache}/gdsuite-delphi"
export CKPT_ROOT="${CKPT_ROOT:-$DEFAULT_CKPT_ROOT}"
export VENV_DIR="${VENV_DIR:-$CKPT_ROOT/gdsuite-venv}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$CKPT_ROOT/delphi-outputs}"
export GDSUITE_DIR
LOG_DIR="${LOG_DIR:-$CKPT_ROOT/logs}"
MODELS_FILE="${MODELS_FILE:-$GDSUITE_DIR/delphi_models.txt}"
export MODELS_FILE

SMALL_GRES="${SMALL_GRES:-gpu:1}"
BIG_GRES="${BIG_GRES:-gpu:1}"
SMALL_MEM="${SMALL_MEM:-96G}"
BIG_MEM="${BIG_MEM:-160G}"
SMALL_CONSTRAINT="${SMALL_CONSTRAINT:-}"
BIG_CONSTRAINT="${BIG_CONSTRAINT:-}"
BIG_THRESHOLD_B="${BIG_THRESHOLD_B:-14}"
SMALL_TIME="${SMALL_TIME:-06:00:00}"
BIG_TIME="${BIG_TIME:-12:00:00}"
# A model with this many result JSONs is considered finished and skipped.
DONE_JSON_COUNT="${DONE_JSON_COUNT:-34}"

# Load the model list (same order eval_delphi.sbatch indexes into).
MODELS=()
while IFS= read -r line; do
    MODELS+=("$line")
done < <(grep -vE '^[[:space:]]*(#|$)' "$MODELS_FILE")
if [[ "${#MODELS[@]}" -eq 0 ]]; then
    echo "no models found in $MODELS_FILE" >&2
    exit 1
fi

# Classify each model (by array index) as small- or big-GPU, by the
# parameter count parsed from its name. Models already finished (a full
# set of result JSONs) are dropped unless RESUBMIT_ALL=1.
small_idx=()
big_idx=()
finished=0
for i in "${!MODELS[@]}"; do
    model_dir="$OUTPUT_ROOT/$(basename "${MODELS[$i]}")"
    if [[ "${RESUBMIT_ALL:-0}" != "1" && -d "$model_dir" ]]; then
        n_json=$(find "$model_dir" -type f -name '*.json' 2>/dev/null | wc -l)
        if [[ "$n_json" -ge "$DONE_JSON_COUNT" ]]; then
            finished=$((finished + 1))
            continue
        fi
    fi
    pb=$(python3 - "${MODELS[$i]}" <<'PY'
import re, sys
m = re.search(r'([\d.]+)([BM])params', sys.argv[1])
if not m:
    print(0.0)
else:
    v = float(m.group(1))
    print(v if m.group(2) == 'B' else v / 1000)
PY
)
    if awk "BEGIN{exit !($pb > $BIG_THRESHOLD_B)}"; then
        big_idx+=("$i")
    else
        small_idx+=("$i")
    fi
done

join_csv() { local IFS=,; echo "$*"; }
SMALL_ARRAY="$(join_csv "${small_idx[@]:-}")"
BIG_ARRAY="$(join_csv "${big_idx[@]:-}")"
[[ -n "${MAX_PARALLEL:-}" && -n "$SMALL_ARRAY" ]] && \
    SMALL_ARRAY="${SMALL_ARRAY}%${MAX_PARALLEL}"

echo "Models:       ${#MODELS[@]} total  (finished: ${finished} skipped, "\
"small-GPU: ${#small_idx[@]}, big-GPU: ${#big_idx[@]})"
echo "Scratch root: $CKPT_ROOT"
echo "Results:      $OUTPUT_ROOT"
echo "Logs:         $LOG_DIR"

if [[ -z "$SMALL_ARRAY" && -z "$BIG_ARRAY" ]]; then
    echo "Nothing to submit — all models finished. (RESUBMIT_ALL=1 to force.)"
    exit 0
fi

add_optional_sbatch_arg() {
    local option="$1" value="$2"
    if [[ -n "$value" ]]; then
        cmd+=("$option=$value")
    fi
}

submit_group() {
    local label="$1" array="$2" constraint="$3" time_limit="$4" gres="$5" mem="$6"
    if [[ -z "$array" ]]; then
        echo "  [$label] no models — skipped"
        return
    fi
    local cmd=( sbatch --array="$array" --chdir="$GDSUITE_DIR" )
    add_optional_sbatch_arg "--partition" "${SLURM_PARTITION:-}"
    add_optional_sbatch_arg "--account" "${SLURM_ACCOUNT:-}"
    add_optional_sbatch_arg "--qos" "${SLURM_QOS:-}"
    add_optional_sbatch_arg "--constraint" "$constraint"
    cmd+=( --gres="$gres"
           --mem="$mem"
           --time="$time_limit"
           --output="$LOG_DIR/delphi_%A_%a.out"
           --error="$LOG_DIR/delphi_%A_%a.err"
           "$GDSUITE_DIR/eval_delphi.sbatch" )
    echo "  [$label] array=$array  gres=$gres  mem=$mem  time=$time_limit"
    [[ -n "$constraint" ]] && echo "    constraint=$constraint"
    echo "    ${cmd[*]}"
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        "${cmd[@]}"
    fi
}

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
fi

submit_group "small-GPU <=${BIG_THRESHOLD_B}B" "$SMALL_ARRAY" "$SMALL_CONSTRAINT" "$SMALL_TIME" "$SMALL_GRES" "$SMALL_MEM"
submit_group "big-GPU >${BIG_THRESHOLD_B}B"    "$BIG_ARRAY"   "$BIG_CONSTRAINT"  "$BIG_TIME" "$BIG_GRES" "$BIG_MEM"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] not submitting."
else
    echo
    echo "Submitted. Monitor with:  squeue --me"
    echo "Cancel all with:          scancel --name=delphi_eval"
fi
