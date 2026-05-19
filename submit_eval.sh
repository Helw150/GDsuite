#!/bin/bash
# Submit a release eval run via SLURM (single node, 1 GPU by default).
#
# Usage:
#   bash release/submit_eval.sh <model> [revision] [tp_size]
#
# Examples:
#   bash release/submit_eval.sh allenai/Olmo-3-1025-7B main
#   bash release/submit_eval.sh allenai/OLMo-3-1125-32B main 4
#   SLURM_QOS=normal bash release/submit_eval.sh allenai/Olmo-3-1025-7B main

set -euo pipefail
cd /workspace-vast/wenj/FT-generalization

MODEL="${1:?usage: submit_eval.sh <model> [revision] [tp_size]}"
REVISION="${2:-main}"
TP_SIZE="${3:-1}"
QOS="${SLURM_QOS:-high}"

MODEL_SHORT=$(basename "$MODEL")
RUN_NAME="${MODEL_SHORT}_${REVISION}"

# Memory based on model footprint (bf16: ~2GB/B-param + KV cache).
if   [[ "$MODEL" == *70B* || "$MODEL" == *72B* ]]; then MEM="320G"
elif [[ "$MODEL" == *32B*                       ]]; then MEM="160G"
else                                                    MEM="80G"
fi

mkdir -p release/logs

sbatch \
    --job-name="gen_eval_${RUN_NAME}" \
    --partition=general \
    --qos="$QOS" \
    --gres="gpu:${TP_SIZE}" \
    --cpus-per-task=8 \
    --mem="$MEM" \
    --time=2-00:00:00 \
    --exclude=node-2,node-4,node-9 \
    --output="release/logs/gen_eval_${RUN_NAME}_%j.out" \
    --wrap="umask 0000 && \
            cd /workspace-vast/wenj/FT-generalization && \
            export HF_HOME=/workspace-vast/pretrained_ckpts && \
            export HF_HUB_CACHE=/workspace-vast/pretrained_ckpts/hub && \
            export HF_HUB_DOWNLOAD_TIMEOUT=120 && \
            .venv/bin/python release/run_eval.py \
                --model_name '${MODEL}' \
                --revision '${REVISION}' \
                --tensor_parallel_size ${TP_SIZE} \
                --output_dir release/outputs/${RUN_NAME}"

echo "Submitted ${RUN_NAME}  qos=${QOS}  TP=${TP_SIZE}  mem=${MEM}"
