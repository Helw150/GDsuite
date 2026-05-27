#!/bin/bash
# Pin every cache + scratch path under CKPT_ROOT. Sourced by setup_env.sh
# and run_delphi_model.sh — not run on its own.
#
# Set CKPT_ROOT to a fast shared filesystem or node-local scratch location
# before running setup_env.sh / submit_delphi_evals.sh. If SCRATCH is set,
# use it; otherwise fall back to a user cache directory.
#
# Setting HF_HOME alone is NOT enough — each library below has its own env
# var and otherwise silently falls back to ~/.cache.

DEFAULT_CKPT_ROOT="${SCRATCH:-$HOME/.cache}/gdsuite-delphi"
CKPT_ROOT="${CKPT_ROOT:-$DEFAULT_CKPT_ROOT}"

# HuggingFace: model snapshots, dataset arrow cache, xet chunk cache, locks.
export HF_HOME="$CKPT_ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

# vLLM + compiler caches (Triton kernels, torch.compile / inductor).
export VLLM_CACHE_ROOT="$CKPT_ROOT/vllm-cache"
export TRITON_CACHE_DIR="$CKPT_ROOT/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$CKPT_ROOT/torchinductor-cache"

# uv: wheel cache (multi-GB — torch, vllm) + downloaded python interpreters.
export UV_CACHE_DIR="$CKPT_ROOT/uv-cache"
export UV_PYTHON_INSTALL_DIR="$CKPT_ROOT/uv-python"

# Catch-all for anything else that honors the XDG base-dir spec.
export XDG_CACHE_HOME="$CKPT_ROOT/xdg-cache"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_XET_CACHE" \
         "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME"
