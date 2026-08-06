#!/usr/bin/env bash
# VP-GRPO evaluation: reuse the VP-Adapter SFT pipeline, skip training
# EVAL_CKPT 指向 GRPO 输出的 best checkpoint（训练器只写 current/ 与 best/）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."

EVAL_ONLY=1 \
EVAL_CKPT="${EVAL_CKPT:-${PROJECT_DIR}/outputs/nig_rl_vp_v3.3/best}" \
SKIP_TRAIN=1 \
SKIP_EVAL=0 \
SKIP_SCORE=0 \
KEEP_EVAL_SHARDS=0 \
EVAL_CUDA="${EVAL_CUDA:-0,1,2,3,4,5,6,7}" \
bash "${PROJECT_DIR}/vp_adapter/run_vp_sft_pipeline.sh"
