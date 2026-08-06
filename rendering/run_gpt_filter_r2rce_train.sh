#!/usr/bin/env bash
# R2RCE train set 指令质量过滤（纯文本，Qwen 打分）
# 流程: build to_score.json -> Qwen 打分 scored_instructions.json -> 阈值过滤 filtered_ult_gt6.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_DIR="${TRAIN_DIR:-${PROJECT_DIR}/outputs/nig_sample_r2rce_detail_train}"
TO_SCORE="${TO_SCORE:-${SCRIPT_DIR}/to_score.json}"
SCORED="${SCORED:-${SCRIPT_DIR}/scored_instructions.json}"
FILTERED="${FILTERED:-${SCRIPT_DIR}/filtered_ult_gt6.json}"
MODEL="${MODEL:-qwen3-max}"
THRESHOLD="${THRESHOLD:-6}"
BATCH_SIZE="${BATCH_SIZE:-20}"
MAX_ITEMS="${MAX_ITEMS:-0}"

echo "[1/3] 构建 to_score.json"
python "${SCRIPT_DIR}/build_to_score_r2rce_train.py" \
  --train_dir "${TRAIN_DIR}" \
  --output "${TO_SCORE}"

echo "[2/3] Qwen 打分 (model=${MODEL})"
python "${SCRIPT_DIR}/filter_reference.py" \
  --input "${TO_SCORE}" \
  --output "${SCORED}" \
  --model "${MODEL}" \
  --batch_size "${BATCH_SIZE}" \
  --max_items "${MAX_ITEMS}" \
  --resume

echo "[3/3] 阈值过滤 (threshold>=${THRESHOLD})"
python "${SCRIPT_DIR}/filter_results.py" \
  --input "${SCORED}" \
  --output "${FILTERED}" \
  --threshold "${THRESHOLD}"

echo "[DONE] 保留集合: ${FILTERED}"
