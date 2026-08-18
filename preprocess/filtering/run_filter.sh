#!/usr/bin/env bash
# Optional: score R2R-CE train instructions and keep those above a threshold.
# Independent of Habitat rendering — only needs episode_*/sample.json (or any
# directory in that layout). The paper ships r2rce_train_filtered.json; rerun
# this only if you want a different threshold or scorer.
#
#   collect_instructions.py  →  to_score.json
#   score_instructions.py    →  scored_instructions.json   (DashScope API)
#   apply_threshold.py       →  r2rce_train_filtered.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TRAIN_DIR="${TRAIN_DIR:-${PROJECT_DIR}/data/R2RCE_visual/r2rce_train_visual}"
TO_SCORE="${TO_SCORE:-${SCRIPT_DIR}/to_score.json}"
SCORED="${SCORED:-${SCRIPT_DIR}/scored_instructions.json}"
FILTERED="${FILTERED:-${SCRIPT_DIR}/r2rce_train_filtered.json}"
MODEL="${MODEL:-qwen3-max}"
THRESHOLD="${THRESHOLD:-6}"
BATCH_SIZE="${BATCH_SIZE:-20}"
MAX_ITEMS="${MAX_ITEMS:-0}"

echo "[1/3] collect instructions from ${TRAIN_DIR}"
python "${SCRIPT_DIR}/collect_instructions.py" \
  --train_dir "${TRAIN_DIR}" \
  --output "${TO_SCORE}"

echo "[2/3] score with ${MODEL}"
python "${SCRIPT_DIR}/score_instructions.py" \
  --input "${TO_SCORE}" \
  --output "${SCORED}" \
  --model "${MODEL}" \
  --batch_size "${BATCH_SIZE}" \
  --max_items "${MAX_ITEMS}" \
  --resume

echo "[3/3] keep score >= ${THRESHOLD}"
python "${SCRIPT_DIR}/apply_threshold.py" \
  --input "${SCORED}" \
  --output "${FILTERED}" \
  --threshold "${THRESHOLD}"

echo "[DONE] keep-list: ${FILTERED}"
