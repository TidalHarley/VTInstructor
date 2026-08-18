#!/usr/bin/env bash
# Stage 1 — R2R-CE clean panorama keyframes (train + val_unseen).
#
# Renders directly into the canonical layout consumed by Stage 2/3/4:
#   data/R2RCE_visual/r2rce_train_visual
#   data/R2RCE_visual/r2rce_valunseen_visual
# Stage 2 (visual_prompt/render_masks.py) then edits these
# directories in place, adding overlays and vp_masks.
#
# Geometry below matches the released training data (vp_768x384_h075_maxparts5):
# 3 x 256x384 panorama faces, camera_height 0.75, max_parts 5. Do not change
# these unless you intend to build a different dataset.
set -euo pipefail

export JAVA_HOME="${JAVA_HOME:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$(cd "${PROJECT_DIR}/.." && pwd)}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  TARGET_ENV="${CONDA_ENV_NAME:-${CONDA_DEFAULT_ENV:-}}"
  if [[ -n "${TARGET_ENV}" && "${CONDA_DEFAULT_ENV:-}" != "${TARGET_ENV}" ]]; then
    if ! conda activate "${TARGET_ENV}"; then
      echo "[WARN] Cannot activate conda env: ${TARGET_ENV}. Continue with current environment."
    fi
  fi
fi

TRAIN_JSON="${TRAIN_JSON:-${DATASET_ROOT}/R2R_VLNCE_v1-3/train/train.json.gz}"
VAL_UNSEEN_JSON="${VAL_UNSEEN_JSON:-${DATASET_ROOT}/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz}"
SCENES_ROOT="${SCENES_ROOT:-${DATASET_ROOT}/scenes_root}"

OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/data/R2RCE_visual}"
OUT_TRAIN="${OUT_TRAIN:-${OUT_ROOT}/r2rce_train_visual}"
OUT_VALUNSEEN="${OUT_VALUNSEEN:-${OUT_ROOT}/r2rce_valunseen_visual}"

LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs}"
LOG_TRAIN="${LOG_TRAIN:-${LOG_ROOT}/render_r2rce_train.log}"
LOG_VALUNSEEN="${LOG_VALUNSEEN:-${LOG_ROOT}/render_r2rce_valunseen.log}"
MAX_EPISODES="${MAX_EPISODES:-0}"

mkdir -p "$OUT_TRAIN" "$OUT_VALUNSEEN" "$LOG_ROOT"

COMMON_ARGS=(
  --scenes_root "$SCENES_ROOT"
  --max_episodes "$MAX_EPISODES"
  --sample_mode event
  --frame_stride 2
  --goal_radius 0.5
  --max_steps 500
  --forward_step 0.25
  --turn_angle 30.0
  --small_fwd_m 0.5
  --small_turn_deg 30.0
  --split_forward_threshold_m 6.0
  --max_parts 5
  --width 256
  --height 384
  --camera_height 0.75
  --panorama
  --panorama_hfov 90.0
  --panorama_step 90.0
)

echo "[Stage 1] R2R-CE train      -> ${OUT_TRAIN}"
python "${SCRIPT_DIR}/render_r2rce.py" \
  --train_json "$TRAIN_JSON" \
  --output_dir "$OUT_TRAIN" \
  --log_path "$LOG_TRAIN" \
  "${COMMON_ARGS[@]}" \
  "$@"

echo "[Stage 1] R2R-CE val_unseen -> ${OUT_VALUNSEEN}"
python "${SCRIPT_DIR}/render_r2rce.py" \
  --train_json "$VAL_UNSEEN_JSON" \
  --output_dir "$OUT_VALUNSEEN" \
  --log_path "$LOG_VALUNSEEN" \
  "${COMMON_ARGS[@]}" \
  "$@"
