#!/usr/bin/env bash
set -euo pipefail

export JAVA_HOME="${JAVA_HOME:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
DATA_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  TARGET_ENV="${CONDA_ENV_NAME:-${CONDA_DEFAULT_ENV:-}}"
  if [[ -n "${TARGET_ENV}" && "${CONDA_DEFAULT_ENV:-}" != "${TARGET_ENV}" ]]; then
    if ! conda activate "${TARGET_ENV}"; then
      echo "[WARN] Cannot activate conda env: ${TARGET_ENV}. Continue with current environment."
    fi
  fi
fi

TRAIN_JSON="${TRAIN_JSON:-${DATA_ROOT}/R2R_VLNCE_v1-3/train/train.json.gz}"
VAL_UNSEEN_JSON="${VAL_UNSEEN_JSON:-${DATA_ROOT}/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz}"
SCENES_ROOT="${SCENES_ROOT:-${DATA_ROOT}/scenes_root}"
OUT_TRAIN="${OUT_TRAIN:-${PROJECT_DIR}/outputs/nig_sample_r2rce_detail_train}"
OUT_VALUNSEEN="${OUT_VALUNSEEN:-${PROJECT_DIR}/outputs/nig_sample_r2rce_detail_valunseen}"
LOG_TRAIN="${LOG_TRAIN:-${PROJECT_DIR}/outputs/nig_render_r2rce_detail_train.log}"
LOG_VALUNSEEN="${LOG_VALUNSEEN:-${PROJECT_DIR}/outputs/nig_render_r2rce_detail_valunseen.log}"
MAX_EPISODES="${MAX_EPISODES:-0}"

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
  --max_parts 3
  --width 256
  --height 256
  --panorama
  --panorama_hfov 90.0
  --panorama_step 90.0
)

python "${SCRIPT_DIR}/nig_render_dataset_r2rce_detail.py" \
  --train_json "$TRAIN_JSON" \
  --output_dir "$OUT_TRAIN" \
  --log_path "$LOG_TRAIN" \
  "${COMMON_ARGS[@]}" \
  "$@"

python "${SCRIPT_DIR}/nig_render_dataset_r2rce_detail.py" \
  --train_json "$VAL_UNSEEN_JSON" \
  --output_dir "$OUT_VALUNSEEN" \
  --log_path "$LOG_VALUNSEEN" \
  "${COMMON_ARGS[@]}" \
  "$@"
