#!/usr/bin/env bash
set -euo pipefail

export JAVA_HOME="${JAVA_HOME:-}"

# R2R-CE visual prompting 渲染脚本
# 输出结构：
#   PATH/TO/outputs/R2RCE_visual/r2rce_train_visual
#   PATH/TO/outputs/R2RCE_visual/r2rce_valunseen_visual
#
# 用法：
#   bash PATH/TO/run_nig_render_r2rce_visual.sh
# 可选：
#   MAX_EPISODES=100 bash PATH/TO/run_nig_render_r2rce_visual.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-habitat}"
fi

TRAIN_JSON="${TRAIN_JSON:-${PROJECT_DIR}/data/R2RCE/train.json.gz}"
VAL_UNSEEN_JSON="${VAL_UNSEEN_JSON:-${PROJECT_DIR}/data/R2RCE/val_unseen.json.gz}"
SCENES_ROOT="${SCENES_ROOT:-${PROJECT_DIR}/data/scenes}"

OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/outputs/R2RCE_visual}"
OUT_TRAIN="${OUT_ROOT}/r2rce_train_visual"
OUT_VALUNSEEN="${OUT_ROOT}/r2rce_valunseen_visual"

LOG_TRAIN="${OUT_ROOT}/r2rce_train_visual.log"
LOG_VALUNSEEN="${OUT_ROOT}/r2rce_valunseen_visual.log"

MAX_EPISODES="${MAX_EPISODES:-0}"

mkdir -p "$OUT_ROOT" "$OUT_TRAIN" "$OUT_VALUNSEEN"

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
  --height 640
  --camera_height 1.2
  --panorama
  --panorama_hfov 90.0
  --panorama_step 90.0
  --vp_dropout_prob 0.0
  --vp_alpha 0.55
  --vp_mixed_mode_prob 0.0
  --vp_seed 42
)

python "${SCRIPT_DIR}/render_with_vp.py" \
  --train_json "$TRAIN_JSON" \
  --output_dir "$OUT_TRAIN" \
  --log_path "$LOG_TRAIN" \
  "${COMMON_ARGS[@]}" \
  "$@"

python "${SCRIPT_DIR}/render_with_vp.py" \
  --train_json "$VAL_UNSEEN_JSON" \
  --output_dir "$OUT_VALUNSEEN" \
  --log_path "$LOG_VALUNSEEN" \
  "${COMMON_ARGS[@]}" \
  "$@"

echo "渲染完成："
echo "  train     -> $OUT_TRAIN"
echo "  valunseen -> $OUT_VALUNSEEN"
