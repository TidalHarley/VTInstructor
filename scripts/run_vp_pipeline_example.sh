#!/usr/bin/env bash
# ============================================================================
# VTInstructor — end-to-end pipeline example
#
# This is a minimal reference orchestration that runs the four stages in order.
# Set the environment variables below for your machine, then run:
#
#     bash scripts/run_vp_pipeline_example.sh
#
# Each stage can also be launched independently via the scripts referenced.
# ============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- Configure these for your environment ------------------------------------
export PY="${PY:-python}"
export MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-VL-8B-Instruct}"
export SCENES_ROOT="${SCENES_ROOT:-/path/to/habitat/scenes}"
export DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# On machines with both Mesa and NVIDIA EGL ICDs, force NVIDIA for habitat-sim:
# export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
# ------------------------------------------------------------------------------

echo "[VTInstructor] PROJECT_DIR=${PROJECT_DIR}"

# Stage 1 — Rendering: clean panorama keyframes (no VTP yet).
# Both scripts write into ${DATA_ROOT} using the canonical split layout below.
echo "== Stage 1/4: Rendering =="
OUT_ROOT="${DATA_ROOT}/R2RCE_visual" bash "${PROJECT_DIR}/rendering/run_render_r2rce.sh"
OUT_ROOT="${DATA_ROOT}/RXRCE_visual" bash "${PROJECT_DIR}/rendering/run_render_rxrce.sh"

# Stage 2 — VP overlays + semantic masks (THIS is what training and eval consume).
# Do NOT use visual_prompt/run_nig_render_*_visual.sh here — those do not write vp_masks.
#
# All four splits need masks: the val_unseen splits too, because the VP-Adapter
# consumes vp_masks at inference time exactly as it does during training. Running
# eval on a split without masks silently degrades to zero VP input.
echo "== Stage 2/4: VP overlay + semantic masks =="
VP_SPLITS=(
  "${DATA_ROOT}/R2RCE_visual/r2rce_train_visual"
  "${DATA_ROOT}/R2RCE_visual/r2rce_valunseen_visual"
  "${DATA_ROOT}/RXRCE_visual/rxrce_train_visual"
  "${DATA_ROOT}/RXRCE_visual/rxrce_valunseen_visual"
)

missing=()
for split in "${VP_SPLITS[@]}"; do
  [[ -d "$split" ]] || missing+=("$split")
done
if (( ${#missing[@]} > 0 )); then
  echo "[FATAL] Stage 1 output missing — Stage 2 cannot run on:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo "Re-run Stage 1, or edit VP_SPLITS if you intentionally use a subset." >&2
  exit 2
fi

for split in "${VP_SPLITS[@]}"; do
  echo "[Stage 2] ${split}"
  "${PY}" "${PROJECT_DIR}/vp_adapter/render_vp_masks_from_samples.py" \
    --data_dir "$split" \
    --scenes_root "$SCENES_ROOT" \
    --width 256 --height 384 --camera_height 0.75 \
    --hfov 90.0 --panorama_step 90.0
done

# Stage 3 — VP-Adapter SFT (train -> eval -> score)
echo "== Stage 3/4: VP-Adapter SFT =="
export R2RCE_TRAIN_DATA_DIR="${VP_SPLITS[0]}"
export R2RCE_EVAL_DATA_DIR="${VP_SPLITS[1]}"
export RXRCE_TRAIN_DATA_DIR="${VP_SPLITS[2]}"
export RXRCE_EVAL_DATA_DIR="${VP_SPLITS[3]}"
bash "${PROJECT_DIR}/vp_adapter/run_vp_sft_pipeline.sh"

# Stage 4 — VP-GRPO reinforcement learning
echo "== Stage 4/4: VP-GRPO =="
SFT_CKPT="${SFT_CKPT:-${PROJECT_DIR}/outputs/nig_vp_adapter_v3.1_sft}" \
  bash "${PROJECT_DIR}/vp_grpo/run_grpo_vp_v3.1.sh"

echo "[VTInstructor] pipeline example finished."
