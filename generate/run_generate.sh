#!/usr/bin/env bash
# Generate navigation instructions over a rendered split, for data augmentation.
#
# This is *not* the official benchmark. Output records use
# `generated_instruction` and have no ground-truth `references`, so they
# cannot be passed to metrics/evaluate.py. For paper scores use eval/run_eval.sh.
#
#   CKPT=/path/to/ckpt DATA_DIR=data/R2RCE_visual/r2rce_train_visual \
#     bash generate/run_generate.sh
#
# Emit a drop-in VLN-CE split as well:
#   CKPT=... DATA_DIR=... \
#     VLNCE_TEMPLATE=/path/to/R2R_VLNCE_v1-3/train/train.json.gz \
#     OUT_VLNCE_GZ=outputs/augment/train_generated.json.gz \
#     bash generate/run_generate.sh
set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "[FATAL] This script requires bash."
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PY:-python}"
GEN_SCRIPT="${SCRIPT_DIR}/generate_instructions.py"

CKPT="${CKPT:-}"
MODEL_DIR="${MODEL_DIR:-}"
DATA_DIR="${DATA_DIR:-}"
OUT_JSON="${OUT_JSON:-${PROJECT_DIR}/outputs/augment/generated_$(date +%Y%m%d_%H%M%S).json}"

VP_MODE="${VP_MODE:-auto}"
FRAME_SOURCE="${FRAME_SOURCE:-auto}"
GRANULARITY="${GRANULARITY:-trajectory}"
DATASET_TYPE="${DATASET_TYPE:-auto}"
MAX_FRAMES="${MAX_FRAMES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-150}"
TEMPERATURE="${TEMPERATURE:-0.0}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
GEN_CUDA="${GEN_CUDA:-0}"
VLNCE_TEMPLATE="${VLNCE_TEMPLATE:-}"
OUT_VLNCE_GZ="${OUT_VLNCE_GZ:-}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[FATAL] Python not found: $PY"
  exit 127
fi
if [[ -z "$CKPT" || -z "$DATA_DIR" ]]; then
  echo "[FATAL] CKPT and DATA_DIR are required."
  echo "        CKPT=/path/to/ckpt DATA_DIR=/path/to/rendered_split bash $0"
  exit 2
fi
for p in "$CKPT" "$DATA_DIR" "$GEN_SCRIPT"; do
  if [[ ! -e "$p" ]]; then
    echo "[FATAL] Missing path: $p"
    exit 2
  fi
done
if [[ -n "$OUT_VLNCE_GZ" && ! -e "$VLNCE_TEMPLATE" ]]; then
  echo "[FATAL] OUT_VLNCE_GZ needs a valid VLNCE_TEMPLATE (.json.gz), got: ${VLNCE_TEMPLATE:-<unset>}"
  exit 2
fi

COMMON_ARGS=(
  --model_dir "$CKPT"
  --data_dir "$DATA_DIR"
  --vp_mode "$VP_MODE"
  --frame_source "$FRAME_SOURCE"
  --granularity "$GRANULARITY"
  --dataset_type "$DATASET_TYPE"
  --max_frames "$MAX_FRAMES"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --num_samples "$NUM_SAMPLES"
)
[[ -n "$MODEL_DIR" ]] && COMMON_ARGS+=(--processor_dir "$MODEL_DIR")

mkdir -p "$(dirname "$OUT_JSON")"

IFS=',' read -r -a GPUS <<< "$GEN_CUDA"
NUM_SHARDS="${#GPUS[@]}"

echo "=========================================="
echo "VTInstructor instruction generation (augmentation)"
echo "  ckpt:    $CKPT"
echo "  data:    $DATA_DIR"
echo "  out:     $OUT_JSON"
echo "  vp_mode: $VP_MODE   frames: $FRAME_SOURCE   unit: $GRANULARITY"
echo "  gpus:    $GEN_CUDA  (${NUM_SHARDS} shard(s))"
echo "=========================================="

if [[ "$NUM_SHARDS" -le 1 ]]; then
  EXTRA=()
  if [[ -n "$OUT_VLNCE_GZ" ]]; then
    EXTRA+=(--vlnce_template "$VLNCE_TEMPLATE" --out_vlnce_gz "$OUT_VLNCE_GZ")
  fi
  CUDA_VISIBLE_DEVICES="$GEN_CUDA" "$PY" "$GEN_SCRIPT" \
    "${COMMON_ARGS[@]}" --out_json "$OUT_JSON" "${EXTRA[@]}"
  echo "[AUG] Done: $OUT_JSON"
  exit 0
fi

SHARD_DIR="${OUT_JSON}.shards"
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/shard_*.json

pids=()
for idx in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$idx]}" "$PY" "$GEN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --out_json "${SHARD_DIR}/shard_${idx}.json" \
    --num_shards "$NUM_SHARDS" --shard_idx "$idx" &
  pids+=("$!")
  sleep 2
done

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[ERROR] Shard $i (GPU ${GPUS[$i]}) failed."
    failed=$((failed + 1))
  fi
done
if [[ "$failed" -gt 0 ]]; then
  echo "[FATAL] ${failed} shard(s) failed; not merging partial results."
  exit 1
fi

MERGE_ARGS=(--merge_shards "$SHARD_DIR" --out_json "$OUT_JSON")
if [[ -n "$OUT_VLNCE_GZ" ]]; then
  MERGE_ARGS+=(--vlnce_template "$VLNCE_TEMPLATE" --out_vlnce_gz "$OUT_VLNCE_GZ")
fi
"$PY" "$GEN_SCRIPT" --model_dir "$CKPT" --data_dir "$DATA_DIR" "${MERGE_ARGS[@]}"

echo "[AUG] Done: $OUT_JSON"
