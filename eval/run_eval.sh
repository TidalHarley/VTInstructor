#!/usr/bin/env bash
# Benchmark inference + NLG scoring.
#
#   EVAL_CKPT=/path/to/checkpoint bash eval/run_eval.sh
#
# Step 1: eval/infer.py writes a JSON list of {prediction, references, ...}
# Step 2: metrics/evaluate.py reads that JSON and writes BLEU/METEOR/ROUGE-L/CIDEr/SPICE
#
# This is the only official scoring path. generate/ is data augmentation and
# does not produce a file metrics/evaluate.py can score.
if [ -z "${BASH_VERSION:-}" ]; then
  echo "[FATAL] This script requires bash. Please run: bash eval/run_eval.sh"
  exit 2
fi

set -euo pipefail

PY="${PY:-python}"
VP_TRANSFORMERS_VERSION="${VP_TRANSFORMERS_VERSION:-4.57.6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-VL-8B-Instruct}"
R2RCE_EVAL_DATA_DIR="${R2RCE_EVAL_DATA_DIR:-${PROJECT_DIR}/data/R2RCE_visual/r2rce_valunseen_visual}"
RXRCE_EVAL_DATA_DIR="${RXRCE_EVAL_DATA_DIR:-${PROJECT_DIR}/data/RXRCE_visual/rxrce_valunseen_visual}"
R2R_VAL_UNSEEN_JSON="${R2R_VAL_UNSEEN_JSON:-${PROJECT_DIR}/metrics/R2R_val_unseen.json}"
R2RCE_VAL_JSON_GZ="${R2RCE_VAL_JSON_GZ:-${PROJECT_DIR}/data/R2RCE/val_unseen.json.gz}"

OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/eval}"
EVAL_SCRIPT="${SCRIPT_DIR}/infer.py"
SCORE_SCRIPT="${SCORE_SCRIPT:-${PROJECT_DIR}/metrics/evaluate.py}"

EVAL_CUDA="${EVAL_CUDA:-0,1,2,3,4,5,6,7}"
SKIP_SCORE="${SKIP_SCORE:-0}"
KEEP_EVAL_SHARDS="${KEEP_EVAL_SHARDS:-0}"
MAX_FRAMES="${MAX_FRAMES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-150}"
R2RCE_EVAL_TEMPERATURE="${R2RCE_EVAL_TEMPERATURE:-0.0}"
RXRCE_EVAL_TEMPERATURE="${RXRCE_EVAL_TEMPERATURE:-1.0}"
RXRCE_SCORE_MAX_WORDS="${RXRCE_SCORE_MAX_WORDS:-150}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[FATAL] Python not found: $PY"
  exit 127
fi

if [[ -z "${EVAL_CKPT:-}" ]]; then
  echo "[FATAL] EVAL_CKPT is not set."
  echo "        Point it at a checkpoint directory, e.g."
  echo "        EVAL_CKPT=/path/to/nig_rl_vp_v3.4 bash eval/run_eval.sh"
  exit 2
fi
if [[ ! -d "$EVAL_CKPT" ]]; then
  echo "[FATAL] EVAL_CKPT is not a directory: $EVAL_CKPT"
  exit 2
fi

PYCOCOEVALCAP_PARENT="${PYCOCOEVALCAP_PARENT:-${PROJECT_DIR}/third_party/C-Instructor}"
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="${PROJECT_DIR}:${PYCOCOEVALCAP_PARENT}:${PYTHONPATH:-}"

CURRENT_TF_VER=$("$PY" -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "0.0.0")
echo "[DEP] Current transformers: ${CURRENT_TF_VER}, required: ${VP_TRANSFORMERS_VERSION}"
if [[ "$CURRENT_TF_VER" != "$VP_TRANSFORMERS_VERSION" ]]; then
  echo "[DEP] Upgrading transformers ${CURRENT_TF_VER} → ${VP_TRANSFORMERS_VERSION} ..."
  "$PY" -m pip install -q "transformers==${VP_TRANSFORMERS_VERSION}" 2>&1 | tail -3
  "$PY" -m pip install -q "accelerate>=1.2.0" 2>&1 | tail -3
else
  echo "[DEP] transformers version OK."
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

for p in "$EVAL_CKPT" "$MODEL_DIR" "$R2RCE_EVAL_DATA_DIR" "$RXRCE_EVAL_DATA_DIR" "$EVAL_SCRIPT" "$R2R_VAL_UNSEEN_JSON"; do
  if [[ ! -e "$p" ]]; then
    echo "[FATAL] Missing path: $p"
    exit 2
  fi
done

mkdir -p "$OUT_DIR"

JAVA_READY=0
if command -v java &>/dev/null; then
  JAVA_READY=1
  echo "[INFO] Java found: $(java -version 2>&1 | head -1)"
else
  echo "[WARN] Java not found — METEOR/SPICE may be skipped or fall back."
fi

gpu_cleanup() {
  echo "[CLEANUP] Releasing GPU memory ..."
  "$PY" -c "import torch, gc; gc.collect(); [torch.cuda.empty_cache() for _ in range(1) if torch.cuda.is_available()]" 2>/dev/null || true
  sleep 3
}

can_score_json() {
  if [[ "$SKIP_SCORE" == "1" ]]; then return 1; fi
  if [[ ! -f "$SCORE_SCRIPT" ]]; then
    echo "[WARN] Score script missing: $SCORE_SCRIPT"
    return 1
  fi
  "$PY" - <<'PY' >/dev/null 2>&1
import importlib.util, shutil
ok = importlib.util.find_spec("pycocoevalcap") is not None and shutil.which("java") is not None
raise SystemExit(0 if ok else 1)
PY
}

count_eval_gpus() {
  awk -F',' '{print NF}' <<< "$1"
}

run_eval_sharded() {
  local eval_mode="$1" dataset_type="$2" data_dir="$3" out_json="$4" temperature="$5"
  shift 5
  local extra_args=("$@")
  local num_shards
  num_shards="$(count_eval_gpus "$EVAL_CUDA")"

  local common_args=(
    --model_dir "$EVAL_CKPT" --processor_dir "$MODEL_DIR"
    --data_dir "$data_dir" --out_json "__PLACEHOLDER__"
    --max_frames $MAX_FRAMES
    --max_new_tokens $MAX_NEW_TOKENS
    --eval_mode "$eval_mode" --dataset_type "$dataset_type"
    --temperature "$temperature"
  )

  if [[ "$num_shards" -le 1 ]]; then
    common_args=("${common_args[@]/__PLACEHOLDER__/$out_json}")
    CUDA_VISIBLE_DEVICES="$EVAL_CUDA" "$PY" "$EVAL_SCRIPT" \
      "${common_args[@]}" "${extra_args[@]}"
    return 0
  fi

  local shard_dir="${out_json}.shards"
  mkdir -p "$shard_dir"
  rm -f "$shard_dir"/shard_*.json

  local pids=() gpus=() idx=0
  IFS=',' read -r -a gpus <<< "$EVAL_CUDA"
  for gpu in "${gpus[@]}"; do
    local shard_args=("${common_args[@]/__PLACEHOLDER__/${shard_dir}/shard_${idx}.json}")
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EVAL_SCRIPT" \
      "${shard_args[@]}" \
      --num_shards "$num_shards" --shard_idx "$idx" "${extra_args[@]}" &
    pids+=("$!")
    idx=$((idx + 1))
    sleep 2
  done

  local failed_indices=()
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      failed_indices+=("$i")
      echo "[WARN] Shard $i (GPU ${gpus[$i]}) failed, will retry."
    fi
  done

  if [[ ${#failed_indices[@]} -gt 0 ]]; then
    echo "[INFO] Retrying ${#failed_indices[@]} failed shard(s) sequentially..."
    local still_failed=0
    for i in "${failed_indices[@]}"; do
      echo "[RETRY] Shard $i on GPU ${gpus[$i]} ..."
      local retry_args=("${common_args[@]/__PLACEHOLDER__/${shard_dir}/shard_${i}.json}")
      if ! CUDA_VISIBLE_DEVICES="${gpus[$i]}" "$PY" "$EVAL_SCRIPT" \
            "${retry_args[@]}" \
            --num_shards "$num_shards" --shard_idx "$i" "${extra_args[@]}"; then
        echo "[ERROR] Shard $i failed again on retry."
        still_failed=$((still_failed + 1))
      else
        echo "[OK] Shard $i succeeded on retry."
      fi
    done
    if [[ "$still_failed" -gt 0 ]]; then
      echo "[WARN] $still_failed shard(s) failed even after retry, merging available results."
    fi
  fi

  "$PY" - "$shard_dir" "$out_json" <<'PY'
import json, os, sys
shard_dir, out_json = sys.argv[1], sys.argv[2]
files = sorted(
    os.path.join(shard_dir, x)
    for x in os.listdir(shard_dir)
    if x.startswith("shard_") and x.endswith(".json")
)
merged = []
for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        merged.extend(json.load(f))

def sort_key(rec):
    return (
        int(rec.get("trajectory_id", -1)) if rec.get("trajectory_id") is not None else -1,
        int(rec.get("episode_id", -1)) if rec.get("episode_id") is not None else -1,
    )

merged.sort(key=sort_key)
os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
print(f"[INFO] merged {len(merged)} records -> {out_json}")
PY

  if [[ "$KEEP_EVAL_SHARDS" != "1" ]]; then
    rm -rf "$shard_dir"
    echo "[INFO] Removed shard dir: $shard_dir"
  else
    echo "[INFO] Kept shard dir: $shard_dir"
  fi
}

CKPT="$EVAL_CKPT"
echo "=========================================="
echo "VTInstructor eval — infer then score"
echo "=========================================="
echo "Checkpoint:  $CKPT"
echo "Processor:   $MODEL_DIR"
echo "R2R-CE data: $R2RCE_EVAL_DATA_DIR"
echo "RxR-CE data: $RXRCE_EVAL_DATA_DIR"
echo "Refs:        $R2R_VAL_UNSEEN_JSON"
echo "Score script:$SCORE_SCRIPT"
echo "GPUs:        $EVAL_CUDA"
echo "Java ready:  $JAVA_READY"
echo "=========================================="

echo ""
echo "=========================================="
echo "Infer: R2R-CE val_unseen (r2r_multiref)"
echo "=========================================="
RUN_TS_R2R="$(date +%Y%m%d_%H%M%S)"
PRED_JSON_R2R="${OUT_DIR}/eval_r2rce_valunseen_${RUN_TS_R2R}.json"
METRIC_JSON_R2R="${OUT_DIR}/eval_r2rce_valunseen_${RUN_TS_R2R}_metrics.json"

R2R_EXTRA=(--r2r_val_json "$R2R_VAL_UNSEEN_JSON")
if [[ -e "$R2RCE_VAL_JSON_GZ" ]]; then
  R2R_EXTRA+=(--vlnce_val_json_gz "$R2RCE_VAL_JSON_GZ")
fi
run_eval_sharded \
  r2r_multiref r2rce "$R2RCE_EVAL_DATA_DIR" "$PRED_JSON_R2R" "$R2RCE_EVAL_TEMPERATURE" \
  "${R2R_EXTRA[@]}"
echo "[INFO] R2R-CE predictions: ${PRED_JSON_R2R}"

echo ""
echo "=========================================="
echo "Infer: RxR-CE val_unseen (rxr_multiref)"
echo "=========================================="
RUN_TS_RXR="$(date +%Y%m%d_%H%M%S)"
PRED_JSON_RXR="${OUT_DIR}/eval_rxrce_valunseen_${RUN_TS_RXR}.json"
METRIC_JSON_RXR="${OUT_DIR}/eval_rxrce_valunseen_${RUN_TS_RXR}_metrics.json"

run_eval_sharded \
  rxr_multiref rxrce "$RXRCE_EVAL_DATA_DIR" "$PRED_JSON_RXR" "$RXRCE_EVAL_TEMPERATURE"
echo "[INFO] RxR-CE predictions: ${PRED_JSON_RXR}"

gpu_cleanup

if can_score_json; then
  echo ""
  echo "=========================================="
  echo "Score prediction JSONs with metrics/evaluate.py"
  echo "=========================================="

  if [[ -f "${PRED_JSON_R2R:-}" ]]; then
    "$PY" "$SCORE_SCRIPT" \
      --pred_json "$PRED_JSON_R2R" \
      --out_json "$METRIC_JSON_R2R" \
      --print_ref_stats
    echo "[INFO] R2R-CE metrics: ${METRIC_JSON_R2R}"
  fi
  if [[ -f "${PRED_JSON_RXR:-}" ]]; then
    "$PY" "$SCORE_SCRIPT" \
      --pred_json "$PRED_JSON_RXR" \
      --out_json "$METRIC_JSON_RXR" \
      --max_caption_words "$RXRCE_SCORE_MAX_WORDS"
    echo "[INFO] RxR-CE metrics: ${METRIC_JSON_RXR}"
  fi
else
  echo "[WARN] Scoring deps not ready (or SKIP_SCORE=1). Predictions were written;"
  echo "       score later with:"
  echo "       $PY $SCORE_SCRIPT --pred_json $PRED_JSON_R2R --out_json ${METRIC_JSON_R2R}"
fi

echo ""
echo "[DONE] Eval finished."
echo "  R2R-CE pred: ${PRED_JSON_R2R}"
echo "  RxR-CE pred: ${PRED_JSON_RXR}"
