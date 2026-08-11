#!/usr/bin/env bash

# Guard against running with /bin/sh (dash), which cannot parse bash-specific syntax.
if [ -z "${BASH_VERSION:-}" ]; then
  echo "[FATAL] This script requires bash. Please run: bash vp_adapter/run_vp_sft_pipeline.sh"
  exit 2
fi

set -euo pipefail

# ============================================================================
# VP-Adapter v3.0 SFT Pipeline: SFT (with VP-Adapter v3.0) → Evaluate
#
# v3.0 changes: spatial modulation replaces cross-attention, vp_dim 384, layer 7 only
#
#   Stage 1: SFT with VP-Adapter v2.0 (R2RCE + RXRCE 混合训练)
#   Stage 2: 评测 (R2RCE val_unseen multiref)
#   Stage 3: 评测 (RXRCE val_unseen multiref)
#   Stage 4: 打分 (BLEU / ROUGE-L / CIDEr / METEOR)
#
# 用法：
#   bash run_vp_sft_pipeline.sh                              # 完整运行
#   SKIP_TRAIN=1 bash run_vp_sft_pipeline.sh                 # 跳过训练
#   SKIP_EVAL=1 bash run_vp_sft_pipeline.sh                  # 跳过评测
#   EVAL_ONLY=1 EVAL_CKPT=/path/to/ckpt bash run_vp_sft_pipeline.sh  # 仅评测
# ============================================================================

PY="${PY:-python}"
VP_TRANSFORMERS_VERSION="${VP_TRANSFORMERS_VERSION:-4.57.6}"

# ── 将所有缓存 / 临时文件重定向到 /mnt，避免写满 /home ──
_MNT_CACHE="${VT_CACHE_DIR:-${HOME}/.cache/vt-instructor}"
mkdir -p "$_MNT_CACHE"
export XDG_CACHE_HOME="$_MNT_CACHE"
export TMPDIR="${VT_TMP_DIR:-${_MNT_CACHE}/tmp}"; mkdir -p "$TMPDIR"
export TEMP="$TMPDIR" TMP="$TMPDIR"
export HF_HOME="$_MNT_CACHE/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TORCH_HOME="$_MNT_CACHE/torch"
export TORCH_EXTENSIONS_DIR="$_MNT_CACHE/torch_extensions"
export TRITON_CACHE_DIR="$_MNT_CACHE/triton"
export NUMBA_CACHE_DIR="$_MNT_CACHE/numba"
export PIP_CACHE_DIR="$_MNT_CACHE/pip"
export MPLCONFIGDIR="$_MNT_CACHE/matplotlib"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${SRC_DIR:-${PROJECT_DIR}/common}"

# ── 模型路径 ──
MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-VL-8B-Instruct}"

# ── 数据路径 ──
R2RCE_TRAIN_DATA_DIR="${R2RCE_TRAIN_DATA_DIR:-${PROJECT_DIR}/data/R2RCE_visual/r2rce_train_visual}"
RXRCE_TRAIN_DATA_DIR="${RXRCE_TRAIN_DATA_DIR:-${PROJECT_DIR}/data/RXRCE_visual/rxrce_train_visual}"
R2RCE_EVAL_DATA_DIR="${R2RCE_EVAL_DATA_DIR:-${PROJECT_DIR}/data/R2RCE_visual/r2rce_valunseen_visual}"
RXRCE_EVAL_DATA_DIR="${RXRCE_EVAL_DATA_DIR:-${PROJECT_DIR}/data/RXRCE_visual/rxrce_valunseen_visual}"
# filtered_ult_gt6.json 随仓库发布（rendering/），也可用 data/ 下自己重新生成的版本
FILTERED_JSON="${FILTERED_JSON:-${PROJECT_DIR}/rendering/filtered_ult_gt6.json}"

# ── 评测参考文件 ──
# R2R_val_unseen.json 随仓库发布（metrics/），是 R2R 官方标注筛到 VLN-CE val_unseen 的 613 条轨迹
R2R_VAL_UNSEEN_JSON="${R2R_VAL_UNSEEN_JSON:-${PROJECT_DIR}/metrics/R2R_val_unseen.json}"
R2RCE_VAL_JSON_GZ="${R2RCE_VAL_JSON_GZ:-${PROJECT_DIR}/data/R2RCE/val_unseen.json.gz}"

# ── 输出目录 ──
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/nig_vp_adapter_v3.1_sft}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"

# ── 训练/评测脚本 ──
TRAIN_SCRIPT="${SCRIPT_DIR}/train_sft_with_vp_adapter.py"
EVAL_SCRIPT="${SCRIPT_DIR}/nig_eval_vp.py"
SCORE_SCRIPT="${SCORE_SCRIPT:-${PROJECT_DIR}/metrics/nig_eval_r2rce_landmarks_from_json.py}"

# ── GPU 设置 ──
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EVAL_CUDA="${EVAL_CUDA:-0,1,2,3,4,5,6,7}"

# ── 控制开关 ──
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_SCORE="${SKIP_SCORE:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
KEEP_EVAL_SHARDS="${KEEP_EVAL_SHARDS:-0}"

# ── SFT 超参 ──
BATCH_SIZE=1
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-96}"
GRAD_ACCUM="${GRAD_ACCUM:-$(( TARGET_GLOBAL_BATCH / (NPROC_PER_NODE * BATCH_SIZE) ))}"
NUM_EPOCHS=3
LR=3e-5
VP_MODULE_LR=5e-4
WARMUP_RATIO=0.1
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
MAX_FRAMES=32

# ── 评测参数 ──
MAX_NEW_TOKENS=150
R2RCE_EVAL_TEMPERATURE="${R2RCE_EVAL_TEMPERATURE:-0.0}"
RXRCE_EVAL_TEMPERATURE="${RXRCE_EVAL_TEMPERATURE:-1.0}"
RXRCE_SCORE_MAX_WORDS="${RXRCE_SCORE_MAX_WORDS:-150}"

# ============================================================================
# Pre-checks
# ============================================================================

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[FATAL] Python not found: $PY"
  exit 127
fi

PYCOCOEVALCAP_PARENT="${PYCOCOEVALCAP_PARENT:-${PROJECT_DIR}/third_party/C-Instructor}"
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="${SCRIPT_DIR}:${SRC_DIR}:${PYCOCOEVALCAP_PARENT}:${PYTHONPATH:-}"
echo "[INFO] SCRIPT_DIR=$SCRIPT_DIR"
echo "[INFO] PYTHONPATH=$PYTHONPATH"

# ===========================================================================
# Hard dependency: transformers >= 4.51 (for Qwen3VLForConditionalGeneration)
# ===========================================================================
CURRENT_TF_VER=$("$PY" -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "0.0.0")
echo "[DEP] Current transformers: ${CURRENT_TF_VER}, required: ${VP_TRANSFORMERS_VERSION}"
if [[ "$CURRENT_TF_VER" != "$VP_TRANSFORMERS_VERSION" ]]; then
  echo "[DEP] Upgrading transformers ${CURRENT_TF_VER} → ${VP_TRANSFORMERS_VERSION} ..."
  "$PY" -m pip install -q "transformers==${VP_TRANSFORMERS_VERSION}" 2>&1 | tail -3
  echo "[DEP] Also ensuring compatible accelerate ..."
  "$PY" -m pip install -q "accelerate>=1.2.0" 2>&1 | tail -3
  echo "[DEP] transformers upgrade done."
else
  echo "[DEP] transformers version OK."
fi

# ---- CUDA / NCCL environment for bare-metal multi-GPU ----
echo "[DEP] /dev/shm: $(df -h /dev/shm 2>/dev/null | awk 'NR==2{print $2"/"$4" avail"}' || echo 'N/A')"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_ASYNC_ERROR_HANDLING=1
echo "[DEP] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

# ---- Sanity check ----
echo "=========================================="
echo "[DEP] Sanity checks"
echo "=========================================="
"$PY" - <<'PYCHECK'
import sys, importlib, shutil

fail = False

def check(label, fn):
    global fail
    try:
        result = fn()
        print(f"  [OK]   {label}: {result}")
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        fail = True

check("torch",        lambda: __import__("torch").__version__)
check("CUDA",         lambda: "available" if __import__("torch").cuda.is_available() else (_ for _ in ()).throw(RuntimeError("No CUDA")))
check("GPU count",    lambda: __import__("torch").cuda.device_count())
check("transformers", lambda: __import__("transformers").__version__)
check("accelerate",   lambda: __import__("accelerate").__version__)
check("Qwen3VLForConditionalGeneration", lambda: (
    getattr(__import__("transformers", fromlist=["Qwen3VLForConditionalGeneration"]),
            "Qwen3VLForConditionalGeneration"),
    "importable")[1])
check("numpy",        lambda: __import__("numpy").__version__)
check("PIL",          lambda: __import__("PIL").__version__)
check("tqdm",         lambda: __import__("tqdm").__version__)
check("pycocoevalcap", lambda: (
    __import__("pycocoevalcap.bleu.bleu", fromlist=["Bleu"]),
    "available")[1])

java_ok = shutil.which("java") is not None
check("java", lambda: "found" if java_ok else "NOT FOUND (METEOR will use fallback)")

if fail:
    print("\n[WARN] Some checks failed. Pipeline may still work if failures are optional.")
else:
    print("\n[DEP] All sanity checks passed!")
PYCHECK

JAVA_READY=0
ensure_java() {
  if command -v java &>/dev/null; then
    JAVA_READY=1
    return 0
  fi
  local conda_root="${PY%/envs/*/bin/python}"
  local conda_bin="${conda_root}/bin/conda"
  local conda_env="$(basename "$(dirname "$(dirname "$PY")")")"
  if [[ -x "$conda_bin" ]]; then
    echo "[INFO] Installing OpenJDK via conda ..."
    "$conda_bin" install -y -n "$conda_env" -c conda-forge openjdk || true
  fi
  export PATH="$(dirname "$PY"):$PATH"
  if command -v java &>/dev/null; then JAVA_READY=1; fi
}
ensure_java

if [[ "$EVAL_ONLY" == "1" ]]; then
  SKIP_TRAIN=1
  SKIP_EVAL=0
fi

# ── Path pre-checks, scoped to the stages that will actually run ──
# Inference-only users (EVAL_ONLY=1 / SKIP_TRAIN=1) do not need the ~44 GB of
# training data, so those paths are only required when training.
REQUIRED_PATHS=("$MODEL_DIR")
if [[ "$SKIP_TRAIN" != "1" ]]; then
  REQUIRED_PATHS+=("$R2RCE_TRAIN_DATA_DIR" "$RXRCE_TRAIN_DATA_DIR" "$FILTERED_JSON")
fi
if [[ "$SKIP_EVAL" != "1" ]]; then
  REQUIRED_PATHS+=("$R2RCE_EVAL_DATA_DIR" "$RXRCE_EVAL_DATA_DIR")
fi

for p in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "$p" ]]; then
    echo "[FATAL] Missing path: $p"
    exit 2
  fi
done

if [[ "$SKIP_TRAIN" == "1" && "$SKIP_EVAL" != "1" ]]; then
  if [[ -z "${EVAL_CKPT:-}" ]]; then
    echo "[FATAL] Training is skipped but EVAL_CKPT is not set."
    echo "        Point it at a checkpoint directory, e.g."
    echo "        EVAL_CKPT=/path/to/nig_rl_vp_v3.4 bash vp_adapter/run_vp_sft_pipeline.sh"
    exit 2
  fi
  if [[ ! -d "$EVAL_CKPT" ]]; then
    echo "[FATAL] EVAL_CKPT is not a directory: $EVAL_CKPT"
    exit 2
  fi
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "=========================================="
echo "VP-Adapter v3.0 SFT Pipeline (R2RCE + RXRCE)"
echo "=========================================="
echo "Stage 1 - SFT Train:       $([ "$SKIP_TRAIN" == "1" ] && echo 'SKIP' || echo 'RUN')"
echo "Stage 2 - Eval R2RCE:      $([ "$SKIP_EVAL" == "1" ] && echo 'SKIP' || echo 'RUN')"
echo "Stage 3 - Eval RXRCE:      $([ "$SKIP_EVAL" == "1" ] && echo 'SKIP' || echo 'RUN')"
echo "Stage 4 - Score:           $([ "$SKIP_SCORE" == "1" ] && echo 'SKIP' || echo 'AUTO')"
echo "------------------------------------------"
echo "GPU数量:          ${NPROC_PER_NODE}"
echo "目标全局 batch:   ${TARGET_GLOBAL_BATCH}"
echo "单卡 batch size:  ${BATCH_SIZE}"
echo "梯度累积步数:     ${GRAD_ACCUM}"
echo "Epochs:           ${NUM_EPOCHS}"
echo "LR (LM):          ${LR}"
echo "LR (VP modules):  ${VP_MODULE_LR}"
echo "Save steps:       ${SAVE_STEPS}"
echo "Save total limit: ${SAVE_TOTAL_LIMIT}"
echo "Batch(global):    $((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM))"
echo "Java ready:       ${JAVA_READY}"
echo "=========================================="

# ============================================================================
# Utility functions
# ============================================================================

gpu_cleanup() {
  echo "[CLEANUP] Releasing GPU memory between stages ..."
  "$PY" -c "import torch, gc; gc.collect(); [torch.cuda.empty_cache() for _ in range(1) if torch.cuda.is_available()]" 2>/dev/null || true
  sleep 3
  echo "[CLEANUP] Done."
}

latest_ckpt() {
  local d="$1"
  ls -1d "$d"/checkpoint-* 2>/dev/null \
    | sed -E 's@.*/checkpoint-([0-9]+)$@\1 \0@' \
    | sort -n | tail -1 | awk '{print $2}'
}

prune_checkpoints() {
  local d="$1"
  local keep
  keep="$(latest_ckpt "$d" || true)"
  if [[ -z "${keep:-}" ]]; then return 0; fi
  shopt -s nullglob
  for ck in "$d"/checkpoint-*; do
    if [[ "$ck" != "$keep" ]]; then rm -rf "$ck"; fi
  done
  shopt -u nullglob
  echo "[INFO] Kept latest checkpoint: $keep"
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
    --model_dir "$CKPT" --processor_dir "$MODEL_DIR"
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

# ============================================================================
# Stage 1: SFT with VP-Adapter
# ============================================================================

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo ""
  echo "=========================================="
  echo "Stage 1: SFT with VP-Adapter v3.0 (${NPROC_PER_NODE} GPU)"
  echo "=========================================="

  RESUME_CKPT="${RESUME_CKPT:-}"
  if [[ -z "${RESUME_CKPT}" ]]; then
    RESUME_CKPT="$(latest_ckpt "$OUT_DIR" || true)"
  fi
  if [[ -n "${RESUME_CKPT}" ]]; then
    echo "[INFO] Auto resume from latest checkpoint: ${RESUME_CKPT}"
  else
    echo "[INFO] No existing checkpoint found, start from base model."
  fi

  "$PY" -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" "$TRAIN_SCRIPT" \
    --model_name "$MODEL_DIR" \
    --processor_name "$MODEL_DIR" \
    --r2rce_data_dir "$R2RCE_TRAIN_DATA_DIR" \
    --rxrce_data_dir "$RXRCE_TRAIN_DATA_DIR" \
    --filtered_json "$FILTERED_JSON" \
    --output_dir "$OUT_DIR" \
    --logging_dir "$LOG_DIR" \
    --max_frames $MAX_FRAMES \
    --max_samples 0 \
    --max_length 0 \
    --shuffle \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --vp_module_lr $VP_MODULE_LR \
    --vp_dim 384 \
    --adapter_layers "7" \
    --lr_scheduler_type cosine \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay 0.01 \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --logging_steps 10 \
    --no_freeze_vit

  echo "[INFO] SFT with VP-Adapter complete!"
  gpu_cleanup
else
  echo "[INFO] Skipping Stage 1 (SKIP_TRAIN=1)"
fi

# ── 确定评测 checkpoint ──
if [[ -n "${EVAL_CKPT:-}" ]]; then
  CKPT="$EVAL_CKPT"
else
  CKPT="$(latest_ckpt "$OUT_DIR" || true)"
  if [[ -z "${CKPT:-}" ]]; then
    CKPT="$OUT_DIR"
  fi
fi

# ============================================================================
# Stage 2 & 3: Evaluation
# ============================================================================

if [[ "$SKIP_EVAL" != "1" ]]; then
  echo ""
  echo "=========================================="
  echo "Stage 2: Eval R2RCE val_unseen (VP-Adapter v3.0)"
  echo "=========================================="
  echo "[INFO] Using checkpoint: $CKPT"

  RUN_TS_R2R="$(date +%Y%m%d_%H%M%S)"
  PRED_JSON_R2R="${OUT_DIR}/nig_eval_vp_v3.1_r2rce_valunseen_${RUN_TS_R2R}.json"
  METRIC_JSON_R2R="${OUT_DIR}/nig_eval_vp_v3.1_r2rce_valunseen_${RUN_TS_R2R}_metrics.json"

  run_eval_sharded \
    r2r_multiref r2rce "$R2RCE_EVAL_DATA_DIR" "$PRED_JSON_R2R" "$R2RCE_EVAL_TEMPERATURE" \
    --r2r_val_json "$R2R_VAL_UNSEEN_JSON" \
    --vlnce_val_json_gz "$R2RCE_VAL_JSON_GZ"
  echo "[INFO] R2RCE predictions: ${PRED_JSON_R2R}"

  echo ""
  echo "=========================================="
  echo "Stage 3: Eval RXRCE val_unseen (VP-Adapter v3.0)"
  echo "=========================================="

  RUN_TS_RXR="$(date +%Y%m%d_%H%M%S)"
  PRED_JSON_RXR="${OUT_DIR}/nig_eval_vp_v3.1_rxrce_valunseen_${RUN_TS_RXR}.json"
  METRIC_JSON_RXR="${OUT_DIR}/nig_eval_vp_v3.1_rxrce_valunseen_${RUN_TS_RXR}_metrics.json"

  run_eval_sharded \
    rxr_multiref rxrce "$RXRCE_EVAL_DATA_DIR" "$PRED_JSON_RXR" "$RXRCE_EVAL_TEMPERATURE"
  echo "[INFO] RXRCE predictions: ${PRED_JSON_RXR}"

  gpu_cleanup

  # ================================================================
  # Stage 4: Score
  # ================================================================
  if can_score_json; then
    echo ""
    echo "=========================================="
    echo "Stage 4: Scoring prediction JSONs"
    echo "=========================================="

    if [[ -f "${PRED_JSON_R2R:-}" ]]; then
      "$PY" "$SCORE_SCRIPT" \
        --pred_json "$PRED_JSON_R2R" \
        --out_json "$METRIC_JSON_R2R" \
        --print_ref_stats
      echo "[INFO] R2RCE metrics: ${METRIC_JSON_R2R}"
    fi
    if [[ -f "${PRED_JSON_RXR:-}" ]]; then
      "$PY" "$SCORE_SCRIPT" \
        --pred_json "$PRED_JSON_RXR" \
        --out_json "$METRIC_JSON_RXR" \
        --max_caption_words "$RXRCE_SCORE_MAX_WORDS"
      echo "[INFO] RXRCE metrics: ${METRIC_JSON_RXR}"
    fi
  else
    echo "[WARN] Scoring deps not ready (or SKIP_SCORE=1), skip scoring."
  fi
else
  echo "[INFO] Skipping evaluation (SKIP_EVAL=1)"
fi

echo ""
echo "=========================================="
echo "[DONE] VP-Adapter v3.0 SFT pipeline complete!"
echo "=========================================="
