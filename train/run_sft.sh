#!/usr/bin/env bash
# Stage 3: VP-Adapter SFT (training only).
#
#   bash train/run_sft.sh
#
# Evaluation is a separate step: bash eval/run_eval.sh
if [ -z "${BASH_VERSION:-}" ]; then
  echo "[FATAL] This script requires bash. Please run: bash train/run_sft.sh"
  exit 2
fi

set -euo pipefail

PY="${PY:-python}"
VP_TRANSFORMERS_VERSION="${VP_TRANSFORMERS_VERSION:-4.57.6}"

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

MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-VL-8B-Instruct}"
R2RCE_TRAIN_DATA_DIR="${R2RCE_TRAIN_DATA_DIR:-${PROJECT_DIR}/data/R2RCE_visual/r2rce_train_visual}"
RXRCE_TRAIN_DATA_DIR="${RXRCE_TRAIN_DATA_DIR:-${PROJECT_DIR}/data/RXRCE_visual/rxrce_train_visual}"
FILTERED_JSON="${FILTERED_JSON:-${PROJECT_DIR}/preprocess/filtering/r2rce_train_filtered.json}"

OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/nig_vp_adapter_v3.1_sft}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
TRAIN_SCRIPT="${SCRIPT_DIR}/sft.py"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
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

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[FATAL] Python not found: $PY"
  exit 127
fi

export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
echo "[INFO] PROJECT_DIR=$PROJECT_DIR"
echo "[INFO] PYTHONPATH=$PYTHONPATH"

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

echo "[DEP] /dev/shm: $(df -h /dev/shm 2>/dev/null | awk 'NR==2{print $2"/"$4" avail"}' || echo 'N/A')"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_ASYNC_ERROR_HANDLING=1

echo "=========================================="
echo "[DEP] Sanity checks"
echo "=========================================="
"$PY" - <<'PYCHECK'
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

if fail:
    print("\n[WARN] Some checks failed. Training may still work if failures are optional.")
else:
    print("\n[DEP] All sanity checks passed!")
PYCHECK

for p in "$MODEL_DIR" "$R2RCE_TRAIN_DATA_DIR" "$RXRCE_TRAIN_DATA_DIR" "$FILTERED_JSON" "$TRAIN_SCRIPT"; do
  if [[ ! -e "$p" ]]; then
    echo "[FATAL] Missing path: $p"
    exit 2
  fi
done

mkdir -p "$OUT_DIR" "$LOG_DIR"

latest_ckpt() {
  local d="$1"
  ls -1d "$d"/checkpoint-* 2>/dev/null \
    | sed -E 's@.*/checkpoint-([0-9]+)$@\1 \0@' \
    | sort -n | tail -1 | awk '{print $2}'
}

echo "=========================================="
echo "VTInstructor Stage 3 — VP-Adapter SFT"
echo "=========================================="
echo "GPU count:        ${NPROC_PER_NODE}"
echo "Target global bs: ${TARGET_GLOBAL_BATCH}"
echo "Per-GPU batch:    ${BATCH_SIZE}"
echo "Grad accum:       ${GRAD_ACCUM}"
echo "Epochs:           ${NUM_EPOCHS}"
echo "LR (LM):          ${LR}"
echo "LR (VP modules):  ${VP_MODULE_LR}"
echo "Save steps:       ${SAVE_STEPS}"
echo "Batch(global):    $((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM))"
echo "Output:           ${OUT_DIR}"
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

echo "[DONE] SFT complete. Checkpoint: $OUT_DIR"
echo "[INFO] To score a checkpoint: EVAL_CKPT=<ckpt> bash ${PROJECT_DIR}/eval/run_eval.sh"
