#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."

# ── VP-Adapter 代码路径 ──
VP_ADAPTER_DIR="${VP_ADAPTER_DIR:-${PROJECT_DIR}/vp_adapter}"

# ── SFT Checkpoint（GRPO 的起点）──
SFT_CKPT="${SFT_CKPT:-${PROJECT_DIR}/outputs/nig_vp_adapter_v3.1_sft}"

# ── 基座模型 Processor ──
MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-VL-8B-Instruct}"

# ── 数据路径（带 VP masks 的 visual 版本）──
R2RCE_TRAIN_DATA_DIR="${R2RCE_TRAIN_DATA_DIR:-${PROJECT_DIR}/data/R2RCE_visual/r2rce_train_visual}"
RXRCE_TRAIN_DATA_DIR="${RXRCE_TRAIN_DATA_DIR:-${PROJECT_DIR}/data/RXRCE_visual/rxrce_train_visual}"
# filtered_ult_gt6.json 随仓库发布（rendering/）
FILTERED_JSON="${FILTERED_JSON:-${PROJECT_DIR}/rendering/filtered_ult_gt6.json}"
# 可选：R2R 官方 train 标注，用于补充多参考。留空也可以——同一轨迹的 3 条指令会从
# 各 episode 的 sample.json 里按 trajectory_id 自动聚合（实测 avg 3.1 refs/traj）
R2R_TRAIN_JSON="${R2R_TRAIN_JSON:-}"

# ── 输出路径 ──
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/nig_rl_vp_v3.3}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/outputs/nig_rl_vp_v3.3_logs}"

# ── RL 训练脚本 ──
RL_SCRIPT="${SCRIPT_DIR}/grpo_rl_train_vp.py"
DS_CONFIG="${SCRIPT_DIR}/ds_zero2_config.json"

# ── GPU 配置 ──
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# ── GRPO 超参数 ──
RL_NUM_EPOCHS="${RL_NUM_EPOCHS:-1}"
RL_LR="${RL_LR:-1e-6}"
RL_GROUP_SIZE="${RL_GROUP_SIZE:-8}"
RL_GRAD_ACCUM="${RL_GRAD_ACCUM:-4}"
RL_KL_BETA="${RL_KL_BETA:-0.04}"
RL_CLIP_EPS="${RL_CLIP_EPS:-0.2}"
RL_R2RCE_TEMPERATURE="${RL_R2RCE_TEMPERATURE:-0.5}"
RL_RXRCE_TEMPERATURE="${RL_RXRCE_TEMPERATURE:-1.0}"
RL_TOP_P="${RL_TOP_P:-0.9}"
RL_SAVE_STEPS="${RL_SAVE_STEPS:-0}"
RL_LOGGING_STEPS="${RL_LOGGING_STEPS:-5}"
RL_GEN_BATCH_SIZE="${RL_GEN_BATCH_SIZE:-16}"
RL_LP_BATCH_SIZE="${RL_LP_BATCH_SIZE:-8}"
RL_MAX_RXRCE="${RL_MAX_RXRCE:-5800}"
RL_MAX_R2RCE="${RL_MAX_R2RCE:-0}"
RL_MAX_SAMPLES="${RL_MAX_SAMPLES:-0}"
MAX_FRAMES="${MAX_FRAMES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-150}"

# ── Reward 权重（CIDEr 尺度归一化：CIDEr 值域 0-10，其余 0-1）──
RL_W_BLEU1="${RL_W_BLEU1:-0.25}"
RL_W_BLEU4="${RL_W_BLEU4:-0.25}"
RL_W_CIDER="${RL_W_CIDER:-0.05}"
RL_W_METEOR="${RL_W_METEOR:-0.25}"
RL_W_ROUGE_L="${RL_W_ROUGE_L:-0.20}"

# ── VP-Adapter 配置 ──
VP_ADAPTER_LAYERS="${VP_ADAPTER_LAYERS:-7}"
VP_DIM="${VP_DIM:-384}"

# ── VP-GRPO: Gate-only + Contrastive Gate Loss ──
GATE_ONLY="${GATE_ONLY:-1}"
GATE_CONTRAST_ALPHA="${GATE_CONTRAST_ALPHA:-0.01}"
GATE_CONTRAST_TAU="${GATE_CONTRAST_TAU:-1.0}"
GATE_LR_SCALE="${GATE_LR_SCALE:-0.1}"

# ── 其他选项 ──
RL_NO_REF_MODEL="${RL_NO_REF_MODEL:-0}"
RL_BATCHED_TRAIN="${RL_BATCHED_TRAIN:-1}"

# ── 检查路径 ──
if [[ ! -x "$PY" ]]; then
  echo "[FATAL] Python not found: $PY"
  exit 127
fi

for p in "$SFT_CKPT" "$R2RCE_TRAIN_DATA_DIR" "$RXRCE_TRAIN_DATA_DIR" "$FILTERED_JSON" "$MODEL_DIR" "$RL_SCRIPT" "$DS_CONFIG"; do
  if [[ ! -e "$p" ]]; then
    echo "[FATAL] Missing path: $p"
    exit 2
  fi
done

# R2R_TRAIN_JSON 是可选的，只有显式指定时才校验
if [[ -n "$R2R_TRAIN_JSON" && ! -e "$R2R_TRAIN_JSON" ]]; then
  echo "[FATAL] Missing path: $R2R_TRAIN_JSON"
  exit 2
fi

# ── 确保 Java（METEOR 需要）──
JAVA_READY=0
if command -v java &>/dev/null; then
  JAVA_READY=1
  echo "[INFO] Java found: $(java -version 2>&1 | head -1)"
else
  echo "[WARN] Java not found — METEOR will use fallback"
fi

# ── 环境变量 ──
PYCOCOEVALCAP_PARENT="${PYCOCOEVALCAP_PARENT:-${PROJECT_DIR}/third_party/C-Instructor}"
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="${SCRIPT_DIR}:${VP_ADAPTER_DIR}:${PYCOCOEVALCAP_PARENT}:${PYTHONPATH:-}"
export VP_ADAPTER_DIR

mkdir -p "$OUT_DIR" "$LOG_DIR"

# ── 打印配置 ──
echo "=========================================="
echo "VP-GRPO v3.3 for VP-Adapter"
echo "=========================================="
echo "SFT Checkpoint: $SFT_CKPT"
echo "Processor:      $MODEL_DIR"
echo "R2RCE data:     $R2RCE_TRAIN_DATA_DIR"
echo "RXRCE data:     $RXRCE_TRAIN_DATA_DIR"
echo "Filtered JSON:  $FILTERED_JSON"
echo "Output:         $OUT_DIR"
echo "VP adapter:     layers=${VP_ADAPTER_LAYERS}, dim=${VP_DIM}"
echo "------------------------------------------"
echo "GPUs:           ${NPROC_PER_NODE}"
echo "RL Epochs:      ${RL_NUM_EPOCHS}"
echo "RL LR:          ${RL_LR}"
echo "Group size:     ${RL_GROUP_SIZE}"
echo "Gen batch:      ${RL_GEN_BATCH_SIZE}"
echo "LP batch:       ${RL_LP_BATCH_SIZE}"
echo "Max RXRCE:      ${RL_MAX_RXRCE}"
echo "Batched train:  ${RL_BATCHED_TRAIN}"
echo "No ref model:   ${RL_NO_REF_MODEL}"
echo "KL beta:        ${RL_KL_BETA}"
echo "R2RCE temp:     ${RL_R2RCE_TEMPERATURE}"
echo "RXRCE temp:     ${RL_RXRCE_TEMPERATURE}"
echo "------------------------------------------"
echo "Reward weights: BLEU1=${RL_W_BLEU1} BLEU4=${RL_W_BLEU4} CIDEr=${RL_W_CIDER} METEOR=${RL_W_METEOR} ROUGE-L=${RL_W_ROUGE_L}"
echo "------------------------------------------"
echo "Gate-only:      ${GATE_ONLY}"
echo "Gate α:         ${GATE_CONTRAST_ALPHA}"
echo "Gate τ:         ${GATE_CONTRAST_TAU}"
echo "Gate LR scale:  ${GATE_LR_SCALE}"
echo "------------------------------------------"
echo "Java ready:     ${JAVA_READY}"
echo "=========================================="

# ── 构建额外参数 ──
RL_EXTRA_ARGS=()
if [[ "$RL_NO_REF_MODEL" == "1" ]]; then
  RL_EXTRA_ARGS+=(--no_ref_model)
fi
if [[ "$RL_BATCHED_TRAIN" == "1" ]]; then
  RL_EXTRA_ARGS+=(--batched_train_update)
fi
if [[ "$GATE_ONLY" == "1" ]]; then
  RL_EXTRA_ARGS+=(--gate_only)
  RL_EXTRA_ARGS+=(--gate_contrast_alpha "$GATE_CONTRAST_ALPHA")
  RL_EXTRA_ARGS+=(--gate_contrast_tau "$GATE_CONTRAST_TAU")
  RL_EXTRA_ARGS+=(--gate_lr_scale "$GATE_LR_SCALE")
fi

# ── 启动 GRPO RL ──
echo ""
echo "[INFO] Starting VP-GRPO v3.3 training ..."
"$PY" -m deepspeed.launcher.runner \
  --num_gpus="$NPROC_PER_NODE" \
  "$RL_SCRIPT" \
  --sft_checkpoint "$SFT_CKPT" \
  --processor_dir "$MODEL_DIR" \
  --r2rce_data_dir "$R2RCE_TRAIN_DATA_DIR" \
  --rxrce_data_dir "$RXRCE_TRAIN_DATA_DIR" \
  --filtered_json "$FILTERED_JSON" \
  --r2r_train_json "$R2R_TRAIN_JSON" \
  --output_dir "$OUT_DIR" \
  --logging_dir "$LOG_DIR" \
  --ds_config "$DS_CONFIG" \
  --max_frames $MAX_FRAMES \
  --max_new_tokens $MAX_NEW_TOKENS \
  --num_epochs "$RL_NUM_EPOCHS" \
  --lr "$RL_LR" \
  --grad_accum "$RL_GRAD_ACCUM" \
  --group_size "$RL_GROUP_SIZE" \
  --gen_batch_size "$RL_GEN_BATCH_SIZE" \
  --lp_batch_size "$RL_LP_BATCH_SIZE" \
  --clip_eps "$RL_CLIP_EPS" \
  --kl_beta "$RL_KL_BETA" \
  --r2rce_temperature "$RL_R2RCE_TEMPERATURE" \
  --rxrce_temperature "$RL_RXRCE_TEMPERATURE" \
  --top_p "$RL_TOP_P" \
  --save_steps "$RL_SAVE_STEPS" \
  --logging_steps "$RL_LOGGING_STEPS" \
  --w_bleu1 "$RL_W_BLEU1" \
  --w_bleu4 "$RL_W_BLEU4" \
  --w_cider "$RL_W_CIDER" \
  --w_meteor "$RL_W_METEOR" \
  --w_rouge_l "$RL_W_ROUGE_L" \
  --max_rxrce "$RL_MAX_RXRCE" \
  --max_r2rce "$RL_MAX_R2RCE" \
  --max_samples "$RL_MAX_SAMPLES" \
  --vp_adapter_layers "$VP_ADAPTER_LAYERS" \
  --vp_dim "$VP_DIM" \
  "${RL_EXTRA_ARGS[@]}"

echo ""
echo "[DONE] VP-GRPO v3.3 for VP-Adapter complete."
echo "Output: $OUT_DIR"
