# [ACMMM 2026] VTInstructor: Visual Trajectory Prompting for Navigation Instruction Generation in Continuous Environments

<p align="center">
  <img src="vt-instructor.png" alt="VTInstructor overview" width="90%"/>
</p>

## Introduction

This repository provides the full open-source pipeline for **VTInstructor**: generating natural-language navigation instructions for trajectories in continuous environments (R2R-CE, RxR-CE).

It covers:

- **Rendering** Habitat egocentric panoramas along VLN-CE trajectories
- **Visual Trajectory Prompt (VTP)** overlays and 3-channel semantic masks (ribbon / arrow / endpoint)
- **VP-Adapter SFT** on Qwen3-VL-8B-Instruct
- **VP-GRPO** gate-only RL refinement
- **Evaluation** with BLEU / CIDEr / METEOR / ROUGE-L
- **Inference for data augmentation**, with VTP rendering optional

```
Stage 1  Rendering      →  clean panorama keyframes (Habitat)
Stage 2  VP masks       →  overlay frames + 3-channel semantic masks
Stage 3  VP-Adapter SFT →  supervised fine-tuning
Stage 4  VP-GRPO        →  RL refinement (gate-only)
Stage 5  Generation     →  instructions for data augmentation (VTP optional)
```

## Contribution

- **The first VLN instruction generation framework for continuous environments.** VTInstructor generates instructions from ego-centric RGB trajectories paired with action sequences; the model itself receives only RGB frames as visual input, without navigation graphs, pre-built maps, or scene reconstruction.
- **A visual trajectory prompting framework for explicit spatial grounding.** We convert implicit trajectory geometry in dense RGB streams into explicit spatial cues through EDTC for navigation-critical keyframe selection, VTP for path/turn/goal prompting on these anchors, VTMod for trajectory-aware visual encoding, and VT-GRPO for reward-driven calibration of spatial signal injection.
- **State-of-the-art performance with practical utility.** VTInstructor achieves state-of-the-art results on the R2R-CE and RxR-CE Val Unseen benchmarks, surpassing the strongest baseline by +0.357 and +0.109 CIDEr, respectively, improving frozen-follower success by 14.7 percentage points, and delivering +3 SR-point data augmentation gains on both benchmarks.
- **Fully open-sourced codebase.** All code for rendering, VTP construction, VP-Adapter SFT, VP-GRPO, and evaluation is released in this repository.

## Repository layout

```
VTInstructor/
├── rendering/         # Stage 1: Habitat keyframes + instruction filter
│                      #   ships filtered_ult_gt6.json (9,264 R2R-CE train eps)
├── visual_prompt/     # VTP geometry / overlay library (used by Stage 2)
├── vp_adapter/        # Stage 2 masks + Stage 3 SFT + eval + generation
│   ├── render_vp_masks_from_samples.py   # ★ Stage 2 entry
│   ├── train_sft_with_vp_adapter.py      # ★ Stage 3 entry
│   ├── nig_eval_vp.py
│   ├── generate_instructions.py          # ★ inference for data augmentation
│   ├── run_generate_instructions.sh
│   └── run_vp_sft_pipeline.sh
├── vp_grpo/           # Stage 4 VP-GRPO
│   ├── grpo_rl_train_vp.py               # ★ Stage 4 entry
│   ├── run_grpo_vp_v3.1.sh, eval_vp_v3.1.sh
│   └── ds_zero2_config.json
├── common/            # Shared dataset / eval helpers
├── metrics/           # Scoring + R2R_val_unseen.json (613 traj × 3 refs)
├── environment/       # Exact SFT / GRPO pip freezes
├── ENVIRONMENT.md     # Two-env setup; DeepSpeed 0.14.4 pin
├── scripts/           # Example launcher
└── requirements.txt
```

**Shipped artifacts** (non-deterministic if regenerated):

| File | Role |
|------|------|
| `rendering/filtered_ult_gt6.json` | R2R-CE train filter for SFT / GRPO |
| `metrics/R2R_val_unseen.json` | Eval multi-references |

Checkpoints are not in git; point `SFT_CKPT` / `EVAL_CKPT` at your local weights.

## Pipeline

### Prerequisites

| Need | Stages |
|------|--------|
| Matterport3D scenes (`.glb` + `.navmesh`) | 1–2 |
| [habitat-sim](https://github.com/facebookresearch/habitat-sim) | 1–2 |
| Qwen3-VL-8B-Instruct (local HF snapshot) | 3–5 / eval |
| [pycocoevalcap](https://github.com/salaniz/pycocoevalcap) + JRE | GRPO reward + scoring |
| Envs from `environment/sft_env.txt` and `environment/grpo_env.txt` | 3 / 4 |

SFT and GRPO use **separate** Python envs. GRPO needs DeepSpeed **0.14.4** exactly — see [`ENVIRONMENT.md`](ENVIRONMENT.md).

Before the first run you can check the VP-Adapter wiring — encoder shapes, adapter registration, save/load round-trip, and gate gradient flow — against your local backbone:

```bash
MODEL_DIR=/path/to/Qwen3-VL-8B-Instruct python vp_adapter/test_pipeline.py
```

If Habitat fails with a CUDA/EGL device error:

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

### 1. Rendering — clean panoramas

```bash
# R2R-CE (paper defaults: 256×384 faces → 768×384 pano, camera_height=0.75)
python rendering/nig_render_dataset_r2rce_detail.py \
  --train_json /path/to/R2R_VLNCE_v1-3/train/train.json.gz \
  --scenes_root /path/to/scenes_root \
  --output_dir data/R2RCE_visual/r2rce_train_visual \
  --log_path logs/r2rce_train_base.log \
  --max_episodes 0 \
  --sample_mode event --frame_stride 2 \
  --width 256 --height 384 --camera_height 0.75 \
  --max_parts 5 \
  --panorama --panorama_hfov 90.0 --panorama_step 90.0

# RxR-CE
python rendering/nig_render_rxr_dataset.py \
  --train_json /path/to/RxR_VLNCE_v0/train/train_guide.json.gz \
  --scenes_root /path/to/scenes_root \
  --output_dir data/RXRCE_visual/rxrce_train_visual \
  --log_path logs/rxrce_train_base.log \
  --width 256 --height 384 --camera_height 0.75 \
  --panorama --panorama_hfov 90.0 --panorama_step 90.0
```

`rendering/run_render_r2rce.sh` and `rendering/run_render_rxrce.sh` wrap the two commands above with the paper geometry and render both `train` and `val_unseen` into the four canonical directories used by every later stage. Set `DATASET_ROOT` (or `TRAIN_JSON` / `VAL_UNSEEN_JSON` / `SCENES_ROOT`) to point them at your data.

Optional: regenerate the R2R filter with `bash rendering/run_gpt_filter_r2rce_train.sh` (API key required). The paper ships `rendering/filtered_ult_gt6.json`.

### 2. VP masks

**Use this as the Stage-2 entry** — it writes overlays **and** `vp_masks`, editing each episode directory in place. Do **not** use `visual_prompt/render_with_vp*.py` here (no masks → zero-mask training).

Run it on **all four** splits, `val_unseen` included: the VP-Adapter consumes `vp_masks` at inference exactly as it does during training, so reproducing the reported numbers needs masks on the eval splits too.

```bash
for split in \
  data/R2RCE_visual/r2rce_train_visual \
  data/R2RCE_visual/r2rce_valunseen_visual \
  data/RXRCE_visual/rxrce_train_visual \
  data/RXRCE_visual/rxrce_valunseen_visual
do
  python vp_adapter/render_vp_masks_from_samples.py \
    --data_dir "$split" \
    --scenes_root /path/to/scenes_root \
    --width 256 --height 384 --camera_height 0.75 \
    --hfov 90.0 --panorama_step 90.0
done
```

If masks are missing the code falls back to all-zero masks and prints a `[VP]` line naming the split. During **training** that is a `WARN` and almost certainly means Stage 2 was skipped. During **evaluation or inference** it is an `INFO`: running without visual trajectory prompting is a supported mode, it just measures the no-VTP setting. `EVAL_ZERO_VP=1` requests that ablation explicitly.

For data augmentation this stage is optional — see [Inference for data augmentation](#5-inference-for-data-augmentation).

### 3. VP-Adapter SFT

```bash
export MODEL_DIR=/path/to/Qwen3-VL-8B-Instruct
export R2RCE_TRAIN_DATA_DIR=data/R2RCE_visual/r2rce_train_visual
export RXRCE_TRAIN_DATA_DIR=data/RXRCE_visual/rxrce_train_visual
export R2RCE_EVAL_DATA_DIR=data/R2RCE_visual/r2rce_valunseen_visual
export RXRCE_EVAL_DATA_DIR=data/RXRCE_visual/rxrce_valunseen_visual
bash vp_adapter/run_vp_sft_pipeline.sh
```

All five paths already default to the layout Stage 1 produces, so the exports are only needed if your data lives elsewhere. Uses `torch.distributed.run` (HF Trainer + DDP); `FILTERED_JSON` / `R2R_VAL_UNSEEN_JSON` default to the shipped files.

To evaluate an existing checkpoint without any training data on disk:

```bash
EVAL_ONLY=1 EVAL_CKPT=/path/to/checkpoint bash vp_adapter/run_vp_sft_pipeline.sh
```

### 4. VP-GRPO + eval

```bash
# train (gate-only GRPO on the SFT checkpoint)
export MODEL_DIR=/path/to/Qwen3-VL-8B-Instruct
export SFT_CKPT=/path/to/outputs/nig_vp_adapter_v3.1_sft/checkpoint-NNNN
bash vp_grpo/run_grpo_vp_v3.1.sh

# eval
EVAL_CKPT=/path/to/rl/best bash vp_grpo/eval_vp_v3.1.sh
```

`R2R_TRAIN_JSON` is optional; without it, GRPO aggregates multi-refs by `trajectory_id`.

### 5. Inference for data augmentation

If you want to use VTInstructor for data augmentation, **VTP rendering is optional**. The model supports instruction generation from RGB renders alone, because the VTP contextual information is already learned into the weights during training. Supplying VTP renders (Stage 2) still gives more accurate instructions, so use them when you can.

Generation runs on any directory of rendered trajectories in the Stage-1 layout (`episode_*/sample.json`), and `DATASET_TYPE` picks the output style: `r2rce` for R2R-style instructions (25–40 words, 2–3 landmarks), `rxrce` for RxR-style narration (35–60 words, richer spatial detail).

```bash
# R2R-style instructions for a rendered split
CKPT=/path/to/checkpoint \
MODEL_DIR=/path/to/Qwen3-VL-8B-Instruct \
DATA_DIR=/path/to/rendered_split \
DATASET_TYPE=r2rce \
OUT_JSON=outputs/augment/r2r_style.json \
GEN_CUDA=0,1,2,3 \
  bash vp_adapter/run_generate_instructions.sh

# RxR-style narration for the same trajectories
DATASET_TYPE=rxrce OUT_JSON=outputs/augment/rxr_style.json \
CKPT=... MODEL_DIR=... DATA_DIR=... \
  bash vp_adapter/run_generate_instructions.sh
```

`VP_MODE=auto` (the default) uses `vp_masks` when the split has them and drops to the RGB-only path when it does not, so one command covers both. Other useful knobs: `GRANULARITY` (`trajectory` or `episode`), `TEMPERATURE` (raise to ~1.0 for diverse augmentations), `GEN_CUDA` (multi-GPU, shards merged automatically).

Output is a records JSON with `generated_instruction`, `trajectory_id`, `episode_ids` and `scene_id`. To get a split a VLN-CE dataloader can load directly, give the original `.json.gz` as a template — episode geometry is cloned and only the instruction text is replaced:

```bash
CKPT=... DATA_DIR=... \
VLNCE_TEMPLATE=/path/to/R2R_VLNCE_v1-3/train/train.json.gz \
OUT_VLNCE_GZ=outputs/augment/train_generated.json.gz \
  bash vp_adapter/run_generate_instructions.sh
```

Stale `instruction_tokens` are dropped from the emitted split, so re-tokenise with your follower's vocabulary.
