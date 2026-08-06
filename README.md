# VTInstructor

**Visual Trajectory Prompting for Navigation Instruction Generation in Continuous Environments**

VTInstructor generates natural-language navigation instructions for trajectories in
continuous environments (R2R-CE, RxR-CE). Its core idea is the **Visual Trajectory
Prompt (VTP)**: the planned route is rendered as semantic overlays (ribbon / arrows /
endpoint masks) onto egocentric keyframes, and a lightweight **VP-Adapter** injects
this structured spatial signal into a Qwen3-VL-8B-Instruct backbone. The model is first
trained with supervised fine-tuning (VP-Adapter SFT) and then refined with
reinforcement learning (VP-GRPO).

```
Stage 1  Rendering          →  clean panorama keyframes (Habitat)
Stage 2  VP masks           →  overlay frames + 3-channel semantic masks
Stage 3  VP-Adapter SFT     →  supervised fine-tuning
Stage 4  VP-GRPO            →  RL refinement (gate-only)
```

## Repository layout

```
VTInstructor/
├── rendering/            # Stage 1: Habitat keyframe rendering + instruction filter
│   ├── nig_render_dataset_r2rce_detail.py
│   ├── nig_render_rxr_dataset.py
│   ├── build_to_score_r2rce_train.py / filter_reference.py / filter_results.py
│   ├── filtered_ult_gt6.json          # shipped: 9,264 filtered R2R-CE train episodes
│   └── run_render_*.sh, run_gpt_filter_*.sh
│
├── visual_prompt/        # VTP geometry / overlay library (imported by Stage 2)
│   ├── projection.py, ribbon.py, overlay.py, vp_overlay_with_mask.py
│   └── augmentation.py, config.py
│
├── vp_adapter/           # Stage 2 masks + Stage 3 SFT + evaluation
│   ├── render_vp_masks_from_samples.py   # ★ Stage 2 entry (overlay + masks)
│   ├── train_sft_with_vp_adapter.py      # Stage 3 entry
│   ├── nig_eval_vp.py
│   ├── vp_{config,encoder,gated_adapter,model_wrapper,dataset}.py
│   └── run_vp_sft_pipeline.sh
│
├── vp_grpo/              # Stage 4 VP-GRPO
│   ├── grpo_rl_train_vp.py               # Stage 4 entry
│   ├── grpo_rl_train.py, grpo_reward.py
│   ├── ds_zero2_config.json
│   └── run_grpo_vp_v3.1.sh, eval_vp_v3.1.sh
│
├── common/               # Shared dataset / eval helpers (used by vp_adapter)
├── metrics/              # NLG scoring + shipped R2R val_unseen references
│   ├── nig_eval_r2rce_landmarks_from_json.py
│   └── R2R_val_unseen.json               # shipped: 613 traj × 3 refs
│
├── baseline_no_vp/       # No-VTP ablation (SFT + GRPO without Visual Prompt)
├── environment/          # Exact pip freezes from the paper runs (SFT / GRPO)
├── ENVIRONMENT.md        # Why two envs + DeepSpeed 0.14.4 pin
├── scripts/              # End-to-end example launcher
└── requirements.txt
```

## Pipeline

### 1. Rendering — clean panoramas (`rendering/`)

Render egocentric panorama keyframes along each VLN-CE trajectory in Habitat
(event-based keyframe sampling, **no VTP yet**):

```bash
# R2R-CE (defaults match the paper data: 768×384, camera_height=0.75, max_parts=5)
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

Optionally filter low-quality R2R references with `bash rendering/run_gpt_filter_r2rce_train.sh`
(requires a DashScope / Qwen API key). The paper already ships the resulting
`rendering/filtered_ult_gt6.json`, so this step is only needed if you want to regenerate it.

**Habitat / EGL tip.** On machines that have both Mesa and NVIDIA EGL ICDs installed,
habitat-sim may fail with `unable to find CUDA device 0 among N EGL devices`. Force the
NVIDIA ICD before launching:

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

### 2. Visual Prompt masks (`vp_adapter/render_vp_masks_from_samples.py`)

This is the script that produces the training data used in the paper. It walks each
already-rendered episode, writes `frame_*_overlay.jpg` (model input) and
`frame_*_vpmask.png` (3-channel semantic mask: ribbon / arrow / endpoint), and updates
`sample.json` with `clean_frames`, `overlay_frames`, `vp_masks`, and `vp_mask_schema`.

```bash
python vp_adapter/render_vp_masks_from_samples.py \
  --data_dir data/R2RCE_visual/r2rce_train_visual \
  --scenes_root /path/to/scenes_root \
  --width 256 --height 384 --camera_height 0.75 \
  --hfov 90.0 --panorama_step 90.0

# same for RxR-CE and for val_unseen splits
```

> **Do not use `visual_prompt/render_with_vp*.py` as the Stage-2 entry for training data.**
> Those scripts only bake an RGB overlay and do **not** write `vp_masks`. Training would
> then silently fall back to zero masks and the VP-Adapter would receive no signal.
> `visual_prompt/` remains the geometry / overlay library that Stage 2 imports.

Bit-level reproducibility of Stage 1+2 against the paper's training data has been verified
on R2R-CE and RxR-CE `episode_1` (all frames, overlays, masks, and `sample.json` fields).

### 3. VP-Adapter SFT (`vp_adapter/`)

Supervised fine-tuning of Qwen3-VL-8B-Instruct with the VP-Adapter
(`vp_dim=384`, `adapter_layers=7`, gate initialized to `0.002`):

```bash
export MODEL_DIR=/path/to/Qwen3-VL-8B-Instruct
export R2RCE_TRAIN_DATA_DIR=data/R2RCE_visual/r2rce_train_visual
export RXRCE_TRAIN_DATA_DIR=data/RXRCE_visual/rxrce_train_visual
export R2RCE_EVAL_DATA_DIR=data/R2RCE_visual/r2rce_valunseen_visual
# FILTERED_JSON and R2R_VAL_UNSEEN_JSON default to the shipped files
bash vp_adapter/run_vp_sft_pipeline.sh
```

Uses `torch.distributed.run` (HF Trainer + DDP). See `ENVIRONMENT.md` for the SFT env.

### 4. VP-GRPO (`vp_grpo/`)

GRPO reinforcement learning on top of the SFT checkpoint. Only the VP-Adapter gate
parameters are unfrozen (`--gate_only`), optimized with an NLG-weighted reward
(BLEU-1/4, CIDEr, METEOR, ROUGE-L) plus a contrastive gate loss. Includes Dr.GRPO
advantage estimation, k3 KL penalty, distributed KL early-stopping, and a numerical
loss guard:

```bash
export MODEL_DIR=/path/to/Qwen3-VL-8B-Instruct
export SFT_CKPT=/path/to/outputs/nig_vp_adapter_v3.1_sft/checkpoint-NNNN
bash vp_grpo/run_grpo_vp_v3.1.sh
```

Post-GRPO eval (defaults to `outputs/nig_rl_vp_v3.3/best`):

```bash
EVAL_CKPT=/path/to/rl/best bash vp_grpo/eval_vp_v3.1.sh
```

**GRPO requires its own Python environment** (torch 2.7.1 + DeepSpeed **exactly** 0.14.4).
Newer DeepSpeed versions corrupt all trainable weights on the first step under this
gate-only + ZeRO-2 setup — details and the exact `pip` recipe are in `ENVIRONMENT.md`.

## Included data artifacts

Two small artifacts ship with the repository because they cannot be regenerated
deterministically, and any difference in them changes the training / eval set:

| File | Contents | Used by |
|------|----------|---------|
| `rendering/filtered_ult_gt6.json` | 9,264 R2R-CE train episodes with instruction score > 6 | SFT + GRPO (`FILTERED_JSON`) |
| `metrics/R2R_val_unseen.json` | 613 R2R val_unseen trajectories × 3 human references | Eval (`R2R_VAL_UNSEEN_JSON`) |

`R2R_TRAIN_JSON` is optional and defaults to empty: GRPO reconstructs multi-reference
targets by grouping episodes on `trajectory_id` (~3.1 refs/traj without it).

## External dependencies

| Dependency | Needed for | Notes |
|------------|------------|-------|
| Qwen3-VL-8B-Instruct | SFT / GRPO / eval | Local HF snapshot; loaders use `local_files_only=True` |
| R2R-CE / RxR-CE episode JSONs | Stage 1 | VLN-CE releases |
| Matterport3D scenes (`.glb` + `.navmesh`) | Stage 1–2 | Missing `.navmesh` raises `RuntimeError` |
| [habitat-sim](https://github.com/facebookresearch/habitat-sim) | Stage 1–2 only | Not required for SFT/GRPO/eval |
| [pycocoevalcap](https://github.com/salaniz/pycocoevalcap) + JRE | GRPO reward + scoring | Without Java, METEOR silently falls back and the reward changes |
| Python packages | all stages | Prefer `environment/sft_env.txt` and `environment/grpo_env.txt` over the loose `requirements.txt` pins |

## Notes

- Hardcoded dataset/model locations in argument defaults are shown as `PATH/TO/...`
  placeholders — override them via CLI args or the shell scripts' environment variables.
- `baseline_no_vp/` is the no-VTP ablation baseline, kept for comparison.
- Checkpoints are **not** in this repository (see `.gitignore`). Publish SFT / GRPO
  weights separately (e.g. Hugging Face Hub) and point `SFT_CKPT` / `EVAL_CKPT` at them.
