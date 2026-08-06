# Training Environment

Exact environments used to train VTInstructor. Machine-readable package lists live in
[`environment/sft_env.txt`](environment/sft_env.txt) and
[`environment/grpo_env.txt`](environment/grpo_env.txt).

## Hardware and base image

| | |
|---|---|
| GPU | 8 x NVIDIA H200 (143 GB each) |
| Driver / CUDA | 570.195.03 / CUDA 12.8 |
| OS | Ubuntu 22.04.5 LTS |
| Container image | `embody-h2-cn-north-1.jcr.service.jdcloud.com/mingxu-robocasa:v4.0` (md5 `5b367305037042691065a447d1df3aef`) |
| Python | 3.10.19 |

The container image is a shared team base image, not a VTInstructor-specific one. The
Python environments below are created inside it.

## Two environments, on purpose

The SFT and GRPO stages run in **separate** Python environments. This is required, not
cosmetic.

| | Stage 3 — VP-Adapter SFT | Stage 4 — VP-GRPO |
|---|---|---|
| Launcher | `torch.distributed.run` (HF Trainer + DDP) | `deepspeed.launcher.runner` (ZeRO-2, bf16) |
| torch | 2.10.0+cu128 | **2.7.1** |
| DeepSpeed | not used | **0.14.4** |
| transformers | 4.57.6 | 4.57.6 |

SFT never touches DeepSpeed, so it is tolerant of a recent torch. GRPO is the only stage
that uses ZeRO-2, and it is version-sensitive.

## Why DeepSpeed 0.14.4 is pinned

Running GRPO with DeepSpeed 0.16.9 makes **every trainable parameter non-finite on the
very first optimizer step**, after which any sampling call dies with:

```
Assertion `probability tensor contains either `inf`, `nan` or element < 0` failed.
```

This is not a diverging-training problem. It was diagnosed by hooking
`DeepSpeedEngine.step` and scanning parameter finiteness after each update:

| | DeepSpeed 0.16.9 + torch 2.10 | DeepSpeed 0.14.4 + torch 2.7.1 |
|---|---|---|
| reported grad norm at update #1 | `1.0` (the post-clip value, which hides a non-finite pre-clip norm) | `0.1449` (genuine, below the clip threshold) |
| non-finite params at update #1 | all trainable tensors | `0` |
| `gate_mean` | `nan` | `0.00145`, stable |

Surrounding conditions that make this fatal rather than merely noisy:

- GRPO freezes the vision tower (`model.visual.requires_grad_(False)`, ~577 M of
  8.77 B parameters), so a large fraction of parameters have `grad=None`. This
  frozen-plus-trainable mix under ZeRO-2 is what the newer DeepSpeed mishandles.
- With `bf16` there is no loss scaling, so DeepSpeed does not skip the step on gradient
  overflow the way it does for `fp16`. A single bad gradient is written straight into the
  weights.
- The loss guard in the training script only checks that the *loss* is finite. The loss
  stays at ~1e-4 throughout, so the guard never fires; the corruption is in the gradients.

Hardware was ruled out: ECC error count 0, all NVLink lanes at 26.562 GB/s, and an 8-GPU
NCCL all-reduce stress test returned exact results on 30/30 iterations.

## Reproducing the GRPO environment

```bash
python -m venv /path/to/vt_rl          # clean venv: do NOT use --system-site-packages
source /path/to/vt_rl/bin/activate
pip install torch==2.7.1 torchvision==0.22.1
pip install numpy==1.26.4 pillow opencv-python-headless safetensors \
            'huggingface_hub>=0.34.0,<1.0' accelerate==1.5.2 \
            sentencepiece protobuf tensorboard
pip install transformers==4.57.6 tokenizers==0.22.2
pip install pycocoevalcap==1.2 einops hjson py-cpuinfo ninja nvidia-ml-py msgpack pydantic
DS_BUILD_OPS=0 pip install deepspeed==0.14.4
```

Two constraints worth calling out:

- The venv must be clean. Inheriting site-packages from an environment that already has a
  newer torch will silently shadow torch 2.7.1.
- `huggingface_hub` must be `<1.0`; transformers 4.57.6 rejects 1.x at import time.

Java (OpenJDK 8) must also be on `PATH` because the METEOR reward and metric shell out to
it. Without it METEOR silently falls back and the GRPO reward changes meaning.

## Runtime environment variables

The original June runs set these, and they are kept:

```bash
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Checkpoint portability

`checkpoints/nig_vp_adapter_v3.1_sft/` was produced under the SFT environment (torch 2.10)
and is loaded by GRPO under torch 2.7.1. This works because the weights are
`safetensors`, which is independent of the torch version. The VP modules
(`vp_encoder.pt`, `vp_adapters.pt`) are plain `state_dict` files and are likewise portable.
