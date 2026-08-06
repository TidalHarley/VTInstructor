#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) RL training for Qwen3-VL
Navigation Instruction Generation.

Uses DeepSpeed ZeRO-2 for memory-efficient training.
"""

import argparse
import inspect
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler

try:
    import deepspeed
except ImportError:
    raise ImportError("deepspeed is required. Install with: pip install deepspeed")

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grpo_reward import RewardComputer, DEFAULT_WEIGHTS, _normalize


def patch_deepspeed_grad_count_for_torch_compat() -> None:
    """
    Work around a Torch/DeepSpeed compatibility issue where
    torch.autograd.graph._get_grad_fn_or_grad_acc(param) may raise:
      AttributeError: 'NoneType' object has no attribute 'next_functions'

    DeepSpeed ZeRO-1/2 calls this helper during backward hooks.
    We replace it with a defensive variant that skips problematic params.
    """
    import deepspeed.runtime.utils as ds_utils
    import deepspeed.runtime.zero.stage_1_and_2 as ds_zero_stage12
    from torch.autograd.graph import _get_grad_fn_or_grad_acc

    def _safe_count_used_parameters_in_backward(parameters):
        if torch._C._current_graph_task_id() == -1:
            raise RuntimeError("count_used_parameters_in_backward must be called during backward execution")

        seen_nodes = set()
        for param in parameters:
            if not isinstance(param, torch.Tensor) or not param.requires_grad:
                continue
            try:
                grad_fn = _get_grad_fn_or_grad_acc(param)
            except Exception:
                # Skip params that trigger internal autograd edge-case.
                continue
            if grad_fn is None or grad_fn in seen_nodes:
                continue
            seen_nodes.add(grad_fn)

        if not seen_nodes:
            return 0
        return int(sum(map(torch._C._will_engine_execute_node, seen_nodes)))

    ds_utils.count_used_parameters_in_backward = _safe_count_used_parameters_in_backward
    ds_zero_stage12.count_used_parameters_in_backward = _safe_count_used_parameters_in_backward

# ============================================================================
# Dataset-specific System Prompts (must match SFT strategy)
# ============================================================================
R2RCE_SYSTEM_PROMPT = """You are an R2R-style indoor navigation instruction writer.

Task:
- Given a first-person trajectory with action snippets and frame observations, write one concise instruction that matches the full path.

Strict constraints:
1) Use 25-40 words.
2) Mention 2-3 concrete landmarks.
3) Explicitly say "go up the stairs" or "go down the stairs" when stairs are involved.
4) End with a precise stop location next to a visible object.
5) Avoid loops, filler words, and repeated phrases.

Output one final instruction only."""

RXRCE_SYSTEM_PROMPT = """You are an RXR-style multilingual route narrator in English.

Goal:
- Describe the shown route as a step-by-step path grounded in scene details and action sequence.

Requirements:
1) Prefer richer spatial detail than R2R (about 35-60 words).
2) Use explicit orientation and transition cues (e.g., slight right, pass through, continue along).
3) Mention distinctive visual anchors, not generic placeholders.
4) Keep temporal order faithful to the trajectory.
5) End with a concrete waiting point.
6) No repetitive template sentences.

Return exactly one route instruction."""

PANORAMA_SYSTEM_PROMPT = R2RCE_SYSTEM_PROMPT


def infer_dataset_type_from_path(path: str) -> str:
    p = (path or "").lower()
    if "rxr" in p:
        return "rxrce"
    return "r2rce"


def get_system_prompt(dataset_type: str) -> str:
    t = (dataset_type or "r2rce").lower()
    if t == "rxrce":
        return RXRCE_SYSTEM_PROMPT
    return R2RCE_SYSTEM_PROMPT


# ============================================================================
# Data utilities (self-contained, no external dependency on src/)
# ============================================================================

def safe_load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def group_actions(actions_text: List[str], group_size: int) -> List[str]:
    if group_size <= 0:
        group_size = 1
    grouped = []
    for i in range(0, len(actions_text), group_size):
        grouped.append(" / ".join(actions_text[i : i + group_size]))
    return grouped


def build_inference_prompt(
    images: List[Image.Image],
    actions_list: List[str],
    system_prompt: Optional[str] = None,
) -> list:
    """Build multimodal user_content in the same format used during SFT."""
    n = min(len(images) - 1, len(actions_list))
    images = images[: n + 1]
    actions_list = actions_list[:n]

    user_content: list = [{"type": "text", "text": system_prompt or PANORAMA_SYSTEM_PROMPT}]
    user_content.append({"type": "image", "image": images[0]})
    for i, action in enumerate(actions_list):
        user_content.append({"type": "text", "text": f"[Action {i + 1}: {action}]"})
        user_content.append({"type": "image", "image": images[i + 1]})
    return user_content


# ============================================================================
# RL Dataset
# ============================================================================

class RLPanoramaDataset(Dataset):
    """
    Loads panorama episodes from one or more data directories.

    ``filtered_json`` is only applied to **R2RCE** episodes (inferred from
    the data directory path).  RXRCE episodes are loaded without any filter,
    matching the behaviour of the SFT training script.
    """

    def __init__(
        self,
        data_dirs: List[str],
        filtered_json: Optional[str] = None,
        max_frames: int = 32,
        max_samples: int = 0,
        max_rxrce: int = 0,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.max_frames = max_frames
        r2rce_files: List[str] = []
        rxrce_files: List[str] = []
        self.sample_dataset_type: Dict[str, str] = {}

        allowed_eps: Optional[set] = None
        if filtered_json and os.path.exists(filtered_json):
            with open(filtered_json, "r", encoding="utf-8") as f:
                fdata = json.load(f)
            if "episodes" in fdata:
                allowed_eps = set(str(k) for k in fdata["episodes"].keys())

        for data_dir in data_dirs:
            if not os.path.isdir(data_dir):
                continue
            ds_type = infer_dataset_type_from_path(data_dir)
            apply_filter = (allowed_eps is not None) and (ds_type == "r2rce")
            for name in os.listdir(data_dir):
                if not name.startswith("episode_"):
                    continue
                ep_id = name[len("episode_"):]
                if apply_filter and ep_id not in allowed_eps:
                    continue
                sp = os.path.join(data_dir, name, "sample.json")
                if os.path.exists(sp):
                    self.sample_dataset_type[sp] = ds_type
                    if ds_type == "rxrce":
                        rxrce_files.append(sp)
                    else:
                        r2rce_files.append(sp)

        if max_rxrce > 0 and len(rxrce_files) > max_rxrce:
            rxrce_files = random.Random(seed).sample(rxrce_files, max_rxrce)
            print(f"[RLDataset] RXRCE subsampled: {len(rxrce_files)}/{max_rxrce}")

        self.sample_files = r2rce_files + rxrce_files

        valid: List[str] = []
        for p in self.sample_files:
            s = safe_load_json(p)
            if s is not None and s.get("frames") and isinstance(s["frames"][0], str):
                valid.append(p)
        self.sample_files = valid

        if shuffle:
            random.Random(seed).shuffle(self.sample_files)
        if max_samples > 0:
            self.sample_files = self.sample_files[:max_samples]

        r2rce_count = sum(1 for f in self.sample_files if self.sample_dataset_type.get(f) == "r2rce")
        rxrce_count = sum(1 for f in self.sample_files if self.sample_dataset_type.get(f) == "rxrce")
        print(f"[RLDataset] {len(self.sample_files)} samples from {len(data_dirs)} dirs "
              f"(R2RCE={r2rce_count}, RXRCE={rxrce_count})")

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> dict:
        sample = safe_load_json(self.sample_files[idx])
        if sample is None:
            raise ValueError(f"Cannot load: {self.sample_files[idx]}")

        if "action_events_text" in sample:
            grouped = sample["action_events_text"]
        else:
            raw = sample.get("actions_text") or [str(a) for a in sample.get("actions", [])]
            gs = int(sample.get("actions_per_frame") or sample.get("frame_stride") or 2)
            grouped = group_actions(raw, gs)

        nf = min(len(sample["frames"]), len(grouped) + 1, self.max_frames)
        frames = sample["frames"][:nf]
        actions_list = grouped[: max(0, nf - 1)]
        images = [Image.open(p).convert("RGB") for p in frames]

        return {
            "images": images,
            "actions_list": actions_list,
            "system_prompt": get_system_prompt(self.sample_dataset_type.get(self.sample_files[idx], "r2rce")),
            "instruction": sample["instruction"],
            "episode_id": sample.get("episode_id"),
            "trajectory_id": sample.get("trajectory_id"),
            "dataset_type": self.sample_dataset_type.get(self.sample_files[idx], "r2rce"),
        }


def _rl_collate(batch: list):
    """DataLoader with batch_size=1: just unpack the single element."""
    return batch[0]


# ============================================================================
# Generation
# ============================================================================

@torch.no_grad()
def generate_completions(
    model: torch.nn.Module,
    processor,
    sample: dict,
    num_completions: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    gen_batch_size: int = 8,
) -> List[str]:
    """Generate *num_completions* sampled completions in mini-batches.

    Uses ``num_return_sequences`` so the prompt (including vision encoding)
    is processed once per mini-batch instead of once per completion.
    With gen_batch_size=8 and G=16 this reduces generation calls from 16→2.
    """
    images = sample["images"]
    actions_list = sample["actions_list"]
    system_prompt = sample.get("system_prompt", PANORAMA_SYSTEM_PROMPT)

    user_content = build_inference_prompt(images, actions_list, system_prompt=system_prompt)
    messages = [{"role": "user", "content": user_content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_len = inputs["input_ids"].shape[1]
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    completions: List[str] = []
    remaining = num_completions
    while remaining > 0:
        batch_n = min(gen_batch_size, remaining)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            num_return_sequences=batch_n,
        )
        for i in range(batch_n):
            comp_ids = gen_ids[i, prompt_len:]
            text = processor.tokenizer.decode(comp_ids, skip_special_tokens=True).strip()
            completions.append(text if text else ".")
        remaining -= batch_n
    return completions


# ============================================================================
# Log-probability computation
# ============================================================================

def _get_prompt_len(processor, sample: dict) -> int:
    """Compute prompt token length once per sample (shared by all completions)."""
    images = sample["images"]
    actions_list = sample["actions_list"]
    system_prompt = sample.get("system_prompt", PANORAMA_SYSTEM_PROMPT)
    user_content = build_inference_prompt(images, actions_list, system_prompt=system_prompt)
    msg_prompt = [{"role": "user", "content": user_content}]
    prompt_inputs = processor.apply_chat_template(
        msg_prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return prompt_inputs["input_ids"].shape[1]


def _prepare_full_inputs(
    processor, sample: dict, completion_text: str,
    cached_prompt_len: Optional[int] = None,
    cached_pixel_values: Optional[torch.Tensor] = None,
    cached_image_grid_thw: Optional[torch.Tensor] = None,
):
    """
    Tokenise (prompt + completion) and return
    ``(full_inputs_dict, prompt_length)``.

    When *cached_prompt_len* is given the prompt-only tokenisation is skipped
    (saves one ``apply_chat_template`` call per completion).
    When *cached_pixel_values* / *cached_image_grid_thw* are given the
    returned inputs reuse them (avoids duplicate large tensors in memory).
    """
    images = sample["images"]
    actions_list = sample["actions_list"]
    system_prompt = sample.get("system_prompt", PANORAMA_SYSTEM_PROMPT)
    user_content = build_inference_prompt(images, actions_list, system_prompt=system_prompt)

    msg_full = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": completion_text}]},
    ]
    full_inputs = processor.apply_chat_template(
        msg_full,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )

    if cached_pixel_values is not None and "pixel_values" in full_inputs:
        full_inputs["pixel_values"] = cached_pixel_values
    if cached_image_grid_thw is not None and "image_grid_thw" in full_inputs:
        full_inputs["image_grid_thw"] = cached_image_grid_thw

    if cached_prompt_len is not None:
        return full_inputs, cached_prompt_len

    msg_prompt = [{"role": "user", "content": user_content}]
    prompt_inputs = processor.apply_chat_template(
        msg_prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]
    return full_inputs, prompt_len


def compute_log_probs(
    model: torch.nn.Module,
    full_inputs: dict,
    prompt_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Forward-pass and return per-token log-probs of the completion portion.

    Uses ``logits_to_keep`` so the LM head only runs on the completion
    positions instead of the full (prompt + completion) sequence.  For 32-frame
    panorama inputs this saves ~2 GB of GPU memory per forward pass.

    Returns shape ``(completion_length,)``.
    """
    inputs = {k: v.to(device) for k, v in full_inputs.items()}
    seq_len = inputs["input_ids"].shape[1]
    comp_len = seq_len - prompt_len

    try:
        outputs = model(**inputs, logits_to_keep=comp_len + 1)
    except TypeError:
        outputs = model(**inputs)

    logits = outputs.logits  # (1, <=comp_len+1 or full_seq, vocab)

    if logits.shape[1] <= comp_len + 1:
        comp_logits = logits[:, :-1, :]
    else:
        comp_logits = logits[:, prompt_len - 1 : -1, :]

    comp_ids = inputs["input_ids"][:, prompt_len:]
    lp = F.log_softmax(comp_logits, dim=-1)
    per_tok = lp.gather(2, comp_ids.unsqueeze(-1)).squeeze(-1)
    return per_tok.squeeze(0)


def compute_log_probs_batched(
    model: torch.nn.Module,
    full_inputs_list: List[dict],
    prompt_len: int,
    device: torch.device,
    batch_size: int = 8,
    detach_to_cpu: bool = True,
) -> List[torch.Tensor]:
    """Compute per-token log-probs for multiple completions in mini-batches.

    All completions share the same prompt (same images), so *prompt_len* is
    identical.  ``pixel_values`` / ``image_grid_thw`` are taken from the first
    element and repeated for the batch, avoiding redundant large-tensor copies.

    Uses ``logits_to_keep`` so the LM head only runs on completion positions,
    keeping the vocab-projection memory at O(batch × max_comp_len × vocab)
    instead of O(batch × full_seq × vocab).
    """
    results: List[torch.Tensor] = []
    for start in range(0, len(full_inputs_list), batch_size):
        chunk = full_inputs_list[start : start + batch_size]
        n = len(chunk)

        if n == 1:
            lp = compute_log_probs(model, chunk[0], prompt_len, device)
            if detach_to_cpu:
                lp = lp.detach().cpu()
            results.append(lp)
            continue

        seq_lens = [c["input_ids"].shape[1] for c in chunk]
        max_seq = max(seq_lens)
        max_comp = max_seq - prompt_len

        ids_list, attn_list = [], []
        for c, sl in zip(chunk, seq_lens):
            pad = max_seq - sl
            ids_list.append(F.pad(c["input_ids"], (0, pad), value=0))
            attn_list.append(F.pad(
                c.get("attention_mask", torch.ones_like(c["input_ids"])),
                (0, pad), value=0,
            ))

        batch_ids = torch.cat(ids_list, dim=0).to(device)
        batch_attn = torch.cat(attn_list, dim=0).to(device)

        pv = chunk[0]["pixel_values"].to(device)
        igt = chunk[0]["image_grid_thw"].to(device)
        batch_pv = pv.repeat(n, 1)
        batch_igt = igt.repeat(n, 1)

        batch_inputs = {
            "input_ids": batch_ids,
            "attention_mask": batch_attn,
            "pixel_values": batch_pv,
            "image_grid_thw": batch_igt,
        }

        try:
            outputs = model(**batch_inputs, logits_to_keep=max_comp + 1)
        except TypeError:
            outputs = model(**batch_inputs)

        logits = outputs.logits  # (n, K or full_seq, vocab)

        for i in range(n):
            comp_len = seq_lens[i] - prompt_len
            if logits.shape[1] <= max_comp + 1:
                cl = logits[i, :comp_len, :]
            else:
                cl = logits[i, prompt_len - 1 : prompt_len - 1 + comp_len, :]
            tgt = batch_ids[i, prompt_len : prompt_len + comp_len]
            lp = F.log_softmax(cl, dim=-1)
            per_tok = lp.gather(1, tgt.unsqueeze(-1)).squeeze(-1)
            if detach_to_cpu:
                per_tok = per_tok.detach().cpu()
            results.append(per_tok)

        del logits, outputs, batch_inputs, batch_pv, batch_igt
    return results


# ============================================================================
# GRPO loss
# ============================================================================

def grpo_loss(
    current_lp: torch.Tensor,   # (T,) with grad
    old_lp: torch.Tensor,       # (T,) detached
    ref_lp: torch.Tensor,       # (T,) detached
    advantage: float,
    clip_eps: float = 0.2,
    kl_beta: float = 0.10,
) -> torch.Tensor:
    """Per-token clipped surrogate + correct-k3 KL penalty (Dr.GRPO), averaged over tokens."""
    ratio = torch.exp(current_lp - old_lp)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    adv_t = torch.tensor(advantage, dtype=current_lp.dtype, device=current_lp.device)
    surrogate = torch.min(ratio * adv_t, clipped * adv_t)

    # correct k3 estimator of KL(pi_theta || pi_ref); samples ~ pi_theta => r = pi_ref/pi_theta
    log_ratio = (ref_lp - current_lp).clamp(-10.0, 10.0)
    kl = torch.exp(log_ratio) - 1 - log_ratio
    return (-surrogate + kl_beta * kl).mean()


# ============================================================================
# Logging helpers
# ============================================================================

class RLMetricsLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self._rewards: List[float] = []
        self._losses: List[float] = []
        self._kl: List[float] = []
        self._comp_lens: List[int] = []
        self._metric_sums: Dict[str, float] = {}
        self._metric_cnt: int = 0

    def add(self, reward: float, loss: float, kl: float, comp_len: int, metrics: Dict[str, float]):
        self._rewards.append(reward)
        self._losses.append(loss)
        self._kl.append(kl)
        self._comp_lens.append(comp_len)
        for k, v in metrics.items():
            self._metric_sums[k] = self._metric_sums.get(k, 0.0) + v
        self._metric_cnt += 1

    def summary(self) -> str:
        n = max(len(self._rewards), 1)
        parts = [
            f"reward={sum(self._rewards) / n:.4f}",
            f"loss={sum(self._losses) / n:.4f}",
            f"kl={sum(self._kl) / n:.4f}",
            f"comp_len={sum(self._comp_lens) / n:.1f}",
        ]
        if self._metric_cnt > 0:
            for k in sorted(self._metric_sums):
                parts.append(f"{k}={self._metric_sums[k] / self._metric_cnt:.4f}")
        return "  ".join(parts)


# ============================================================================
# Main training loop
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GRPO RL for Qwen3-VL NIG")

    # ----- paths -----
    p.add_argument("--sft_checkpoint", required=True, help="SFT checkpoint directory")
    p.add_argument("--processor_dir", default="", help="Processor dir (defaults to sft_checkpoint)")
    p.add_argument("--r2rce_data_dir", required=True, help="R2RCE training data directory")
    p.add_argument("--rxrce_data_dir", required=True, help="RXRCE training data directory")
    p.add_argument("--filtered_json", default="", help="Optional episode-filter JSON")
    p.add_argument("--r2r_train_json", default="", help="R2R_train.json for multi-reference (3 refs per path)")
    p.add_argument("--output_dir", required=True, help="RL output / checkpoint directory")
    p.add_argument("--logging_dir", default="", help="TensorBoard log directory")
    p.add_argument("--ds_config", required=True, help="DeepSpeed ZeRO-2 JSON config")

    # ----- data -----
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=42)

    # ----- RL hyper-params -----
    p.add_argument("--group_size", type=int, default=16, help="G: completions per prompt (larger = better advantage estimates)")
    p.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
    p.add_argument("--kl_beta", type=float, default=0.04, help="KL penalty coefficient")
    p.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    p.add_argument("--max_new_tokens", type=int, default=150)

    # ----- optimiser -----
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4, help="Inter-sample gradient accumulation")

    # ----- reward weights (equal by default) -----
    p.add_argument("--w_bleu1", type=float, default=0.2)
    p.add_argument("--w_bleu4", type=float, default=0.2)
    p.add_argument("--w_cider", type=float, default=0.2)
    p.add_argument("--w_meteor", type=float, default=0.2)
    p.add_argument("--w_rouge_l", type=float, default=0.2)

    # ----- misc -----
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=0, help="Save every N optimizer steps (0=epoch only)")
    p.add_argument("--local_rank", type=int, default=-1, help="Set by DeepSpeed launcher")
    p.add_argument("--gen_batch_size", type=int, default=16,
                   help="Mini-batch size for generation (uses num_return_sequences; "
                        "16 is safe on H200 even for 32-frame RXRCE worst case)")
    p.add_argument("--lp_batch_size", type=int, default=8,
                   help="Mini-batch size for batched log-prob forward passes "
                        "(old_lp and ref_lp); 8 is safe on H200")
    p.add_argument("--r2rce_temperature", type=float, default=0.5,
                   help="Sampling temperature for R2RCE during GRPO generation")
    p.add_argument("--rxrce_temperature", type=float, default=1.0,
                   help="Sampling temperature for RXRCE during GRPO generation")
    p.add_argument("--max_rxrce", type=int, default=0,
                   help="Max RXRCE samples to keep (0 = all). "
                        "Randomly subsamples if total RXRCE exceeds this number.")
    p.add_argument(
        "--batched_train_update",
        action="store_true",
        help="Use one batched forward + one backward per sample for G completions.",
    )
    p.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (enabled by default; disabling will OOM on long sequences).",
    )
    p.add_argument(
        "--no_ref_model",
        action="store_true",
        help="Disable frozen reference model for KL (saves ~16GB GPU; KL will anchor to old_lp instead).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    patch_deepspeed_grad_count_for_torch_compat()

    # ---- distributed setup ----
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank < 0:
        local_rank = 0
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if not args.processor_dir:
        args.processor_dir = args.sft_checkpoint

    reward_weights = {
        "BLEU-1": args.w_bleu1,
        "BLEU-4": args.w_bleu4,
        "CIDEr": args.w_cider,
        "METEOR": args.w_meteor,
        "ROUGE-L": args.w_rouge_l,
    }

    if rank == 0:
        print("=" * 60)
        print("GRPO RL Training Configuration")
        print("=" * 60)
        print(f"SFT checkpoint:  {args.sft_checkpoint}")
        print(f"Processor dir:   {args.processor_dir}")
        print(f"R2RCE data dir:  {args.r2rce_data_dir}")
        print(f"RXRCE data dir:  {args.rxrce_data_dir}")
        print(f"Max RXRCE:       {args.max_rxrce if args.max_rxrce > 0 else 'ALL'}")
        print(f"Output dir:      {args.output_dir}")
        print(f"Group size (G):  {args.group_size}")
        print(f"Clip eps:        {args.clip_eps}")
        print(f"KL beta:         {args.kl_beta}")
        print(f"LR:              {args.lr}")
        print(f"Epochs:          {args.num_epochs}")
        print(f"Grad accum:      {args.grad_accum}")
        print(f"Gen batch size:  {args.gen_batch_size}")
        print(f"R2RCE temp:      {args.r2rce_temperature}")
        print(f"RXRCE temp:      {args.rxrce_temperature}")
        print(f"Batched train:   {'ON' if args.batched_train_update else 'OFF'}")
        print(f"Reward weights:  {reward_weights}")
        print(f"Ref model:       {'ENABLED (frozen SFT)' if not args.no_ref_model else 'DISABLED'}")
        print(f"World size:      {world_size}")
        print("=" * 60)

    # ---- load processor ----
    if rank == 0:
        print("[INFO] Loading processor …")
    processor = AutoProcessor.from_pretrained(
        args.processor_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    # ---- load policy model ----
    if rank == 0:
        print(f"[INFO] Loading policy model from {args.sft_checkpoint} …")
    policy_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.sft_checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    # Freeze vision encoder — its 27 ViT layers do not need RL gradients
    # and skipping their backward saves tens of GB of activation memory.
    policy_model.model.visual.requires_grad_(False)

    if not args.no_gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
        policy_model.config.use_cache = False
        if rank == 0:
            print("[INFO] Gradient checkpointing ENABLED (saves ~30 GB activation memory)")
    else:
        if rank == 0:
            print("[WARN] Gradient checkpointing DISABLED — may OOM on long sequences")

    # ---- load reference model (frozen, for KL regularisation) ----
    ref_model = None
    if not args.no_ref_model:
        if rank == 0:
            print(f"[INFO] Loading frozen reference model from {args.sft_checkpoint} …")
        ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.sft_checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        )
        ref_model.requires_grad_(False)
        ref_model.eval()
        ref_model.to(device)
        if rank == 0:
            alloc_gb = torch.cuda.memory_allocated(device) / 1e9
            total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            print(f"[INFO] Reference model loaded on {device} "
                  f"(GPU mem: {alloc_gb:.1f}/{total_gb:.0f} GB)")
    else:
        if rank == 0:
            print("[INFO] Reference model DISABLED (--no_ref_model); KL will anchor to old_lp")

    # ---- dataset & dataloader ----
    data_dirs = [args.r2rce_data_dir, args.rxrce_data_dir]
    dataset = RLPanoramaDataset(
        data_dirs=data_dirs,
        filtered_json=args.filtered_json or None,
        max_frames=args.max_frames,
        max_samples=args.max_samples,
        max_rxrce=args.max_rxrce,
        shuffle=True,
        seed=args.seed,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=_rl_collate, num_workers=0)

    # ---- Build multi-reference mapping & CIDEr-D IDF ----
    if rank == 0:
        print("[INFO] Building multi-reference mapping for reward computation ...")

    multiref_mapping: Dict[int, List[str]] = {}

    if args.r2r_train_json and os.path.exists(args.r2r_train_json):
        with open(args.r2r_train_json, "r", encoding="utf-8") as f:
            r2r_data = json.load(f)
        for item in r2r_data:
            path_id = item.get("path_id")
            insts = item.get("instructions", [])
            if path_id is not None and insts:
                clean = [_normalize(x) for x in insts if _normalize(x)]
                if clean:
                    multiref_mapping[int(path_id)] = clean
        if rank == 0:
            print(f"[INFO] R2R multiref: {len(multiref_mapping)} trajectories from {args.r2r_train_json}")

    traj_refs: Dict[int, set] = {}
    for sf in dataset.sample_files:
        s = safe_load_json(sf)
        if not s:
            continue
        traj_id = s.get("trajectory_id")
        instruction = s.get("instruction", "")
        if traj_id is not None and instruction.strip():
            tid = int(traj_id)
            if tid not in traj_refs:
                traj_refs[tid] = set()
            traj_refs[tid].add(_normalize(instruction))
    added = 0
    for tid, refs in traj_refs.items():
        if tid not in multiref_mapping:
            multiref_mapping[tid] = list(refs)
            added += 1
        else:
            existing = set(multiref_mapping[tid])
            for ref in refs:
                if ref not in existing:
                    multiref_mapping[tid].append(ref)
    if rank == 0:
        n_total = len(multiref_mapping)
        avg_refs = sum(len(v) for v in multiref_mapping.values()) / max(n_total, 1)
        print(f"[INFO] Multiref mapping: {n_total} trajectories "
              f"({n_total - added} from R2R, {added} from training data), "
              f"avg {avg_refs:.1f} refs/traj")
    del traj_refs

    reward_computer = RewardComputer()
    reward_computer.build_cider_idf(multiref_mapping)

    per_gpu_samples = len(dataloader)
    G = args.group_size
    if args.batched_train_update:
        micro_batches_per_epoch = per_gpu_samples
        ds_grad_accum = args.grad_accum
    else:
        micro_batches_per_epoch = per_gpu_samples * G
        ds_grad_accum = G * args.grad_accum
    total_optimizer_steps = (per_gpu_samples * args.num_epochs) // args.grad_accum
    total_optimizer_steps = max(total_optimizer_steps, 1)
    warmup_steps = max(int(total_optimizer_steps * args.warmup_ratio), 1)

    if rank == 0:
        print(f"[INFO] Samples per GPU: {per_gpu_samples}")
        print(f"[INFO] Micro-batches per epoch: {micro_batches_per_epoch}")
        print(f"[INFO] DS grad_accum (G × accum): {ds_grad_accum}")
        print(f"[INFO] Total optimizer steps: {total_optimizer_steps}")
        print(f"[INFO] Warmup steps: {warmup_steps}")

    # ---- DeepSpeed config ----
    with open(args.ds_config, "r") as f:
        ds_config = json.load(f)
    ds_config["train_micro_batch_size_per_gpu"] = 1
    ds_config["gradient_accumulation_steps"] = ds_grad_accum
    ds_config["train_batch_size"] = world_size * ds_grad_accum
    ds_config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": args.lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": args.weight_decay,
        },
    }
    # DeepSpeed scheduler API differs across versions:
    #   - older: warmup_min_lr / warmup_max_lr
    #   - newer (e.g. 0.18.x): warmup_min_ratio / cos_min_ratio
    cosine_init_sig = inspect.signature(deepspeed.runtime.lr_schedules.WarmupCosineLR.__init__)
    if "warmup_min_ratio" in cosine_init_sig.parameters:
        scheduler_params = {
            "total_num_steps": total_optimizer_steps,
            "warmup_num_steps": warmup_steps,
            "warmup_min_ratio": 0.0,
            "cos_min_ratio": 0.0,
            "warmup_type": "linear",
        }
    else:
        scheduler_params = {
            "warmup_min_lr": 0.0,
            "warmup_max_lr": args.lr,
            "warmup_num_steps": warmup_steps,
            "total_num_steps": total_optimizer_steps,
        }

    ds_config["scheduler"] = {
        "type": "WarmupCosineLR",
        "params": scheduler_params,
    }

    # ---- DeepSpeed init ----
    model_engine, _, _, lr_scheduler = deepspeed.initialize(
        model=policy_model,
        config=ds_config,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    if args.logging_dir:
        os.makedirs(args.logging_dir, exist_ok=True)

    # Optional TensorBoard
    tb_writer = None
    if rank == 0 and args.logging_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=args.logging_dir)
        except ImportError:
            pass

    # ---- training ----
    global_micro_step = 0
    global_opt_step = 0
    logger = RLMetricsLogger()

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print(f"{'=' * 60}")

        for sample_idx, sample in enumerate(dataloader):
            t0 = time.time()

            try:
                # ---------- Phase 1: Generate G completions ----------
                t_gen_start = time.time()
                model_engine.eval()
                ds_type = sample.get("dataset_type", "r2rce")
                sample_temp = (args.rxrce_temperature
                               if ds_type == "rxrce"
                               else args.r2rce_temperature)
                completions = generate_completions(
                    model_engine.module,
                    processor,
                    sample,
                    num_completions=G,
                    max_new_tokens=args.max_new_tokens,
                    temperature=sample_temp,
                    top_p=args.top_p,
                    gen_batch_size=args.gen_batch_size,
                )
                t_gen = time.time() - t_gen_start

                torch.cuda.empty_cache()

                # ---------- Phase 2: Compute old & ref log-probs ----------
                t_lp_start = time.time()
                full_inputs_list: List[dict] = []
                prompt_len_list: List[int] = []

                cached_pl = _get_prompt_len(processor, sample)
                cached_pv: Optional[torch.Tensor] = None
                cached_igt: Optional[torch.Tensor] = None

                for idx_c, text in enumerate(completions):
                    fi, pl = _prepare_full_inputs(
                        processor, sample, text,
                        cached_prompt_len=cached_pl,
                        cached_pixel_values=cached_pv,
                        cached_image_grid_thw=cached_igt,
                    )
                    if idx_c == 0:
                        cached_pv = fi.get("pixel_values")
                        cached_igt = fi.get("image_grid_thw")
                    full_inputs_list.append(fi)
                    prompt_len_list.append(pl)

                with torch.no_grad():
                    old_lp_list = compute_log_probs_batched(
                        model_engine, full_inputs_list, cached_pl,
                        device, batch_size=args.lp_batch_size,
                    )

                ref_lp_list: List[torch.Tensor] = []
                if ref_model is not None:
                    with torch.no_grad():
                        try:
                            ref_lp_list = compute_log_probs_batched(
                                ref_model, full_inputs_list, cached_pl,
                                device, batch_size=args.lp_batch_size,
                            )
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            ref_lp_list = [lp.clone() for lp in old_lp_list]

                t_lp = time.time() - t_lp_start
                torch.cuda.empty_cache()

                # ---------- Phase 3: Rewards & advantages ----------
                traj_id = sample.get("trajectory_id")
                if traj_id is not None and int(traj_id) in multiref_mapping:
                    refs = multiref_mapping[int(traj_id)]
                else:
                    refs = [sample["instruction"]]

                rewards, all_metrics = reward_computer.compute_rewards(
                    completions, refs, weights=reward_weights,
                )

                rewards_t = torch.tensor(rewards, dtype=torch.float32)
                if rewards_t.std() > 1e-6:
                    advantages = ((rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)).tolist()
                else:
                    advantages = [0.0] * G

            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print(f"[WARN] OOM during generation/old_lp for sample {sample_idx} — skipping entire sample")
                torch.cuda.empty_cache()
                # Feed dummy step(s) to keep DS gradient accumulation counters in sync.
                model_engine.train()
                dummy_steps = 1 if args.batched_train_update else G
                for _ in range(dummy_steps):
                    model_engine.step()
                    global_micro_step += 1
                continue

            # ---------- Phase 4: GRPO training step ----------
            t_train_start = time.time()
            model_engine.train()
            if args.batched_train_update:
                try:
                    current_lp_list = compute_log_probs_batched(
                        model_engine,
                        full_inputs_list,
                        cached_pl,
                        device,
                        batch_size=G,      # one forward for all completions
                        detach_to_cpu=False,
                    )
                    loss_list: List[torch.Tensor] = []
                    for g in range(G):
                        old_lp_g = old_lp_list[g].to(device)
                        ref_lp_g = ref_lp_list[g].to(device) if ref_lp_list else old_lp_g
                        current_lp = current_lp_list[g]
                        loss_g = grpo_loss(
                            current_lp,
                            old_lp_g,
                            ref_lp_g,
                            advantages[g],
                            clip_eps=args.clip_eps,
                            kl_beta=args.kl_beta,
                        )
                        loss_list.append(loss_g)

                        kl_val = (current_lp.detach() - ref_lp_g).mean().item()
                        logger.add(
                            reward=rewards[g],
                            loss=loss_g.detach().item(),
                            kl=kl_val,
                            comp_len=len(completions[g].split()),
                            metrics=all_metrics[g],
                        )

                    total_loss = torch.stack(loss_list).mean()
                    model_engine.backward(total_loss)
                    model_engine.step()
                    global_micro_step += 1
                except torch.cuda.OutOfMemoryError:
                    if rank == 0:
                        print(f"[WARN] OOM on batched-train sample {sample_idx} — skipping")
                    torch.cuda.empty_cache()
                    model_engine.step()
                    global_micro_step += 1
            else:
                for g in range(G):
                    old_lp_g = old_lp_list[g].to(device)
                    ref_lp_g = ref_lp_list[g].to(device) if ref_lp_list else old_lp_g

                    try:
                        current_lp = compute_log_probs(
                            model_engine, full_inputs_list[g], prompt_len_list[g], device
                        )

                        loss = grpo_loss(
                            current_lp,
                            old_lp_g,
                            ref_lp_g,
                            advantages[g],
                            clip_eps=args.clip_eps,
                            kl_beta=args.kl_beta,
                        )

                        model_engine.backward(loss)
                        model_engine.step()
                    except torch.cuda.OutOfMemoryError:
                        if rank == 0:
                            print(f"[WARN] OOM on sample {sample_idx}, completion {g} — skipping")
                        torch.cuda.empty_cache()
                        model_engine.step()
                        global_micro_step += 1
                        continue

                    global_micro_step += 1

                    kl_val = (current_lp.detach() - ref_lp_g).mean().item()
                    logger.add(
                        reward=rewards[g],
                        loss=loss.detach().item(),
                        kl=kl_val,
                        comp_len=len(completions[g].split()),
                        metrics=all_metrics[g],
                    )

            # Check if an optimizer step happened
            if global_micro_step % ds_grad_accum == 0:
                global_opt_step += 1

            # ---------- Logging ----------
            t_train = time.time() - t_train_start
            elapsed = time.time() - t0
            if rank == 0 and (sample_idx + 1) % args.logging_steps == 0:
                print(
                    f"[Epoch {epoch + 1}] sample {sample_idx + 1}/{per_gpu_samples}  "
                    f"opt_step={global_opt_step}  {logger.summary()}  "
                    f"time={elapsed:.1f}s  "
                    f"[gen={t_gen:.1f}s  lp={t_lp:.1f}s  train={t_train:.1f}s]"
                )
                if tb_writer is not None and logger._rewards:
                    s = global_opt_step
                    n = max(len(logger._rewards), 1)
                    tb_writer.add_scalar("rl/reward", sum(logger._rewards) / n, s)
                    tb_writer.add_scalar("rl/loss", sum(logger._losses) / n, s)
                logger.reset()

            # ---------- Mid-epoch checkpoint ----------
            if args.save_steps > 0 and global_opt_step > 0 and global_opt_step % args.save_steps == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_opt_step}")
                if rank == 0:
                    print(f"[INFO] Saving checkpoint → {ckpt_dir}")
                    model_engine.module.save_pretrained(ckpt_dir)
                    processor.save_pretrained(ckpt_dir)
                dist.barrier()

        # ---------- End-of-epoch checkpoint ----------
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch + 1}")
        if rank == 0:
            print(f"[INFO] Saving epoch checkpoint → {ckpt_dir}")
            model_engine.module.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
        dist.barrier()

    # ---- save final model ----
    final_dir = os.path.join(args.output_dir, "final")
    if rank == 0:
        print(f"[INFO] Saving final RL model → {final_dir}")
        model_engine.module.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
    dist.barrier()

    if tb_writer is not None:
        tb_writer.close()

    if rank == 0:
        print("[DONE] GRPO RL training complete.")


if __name__ == "__main__":
    main()
