#!/usr/bin/env python3
"""
GRPO RL training with VP-Adapter for Qwen3-VL Navigation Instruction Generation.

Extends grpo_rl_train.py with VP-Adapter integration:
  - Loads VP encoder + adapter weights from the SFT checkpoint
  - Computes VP features from semantic masks before each forward pass
  - Freezes VP modules (GRPO only updates the language model)

Launch:
    deepspeed --num_gpus=N grpo_rl_train_vp.py \
        --sft_checkpoint /path/to/vp_sft_ckpt \
        --r2rce_data_dir /path/to/R2RCE_visual/r2rce_train_visual \
        --rxrce_data_dir /path/to/RXRCE_visual/rxrce_train_visual \
        --output_dir /path/to/rl_output \
        --ds_config ds_zero2_config.json
"""

import argparse
import inspect
import json
import os
import random
import sys
import time
import types
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader, DistributedSampler

try:
    import deepspeed
except ImportError:
    raise ImportError("deepspeed is required. Install with: pip install deepspeed")

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ---- Import utilities from the original GRPO script ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grpo_rl_train import (
    patch_deepspeed_grad_count_for_torch_compat,
    R2RCE_SYSTEM_PROMPT,
    RXRCE_SYSTEM_PROMPT,
    PANORAMA_SYSTEM_PROMPT,
    infer_dataset_type_from_path,
    get_system_prompt,
    safe_load_json,
    group_actions,
    build_inference_prompt,
    _rl_collate,
    _get_prompt_len,
    _prepare_full_inputs,
    compute_log_probs,
    compute_log_probs_batched,
    grpo_loss,
    RLMetricsLogger,
)
from grpo_reward import RewardComputer

# ---- Import VP-Adapter modules ----
VP_ADAPTER_DIR = os.environ.get(
    "VP_ADAPTER_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vp_adapter"),
)
sys.path.insert(0, VP_ADAPTER_DIR)
from vp_config import VPAdapterConfig
from vp_model_wrapper import (
    attach_vp_adapter,
    load_vp_modules,
    save_vp_modules,
    set_vp_features,
    clear_vp_features,
    _unwrap,
    print_vp_summary,
)

# ============================================================================
# VP Mask loading (adapted from vp_dataset.py)
# ============================================================================

def _load_vp_mask(mask_path: str, episode_dir: str) -> Optional[np.ndarray]:
    """Load a VP semantic mask, return (H, W, 3) float32 in [0, 1]."""
    if not mask_path:
        return None

    candidates = [mask_path]
    candidates.append(os.path.join(episode_dir, os.path.basename(mask_path)))

    # Prefer PNG (much smaller than .npy, 3-channel semantic)
    png_candidates = []
    for c in candidates:
        png_candidates.append(c)
        # Also try .png variant of .npy path
        if c.endswith("_vpmask.npy"):
            png_candidates.append(c.replace("_vpmask.npy", "_vpmask.png"))

    for p in png_candidates:
        if not os.path.exists(p):
            continue
        if p.endswith(".npy"):
            arr = np.load(p).astype(np.float32)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            return arr
        else:
            img = Image.open(p)
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.max() > 1.0:
                arr = arr / 255.0
            return arr

    return None


# ============================================================================
# VP-aware Dataset
# ============================================================================

class RLVPDataset(Dataset):
    """
    RL dataset that loads trajectory frames + VP semantic masks.

    Differences from RLPanoramaDataset:
      - Also loads VP masks from sample.json['vp_masks']
      - Returns vp_masks as a list of numpy arrays in __getitem__
    """

    def __init__(
        self,
        data_dirs: List[str],
        filtered_json: Optional[str] = None,
        max_frames: int = 32,
        max_samples: int = 0,
        max_rxrce: int = 0,
        max_r2rce: int = 0,
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
            print(f"[RLVPDataset] RXRCE subsampled: {len(rxrce_files)}/{max_rxrce}")

        if max_r2rce > 0 and len(r2rce_files) > max_r2rce:
            r2rce_files = random.Random(seed).sample(r2rce_files, max_r2rce)
            print(f"[RLVPDataset] R2RCE subsampled: {len(r2rce_files)}/{max_r2rce}")
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
        print(f"[RLVPDataset] {len(self.sample_files)} samples "
              f"(R2RCE={r2rce_count}, RXRCE={rxrce_count})")

    def __len__(self) -> int:
        return len(self.sample_files)

    @staticmethod
    def _resolve_path(stored_path: str, episode_dir: str) -> str:
        """Resolve a stored sample path, falling back to the episode directory."""
        if os.path.exists(stored_path):
            return stored_path
        return os.path.join(episode_dir, os.path.basename(stored_path))

    def __getitem__(self, idx: int) -> dict:
        n = len(self.sample_files)
        for off in range(n):
            j = (idx + off) % n
            try:
                return self._build_item(j)
            except Exception as e:
                if off == 0:
                    print(f"[RLVPDataset] skip bad sample idx={j}: {e}", flush=True)
                continue
        raise RuntimeError("No valid samples available")

    def _build_item(self, idx: int) -> dict:
        sample_path = self.sample_files[idx]
        sample = safe_load_json(sample_path)
        if sample is None:
            raise ValueError(f"Cannot load: {sample_path}")

        if "action_events_text" in sample:
            grouped = sample["action_events_text"]
        else:
            raw = sample.get("actions_text") or [str(a) for a in sample.get("actions", [])]
            gs = int(sample.get("actions_per_frame") or sample.get("frame_stride") or 2)
            grouped = group_actions(raw, gs)

        nf = min(len(sample["frames"]), len(grouped) + 1, self.max_frames)
        frames = sample["frames"][:nf]
        actions_list = grouped[: max(0, nf - 1)]

        episode_dir = os.path.dirname(sample_path)

        images = []
        for fp in frames:
            p = self._resolve_path(fp, episode_dir)
            images.append(Image.open(p).convert("RGB"))

        # Load VP masks
        vp_mask_paths = sample.get("vp_masks", [])
        vp_masks: List[np.ndarray] = []
        for i in range(nf):
            mask = None
            if i < len(vp_mask_paths) and vp_mask_paths[i]:
                mask = _load_vp_mask(vp_mask_paths[i], episode_dir)
            if mask is None:
                img_arr = np.array(images[i])
                h, w = img_arr.shape[:2]
                mask = np.zeros((h, w, 3), dtype=np.float32)
            vp_masks.append(mask)

        return {
            "images": images,
            "actions_list": actions_list,
            "vp_masks": vp_masks,
            "system_prompt": get_system_prompt(
                self.sample_dataset_type.get(sample_path, "r2rce")),
            "instruction": sample["instruction"],
            "episode_id": sample.get("episode_id"),
            "trajectory_id": sample.get("trajectory_id"),
            "dataset_type": self.sample_dataset_type.get(sample_path, "r2rce"),
        }


# ============================================================================
# VP Feature computation helper
# ============================================================================

def compute_vp_features(
    vp_encoder: torch.nn.Module,
    vp_masks: List[np.ndarray],
    grid_thw: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Compute VP features from masks using the VP encoder.

    Returns (total_patches, vp_dim) tensor, or None if no masks.
    """
    if vp_encoder is None or not vp_masks or grid_thw is None:
        return None

    device = next(vp_encoder.parameters()).device
    mask_tensors = [
        torch.from_numpy(m).permute(2, 0, 1).float()  # (3, H, W)
        for m in vp_masks
    ]

    with torch.no_grad():
        vp_features = vp_encoder(mask_tensors, grid_thw.to(device))

    return vp_features


# ============================================================================
# VP-aware generation
# ============================================================================

@torch.no_grad()
def vp_generate_completions(
    model: torch.nn.Module,
    processor,
    sample: dict,
    vp_encoder: torch.nn.Module,
    num_completions: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    gen_batch_size: int = 8,
) -> List[str]:
    """Generate completions with VP features set during visual encoding."""
    images = sample["images"]
    actions_list = sample["actions_list"]
    system_prompt = sample.get("system_prompt", PANORAMA_SYSTEM_PROMPT)
    vp_masks = sample.get("vp_masks", [])

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

    # Compute and set VP features
    grid_thw = inputs.get("image_grid_thw")
    vp_feat = compute_vp_features(vp_encoder, vp_masks, grid_thw)
    if vp_feat is not None:
        set_vp_features(model, vp_feat)

    completions: List[str] = []
    remaining = num_completions
    try:
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
    finally:
        clear_vp_features(model)

    return completions


# ============================================================================
# VP-aware log-prob computation
# ============================================================================

def vp_compute_log_probs_batched(
    model: torch.nn.Module,
    full_inputs_list: List[dict],
    prompt_len: int,
    device: torch.device,
    vp_encoder: torch.nn.Module,
    vp_masks: List[np.ndarray],
    batch_size: int = 8,
    detach_to_cpu: bool = True,
) -> List[torch.Tensor]:
    """Compute log-probs with VP features set during each forward pass."""

    grid_thw = full_inputs_list[0].get("image_grid_thw")
    vp_feat = compute_vp_features(vp_encoder, vp_masks, grid_thw)

    if vp_feat is not None:
        set_vp_features(model, vp_feat)
    try:
        results = compute_log_probs_batched(
            model, full_inputs_list, prompt_len, device,
            batch_size=batch_size, detach_to_cpu=detach_to_cpu,
        )
    finally:
        clear_vp_features(model)

    return results


# ============================================================================
# Enhanced VP visual forward (with batch-repeat support)
# ============================================================================

def _vp_enhanced_visual_forward_with_repeat(self, hidden_states, grid_thw, **kwargs):
    """
    Drop-in replacement for Qwen3VLVisionModel.forward with VP adapter
    injection + automatic batch-repeat for multi-completion log-prob batching.
    """
    hidden_states = self.patch_embed(hidden_states)

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    vp_features = getattr(self, "_vp_features", None)
    adapter_layer_set = getattr(self, "_vp_adapter_layer_set", set())

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if vp_features is not None and layer_num in adapter_layer_set:
            vp_feat = vp_features
            # Handle batched inputs: hidden_states may be N× larger
            if vp_feat.shape[0] != hidden_states.shape[0]:
                ratio = hidden_states.shape[0] // vp_feat.shape[0]
                if ratio > 1 and hidden_states.shape[0] == vp_feat.shape[0] * ratio:
                    vp_feat = vp_feat.repeat_interleave(ratio, dim=0)
            adapter = self._vp_adapters[str(layer_num)]
            hidden_states = adapter(hidden_states, vp_feat, cu_seqlens)

        if layer_num in self.deepstack_visual_indexes:
            idx_ds = self.deepstack_visual_indexes.index(layer_num)
            deepstack_feature = self.deepstack_merger_list[idx_ds](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)
    return hidden_states, deepstack_feature_lists


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GRPO RL with VP-Adapter for Qwen3-VL NIG")

    # ----- paths -----
    p.add_argument("--sft_checkpoint", required=True, help="VP-SFT checkpoint directory")
    p.add_argument("--processor_dir", default="", help="Processor dir (defaults to base Qwen3-VL)")
    p.add_argument("--r2rce_data_dir", required=True, help="R2RCE visual training data directory")
    p.add_argument("--rxrce_data_dir", required=True, help="RXRCE visual training data directory")
    p.add_argument("--filtered_json", default="", help="Optional episode-filter JSON")
    p.add_argument("--r2r_train_json", default="", help="R2R_train.json for multi-reference")
    p.add_argument("--output_dir", required=True, help="RL output / checkpoint directory")
    p.add_argument("--logging_dir", default="", help="TensorBoard log directory")
    p.add_argument("--ds_config", required=True, help="DeepSpeed ZeRO-2 JSON config")

    # ----- VP adapter config -----
    p.add_argument("--vp_adapter_layers", default="7", help="Comma-separated adapter layer indices")
    p.add_argument("--vp_dim", type=int, default=384, help="VP encoder output dimension")
    p.add_argument("--freeze_vp", action="store_true", default=True,
                   help="Freeze VP encoder + adapters during GRPO (default: True)")
    p.add_argument("--no_freeze_vp", action="store_true",
                   help="Allow VP modules to receive gradients during GRPO")
    p.add_argument("--gate_only", action="store_true",
                   help="Only unfreeze gate params in VP adapters (VP encoder stays frozen)")
    p.add_argument("--gate_contrast_alpha", type=float, default=0.01,
                   help="Weight for contrastive gate loss (default: 0.01, keep small)")
    p.add_argument("--gate_contrast_tau", type=float, default=1.0,
                   help="Temperature for contrastive gate sigmoid (default: 1.0)")
    p.add_argument("--gate_lr_scale", type=float, default=0.1,
                   help="Multiply gate gradient by this factor (effective LR = lr * scale)")

    # ----- data -----
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=42)

    # ----- RL hyper-params -----
    p.add_argument("--group_size", type=int, default=16)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_beta", type=float, default=0.10,
                   help="KL penalty coefficient")
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=150)

    # ----- optimiser -----
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)

    # ----- reward weights -----
    p.add_argument("--w_bleu1", type=float, default=0.2)
    p.add_argument("--w_bleu4", type=float, default=0.2)
    p.add_argument("--w_cider", type=float, default=0.2)
    p.add_argument("--w_meteor", type=float, default=0.2)
    p.add_argument("--w_rouge_l", type=float, default=0.2)

    # ----- misc -----
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--gen_batch_size", type=int, default=16)
    p.add_argument("--lp_batch_size", type=int, default=8)
    p.add_argument("--r2rce_temperature", type=float, default=0.5,
                   help="R2RCE sampling temperature")
    p.add_argument("--rxrce_temperature", type=float, default=1.0,
                   help="RXRCE sampling temperature")
    p.add_argument("--max_rxrce", type=int, default=0)
    p.add_argument("--max_r2rce", type=int, default=0)
    p.add_argument("--batched_train_update", action="store_true")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--no_ref_model", action="store_true")

    # ---- resume (optional; defaults reproduce a clean run from SFT) ----
    p.add_argument(
        "--policy_checkpoint",
        default="",
        help="Policy init weights (defaults to --sft_checkpoint). Use RL current/best to resume.",
    )
    p.add_argument(
        "--ref_checkpoint",
        default="",
        help="Frozen KL reference checkpoint (defaults to --sft_checkpoint).",
    )
    p.add_argument(
        "--skip_samples",
        type=int,
        default=0,
        help="Skip the first N per-rank dataloader samples (resume mid-epoch).",
    )
    p.add_argument(
        "--start_opt_step",
        type=int,
        default=0,
        help="Initialize global_opt_step counter when resuming.",
    )

    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    patch_deepspeed_grad_count_for_torch_compat()

    freeze_vp = not args.no_freeze_vp and not args.gate_only

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
    if not args.policy_checkpoint:
        args.policy_checkpoint = args.sft_checkpoint
    if not args.ref_checkpoint:
        args.ref_checkpoint = args.sft_checkpoint

    reward_weights = {
        "BLEU-1": args.w_bleu1,
        "BLEU-4": args.w_bleu4,
        "CIDEr": args.w_cider,
        "METEOR": args.w_meteor,
        "ROUGE-L": args.w_rouge_l,
    }

    vp_cfg = VPAdapterConfig(
        adapter_layers=tuple(int(x) for x in args.vp_adapter_layers.split(",")),
        vp_dim=args.vp_dim,
        freeze_vit_backbone=True,
    )

    if rank == 0:
        print("=" * 60)
        print("GRPO RL with VP-Adapter")
        print("=" * 60)
        print(f"SFT checkpoint:  {args.sft_checkpoint}")
        print(f"Policy init:     {args.policy_checkpoint}")
        print(f"Ref checkpoint:  {args.ref_checkpoint}")
        print(f"Skip samples:    {args.skip_samples}")
        print(f"Start opt_step:  {args.start_opt_step}")
        print(f"Save steps:      {args.save_steps} (0=no mid-ckpt)")
        print(f"Processor:       {args.processor_dir}")
        print(f"R2RCE data:      {args.r2rce_data_dir}")
        print(f"RXRCE data:      {args.rxrce_data_dir}")
        print(f"Filtered JSON:   {args.filtered_json}")
        print(f"Output:          {args.output_dir}")
        print(f"VP adapter layers: {vp_cfg.adapter_layers}")
        print(f"VP dim:          {vp_cfg.vp_dim}")
        print(f"Freeze VP:       {freeze_vp}")
        print(f"Group size:      {args.group_size}")
        print(f"LR:              {args.lr}")
        print(f"Epochs:          {args.num_epochs}")
        print(f"Grad accum:      {args.grad_accum}")
        print(f"No ref model:    {args.no_ref_model}")
        print(f"Batched train:   {args.batched_train_update}")
        print(f"World size:      {world_size}")
        print(f"R2RCE temp:      {args.r2rce_temperature}")
        print(f"RXRCE temp:      {args.rxrce_temperature}")
        print(f"KL beta:         {args.kl_beta}")
        print("=" * 60)

    # ---- load processor ----
    if rank == 0:
        print("[INFO] Loading processor …")
    processor = AutoProcessor.from_pretrained(
        args.processor_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    # ---- load policy model + VP-adapter ----
    if rank == 0:
        print(f"[INFO] Loading policy model from {args.policy_checkpoint} …")
    policy_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.policy_checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )

    if rank == 0:
        print("[INFO] Attaching VP-Adapter to policy model …")
    vp_encoder, _ = attach_vp_adapter(policy_model, vp_cfg)

    # Re-patch the visual forward to include batch-repeat logic
    visual = _unwrap(policy_model).model.visual
    visual.forward = types.MethodType(
        _vp_enhanced_visual_forward_with_repeat, visual)

    if rank == 0:
        print(f"[INFO] Loading VP weights from {args.policy_checkpoint} …")
    load_vp_modules(vp_encoder, policy_model, args.policy_checkpoint)

    # Freeze vision encoder backbone (always frozen for GRPO)
    policy_model.model.visual.requires_grad_(False)

    if args.gate_only:
        # Gate-only mode: unfreeze ONLY gate params in VP adapters
        vp_encoder.requires_grad_(False)
        if hasattr(visual, "_vp_adapters"):
            visual._vp_adapters.requires_grad_(False)
            for adapter in visual._vp_adapters.values():
                if hasattr(adapter, "gate"):
                    adapter.gate.requires_grad = True
        if rank == 0:
            n_gate = sum(
                adapter.gate.numel()
                for adapter in visual._vp_adapters.values()
                if hasattr(adapter, "gate")
            )
            print(f"[INFO] VP GATE-ONLY mode: {n_gate} gate params trainable, "
                  f"contrast_α={args.gate_contrast_alpha}, "
                  f"contrast_τ={args.gate_contrast_tau}, "
                  f"gate_lr_scale={args.gate_lr_scale}")
    elif freeze_vp:
        vp_encoder.requires_grad_(False)
        if hasattr(visual, "_vp_adapters"):
            visual._vp_adapters.requires_grad_(False)
        if rank == 0:
            print("[INFO] VP modules FROZEN (GRPO only updates language model)")
    else:
        vp_encoder.requires_grad_(True)
        if hasattr(visual, "_vp_adapters"):
            for adapter in visual._vp_adapters.values():
                adapter.requires_grad_(True)
        if rank == 0:
            print("[INFO] VP modules TRAINABLE during GRPO")

    # Register gradient scaling hook on gate params (runs during backward)
    if args.gate_only and args.gate_lr_scale != 1.0:
        _gate_scale = args.gate_lr_scale
        for _adp in visual._vp_adapters.values():
            if hasattr(_adp, "gate") and _adp.gate.requires_grad:
                _adp.gate.register_hook(lambda grad, s=_gate_scale: grad * s)
        if rank == 0:
            print(f"[INFO] Gate gradient hook registered (scale={_gate_scale})")

    if rank == 0:
        print_vp_summary(vp_encoder, policy_model)

    if not args.no_gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
        policy_model.config.use_cache = False
        if rank == 0:
            print("[INFO] Gradient checkpointing ENABLED")

    # Move VP encoder to device
    vp_encoder = vp_encoder.to(device)

    # ---- load reference model (frozen, for KL regularisation) ----
    ref_model = None
    ref_vp_encoder = None
    if not args.no_ref_model:
        if rank == 0:
            print(f"[INFO] Loading frozen reference model from {args.ref_checkpoint} …")
        ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.ref_checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        )
        ref_vp_enc, _ = attach_vp_adapter(ref_model, vp_cfg)
        ref_visual = _unwrap(ref_model).model.visual
        ref_visual.forward = types.MethodType(
            _vp_enhanced_visual_forward_with_repeat, ref_visual)
        load_vp_modules(ref_vp_enc, ref_model, args.ref_checkpoint)

        ref_model.requires_grad_(False)
        ref_vp_enc.requires_grad_(False)
        ref_model.eval()
        ref_model.to(device)
        ref_vp_enc.to(device)
        ref_vp_encoder = ref_vp_enc
        if rank == 0:
            alloc_gb = torch.cuda.memory_allocated(device) / 1e9
            total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            print(f"[INFO] Reference model loaded on {device} "
                  f"(GPU mem: {alloc_gb:.1f}/{total_gb:.0f} GB)")
    else:
        if rank == 0:
            print("[INFO] Reference model DISABLED (--no_ref_model)")

    # ---- dataset & dataloader ----
    data_dirs = [args.r2rce_data_dir, args.rxrce_data_dir]
    dataset = RLVPDataset(
        data_dirs=data_dirs,
        filtered_json=args.filtered_json or None,
        max_frames=args.max_frames,
        max_samples=args.max_samples,
        max_rxrce=args.max_rxrce,
        max_r2rce=args.max_r2rce,
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
        for entry in r2r_data:
            pid = entry.get("path_id")
            if pid is not None:
                pid = int(pid)
                insts = entry.get("instructions", [])
                if insts:
                    if pid not in multiref_mapping:
                        multiref_mapping[pid] = list(insts)
                    else:
                        existing = set(multiref_mapping[pid])
                        for inst in insts:
                            if inst not in existing:
                                multiref_mapping[pid].append(inst)
        if rank == 0:
            print(f"[INFO] Loaded {len(multiref_mapping)} trajectory refs from R2R_train.json")

    traj_refs: Dict[int, set] = {}
    added = 0
    for sample_path in dataset.sample_files:
        s = safe_load_json(sample_path)
        if s is None:
            continue
        tid = s.get("trajectory_id")
        if tid is None:
            continue
        tid = int(tid)
        refs = set()
        inst = s.get("instruction")
        if inst:
            refs.add(inst)
        if tid in traj_refs:
            traj_refs[tid] |= refs
        else:
            traj_refs[tid] = refs
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
        print(f"[INFO] Multiref mapping: {n_total} trajectories, avg {avg_refs:.1f} refs/traj")
    del traj_refs

    reward_computer = RewardComputer()
    reward_computer.build_cider_idf(multiref_mapping)

    per_gpu_samples = len(dataloader)
    G = args.group_size
    if args.batched_train_update:
        ds_grad_accum = args.grad_accum
    else:
        ds_grad_accum = G * args.grad_accum
    total_optimizer_steps = (per_gpu_samples * args.num_epochs) // args.grad_accum
    total_optimizer_steps = max(total_optimizer_steps, 1)
    warmup_steps = max(int(total_optimizer_steps * args.warmup_ratio), 1)

    if rank == 0:
        print(f"[INFO] Samples per GPU: {per_gpu_samples}")
        print(f"[INFO] DS grad_accum: {ds_grad_accum}")
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
            "torch_adam": True,
        },
    }

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

    tb_writer = None
    if rank == 0 and args.logging_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=args.logging_dir)
        except ImportError:
            pass

    # ---- Helper: save checkpoint with VP modules ----
    def _save_checkpoint(ckpt_dir: str):
        if rank == 0:
            print(f"[INFO] Saving checkpoint → {ckpt_dir}")
            model_engine.module.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            save_vp_modules(vp_encoder, model_engine.module, ckpt_dir)
            print(f"[INFO] VP modules saved to {ckpt_dir}")
        dist.barrier()

    # ---- training ----
    global_micro_step = 0
    global_opt_step = int(args.start_opt_step)
    # ---- robustness trackers (Dr.GRPO + KL early-stop + numerical guard) ----
    best_reward = -1.0
    reward_window = []
    kl_accum = []
    kl_bad_streak = 0
    early_stop = False
    _last_saved_opt_step = -1
    KL_STOP = 0.8
    KL_PATIENCE = 3
    LOSS_GUARD = 10.0
    logger = RLMetricsLogger()

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print(f"{'=' * 60}")

        for sample_idx, sample in enumerate(dataloader):
            if sample_idx < args.skip_samples:
                continue
            t0 = time.time()
            vp_masks = sample.get("vp_masks", [])

            ds_type = sample.get("dataset_type", "r2rce")
            sample_temp = args.rxrce_temperature if ds_type == "rxrce" else args.r2rce_temperature
            current_kl_beta = args.kl_beta

            try:
                # ---------- Phase 1: Generate G completions ----------
                t_gen_start = time.time()
                model_engine.eval()
                completions = vp_generate_completions(
                    model_engine.module,
                    processor,
                    sample,
                    vp_encoder,
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

                with torch.no_grad():
                    old_lp_list = vp_compute_log_probs_batched(
                        model_engine, full_inputs_list, cached_pl,
                        device, vp_encoder, vp_masks,
                        batch_size=args.lp_batch_size,
                    )

                ref_lp_list: List[torch.Tensor] = []
                if ref_model is not None:
                    with torch.no_grad():
                        try:
                            ref_lp_list = vp_compute_log_probs_batched(
                                ref_model, full_inputs_list, cached_pl,
                                device, ref_vp_encoder, vp_masks,
                                batch_size=args.lp_batch_size,
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
                # Dr.GRPO: mean-subtract baseline only (NO std normalization), clamp for safety;
                # homogeneous low-variance group -> zero advantage to avoid amplifying noise.
                if rewards_t.std() > 1e-3:
                    advantages = (rewards_t - rewards_t.mean()).clamp(-3.0, 3.0).tolist()
                else:
                    advantages = [0.0] * G

            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print(f"[WARN] OOM during generation/old_lp for sample {sample_idx} — skipping")
                torch.cuda.empty_cache()
                model_engine.train()
                dummy_steps = 1 if args.batched_train_update else G
                for _ in range(dummy_steps):
                    model_engine.step()
                    global_micro_step += 1
                continue

            # ---------- Phase 4: GRPO training step ----------
            t_train_start = time.time()
            model_engine.train()

            best_idx = int(np.argmax(rewards))
            worst_idx = int(np.argmin(rewards))

            # Set VP features for training forward passes
            grid_thw = full_inputs_list[0].get("image_grid_thw")
            vp_feat = compute_vp_features(vp_encoder, vp_masks, grid_thw)

            if args.batched_train_update:
                try:
                    if vp_feat is not None:
                        set_vp_features(model_engine, vp_feat)
                    current_lp_list = compute_log_probs_batched(
                        model_engine,
                        full_inputs_list,
                        cached_pl,
                        device,
                        batch_size=G,
                        detach_to_cpu=False,
                    )
                    clear_vp_features(model_engine)

                    loss_list: List[torch.Tensor] = []
                    _kls_this = []
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
                            kl_beta=current_kl_beta,
                        )
                        loss_list.append(loss_g)

                        kl_val = (current_lp.detach() - ref_lp_g).mean().item()
                        _kls_this.append(kl_val)
                        logger.add(
                            reward=rewards[g],
                            loss=loss_g.detach().item(),
                            kl=kl_val,
                            comp_len=len(completions[g].split()),
                            metrics=all_metrics[g],
                        )

                    kl_accum.append(sum(_kls_this) / max(len(_kls_this), 1))
                    reward_window.append(float(sum(rewards) / max(len(rewards), 1)))
                    if len(reward_window) > 100:
                        reward_window.pop(0)

                    grpo_loss_val = torch.stack(loss_list).mean()

                    # Contrastive gate loss (VP-GRPO): use per-token mean for length invariance
                    gate_contrast_loss = torch.tensor(0.0, device=device)
                    if args.gate_only and rewards[best_idx] > rewards[worst_idx]:
                        lp_best_mean = current_lp_list[best_idx].mean()
                        lp_worst_mean = current_lp_list[worst_idx].mean()
                        gate_contrast_loss = -torch.log(
                            torch.sigmoid(
                                (lp_best_mean - lp_worst_mean) / args.gate_contrast_tau
                            ) + 1e-8
                        )

                    total_loss = grpo_loss_val + args.gate_contrast_alpha * gate_contrast_loss
                    # numerical guard: neutralize non-finite / exploding loss (keeps DDP collectives in sync)
                    if (not torch.isfinite(total_loss)) or (total_loss.detach().abs().item() > LOSS_GUARD):
                        if rank == 0:
                            print(f"[GUARD] opt_step~{global_opt_step}: bad loss={total_loss.detach().item():.3f} neutralized", flush=True)
                        total_loss = total_loss * 0.0
                    model_engine.backward(total_loss)
                    model_engine.step()
                    global_micro_step += 1
                except torch.cuda.OutOfMemoryError:
                    if rank == 0:
                        print(f"[WARN] OOM on batched-train sample {sample_idx} — skipping")
                    torch.cuda.empty_cache()
                    clear_vp_features(model_engine)
                    model_engine.step()
                    global_micro_step += 1
            else:
                for g in range(G):
                    old_lp_g = old_lp_list[g].to(device)
                    ref_lp_g = ref_lp_list[g].to(device) if ref_lp_list else old_lp_g

                    try:
                        if vp_feat is not None:
                            set_vp_features(model_engine, vp_feat)
                        current_lp = compute_log_probs(
                            model_engine, full_inputs_list[g], cached_pl, device
                        )
                        clear_vp_features(model_engine)

                        loss = grpo_loss(
                            current_lp,
                            old_lp_g,
                            ref_lp_g,
                            advantages[g],
                            clip_eps=args.clip_eps,
                            kl_beta=current_kl_beta,
                        )

                        # Contrastive gate loss for non-batched branch:
                        # use per-token mean for length invariance
                        if args.gate_only and rewards[best_idx] > rewards[worst_idx]:
                            if g == best_idx:
                                lp_worst_ref = old_lp_list[worst_idx].to(device).detach().mean()
                                gc_loss = -torch.log(torch.sigmoid(
                                    (current_lp.mean() - lp_worst_ref) / args.gate_contrast_tau
                                ) + 1e-8)
                                loss = loss + args.gate_contrast_alpha * gc_loss
                            elif g == worst_idx:
                                lp_best_ref = old_lp_list[best_idx].to(device).detach().mean()
                                gc_loss = -torch.log(torch.sigmoid(
                                    (lp_best_ref - current_lp.mean()) / args.gate_contrast_tau
                                ) + 1e-8)
                                loss = loss + args.gate_contrast_alpha * gc_loss

                        model_engine.backward(loss)
                        model_engine.step()
                    except torch.cuda.OutOfMemoryError:
                        if rank == 0:
                            print(f"[WARN] OOM on sample {sample_idx}, completion {g} — skipping")
                        torch.cuda.empty_cache()
                        clear_vp_features(model_engine)
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

            if global_micro_step % ds_grad_accum == 0:
                global_opt_step += 1
                # ---- distributed KL early-stop ----
                _kl_step = abs(sum(kl_accum) / len(kl_accum)) if kl_accum else 0.0
                kl_accum.clear()
                _bad = torch.tensor([1.0 if _kl_step > KL_STOP else 0.0], device=device)
                try:
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(_bad, op=dist.ReduceOp.MAX)
                except Exception:
                    pass
                if _bad.item() > 0.5:
                    kl_bad_streak += 1
                else:
                    kl_bad_streak = 0
                if kl_bad_streak >= KL_PATIENCE:
                    if rank == 0:
                        print(f"[EARLYSTOP] |KL|>{KL_STOP} for {KL_PATIENCE} opt-steps @step{global_opt_step}; saving & stopping", flush=True)
                    early_stop = True

            # ---------- Logging ----------
            t_train = time.time() - t_train_start
            elapsed = time.time() - t0
            if rank == 0 and (sample_idx + 1) % args.logging_steps == 0:
                gate_info = ""
                if args.gate_only:
                    with torch.no_grad():
                        _vis = _unwrap(model_engine).model.visual
                        _gmeans = []
                        for _adp in _vis._vp_adapters.values():
                            if hasattr(_adp, "gate"):
                                _gmeans.append(_adp.gate.data.float().mean().item())
                        if _gmeans:
                            gate_info = f"  gate_mean={sum(_gmeans)/len(_gmeans):.5f}"
                print(
                    f"[Epoch {epoch + 1}] sample {sample_idx + 1}/{per_gpu_samples}  "
                    f"opt_step={global_opt_step}  {logger.summary()}  "
                    f"temp={sample_temp:.3f}  kl_β={current_kl_beta:.4f}{gate_info}  "
                    f"time={elapsed:.1f}s  "
                    f"[gen={t_gen:.1f}s  lp={t_lp:.1f}s  train={t_train:.1f}s]"
                )
                if tb_writer is not None and logger._rewards:
                    s = global_opt_step
                    n = max(len(logger._rewards), 1)
                    tb_writer.add_scalar("rl/reward", sum(logger._rewards) / n, s)
                    tb_writer.add_scalar("rl/loss", sum(logger._losses) / n, s)
                    tb_writer.add_scalar("rl/temperature", sample_temp, s)
                    tb_writer.add_scalar("rl/kl_beta", current_kl_beta, s)
                    if args.gate_only:
                        with torch.no_grad():
                            _vis = _unwrap(model_engine).model.visual
                            for _aname, _adp in _vis._vp_adapters.items():
                                if hasattr(_adp, "gate"):
                                    _g = _adp.gate.data.float()
                                    tb_writer.add_scalar(f"vp/gate_{_aname}_mean", _g.mean().item(), s)
                                    tb_writer.add_scalar(f"vp/gate_{_aname}_std", _g.std().item(), s)
                                    tb_writer.add_scalar(f"vp/gate_{_aname}_absmax", _g.abs().max().item(), s)
                logger.reset()

            # ---------- periodic checkpoint: keep ONLY "current" + "best" ----------
            if (args.save_steps > 0 and global_opt_step > 0
                    and global_opt_step % args.save_steps == 0
                    and global_opt_step != _last_saved_opt_step):
                _last_saved_opt_step = global_opt_step
                _cur_reward = (sum(reward_window) / len(reward_window)) if reward_window else 0.0
                _save_checkpoint(os.path.join(args.output_dir, "current"))
                if rank == 0:
                    print(f"[CKPT] current @step{global_opt_step} reward={_cur_reward:.4f} best={best_reward:.4f}", flush=True)
                if _cur_reward > best_reward:
                    best_reward = _cur_reward
                    _save_checkpoint(os.path.join(args.output_dir, "best"))
                    if rank == 0:
                        print(f"[CKPT] new BEST reward={best_reward:.4f} -> saved 'best'", flush=True)

            if early_stop:
                break

        # ---------- End-of-epoch: refresh "current" (+ best if improved) ----------
        _cur_reward = (sum(reward_window) / len(reward_window)) if reward_window else 0.0
        _save_checkpoint(os.path.join(args.output_dir, "current"))
        if _cur_reward > best_reward:
            best_reward = _cur_reward
            _save_checkpoint(os.path.join(args.output_dir, "best"))
        if early_stop:
            break

    # ---- final: keep "current" as last state (only current + best retained) ----
    _save_checkpoint(os.path.join(args.output_dir, "current"))

    if tb_writer is not None:
        tb_writer.close()

    if rank == 0:
        print("[DONE] GRPO RL with VP-Adapter training complete.")


if __name__ == "__main__":
    main()
