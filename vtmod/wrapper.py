"""
Model wrapper that attaches VP-Adapter modules to a Qwen3-VL model.

Strategy:
  1. Create VPEncoder + VPGatedAdapters
  2. Register them ON the visual model so they are part of the state dict
  3. Replace VisionModel.forward with a patched version that calls adapters
  4. Freeze ViT backbone; only VP modules + language model are trainable

The patched forward mirrors the original Qwen3VLVisionModel.forward exactly,
with adapter calls inserted after the designated layers (v3.0: spatial modulation at layer 7).
"""
from __future__ import annotations

import os
import sys
import types
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from .config import VPAdapterConfig
    from .encoder import VPEncoder, build_vp_encoder
    from .adapter import build_vp_adapters
except ImportError:
    from config import VPAdapterConfig
    from encoder import VPEncoder, build_vp_encoder
    from adapter import build_vp_adapters

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_COMMON = os.path.join(_ROOT, "common")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

from common.dataset import build_inference_prompt


def _vp_enhanced_visual_forward(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
) -> tuple:
    """
    Drop-in replacement for Qwen3VLVisionModel.forward.

    Identical to the original except for VP adapter injection after
    designated layers.  When no VP features are set (self._vp_features
    is None), the adapter calls are skipped and behaviour is identical
    to the unpatched model.
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
            adapter = self._vp_adapters[str(layer_num)]
            hidden_states = adapter(hidden_states, vp_features, cu_seqlens)

        if layer_num in self.deepstack_visual_indexes:
            idx = self.deepstack_visual_indexes.index(layer_num)
            deepstack_feature = self.deepstack_merger_list[idx](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)
    return hidden_states, deepstack_feature_lists


def attach_vp_adapter(
    model: nn.Module,
    cfg: VPAdapterConfig,
) -> tuple:
    """
    Attach VP-Adapter modules to a Qwen3VLForConditionalGeneration model.

    Returns:
        (vp_encoder, vp_adapters) — the new trainable modules
    """
    visual = model.model.visual

    vit_dtype = next(visual.parameters()).dtype
    vit_device = next(visual.parameters()).device

    vp_encoder = build_vp_encoder(cfg).to(dtype=vit_dtype, device=vit_device)

    vp_adapters = build_vp_adapters(cfg).to(dtype=vit_dtype, device=vit_device)
    visual._vp_adapters = vp_adapters
    visual._vp_adapter_layer_set = set(int(k) for k in vp_adapters.keys())
    visual._vp_features = None

    # Register VPEncoder on the model so it is part of model.parameters()
    # and automatically included in Trainer's optimizer / DDP wrapping.
    model._vp_encoder = vp_encoder

    visual.forward = types.MethodType(_vp_enhanced_visual_forward, visual)

    if cfg.freeze_vit_backbone:
        for name, param in visual.named_parameters():
            if "_vp_adapters" not in name:
                param.requires_grad = False
        for adapter in vp_adapters.values():
            for param in adapter.parameters():
                param.requires_grad = True

    return vp_encoder, vp_adapters


def _unwrap(model: nn.Module) -> nn.Module:
    """Unwrap DDP / DeepSpeed / FSDP wrappers."""
    return getattr(model, "module", model)


def set_vp_features(model: nn.Module, vp_features: Optional[torch.Tensor]):
    """Set VP features on the visual model for the current forward pass."""
    _unwrap(model).model.visual._vp_features = vp_features


def clear_vp_features(model: nn.Module):
    """Clear VP features after forward pass (for memory)."""
    _unwrap(model).model.visual._vp_features = None


def get_vp_trainable_params(
    model: nn.Module,
    vp_encoder: nn.Module,
) -> List[dict]:
    """
    Return parameter groups for the optimizer:
      - VP modules (encoder + adapters): higher LR
      - Language model: standard LR
    """
    vp_param_ids = set()
    vp_params = []

    for p in vp_encoder.parameters():
        if p.requires_grad:
            vp_params.append(p)
            vp_param_ids.add(id(p))

    visual = _unwrap(model).model.visual
    if hasattr(visual, "_vp_adapters"):
        for p in visual._vp_adapters.parameters():
            if p.requires_grad:
                vp_params.append(p)
                vp_param_ids.add(id(p))

    lm_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in vp_param_ids
    ]

    return vp_params, lm_params


VP_ENCODER_FILENAME = "vp_encoder.pt"
VP_ADAPTERS_FILENAME = "vp_adapters.pt"

def _compact_state_dict(state: dict) -> dict:
    """Give every tensor its own storage before serialising.

    Under DeepSpeed ZeRO the VP parameters are views into one flattened buffer
    that spans the whole model, and torch.save writes out the entire underlying
    storage of a view. Cloning is the difference between a ~1 MB adapter file
    and a ~16 GB one.
    """
    return {
        k: (v.detach().to("cpu").clone() if torch.is_tensor(v) else v)
        for k, v in state.items()
    }


def save_vp_modules(
    vp_encoder: nn.Module,
    model: nn.Module,
    save_dir: str,
):
    """Save VP encoder and adapter weights alongside the main checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        _compact_state_dict(vp_encoder.state_dict()),
        os.path.join(save_dir, VP_ENCODER_FILENAME),
    )

    visual = _unwrap(model).model.visual
    if hasattr(visual, "_vp_adapters"):
        torch.save(
            _compact_state_dict(visual._vp_adapters.state_dict()),
            os.path.join(save_dir, VP_ADAPTERS_FILENAME),
        )


def load_vp_modules(
    vp_encoder: nn.Module,
    model: nn.Module,
    load_dir: str,
    strict: bool = True,
):
    """Load VP encoder and adapter weights."""
    enc_path = os.path.join(load_dir, VP_ENCODER_FILENAME)
    if os.path.exists(enc_path):
        vp_encoder.load_state_dict(torch.load(enc_path, map_location="cpu"), strict=strict)
        print(f"[VP] Loaded VP encoder from {enc_path}")

    adp_path = os.path.join(load_dir, VP_ADAPTERS_FILENAME)
    visual = _unwrap(model).model.visual
    if os.path.exists(adp_path) and hasattr(visual, "_vp_adapters"):
        visual._vp_adapters.load_state_dict(
            torch.load(adp_path, map_location="cpu"), strict=strict
        )
        print(f"[VP] Loaded VP adapters from {adp_path}")


def print_vp_summary(vp_encoder: nn.Module, model: nn.Module):
    """Print parameter counts for VP modules."""
    enc_total = sum(p.numel() for p in vp_encoder.parameters())
    enc_train = sum(p.numel() for p in vp_encoder.parameters() if p.requires_grad)

    adp_total = 0
    adp_train = 0
    visual = _unwrap(model).model.visual
    if hasattr(visual, "_vp_adapters"):
        adp_total = sum(p.numel() for p in visual._vp_adapters.parameters())
        adp_train = sum(p.numel() for p in visual._vp_adapters.parameters() if p.requires_grad)

    adapter_info = {}
    if hasattr(visual, "_vp_adapters"):
        for name, adapter in visual._vp_adapters.items():
            n_p = sum(p.numel() for p in adapter.parameters())
            adapter_info[f"layer_{name}"] = f"{n_p:,} params"
            if hasattr(adapter, "gate"):
                g = adapter.gate.float()
                adapter_info[f"layer_{name}"] += (
                    f" | gate mean={g.mean().item():.6f}"
                    f" min={g.min().item():.6f}"
                    f" max={g.max().item():.6f}"
                    f" std={g.std().item():.6f}"
                )

    model_total = sum(p.numel() for p in model.parameters())
    model_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("VP-Adapter v3.1 Summary (Spatial Modulation)")
    print("=" * 60)
    print(f"  VP Encoder:   {enc_total:>10,} params ({enc_train:>10,} trainable)")
    print(f"  VP Adapters:  {adp_total:>10,} params ({adp_train:>10,} trainable)")
    print(f"  VP Total:     {enc_total + adp_total:>10,} params")
    print(f"  Full Model:   {model_total:>10,} params ({model_train:>10,} trainable)")
    print(f"  VP overhead:  {(enc_total + adp_total) / model_total * 100:.3f}%")
    for k, v in adapter_info.items():
        print(f"  {k}: {v}")
    print("=" * 60)


def generate_with_vp(
    model,
    processor,
    vp_encoder,
    images: List[Image.Image],
    actions_list: List[str],
    vp_masks: List[np.ndarray],
    max_new_tokens: int = 150,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate one instruction, optionally injecting VP features.

    Used by both benchmark inference (`eval/infer.py`) and data-augmentation
    generation (`generate/generate_instructions.py`). An empty `vp_masks` list
    (or a None encoder) skips injection so the vision tower runs unmodified.
    """
    user_content = build_inference_prompt(images, actions_list, system_prompt=system_prompt)
    messages = [{"role": "user", "content": user_content}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )

    model_device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    grid_thw = inputs.get("image_grid_thw")
    if grid_thw is not None and vp_encoder is not None and vp_masks:
        vp_device = next(vp_encoder.parameters()).device
        mask_tensors = [
            torch.from_numpy(m).permute(2, 0, 1).float()
            for m in vp_masks
        ]
        with torch.no_grad():
            vp_features = vp_encoder(mask_tensors, grid_thw.to(vp_device))
        set_vp_features(model, vp_features)

    with torch.no_grad():
        if temperature > 0:
            gen_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature, top_p=0.9,
            )
        else:
            gen_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )

    clear_vp_features(model)

    gen_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()
