"""
VP-Adapter configuration.

All hyper-parameters are collected here so that ablation experiments only
need to touch this single file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class VPAdapterConfig:
    # ── VP Semantic Mask ──
    # 3 channels: C0=ribbon/path, C1=arrow, C2=endpoint
    vp_mask_channels: int = 3

    # ── VP Encoder (Residual CNN) ──
    vp_encoder_dims: Tuple[int, ...] = (64, 128, 256)
    vp_dim: int = 384  # output feature dimension (v2.0: 128→384)

    # ── Gated Cross-Attention Adapter ──
    adapter_num_heads: int = 4
    adapter_layers: Tuple[int, ...] = (7,)  # v2.0: only layer 7

    # ── Qwen3-VL-8B vision encoder constants (from config.json) ──
    vit_hidden_dim: int = 1152
    vit_depth: int = 27
    vit_patch_size: int = 16
    deepstack_indexes: Tuple[int, ...] = (8, 16, 24)

    # ── Training strategy ──
    freeze_vit_backbone: bool = True
    vp_module_lr: float = 2e-5
    lm_lr: float = 5e-6
