"""
VP Encoder v2.0: lightweight CNN that encodes a 3-channel VP semantic mask
into patch-aligned feature vectors compatible with Qwen3-VL's ViT.

Changes from v1.0:
  - Wider channels: (48, 96, 128) → (64, 128, 256)
  - vp_dim: 128 → 384 (reduce information bottleneck)
  - CNN architecture unchanged (Conv-GN-GELU blocks)

Input:  list of VP masks, each (C, H_i, W_i) float32 in [0, 1]
Output: (total_patches, vp_dim) tensor aligned 1-to-1 with ViT hidden states
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .vp_config import VPAdapterConfig
except ImportError:
    from vp_config import VPAdapterConfig


class VPEncoder(nn.Module):
    """Lightweight CNN encoder for VP semantic masks."""

    def __init__(self, cfg: VPAdapterConfig):
        super().__init__()
        dims = cfg.vp_encoder_dims  # default: (64, 128, 256)
        in_ch = cfg.vp_mask_channels  # 3

        layers = []
        prev = in_ch
        for d in dims:
            layers.extend([
                nn.Conv2d(prev, d, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(8, d), d),
                nn.GELU(),
            ])
            prev = d
        layers.append(nn.Conv2d(prev, cfg.vp_dim, kernel_size=1))
        self.features = nn.Sequential(*layers)
        self.vp_dim = cfg.vp_dim

    def forward(
        self,
        vp_mask_list: List[torch.Tensor],
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vp_mask_list: list of N tensors, each (C, H_i, W_i) float in [0,1]
            grid_thw: (N, 3) LongTensor — (temporal, height_patches, width_patches)
        Returns:
            (total_patches, vp_dim) tensor, same sequence length as ViT hidden states.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        all_features: List[torch.Tensor] = []

        for i, mask in enumerate(vp_mask_list):
            t, h, w = grid_thw[i].tolist()
            t, h, w = int(t), int(h), int(w)

            mask_4d = mask.unsqueeze(0).to(device=device, dtype=dtype)
            feat = self.features(mask_4d)                       # (1, vp_dim, H', W')
            feat = F.adaptive_avg_pool2d(feat, (h, w))          # (1, vp_dim, h, w)
            feat = feat.flatten(2).transpose(1, 2).squeeze(0)   # (h*w, vp_dim)

            if t > 1:
                feat = feat.unsqueeze(0).expand(t, -1, -1).reshape(t * h * w, -1)

            all_features.append(feat)

        return torch.cat(all_features, dim=0)


def build_vp_encoder(cfg: VPAdapterConfig) -> VPEncoder:
    return VPEncoder(cfg)
