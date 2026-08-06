"""
VP Spatial Modulation Adapter v3.1.

Changes from v3.0:
  - sigmoid(gate_net(x)) → direct learnable vector gate (no sigmoid, no MLP)
  - Init to 0.002, bf16-friendly (ULP ≈ 1.5e-5 near 0.002)
  - Keeps direct spatial modulation via vp_proj (1:1 aligned)
  - No per-image loop, pure tensor ops → fast
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .vp_config import VPAdapterConfig
except ImportError:
    from vp_config import VPAdapterConfig


_GATE_INIT = 0.002


class VPSpatialAdapter(nn.Module):
    """
    Spatial modulation adapter with per-dimension learnable gate.

    For each token position i:
      vp_proj_i  = Linear(vp_features_i)    → project VP to ViT dim
      output_i   = x_i + gate * vp_proj_i   → gated injection

    The gate is a shared learnable vector (D,) that scales each dimension
    of the projected VP signal independently.
    """

    def __init__(self, hidden_dim: int, vp_dim: int, **kwargs):
        super().__init__()
        self.vp_proj = nn.Linear(vp_dim, hidden_dim)
        self.norm_vp = nn.LayerNorm(hidden_dim)
        self.gate = nn.Parameter(
            torch.full((hidden_dim,), _GATE_INIT)
        )

    def forward(
        self,
        x: torch.Tensor,
        vp_features: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        vp_proj = self.norm_vp(self.vp_proj(vp_features))
        return x + self.gate * vp_proj


def build_vp_adapters(cfg: VPAdapterConfig) -> nn.ModuleDict:
    """Create one VPSpatialAdapter for each insertion layer."""
    adapters = nn.ModuleDict()
    for layer_idx in cfg.adapter_layers:
        adapters[str(layer_idx)] = VPSpatialAdapter(
            hidden_dim=cfg.vit_hidden_dim,
            vp_dim=cfg.vp_dim,
            bottleneck_ratio=4,
        )
    return adapters
