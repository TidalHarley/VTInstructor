"""
训练时的 augmentation：prompt dropout / style jitter / mixed-mode sampling。

目的：防止模型 shortcut（只看 HUD UI 读答案），
      确保模型必须理解场景本身来生成 instruction。
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:
    from .config import VisualPromptConfig
except ImportError:
    from config import VisualPromptConfig


@dataclass
class AugParams:
    dropout: bool = False
    alpha: float = 0.55
    color: Tuple[int, int, int] = (0, 245, 255)
    width_factor: float = 1.0
    force_mode: Optional[str] = None  # None / "A" / "B" / "C"


def sample_augmentation(
    cfg: VisualPromptConfig,
    rng: Optional[np.random.RandomState] = None,
) -> AugParams:
    """根据配置采样一组 augmentation 参数。"""
    if rng is None:
        rng = np.random.RandomState()

    params = AugParams()

    if rng.rand() < cfg.prompt_dropout_prob:
        params.dropout = True
        return params

    alpha = cfg.path_alpha_base + rng.randn() * cfg.style_jitter_alpha_std
    params.alpha = float(np.clip(alpha, *cfg.path_alpha_range))

    r, g, b = cfg.path_color
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h_shift = rng.uniform(-cfg.style_jitter_hue_range, cfg.style_jitter_hue_range) / 360.0
    h_new = (h + h_shift) % 1.0
    rn, gn, bn = colorsys.hsv_to_rgb(h_new, s, v)
    params.color = (int(rn * 255), int(gn * 255), int(bn * 255))

    params.width_factor = float(
        rng.uniform(*cfg.style_jitter_width_factor)
    )

    if rng.rand() < cfg.mixed_mode_prob:
        params.force_mode = rng.choice(["A", "B", "C"])

    return params
