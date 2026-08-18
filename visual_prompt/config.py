"""
Visual Prompt 配置。

显式假设：
  A1. 传感器高度 1.5m（与原始渲染代码一致）。
  A2. 子视图分辨率 256×256，HFOV = 90°，全景步长 = 90°（3 张子视图拼接为 270°）。
  A3. 深度传感器 near = 0.01m，far = 10.0m。
  A4. NavMesh 已加载且可用。
  A5. GreedyGeodesicFollower 产出的轨迹被视为 ground-truth 路径。
  A6. 环境中安装了 habitat-sim ≥ 0.3.x、numpy、Pillow、opencv-python、scipy。
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VisualPromptConfig:
    path_resample_interval: float = 0.08
    path_smooth_sigma_m: float = 0.12
    path_elevation: float = 0.02

    path_thickness_px: int = 14
    path_color: Tuple[int, int, int] = (0, 190, 255)
    path_border_color: Tuple[int, int, int] = (0, 100, 180)
    path_border_width: int = 3
    path_alpha_base: float = 0.82
    path_alpha_range: Tuple[float, float] = (0.70, 0.92)

    occlusion_tau: float = 0.20

    arrow_head_length_px: int = 28
    arrow_head_width_px: int = 24

    # ── Mode A 阈值（下倾 30° 后地面大面积可见，阈值可宽松）──
    mode_a_min_visible_count: int = 2

    beacon_radius_px: int = 14
    beacon_ring_width: int = 4

    hud_margin_bottom_px: int = 18
    hud_arrow_size_px: int = 32
    hud_bg_radius_px: int = 40
    hud_alpha: float = 0.70

    prompt_dropout_prob: float = 0.15
    style_jitter_alpha_std: float = 0.04
    style_jitter_hue_range: float = 12.0
    style_jitter_width_factor: Tuple[float, float] = (0.85, 1.15)
    mixed_mode_prob: float = 0.10

    depth_near: float = 0.01
    depth_far: float = 10.0
