"""
3D 轨迹提取 & 平滑。

核心规则（来自任务约束）：
  - 优先使用真实 event 子轨迹的 pose 序列构造 polyline；
  - 重采样 → 平滑 → navmesh snap（可选）。
  - shortest path 仅作为坏点修复 fallback，不替代 ground-truth。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ────────────────────────── 轨迹提取 ──────────────────────────

def get_current_event_actions(
    event_idx: int,
    current_pos: np.ndarray,
    step_states: List[Dict],
    event_end_steps: List[int],
) -> np.ndarray:
    """
    提取当前 event 内每个 action 对应的地面位置点。
    不跨 event，只取当前 event 的步骤。
    转向动作位置不变，会被去重过滤。

    返回 (M, 3) ndarray，从 current_pos 开始。
    """
    if event_idx >= len(event_end_steps):
        return np.array([np.asarray(current_pos, dtype=np.float64)])

    positions = [np.asarray(current_pos, dtype=np.float64)]

    start_step = 0 if event_idx == 0 else event_end_steps[event_idx - 1] + 1
    end_step = event_end_steps[event_idx]

    for step in range(start_step, min(end_step + 1, len(step_states))):
        pos = np.asarray(step_states[step]["position"], dtype=np.float64)
        if np.linalg.norm(pos - positions[-1]) > 1e-4:
            positions.append(pos)

    return np.array(positions, dtype=np.float64)


def determine_turn_direction(
    event_idx: int, action_events: List[Dict]
) -> str:
    """返回 'left' / 'right' / 'straight'。"""
    if event_idx >= len(action_events):
        return "straight"
    ev = action_events[event_idx]
    action = ev.get("action", "")
    if action == "left":
        return "left"
    if action == "right":
        return "right"
    if action == "combo":
        parts = ev.get("parts", [])
        lc = sum(p.get("count", 0) for p in parts if p.get("action") == "left")
        rc = sum(p.get("count", 0) for p in parts if p.get("action") == "right")
        if lc > rc:
            return "left"
        if rc > lc:
            return "right"
    return "straight"


# ────────────────────────── polyline 工具 ──────────────────────────

def _polyline_length(pts) -> float:
    total = 0.0
    for i in range(len(pts) - 1):
        total += np.linalg.norm(pts[i + 1] - pts[i])
    return total


def deduplicate_polyline(pts: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    if len(pts) <= 1:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > eps:
            keep.append(i)
    return pts[keep]


def resample_polyline(pts: np.ndarray, interval: float) -> np.ndarray:
    """沿折线以固定 interval 重采样。"""
    if len(pts) < 2:
        return pts.copy()

    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(dists)])
    total = cum[-1]
    if total < 1e-6:
        return pts[:1].copy()

    n_samples = max(int(total / interval) + 1, 2)
    sample_dists = np.linspace(0, total, n_samples)

    resampled = np.empty((n_samples, 3), dtype=np.float64)
    j = 0
    for i, sd in enumerate(sample_dists):
        while j < len(cum) - 2 and cum[j + 1] < sd:
            j += 1
        seg_len = cum[j + 1] - cum[j]
        t = (sd - cum[j]) / seg_len if seg_len > 1e-8 else 0.0
        resampled[i] = pts[j] * (1 - t) + pts[min(j + 1, len(pts) - 1)] * t
    return resampled


def smooth_polyline(pts: np.ndarray, sigma_n: float) -> np.ndarray:
    """1D Gaussian 平滑（保留首尾端点）。sigma_n 以点数为单位。"""
    N = len(pts)
    if sigma_n < 0.5 or N < 4:
        return pts.copy()

    k = min(int(math.ceil(3 * sigma_n)), (N - 1) // 2)
    if k < 1:
        return pts.copy()

    x = np.arange(-k, k + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / max(sigma_n, 1e-6)) ** 2)
    kernel /= kernel.sum()

    smoothed = pts.copy()
    for dim in range(3):
        conv = np.convolve(pts[:, dim], kernel, mode="same")
        smoothed[:, dim] = conv[:N]
    smoothed[0] = pts[0]
    smoothed[-1] = pts[-1]
    return smoothed


# ────────────────────────── Ribbon 顶点 ──────────────────────────

def build_ribbon_vertices(
    center: np.ndarray,
    half_width: float,
    elevation: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (left_edge, right_edge)，每个 shape = (N, 3)。

    perpendicular 方向在 XZ 平面上（跟 Y-up 正交）。
    """
    N = len(center)
    if N < 2:
        c = center.copy()
        c[:, 1] += elevation
        return c, c.copy()

    up = np.array([0.0, 1.0, 0.0])
    tangent = np.zeros_like(center)
    tangent[0] = center[1] - center[0]
    tangent[-1] = center[-1] - center[-2]
    for i in range(1, N - 1):
        tangent[i] = center[i + 1] - center[i - 1]

    tangent_xz = tangent.copy()
    tangent_xz[:, 1] = 0.0
    norms = np.linalg.norm(tangent_xz, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    tangent_xz = tangent_xz / norms

    perp = np.cross(tangent_xz, up)
    pnorms = np.linalg.norm(perp, axis=1, keepdims=True)
    pnorms = np.where(pnorms < 1e-8, 1.0, pnorms)
    perp = perp / pnorms

    left = center + perp * half_width
    right = center - perp * half_width
    left[:, 1] = center[:, 1] + elevation
    right[:, 1] = center[:, 1] + elevation
    return left, right


def build_smooth_trajectory(
    trajectory: np.ndarray,
    resample_interval: float,
    smooth_sigma_m: float,
    elevation: float = 0.02,
    pathfinder=None,
) -> Optional[np.ndarray]:
    """
    重采样 + 平滑 + navmesh snap，返回 (N, 3) 中心线。
    若轨迹无效则返回 None。
    """
    pts = deduplicate_polyline(trajectory)
    if len(pts) < 2:
        return None

    pts = resample_polyline(pts, resample_interval)
    if len(pts) < 2:
        return None

    sigma_n = smooth_sigma_m / resample_interval if resample_interval > 0 else 0
    pts = smooth_polyline(pts, sigma_n)

    if pathfinder is not None:
        for i in range(len(pts)):
            snapped = pathfinder.snap_point(pts[i])
            if np.isfinite(snapped).all():
                pts[i] = snapped

    pts[:, 1] += elevation
    return pts


def build_full_ribbon_3d(
    trajectory: np.ndarray,
    half_width: float,
    resample_interval: float,
    smooth_sigma_m: float,
    elevation: float = 0.02,
    pathfinder=None,
) -> Optional[Dict]:
    pts = deduplicate_polyline(trajectory)
    if len(pts) < 2:
        return None

    pts = resample_polyline(pts, resample_interval)
    if len(pts) < 2:
        return None

    sigma_n = smooth_sigma_m / resample_interval if resample_interval > 0 else 0
    pts = smooth_polyline(pts, sigma_n)

    if pathfinder is not None:
        for i in range(len(pts)):
            snapped = pathfinder.snap_point(pts[i])
            if np.isfinite(snapped).all():
                pts[i] = snapped

    left, right = build_ribbon_vertices(pts, half_width, elevation)
    return {"center": pts, "left": left, "right": right}
