"""
Visual Prompt 覆盖渲染器（v5）。

仅在中间子视图（center subview, index=1）上绘制。

Mode dots → 细 ribbon：
  - 从相机脚下（地面投影点）开始，经过可见 action 点，到最后可见点
  - 当前 event 段：cyan；lookahead 段：teal
  - 高斯平滑后绘制，遮挡段跳过
  - 终点（goal）：倒数 4 帧可见时用特殊颜色标注

Mode arc/arrow：
  - 转向：弧线箭头（矢量风格）
  - 前进但看不到点：细直箭头
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from .augmentation import AugParams
    from .config import VisualPromptConfig
    from .projection import (
        check_visibility,
        compute_intrinsics,
        project_points,
        quat_to_rotmat,
    )
except ImportError:
    from augmentation import AugParams
    from config import VisualPromptConfig
    from projection import (
        check_visibility,
        compute_intrinsics,
        project_points,
        quat_to_rotmat,
    )
# ────────────────────────── 颜色 & 参数 ──────────────────────────

# 注意：当前图像是 RGB 顺序，颜色常量也按 RGB 给出
CUR_COLOR = (20, 62, 168)          # 再深一小档蓝
LOOK_COLOR = (24, 98, 46)          # 再深一小档绿
CUR_BORDER = (10, 34, 100)
LOOK_BORDER = (12, 60, 26)
ENDPOINT_COLOR = (50, 50, 255)     # red — 终点
RIBBON_THICK = 4
RIBBON_BORDER_W = 1
RIBBON_ALPHA = 0.68

ARC_COLOR = (30, 220, 255)
ARC_BORDER = (0, 96, 170)
ARC_THICK = 4
ARC_ALPHA = 0.80

OCCLUDED_TAU = 0.20
GOAL_TAU = 0.35
PATH_ELEVATION = 0.02


# ────────────────────────── alpha 合成 ──────────────────────────

def _composite(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    mask = overlay.astype(np.int32).sum(axis=2) > 0
    out = base.copy().astype(np.float32)
    out[mask] = out[mask] * (1 - alpha) + overlay[mask].astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# ────────────────────────── 可见段提取 ──────────────────────────

def _get_visible_segments(vis_mask: np.ndarray) -> List[Tuple[int, int]]:
    segments = []
    cur_start = None
    for i in range(len(vis_mask)):
        if vis_mask[i]:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                segments.append((cur_start, i - 1))
                cur_start = None
    if cur_start is not None:
        segments.append((cur_start, len(vis_mask) - 1))
    return segments


def _densify_uv(points_uv: np.ndarray, step_px: float = 2.0) -> np.ndarray:
    """
    仅做线性加密，不做全局平滑。
    这样直线段保持原来的走向，只在后续局部拐角处理中变圆。
    """
    pts = np.asarray(points_uv, dtype=np.float64)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-6:
        return pts

    n_new = max(2, int(total / max(step_px, 1e-3)) + 1)
    s_new = np.linspace(0.0, total, n_new)
    x_new = np.interp(s_new, cum, pts[:, 0])
    y_new = np.interp(s_new, cum, pts[:, 1])
    return np.stack([x_new, y_new], axis=1)


def _round_corners_uv(
    points_uv: np.ndarray,
    min_turn_deg: float = 10.0,
    cut_ratio: float = 0.22,
    max_cut_px: float = 12.0,
    curve_samples: int = 6,
) -> np.ndarray:
    """
    只圆角化真正的转折处，直线段保持不变。
    使用受限的二次 Bezier，避免过冲或产生额外扭曲。
    """
    pts = np.asarray(points_uv, dtype=np.float64)
    if len(pts) < 3:
        return pts

    out: List[np.ndarray] = [pts[0]]
    min_turn_rad = math.radians(min_turn_deg)

    for i in range(1, len(pts) - 1):
        p_prev = pts[i - 1]
        p = pts[i]
        p_next = pts[i + 1]

        v_in = p - p_prev
        v_out = p_next - p
        len_in = float(np.linalg.norm(v_in))
        len_out = float(np.linalg.norm(v_out))
        if len_in < 1e-4 or len_out < 1e-4:
            if np.linalg.norm(out[-1] - p) > 1e-4:
                out.append(p)
            continue

        dir_in = v_in / len_in
        dir_out = v_out / len_out
        cos_theta = float(np.clip(np.dot(dir_in, dir_out), -1.0, 1.0))
        turn = math.acos(cos_theta)

        # 接近直线时不动，保留原始直线感
        if turn < min_turn_rad:
            if np.linalg.norm(out[-1] - p) > 1e-4:
                out.append(p)
            continue

        cut = min(len_in, len_out) * cut_ratio
        cut = min(max(cut, 2.0), max_cut_px)
        q = p - dir_in * cut
        r = p + dir_out * cut

        if np.linalg.norm(out[-1] - q) > 1e-4:
            out.append(q)

        ts = np.linspace(0.0, 1.0, curve_samples + 2)[1:-1]
        for t in ts:
            bez = ((1.0 - t) ** 2) * q + 2.0 * (1.0 - t) * t * p + (t ** 2) * r
            out.append(bez)

        out.append(r)

    if np.linalg.norm(out[-1] - pts[-1]) > 1e-4:
        out.append(pts[-1])
    return np.asarray(out, dtype=np.float64)


def _smooth_visible_path(points_uv: np.ndarray) -> np.ndarray:
    """
    严格沿可见离散点顺序生成连续线。
    不发明新的全局弧线，只在真正拐角处做局部圆角，直线段尽量不动。
    """
    pts = np.asarray(points_uv, dtype=np.float64)
    if len(pts) < 2:
        return pts
    dense = _densify_uv(pts, step_px=2.0)
    return _round_corners_uv(dense)


def _make_bottom_connector(start_uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    用一个很短的 2D 蓝色连接段，从画面底部中间连到当前可见线段的起点，
    只用于增强连续感，不改变 3D ribbon 的可见性逻辑。
    """
    start = np.asarray(start_uv, dtype=np.float64)
    anchor = np.array([width / 2.0, height - 2.0], dtype=np.float64)
    dist = float(np.linalg.norm(start - anchor))
    if dist < 2.0:
        return np.stack([anchor, start], axis=0)

    ctrl_y = max(start[1], height - min(26.0, 0.22 * dist))
    ctrl = np.array([(anchor[0] + start[0]) * 0.5, ctrl_y], dtype=np.float64)
    ts = np.linspace(0.0, 1.0, 10)
    curve = []
    for t in ts:
        bez = ((1.0 - t) ** 2) * anchor + 2.0 * (1.0 - t) * t * ctrl + (t ** 2) * start
        curve.append(bez)
    return np.asarray(curve, dtype=np.float64)


def _event_parts(event: Dict) -> List[Dict]:
    if not event:
        return []
    if event.get("action") == "combo":
        return event.get("parts", [])
    return [{"action": event.get("action"), "count": event.get("count", 0)}]


def _is_pure_rotation_event(event: Dict) -> bool:
    parts = _event_parts(event)
    if not parts:
        return False
    return all(p.get("action") in ("left", "right") for p in parts)


def _rotation_label(event: Dict, turn_angle_deg: float = 30.0) -> Tuple[str, int]:
    parts = _event_parts(event)
    left_count = sum(int(p.get("count", 0)) for p in parts if p.get("action") == "left")
    right_count = sum(int(p.get("count", 0)) for p in parts if p.get("action") == "right")
    net = left_count - right_count
    if net > 0:
        return "left", int(round(abs(net) * turn_angle_deg))
    if net < 0:
        return "right", int(round(abs(net) * turn_angle_deg))
    return "straight", 0


def _event_step_range(event_idx: int, event_end_steps: list) -> Tuple[int, int]:
    if event_idx >= len(event_end_steps):
        return -1, -1
    start_step = 0 if event_idx == 0 else event_end_steps[event_idx - 1] + 1
    end_step = event_end_steps[event_idx]
    return start_step, end_step


def _forward_world_from_rot(rot) -> np.ndarray:
    R = quat_to_rotmat(rot)
    fwd = -R[:, 2].astype(np.float64)
    fwd[1] = 0.0
    n = np.linalg.norm(fwd)
    if n < 1e-6:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return fwd / n


def _build_forward_probe_positions(
    event_idx: int,
    current_pos: np.ndarray,
    step_states: list,
    event_end_steps: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    当当前 event 无法直接看到 ribbon，且累计旋转为 0 时：
    从 event 终点沿最后朝向向前探测，直到有点进入画面。
    """
    start_step, end_step = _event_step_range(event_idx, event_end_steps)
    if end_step < 0 or end_step >= len(step_states):
        return None, None

    event_positions = get_single_event_positions(
        event_idx, current_pos, step_states, event_end_steps
    )
    end_pos = np.asarray(step_states[end_step]["position"], dtype=np.float64)

    if len(event_positions) >= 2:
        heading = np.asarray(event_positions[-1] - event_positions[-2], dtype=np.float64)
        heading[1] = 0.0
        norm = np.linalg.norm(heading)
        if norm > 1e-6:
            heading /= norm
        else:
            heading = _forward_world_from_rot(step_states[end_step]["rotation"])
    else:
        heading = _forward_world_from_rot(step_states[end_step]["rotation"])

    return end_pos, heading


def _draw_short_forward_fallback(
    rgb: np.ndarray,
    end_pos: np.ndarray,
    heading: np.ndarray,
    cam_pos: np.ndarray,
    R_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_buf: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    累计角度为 0 且无 ribbon 时，直接从画面底部中央画一小截固定短蓝线。
    这里不再做 3D 探测，也不再考虑遮挡，只提供一个很短的前向提示。
    """
    h, w = rgb.shape[:2]
    overlay = np.zeros_like(rgb)
    x0 = int(round(w / 2.0))
    y0 = h - 2
    seg_len = 32
    x1 = x0
    y1 = max(0, y0 - seg_len)
    cv2.line(overlay, (x0, y0), (x1, y1), CUR_BORDER,
             RIBBON_THICK + RIBBON_BORDER_W * 2, cv2.LINE_AA)
    cv2.line(overlay, (x0, y0), (x1, y1), CUR_COLOR,
             RIBBON_THICK, cv2.LINE_AA)
    return _composite(rgb, overlay, RIBBON_ALPHA), True


# ────────────────────────── Ribbon 绘制 ──────────────────────────

def _build_and_draw_ribbon(
    rgb: np.ndarray,
    positions_3d: List[np.ndarray],
    cur_end: int,
    cam_pos: np.ndarray,
    R_cam: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    depth_buf: np.ndarray,
    aug: AugParams,
) -> Tuple[np.ndarray, int]:
    """
    只基于“可见点序列”绘制连续线：
      - 从脚下开始
      - 只连接当前能看到的离散点
      - 当前 event 段用深蓝，lookahead 段用深绿
    这样连续线就是 debug_dots 可见离散点的连续版本，避免 3D 过度平滑带来的怪异扭曲。
    """
    h, w = rgb.shape[:2]

    pts3d = np.array(positions_3d, dtype=np.float64)
    if len(pts3d) < 2:
        return rgb.copy(), 0

    pts3d[:, 1] += PATH_ELEVATION
    uv, depths, in_front = project_points(pts3d, cam_pos, R_cam, fx, fy, cx, cy)
    vis = check_visibility(uv, depths, in_front, depth_buf, w, h, OCCLUDED_TAU)

    foot_uv = uv[0].copy()
    foot_ok = bool(
        vis[0]
        and np.isfinite(foot_uv).all()
        and 0 <= foot_uv[0] < w
        and 0 <= foot_uv[1] < h
    )

    current_uv = [foot_uv] if foot_ok else []
    future_uv = []

    for idx in range(1, len(pts3d)):
        if not vis[idx]:
            continue
        if idx < cur_end:
            current_uv.append(uv[idx])
        else:
            future_uv.append(uv[idx])

    overlay = np.zeros_like(rgb)
    drawn_pts = len(current_uv) - 1 + len(future_uv)

    full_uv = list(current_uv)
    if future_uv:
        full_uv.extend(future_uv)

    if len(full_uv) >= 2:
        full_uv = np.array(full_uv, dtype=np.float64)
        smooth_uv = _smooth_visible_path(full_uv)
        connector_uv = _make_bottom_connector(smooth_uv[0], w, h)

        if len(connector_uv) >= 2:
            conn_pts = connector_uv.astype(np.int32)
            cv2.polylines(overlay, [conn_pts], False, CUR_BORDER,
                          RIBBON_THICK + RIBBON_BORDER_W * 2, cv2.LINE_AA)
            cv2.polylines(overlay, [conn_pts], False, CUR_COLOR,
                          RIBBON_THICK, cv2.LINE_AA)

        # 以“当前可见段在原始路径中的弧长比例”切分颜色，避免在切分点产生 V 字缝合
        if len(current_uv) >= 2:
            seg = np.linalg.norm(full_uv[1:] - full_uv[:-1], axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            total = float(cum[-1])
            split_dist = float(cum[len(current_uv) - 1])
            ratio = split_dist / max(total, 1e-6)
            split_idx = max(1, min(len(smooth_uv) - 1, int(round(ratio * (len(smooth_uv) - 1)))))

            cur_pts = smooth_uv[:split_idx + 1].astype(np.int32)
            cv2.polylines(overlay, [cur_pts], False, CUR_BORDER,
                          RIBBON_THICK + RIBBON_BORDER_W * 2, cv2.LINE_AA)
            cv2.polylines(overlay, [cur_pts], False, CUR_COLOR,
                          RIBBON_THICK, cv2.LINE_AA)

            if future_uv:
                fut_pts = smooth_uv[split_idx:].astype(np.int32)
                if len(fut_pts) >= 2:
                    cv2.polylines(overlay, [fut_pts], False, LOOK_BORDER,
                                  RIBBON_THICK + RIBBON_BORDER_W * 2, cv2.LINE_AA)
                    cv2.polylines(overlay, [fut_pts], False, LOOK_COLOR,
                                  RIBBON_THICK, cv2.LINE_AA)
        else:
            fut_pts = smooth_uv.astype(np.int32)
            cv2.polylines(overlay, [fut_pts], False, LOOK_BORDER,
                          RIBBON_THICK + RIBBON_BORDER_W * 2, cv2.LINE_AA)
            cv2.polylines(overlay, [fut_pts], False, LOOK_COLOR,
                          RIBBON_THICK, cv2.LINE_AA)

    return _composite(rgb, overlay, RIBBON_ALPHA), drawn_pts


# ────────────────────────── 终点标记 ──────────────────────────

def draw_endpoint(rgb, depth_buf, cam_pos, R_cam, fx, fy, cx, cy,
                  goal_pos):
    h, w = rgb.shape[:2]
    pts = np.array([goal_pos], dtype=np.float64)
    uv, depths, in_front = project_points(pts, cam_pos, R_cam, fx, fy, cx, cy)
    if not in_front[0]:
        return rgb, False
    ui, vi = int(round(uv[0, 0])), int(round(uv[0, 1]))
    if not (0 <= ui < w and 0 <= vi < h):
        return rgb, False

    # 终点标记使用局部邻域深度，避免单像素深度抖动导致“明明可见却不标”
    u0, u1 = max(0, ui - 2), min(w, ui + 3)
    v0, v1 = max(0, vi - 2), min(h, vi + 3)
    local_depth = depth_buf[v0:v1, u0:u1]
    valid = (local_depth > 0) & np.isfinite(local_depth)
    if not np.any(valid):
        return rgb, False
    local_max = float(np.max(local_depth[valid]))
    if depths[0] > (local_max + GOAL_TAU):
        return rgb, False

    overlay = np.zeros_like(rgb)
    cv2.circle(overlay, (ui, vi), 12, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(overlay, (ui, vi), 9, CUR_COLOR, -1, cv2.LINE_AA)

    pole_x = ui + 5
    pole_y0 = vi + 8
    pole_y1 = vi - 42
    cv2.line(overlay, (pole_x, pole_y0), (pole_x, pole_y1), (240, 240, 240), 5, cv2.LINE_AA)
    cv2.line(overlay, (pole_x, pole_y0), (pole_x, pole_y1), (120, 120, 120), 2, cv2.LINE_AA)

    # 更高更挺拔的单三角红旗，避免横向显胖
    flag = np.array(
        [
            [pole_x + 1, pole_y1 + 2],
            [pole_x + 28, pole_y1 + 11],
            [pole_x + 1, pole_y1 + 26],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(overlay, flag, (225, 28, 28))
    cv2.polylines(overlay, [flag], True, (255, 240, 240), 2, cv2.LINE_AA)

    highlight = np.array(
        [
            [pole_x + 3, pole_y1 + 3],
            [pole_x + 12, pole_y1 + 5],
            [pole_x + 19, pole_y1 + 7],
        ],
        dtype=np.int32,
    )
    cv2.polylines(overlay, [highlight], False, (255, 120, 120), 2, cv2.LINE_AA)
    return _composite(rgb, overlay, RIBBON_ALPHA), True


def draw_turn_arrow(rgb, turn_dir: str, turn_deg: int):
    h, w = rgb.shape[:2]
    overlay = np.zeros_like(rgb)
    cx, cy = w // 2, h - 52
    half_len = 34
    body_thick_outer = 7
    body_thick_inner = 3
    head_len = 16
    head_half_h = 11

    if turn_dir == "right":
        x0, x1 = cx - half_len, cx + half_len
        cv2.line(overlay, (x0, cy), (x1, cy), ARC_BORDER, body_thick_outer, cv2.LINE_AA)
        cv2.line(overlay, (x0, cy), (x1, cy), ARC_COLOR, body_thick_inner, cv2.LINE_AA)
        tri = np.array(
            [[x1 + head_len, cy], [x1, cy - head_half_h], [x1, cy + head_half_h]],
            dtype=np.int32,
        )
    else:
        x0, x1 = cx + half_len, cx - half_len
        cv2.line(overlay, (x0, cy), (x1, cy), ARC_BORDER, body_thick_outer, cv2.LINE_AA)
        cv2.line(overlay, (x0, cy), (x1, cy), ARC_COLOR, body_thick_inner, cv2.LINE_AA)
        tri = np.array(
            [[x1 - head_len, cy], [x1, cy - head_half_h], [x1, cy + head_half_h]],
            dtype=np.int32,
        )
    cv2.fillConvexPoly(overlay, tri, ARC_COLOR)
    cv2.polylines(overlay, [tri], True, ARC_BORDER, 2, cv2.LINE_AA)

    result = _composite(rgb, overlay, ARC_ALPHA)
    label = f"{turn_dir} {turn_deg} degrees"
    org = (max(8, cx - 86), min(h - 8, cy + 34))
    cv2.putText(result, label, org, cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(result, label, org, cv2.FONT_HERSHEY_SIMPLEX,
                0.68, ARC_BORDER, 2, cv2.LINE_AA)
    return result


def draw_forward_arrow(rgb):
    h, w = rgb.shape[:2]
    overlay = np.zeros_like(rgb)
    cx = w // 2
    y_bot, y_top = h - 25, h - 65
    cv2.line(overlay, (cx, y_bot), (cx, y_top), ARC_BORDER, 6, cv2.LINE_AA)
    cv2.line(overlay, (cx, y_bot), (cx, y_top), ARC_COLOR, 3, cv2.LINE_AA)
    tri = np.array([[cx, y_top - 10], [cx - 8, y_top], [cx + 8, y_top]], np.int32)
    cv2.fillConvexPoly(overlay, tri, ARC_COLOR)
    cv2.polylines(overlay, [tri], True, ARC_BORDER, 1, cv2.LINE_AA)
    return _composite(rgb, overlay, ARC_ALPHA)


# ────────────────────────── Event 位置提取 ──────────────────────────

def get_single_event_positions(event_idx, current_pos, step_states, event_end_steps):
    if event_idx >= len(event_end_steps):
        return []
    positions = [np.asarray(current_pos, dtype=np.float64)]
    start_step = 0 if event_idx == 0 else event_end_steps[event_idx - 1] + 1
    end_step = event_end_steps[event_idx]
    for step in range(start_step, min(end_step + 1, len(step_states))):
        pos = np.asarray(step_states[step]["position"], dtype=np.float64)
        if np.linalg.norm(pos - positions[-1]) > 1e-4:
            positions.append(pos)
    return positions


def extend_one_event(positions, ext_event_idx, step_states, event_end_steps):
    if ext_event_idx >= len(event_end_steps):
        return positions
    extended = list(positions)
    start_step = 0 if ext_event_idx == 0 else event_end_steps[ext_event_idx - 1] + 1
    end_step = event_end_steps[ext_event_idx]
    for step in range(start_step, min(end_step + 1, len(step_states))):
        pos = np.asarray(step_states[step]["position"], dtype=np.float64)
        if np.linalg.norm(pos - extended[-1]) > 1e-4:
            extended.append(pos)
    return extended


def _count_visible(positions, cam_pos, R_cam, fx, fy, cx, cy, depth_buf, w, h):
    if not positions:
        return 0
    pts = np.array(positions, dtype=np.float64)
    uv, depths, in_front = project_points(pts, cam_pos, R_cam, fx, fy, cx, cy)
    vis = check_visibility(uv, depths, in_front, depth_buf, w, h, OCCLUDED_TAU)
    return int(vis.sum())


# ────────────────────────── 中间子视图处理 ──────────────────────────

def process_center_subview(
    rgb: np.ndarray,
    depth_buf: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot,
    hfov_deg: float,
    event_idx: int,
    current_pos: np.ndarray,
    step_states: list,
    event_end_steps: list,
    action_events: list,
    turn_dir: str,
    goal_pos: Optional[np.ndarray],
    is_near_end: bool,
    cfg: VisualPromptConfig,
    aug: AugParams,
) -> Tuple[np.ndarray, str]:
    h, w = rgb.shape[:2]
    R_cam = quat_to_rotmat(cam_rot)
    fx, fy, cx, cy = compute_intrinsics(hfov_deg, w, h)
    if event_idx >= len(action_events):
        result = _maybe_endpoint(rgb, depth_buf, cam_pos, R_cam,
                                 fx, fy, cx, cy, goal_pos, is_near_end)
        return result, "none"

    event = action_events[event_idx]

    # 纯旋转：只保留左右箭头，并显示角度信息
    if _is_pure_rotation_event(event):
        turn_dir, turn_deg = _rotation_label(event)
        if turn_dir in ("left", "right") and turn_deg > 0:
            result = draw_turn_arrow(rgb, turn_dir, turn_deg)
            result = _maybe_endpoint(result, depth_buf, cam_pos, R_cam,
                                     fx, fy, cx, cy, goal_pos, is_near_end)
            return result, f"arc_{turn_dir[0]}_{turn_deg}"
        result = _maybe_endpoint(rgb, depth_buf, cam_pos, R_cam,
                                 fx, fy, cx, cy, goal_pos, is_near_end)
        return result, "none"

    # 非纯旋转：只绘制当前 event + 下一个 event，不再探测更远 future events
    cur_positions = get_single_event_positions(
        event_idx, current_pos, step_states, event_end_steps)
    positions = list(cur_positions)
    cur_end = len(cur_positions)
    positions = extend_one_event(positions, event_idx + 1, step_states, event_end_steps)

    cam_floor = cam_pos.copy()
    cam_floor[1] = current_pos[1]
    ribbon_pts = [cam_floor] + positions[1:]
    result, drawn = _build_and_draw_ribbon(
        rgb, ribbon_pts, cur_end, cam_pos, R_cam,
        fx, fy, cx, cy, depth_buf, aug)

    if drawn > 0:
        result = _maybe_endpoint(result, depth_buf, cam_pos, R_cam,
                                 fx, fy, cx, cy, goal_pos, is_near_end)
        return result, "ribbon_cur_next"

    # 当前 event / next event 都没有可见 ribbon 时：
    # 1) 若该 event 累计旋转不为 0，用同款左右箭头表示
    # 2) 若累计旋转为 0，则从 event 终点沿最终朝向向前探测短蓝线
    turn_dir, turn_deg = _rotation_label(event)
    if turn_dir in ("left", "right") and turn_deg > 0:
        result = draw_turn_arrow(rgb, turn_dir, turn_deg)
        result = _maybe_endpoint(result, depth_buf, cam_pos, R_cam,
                                 fx, fy, cx, cy, goal_pos, is_near_end)
        return result, f"arc_fallback_{turn_dir[0]}_{turn_deg}"

    end_pos, heading = _build_forward_probe_positions(
        event_idx, current_pos, step_states, event_end_steps
    )
    if end_pos is not None and heading is not None:
        probe_result, probe_ok = _draw_short_forward_fallback(
            rgb, end_pos, heading, cam_pos, R_cam, fx, fy, cx, cy, depth_buf
        )
        if probe_ok:
            probe_result = _maybe_endpoint(probe_result, depth_buf, cam_pos, R_cam,
                                           fx, fy, cx, cy, goal_pos, is_near_end)
            return probe_result, "probe_blue_short"

    # 仍然都不可见就不管了
    result = _maybe_endpoint(rgb, depth_buf, cam_pos, R_cam,
                             fx, fy, cx, cy, goal_pos, is_near_end)
    return result, "none"


def _maybe_endpoint(rgb, depth_buf, cam_pos, R_cam, fx, fy, cx, cy,
                    goal_pos, is_near_end):
    if is_near_end and goal_pos is not None:
        rgb, _ = draw_endpoint(rgb, depth_buf, cam_pos, R_cam,
                               fx, fy, cx, cy, goal_pos)
    return rgb


# ────────────────────────── 全景拼接 ──────────────────────────

def overlay_panorama(
    subview_rgbs: List[np.ndarray],
    subview_depths: List[np.ndarray],
    cam_pos: np.ndarray,
    subview_rots: list,
    hfov_deg: float,
    event_idx: int,
    current_pos: np.ndarray,
    step_states: list,
    event_end_steps: list,
    action_events: list,
    turn_dir: str,
    goal_pos: Optional[np.ndarray],
    is_near_end: bool,
    cfg: VisualPromptConfig,
    aug: AugParams,
) -> Tuple[np.ndarray, List[str]]:
    center_idx = 1
    n_views = len(subview_rgbs)
    modes = ["none"] * n_views

    overlaid_center, mode = process_center_subview(
        subview_rgbs[center_idx],
        subview_depths[center_idx],
        cam_pos,
        subview_rots[center_idx],
        hfov_deg,
        event_idx,
        current_pos,
        step_states,
        event_end_steps,
        action_events,
        turn_dir,
        goal_pos,
        is_near_end,
        cfg,
        aug,
    )
    modes[center_idx] = mode

    h, w = subview_rgbs[0].shape[:2]
    pano = np.zeros((h, w * n_views, 3), dtype=np.uint8)
    for i in range(n_views):
        if i == center_idx:
            pano[:, i * w: (i + 1) * w] = overlaid_center
        else:
            pano[:, i * w: (i + 1) * w] = subview_rgbs[i]

    return pano, modes
