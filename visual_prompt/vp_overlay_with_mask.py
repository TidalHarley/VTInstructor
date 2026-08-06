"""
Visual Prompt 覆盖渲染 + 0/1 语义 mask 提取。

ribbon / arrow / endpoint 的绘制风格与 `overlay.py` 完全一致（细线描边版），
在此基础上额外输出一份与 RGB 像素对齐的 3 通道二值 mask，供 VTMod (VPEncoder) 使用：

    C0 = ribbon    (轨迹带：当前段 + lookahead 段 + 底部连接 + 前向短线)
    C1 = arrow     (转向弧箭头 / 前进箭头)
    C2 = endpoint  (终点旗标)

mask 取值为 0/1（无抗锯齿，保持二值），shape = (H, W*n_views, 3) float32。

核心入口：
    overlay_panorama_with_mask(...) -> (overlaid_rgb, modes, pano_mask)

与 `overlay.overlay_panorama` 同签名，便于 `render_vp_masks.py` 直接调用。
该模块自包含（只依赖 projection.py）。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:  # 兼容包模式 / 扁平模式两种 import
    from .projection import (
        check_visibility,
        compute_intrinsics,
        project_points,
        quat_to_rotmat,
    )
except ImportError:  # pragma: no cover
    from projection import (
        check_visibility,
        compute_intrinsics,
        project_points,
        quat_to_rotmat,
    )

# ────────────────────────── 颜色 & 参数（与 overlay.py 对齐） ──────────────────────────

# 注意：当前图像是 RGB 顺序，颜色常量也按 RGB 给出
# ribbon 沿弧长渐变：近端深蓝(RIBBON_NEAR_COLOR) → 远端深绿(RIBBON_FAR_COLOR)
# 起始深蓝先保持 RIBBON_HOLD_FRAC 的距离，之后再开始向深绿过渡
RIBBON_NEAR_COLOR = (18, 42, 138)   # 起始：深蓝
RIBBON_FAR_COLOR = (28, 112, 52)    # 远端：深绿
RIBBON_HOLD_FRAC = 0.35             # 深蓝保持的弧长比例，之后才过渡
RIBBON_SILVER = (208, 210, 216)     # 两侧银色描边
RIBBON_SILVER_W = 2                 # 每侧银线宽度(px)
CUR_COLOR = RIBBON_NEAR_COLOR       # 连接段 / 兜底短线用近端深蓝
LOOK_COLOR = RIBBON_FAR_COLOR
CUR_BORDER = RIBBON_SILVER
LOOK_BORDER = RIBBON_SILVER
ENDPOINT_COLOR = (50, 50, 255)     # red — 终点
RIBBON_THICK = 20                  # 纯色核心宽度（再宽 ~30%）
RIBBON_BORDER_W = RIBBON_SILVER_W
RIBBON_ALPHA = 0.55                # 透明度

ARC_COLOR = (30, 220, 255)
ARC_BORDER = (0, 96, 170)
ARC_THICK = 4
ARC_ALPHA = 0.80

OCCLUDED_TAU = 0.20
GOAL_TAU = 0.35
PATH_ELEVATION = 0.02

# ribbon mask 线宽（覆盖描边外轮廓）
RIBBON_MASK_THICK = RIBBON_THICK + RIBBON_BORDER_W * 2

# mask 通道索引
CH_RIBBON = 0
CH_ARROW = 1
CH_ENDPOINT = 2


# ────────────────────────── alpha 合成 ──────────────────────────

def _composite(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    mask = overlay.astype(np.int32).sum(axis=2) > 0
    out = base.copy().astype(np.float32)
    out[mask] = out[mask] * (1 - alpha) + overlay[mask].astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _paint_mask_polylines(ch: np.ndarray, pts: np.ndarray, thick: int) -> None:
    """在单通道 uint8 缓冲上画二值折线（无抗锯齿，保证 0/1）。"""
    if pts is None or len(pts) < 2:
        return
    cv2.polylines(ch, [pts.astype(np.int32)], False, 1, thick, cv2.LINE_8)


# ────────────────────────── 几何工具（移植自 overlay.py） ──────────────────────────

def _densify_uv(points_uv: np.ndarray, step_px: float = 2.0) -> np.ndarray:
    """仅做线性加密，不做全局平滑（直线段保持原走向）。"""
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
    """只圆角化真正的转折处，直线段保持不变（受限二次 Bezier）。"""
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
    """严格沿可见离散点顺序生成连续线：加密 + 局部圆角。"""
    pts = np.asarray(points_uv, dtype=np.float64)
    if len(pts) < 2:
        return pts
    dense = _densify_uv(pts, step_px=2.0)
    return _round_corners_uv(dense)


def _make_bottom_connector(start_uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """从画面底部中间连到当前可见线段起点的短连接段（仅增强连续感）。"""
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


def _build_forward_probe_positions(event_idx, current_pos, step_states, event_end_steps):
    start_step, end_step = _event_step_range(event_idx, event_end_steps)
    if end_step < 0 or end_step >= len(step_states):
        return None, None
    event_positions = get_single_event_positions(
        event_idx, current_pos, step_states, event_end_steps)
    end_pos = np.asarray(step_states[end_step]["position"], dtype=np.float64)
    if len(event_positions) >= 2:
        heading = np.asarray(event_positions[-1] - event_positions[-2], dtype=np.float64)
        heading[1] = 0.0
        norm = np.linalg.norm(heading)
        heading = heading / norm if norm > 1e-6 else _forward_world_from_rot(
            step_states[end_step]["rotation"])
    else:
        heading = _forward_world_from_rot(step_states[end_step]["rotation"])
    return end_pos, heading


# ────────────────────────── Ribbon 绘制（细线描边 + 二值 mask） ──────────────────────────

def _lerp_color(c0, c1, t: float):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(c0[k] * (1.0 - t) + c1[k] * t)) for k in range(3))


def _ribbon_color_at(t: float):
    """弧长比例 t -> 颜色：先保持深蓝(RIBBON_HOLD_FRAC)，之后再过渡到深绿。"""
    if t <= RIBBON_HOLD_FRAC:
        tt = 0.0
    else:
        tt = (t - RIBBON_HOLD_FRAC) / max(1.0 - RIBBON_HOLD_FRAC, 1e-6)
    return _lerp_color(RIBBON_NEAR_COLOR, RIBBON_FAR_COLOR, tt)


def _draw_gradient_ribbon(overlay: np.ndarray, ch_ribbon: np.ndarray,
                          path: np.ndarray) -> None:
    """沿中心线画 深蓝→深绿 渐变的 ribbon：纯色核心 + 两侧 2px 银色描边，并写入二值 mask。"""
    if path is None or len(path) < 2:
        return
    pts = path.astype(np.int32)
    n = len(pts)
    seg = np.linalg.norm(path[1:] - path[:-1], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1]) or 1e-6
    outer = RIBBON_THICK + RIBBON_SILVER_W * 2

    # 银色描边（整条一次画，作为两侧 2px 银线）
    cv2.polylines(overlay, [pts], False, RIBBON_SILVER, outer, cv2.LINE_AA)
    # 纯色渐变核心覆盖在银边内侧
    for i in range(n - 1):
        col = _ribbon_color_at(cum[i] / total)
        cv2.line(overlay, tuple(pts[i]), tuple(pts[i + 1]), col, RIBBON_THICK, cv2.LINE_AA)

    _paint_mask_polylines(ch_ribbon, path, outer)


def _draw_ribbon(
    rgb: np.ndarray, ch_ribbon: np.ndarray,
    positions_3d: List[np.ndarray], cur_end: int,
    cam_pos: np.ndarray, R_cam: np.ndarray,
    fx: float, fy: float, cx: float, cy: float, depth_buf: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """基于可见点序列绘制连续 ribbon（沿弧长蓝→绿渐变），
    并把 ribbon 足迹写入 ch_ribbon。"""
    h, w = rgb.shape[:2]
    pts3d = np.array(positions_3d, dtype=np.float64)
    if len(pts3d) < 2:
        return rgb.copy(), 0

    pts3d[:, 1] += PATH_ELEVATION
    uv, depths, in_front = project_points(pts3d, cam_pos, R_cam, fx, fy, cx, cy)
    vis = check_visibility(uv, depths, in_front, depth_buf, w, h, OCCLUDED_TAU)

    foot_uv = uv[0].copy()
    foot_ok = bool(vis[0] and np.isfinite(foot_uv).all()
                   and 0 <= foot_uv[0] < w and 0 <= foot_uv[1] < h)

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

        # 把底部连接段与平滑路径拼成一条连续中心线，沿弧长做蓝→绿渐变
        if len(connector_uv) >= 2:
            path = np.concatenate([connector_uv[:-1], smooth_uv], axis=0)
        else:
            path = smooth_uv
        _draw_gradient_ribbon(overlay, ch_ribbon, path)

    return _composite(rgb, overlay, RIBBON_ALPHA), drawn_pts


def _draw_short_forward_fallback(rgb: np.ndarray, ch_ribbon: np.ndarray) -> Tuple[np.ndarray, bool]:
    """累计角度为 0 且无 ribbon 时：从画面底部中央画一小截固定短蓝线。"""
    h, w = rgb.shape[:2]
    overlay = np.zeros_like(rgb)
    x0 = int(round(w / 2.0))
    y0 = h - 2
    seg_len = 32
    x1, y1 = x0, max(0, y0 - seg_len)
    cv2.line(overlay, (x0, y0), (x1, y1), CUR_BORDER,
             RIBBON_THICK + RIBBON_BORDER_W * 2, cv2.LINE_AA)
    cv2.line(overlay, (x0, y0), (x1, y1), CUR_COLOR, RIBBON_THICK, cv2.LINE_AA)
    cv2.line(ch_ribbon, (x0, y0), (x1, y1), 1, RIBBON_MASK_THICK, cv2.LINE_8)
    return _composite(rgb, overlay, RIBBON_ALPHA), True


# ────────────────────────── 箭头 / 终点 ──────────────────────────

def _draw_turn_arrow(rgb: np.ndarray, ch_arrow: np.ndarray,
                     turn_dir: str, turn_deg: int) -> np.ndarray:
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
        tri = np.array([[x1 + head_len, cy], [x1, cy - head_half_h], [x1, cy + head_half_h]],
                       dtype=np.int32)
    else:
        x0, x1 = cx + half_len, cx - half_len
        tri = np.array([[x1 - head_len, cy], [x1, cy - head_half_h], [x1, cy + head_half_h]],
                       dtype=np.int32)

    cv2.line(overlay, (x0, cy), (x1, cy), ARC_BORDER, body_thick_outer, cv2.LINE_AA)
    cv2.line(overlay, (x0, cy), (x1, cy), ARC_COLOR, body_thick_inner, cv2.LINE_AA)
    cv2.fillConvexPoly(overlay, tri, ARC_COLOR)
    cv2.polylines(overlay, [tri], True, ARC_BORDER, 2, cv2.LINE_AA)

    cv2.line(ch_arrow, (x0, cy), (x1, cy), 1, body_thick_outer, cv2.LINE_8)
    cv2.fillConvexPoly(ch_arrow, tri, 1)

    result = _composite(rgb, overlay, ARC_ALPHA)
    label = f"{turn_dir} {turn_deg} degrees"
    org = (max(8, cx - 86), min(h - 8, cy + 34))
    cv2.putText(result, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(result, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.68, ARC_BORDER, 2, cv2.LINE_AA)
    return result


def _draw_forward_arrow(rgb: np.ndarray, ch_arrow: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    overlay = np.zeros_like(rgb)
    cx = w // 2
    y_bot, y_top = h - 25, h - 65
    cv2.line(overlay, (cx, y_bot), (cx, y_top), ARC_BORDER, 6, cv2.LINE_AA)
    cv2.line(overlay, (cx, y_bot), (cx, y_top), ARC_COLOR, 3, cv2.LINE_AA)
    tri = np.array([[cx, y_top - 10], [cx - 8, y_top], [cx + 8, y_top]], np.int32)
    cv2.fillConvexPoly(overlay, tri, ARC_COLOR)
    cv2.polylines(overlay, [tri], True, ARC_BORDER, 1, cv2.LINE_AA)
    cv2.line(ch_arrow, (cx, y_bot), (cx, y_top), 1, 6, cv2.LINE_8)
    cv2.fillConvexPoly(ch_arrow, tri, 1)
    return _composite(rgb, overlay, ARC_ALPHA)


def _draw_endpoint(rgb: np.ndarray, ch_endpoint: np.ndarray, depth_buf: np.ndarray,
                   cam_pos: np.ndarray, R_cam: np.ndarray,
                   fx: float, fy: float, cx: float, cy: float,
                   goal_pos: np.ndarray) -> Tuple[np.ndarray, bool]:
    h, w = rgb.shape[:2]
    pts = np.array([goal_pos], dtype=np.float64)
    uv, depths, in_front = project_points(pts, cam_pos, R_cam, fx, fy, cx, cy)
    if not in_front[0]:
        return rgb, False
    ui, vi = int(round(uv[0, 0])), int(round(uv[0, 1]))
    if not (0 <= ui < w and 0 <= vi < h):
        return rgb, False

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

    flag = np.array([[pole_x + 1, pole_y1 + 2], [pole_x + 28, pole_y1 + 11],
                     [pole_x + 1, pole_y1 + 26]], dtype=np.int32)
    cv2.fillConvexPoly(overlay, flag, (225, 28, 28))
    cv2.polylines(overlay, [flag], True, (255, 240, 240), 2, cv2.LINE_AA)
    highlight = np.array([[pole_x + 3, pole_y1 + 3], [pole_x + 12, pole_y1 + 5],
                          [pole_x + 19, pole_y1 + 7]], dtype=np.int32)
    cv2.polylines(overlay, [highlight], False, (255, 120, 120), 2, cv2.LINE_AA)

    cv2.circle(ch_endpoint, (ui, vi), 12, 1, -1, cv2.LINE_8)
    cv2.line(ch_endpoint, (pole_x, pole_y0), (pole_x, pole_y1), 1, 5, cv2.LINE_8)
    cv2.fillConvexPoly(ch_endpoint, flag, 1)

    return _composite(rgb, overlay, RIBBON_ALPHA), True


def _maybe_endpoint(rgb, ch_endpoint, depth_buf, cam_pos, R_cam, fx, fy, cx, cy,
                    goal_pos, is_near_end):
    if is_near_end and goal_pos is not None:
        rgb, _ = _draw_endpoint(rgb, ch_endpoint, depth_buf, cam_pos, R_cam,
                                fx, fy, cx, cy, goal_pos)
    return rgb


# ────────────────────────── 中间子视图（彩色 + mask） ──────────────────────────

def process_center_subview_with_mask(
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
    cfg,
    aug,
) -> Tuple[np.ndarray, np.ndarray, str]:
    h, w = rgb.shape[:2]
    # 各通道用独立连续缓冲（cv2 无法在 strided 视图上绘制）
    ch_r = np.zeros((h, w), dtype=np.uint8)
    ch_a = np.zeros((h, w), dtype=np.uint8)
    ch_e = np.zeros((h, w), dtype=np.uint8)

    def _stack():
        m = np.zeros((h, w, 3), dtype=np.float32)
        m[:, :, CH_RIBBON] = ch_r
        m[:, :, CH_ARROW] = ch_a
        m[:, :, CH_ENDPOINT] = ch_e
        return m

    R_cam = quat_to_rotmat(cam_rot)
    fx, fy, cx, cy = compute_intrinsics(hfov_deg, w, h)

    if event_idx >= len(action_events):
        result = _maybe_endpoint(rgb, ch_e, depth_buf, cam_pos, R_cam,
                                 fx, fy, cx, cy, goal_pos, is_near_end)
        return result, _stack(), "none"

    event = action_events[event_idx]

    if _is_pure_rotation_event(event):
        td, turn_deg = _rotation_label(event)
        if td in ("left", "right") and turn_deg > 0:
            result = _draw_turn_arrow(rgb, ch_a, td, turn_deg)
            result = _maybe_endpoint(result, ch_e, depth_buf, cam_pos, R_cam,
                                     fx, fy, cx, cy, goal_pos, is_near_end)
            return result, _stack(), f"arc_{td[0]}_{turn_deg}"
        result = _maybe_endpoint(rgb, ch_e, depth_buf, cam_pos, R_cam,
                                 fx, fy, cx, cy, goal_pos, is_near_end)
        return result, _stack(), "none"

    cur_positions = get_single_event_positions(
        event_idx, current_pos, step_states, event_end_steps)
    positions = list(cur_positions)
    cur_end = len(cur_positions)
    positions = extend_one_event(positions, event_idx + 1, step_states, event_end_steps)

    cam_floor = cam_pos.copy()
    cam_floor[1] = current_pos[1]
    ribbon_pts = [cam_floor] + positions[1:]
    result, drawn = _draw_ribbon(rgb, ch_r, ribbon_pts, cur_end, cam_pos, R_cam,
                                 fx, fy, cx, cy, depth_buf)

    if drawn > 0:
        result = _maybe_endpoint(result, ch_e, depth_buf, cam_pos, R_cam,
                                 fx, fy, cx, cy, goal_pos, is_near_end)
        return result, _stack(), "ribbon_cur_next"

    td, turn_deg = _rotation_label(event)
    if td in ("left", "right") and turn_deg > 0:
        result = _draw_turn_arrow(rgb, ch_a, td, turn_deg)
        result = _maybe_endpoint(result, ch_e, depth_buf, cam_pos, R_cam,
                                 fx, fy, cx, cy, goal_pos, is_near_end)
        return result, _stack(), f"arc_fallback_{td[0]}_{turn_deg}"

    end_pos, heading = _build_forward_probe_positions(
        event_idx, current_pos, step_states, event_end_steps)
    if end_pos is not None and heading is not None:
        probe_result, probe_ok = _draw_short_forward_fallback(rgb, ch_r)
        if probe_ok:
            probe_result = _maybe_endpoint(probe_result, ch_e, depth_buf, cam_pos, R_cam,
                                           fx, fy, cx, cy, goal_pos, is_near_end)
            return probe_result, _stack(), "probe_blue_short"

    result = _maybe_endpoint(rgb, ch_e, depth_buf, cam_pos, R_cam,
                             fx, fy, cx, cy, goal_pos, is_near_end)
    return result, _stack(), "none"


# ────────────────────────── 全景拼接（RGB + mask） ──────────────────────────

def overlay_panorama_with_mask(
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
    cfg,
    aug,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """与 overlay_panorama 同签名，额外返回 3 通道 0/1 mask。

    返回 (pano_rgb, modes, pano_mask)，其中
      pano_rgb : (H, W*n, 3) uint8  —— ribbon/arrow/endpoint 叠加后的全景
      modes    : 每个子视图的绘制模式
      pano_mask: (H, W*n, 3) float32 —— C0=ribbon, C1=arrow, C2=endpoint，取值 0/1
    """
    center_idx = 1
    n_views = len(subview_rgbs)
    modes = ["none"] * n_views

    overlaid_center, center_mask, mode = process_center_subview_with_mask(
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
    pano_mask = np.zeros((h, w * n_views, 3), dtype=np.float32)
    for i in range(n_views):
        if i == center_idx:
            pano[:, i * w:(i + 1) * w] = overlaid_center
            pano_mask[:, i * w:(i + 1) * w] = center_mask
        else:
            pano[:, i * w:(i + 1) * w] = subview_rgbs[i]

    return pano, modes, pano_mask
