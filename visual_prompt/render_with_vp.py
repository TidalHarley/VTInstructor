#!/usr/bin/env python3
"""
R2R-CE 事件驱动渲染 + Adaptive World-Grounded Visual Prompting。

在原始渲染流程的基础上，增加：
  1. 深度传感器采集
  2. 基于真实轨迹的 3D ribbon 构造
  3. 每个子视图独立投影 → 遮挡测试 → Mode A/B/C 三层 fallback
  4. Prompt dropout / style jitter / mixed-mode sampling

用法示例：
  python visual_prompt/render_with_vp.py \\
      --train_json <r2rce_train.json.gz> \\
      --output_dir outputs/nig_vp_r2rce_detail_train \\
      --log_path logs/render_vp.log \\
      --panorama --sample_mode event

显式假设：
  A1. 子视图 256×256，HFOV 90°，全景步长 90°（3 张 → 270° 全景）。
  A2. 传感器高度 1.5m（与原始代码一致）。
  A3. forward_step = 0.25m，turn_angle = 30°（默认）。
  A4. 使用 habitat conda 环境（habitat-sim ≥ 0.3.x）。
  A5. 场景根目录为 PATH/TO/data。
"""
import argparse
import gzip
import json
import logging
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import habitat_sim
from habitat_sim.agent import ActionSpec, ActuationSpec
from habitat_sim.nav.greedy_geodesic_follower import GreedyGeodesicFollower
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs

# ── 从原始渲染脚本复用工具函数 ──
_BASE = os.path.dirname(os.path.abspath(__file__))
_SRC_CANDIDATES = [
    os.path.join(_BASE, "..", "rendering"),
]
for _src in _SRC_CANDIDATES:
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

from nig_render_dataset_r2rce_detail import (
    ACTION_ID_TO_NAME,
    EpisodeSample,
    Segment,
    _build_segments_from_actions,
    _normalize_action_name,
    _split_long_forward_segments,
    _segments_to_events,
    _to_action_id,
    format_event_action,
    load_r2r_vlnce,
)

# ── visual prompt 模块 ──
_VP_ROOT = os.path.join(_BASE, "..")
if _VP_ROOT not in sys.path:
    sys.path.insert(0, _VP_ROOT)

try:
    from visual_prompt.augmentation import AugParams, sample_augmentation
    from visual_prompt.config import VisualPromptConfig
    from visual_prompt.overlay import overlay_panorama
    from visual_prompt.ribbon import determine_turn_direction
except ImportError:
    from augmentation import AugParams, sample_augmentation
    from config import VisualPromptConfig
    from overlay import overlay_panorama
    from ribbon import determine_turn_direction


# ═══════════════════════════════════════════════════════════════
#  Simulator 构建（增加 depth sensor）
# ═══════════════════════════════════════════════════════════════

def build_sim_with_depth(
    scene_path: str,
    width: int,
    height: int,
    forward_step: float,
    turn_angle: float,
    hfov: float,
    camera_height: float,
) -> habitat_sim.Simulator:
    """
    构建 Simulator（RGB + Depth），通过 height > width 扩展纵向 FOV。
    HFOV 由 hfov 参数指定；VFOV = 2*atan(height/width * tan(HFOV/2))。
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [height, width]
    rgb_spec.hfov = hfov
    rgb_spec.position = [0.0, camera_height, 0.0]

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [height, width]
    depth_spec.hfov = hfov
    depth_spec.position = [0.0, camera_height, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
    agent_cfg.action_space = {
        "stop": ActionSpec("stop"),
        "move_forward": ActionSpec("move_forward", ActuationSpec(amount=forward_step)),
        "turn_left": ActionSpec("turn_left", ActuationSpec(amount=turn_angle)),
        "turn_right": ActionSpec("turn_right", ActuationSpec(amount=turn_angle)),
    }

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    navmesh_path = os.path.splitext(scene_path)[0] + ".navmesh"
    if os.path.exists(navmesh_path):
        sim.pathfinder.load_nav_mesh(navmesh_path)
    if not sim.pathfinder.is_loaded:
        raise RuntimeError(f"NavMesh 未加载: {navmesh_path}")
    return sim


# ═══════════════════════════════════════════════════════════════
#  子视图采集（RGB + Depth × 3 方向）
# ═══════════════════════════════════════════════════════════════

def capture_subviews(
    sim: habitat_sim.Simulator,
    panorama_step_deg: float,
    camera_height: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, list]:
    """
    采集 3 个子视图的 RGB 和 depth。
    相机水平，HFOV 不变，纵向 FOV 通过 height > width 扩展。

    返回:
        rgb_list   – 3 × (H, W, 3) uint8
        depth_list – 3 × (H, W) float32
        cam_pos    – (3,)  相机世界位置
        rot_list   – 3 个子视图的旋转四元数
    """
    agent = sim.get_agent(0)
    state = agent.get_state()
    base_pos = np.array(state.position, dtype=np.float64)
    base_rot = state.rotation

    cam_pos = base_pos + np.array([0.0, camera_height, 0.0], dtype=np.float64)

    yaw_delta = math.radians(panorama_step_deg)
    angles = [yaw_delta, 0.0, -yaw_delta]

    rgb_list, depth_list, rot_list = [], [], []

    for a in angles:
        rot = quat_from_angle_axis(a, np.array([0.0, 1.0, 0.0])) * base_rot
        agent.set_state(
            habitat_sim.AgentState(position=state.position, rotation=rot)
        )
        obs = sim.get_sensor_observations()

        rgb = np.asarray(obs["rgb"])
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        rgb_list.append(rgb.copy())

        depth = np.asarray(obs["depth"], dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth_list.append(depth.copy())

        rot_list.append(rot)

    agent.set_state(habitat_sim.AgentState(position=state.position, rotation=base_rot))
    return rgb_list, depth_list, cam_pos, rot_list


# ═══════════════════════════════════════════════════════════════
#  轨迹模拟（记录每步的位姿）
# ═══════════════════════════════════════════════════════════════

def walk_trajectory(
    sim: habitat_sim.Simulator,
    sample: EpisodeSample,
    goal_radius: float,
    max_steps: int,
) -> Tuple[List[int], List[Dict]]:
    """
    沿最短路径行走，记录每步的 (position, rotation)。

    返回:
        actions     – 动作 id 列表
        step_states – [{'position': ndarray, 'rotation': quat}, ...]
    """
    agent = sim.get_agent(0)
    initial_state = habitat_sim.AgentState()
    initial_state.position = np.array(sample.start_position, dtype=np.float32)
    if sample.start_rotation:
        initial_state.rotation = quat_from_coeffs(sample.start_rotation)
    agent.set_state(initial_state)

    follower = GreedyGeodesicFollower(sim.pathfinder, agent, goal_radius=goal_radius)

    actions: List[int] = []
    step_states: List[Dict] = []

    for _ in range(max_steps):
        try:
            action = follower.next_action_along(sample.goal_position)
        except Exception:
            break

        name = _normalize_action_name(action)
        if action is None or name == "stop" or action == 0:
            actions.append(0)
            break

        action_id = _to_action_id(action)
        actions.append(action_id)
        sim.step(action)

        st = agent.get_state()
        step_states.append(
            {
                "position": np.array(st.position, dtype=np.float64),
                "rotation": st.rotation,
            }
        )

    return actions, step_states


# ═══════════════════════════════════════════════════════════════
#  Episode 渲染（含 Visual Prompt）
# ═══════════════════════════════════════════════════════════════

def render_episode_with_vp(
    sim: habitat_sim.Simulator,
    sample: EpisodeSample,
    goal_radius: float,
    max_steps: int,
    forward_step: float,
    turn_angle: float,
    panorama_step: float,
    hfov: float,
    split_forward_threshold_m: float,
    small_fwd_m: float,
    small_turn_deg: float,
    max_parts: int,
    vp_cfg: VisualPromptConfig,
    camera_height: float,
    rng: Optional[np.random.RandomState] = None,
    enable_vp: bool = True,
) -> Optional[Dict]:
    """
    完整渲染一个 episode：轨迹 → 事件 → 帧 + Visual Prompt。

    返回 dict 或 None（失败时）。
    """
    # ── 1. 行走轨迹 ──
    actions, step_states = walk_trajectory(sim, sample, goal_radius, max_steps)
    raw_segments = _build_segments_from_actions(actions)
    if not raw_segments:
        return None

    split_segments = _split_long_forward_segments(
        raw_segments, forward_step, split_forward_threshold_m
    )
    action_events, action_events_text, event_end_steps = _segments_to_events(
        split_segments,
        forward_step,
        turn_angle,
        max_parts=max_parts,
        small_fwd_m=small_fwd_m,
        small_turn_deg=small_turn_deg,
    )

    if not event_end_steps:
        return None

    # ── 2. 准备初始状态 ──
    agent = sim.get_agent(0)
    initial_state = habitat_sim.AgentState()
    initial_state.position = np.array(sample.start_position, dtype=np.float32)
    if sample.start_rotation:
        initial_state.rotation = quat_from_coeffs(sample.start_rotation)

    frames: List[Image.Image] = []
    path_points: List[List[float]] = []
    vp_meta: List[Dict] = []

    n_frames = len(event_end_steps) + 1
    goal_pos = np.array(sample.goal_position, dtype=np.float64)

    # ── 辅助函数 ──
    def _render_frame_at(agent_state, event_idx: int,
                         frame_idx: int) -> Tuple[Image.Image, Dict]:
        agent.set_state(agent_state)
        rgb_list, depth_list, cam_pos, rot_list = capture_subviews(
            sim, panorama_step, camera_height
        )

        meta: Dict = {"mode": ["none"] * 3}
        if not enable_vp or event_idx >= len(action_events):
            pano = _stitch(rgb_list)
            pano = _downsample_pano(pano)
            return Image.fromarray(pano), meta

        aug = sample_augmentation(vp_cfg, rng) if rng is not None else AugParams()
        if aug.dropout:
            meta = {"mode": ["dropout"] * 3, "augmentation": _aug_dict(aug)}
            pano = _stitch(rgb_list)
            pano = _downsample_pano(pano)
            return Image.fromarray(pano), meta

        cur_pos = np.array(agent_state.position, dtype=np.float64)
        turn_dir = determine_turn_direction(event_idx, action_events)
        is_near_end = (frame_idx >= n_frames - 5) and (frame_idx < n_frames - 1)

        pano, modes = overlay_panorama(
            rgb_list, depth_list, cam_pos, rot_list, hfov,
            event_idx, cur_pos, step_states, event_end_steps,
            action_events, turn_dir,
            goal_pos if is_near_end else None, is_near_end,
            vp_cfg, aug,
        )
        pano = _downsample_pano(pano)
        meta = {"mode": modes, "augmentation": _aug_dict(aug)}
        return Image.fromarray(pano), meta

    # ── 3. 渲染初始帧（Frame 0 → Visual Prompt for Event 0）──
    frame0, meta0 = _render_frame_at(initial_state, event_idx=0, frame_idx=0)
    frames.append(frame0)
    path_points.append(sample.start_position)
    vp_meta.append(meta0)

    # ── 4. 渲染每个 event boundary 帧 ──
    for i, end_step in enumerate(event_end_steps):
        if end_step >= len(step_states):
            continue

        boundary_state = habitat_sim.AgentState()
        boundary_state.position = step_states[end_step]["position"].astype(np.float32)
        boundary_state.rotation = step_states[end_step]["rotation"]

        next_event_idx = i + 1
        frame_i, meta_i = _render_frame_at(boundary_state,
                                            event_idx=next_event_idx,
                                            frame_idx=next_event_idx)
        frames.append(frame_i)
        path_points.append(step_states[end_step]["position"].tolist())
        vp_meta.append(meta_i)

    # ── 5. 组装返回 ──
    action_events_raw = [{"action": s.action, "count": s.count} for s in split_segments]
    action_events_text_raw = [
        format_event_action(s.action, s.count, forward_step, turn_angle) for s in split_segments
    ]

    return {
        "frames": frames,
        "actions": actions,
        "path_points": path_points,
        "action_events": action_events,
        "action_events_text": action_events_text,
        "action_events_raw": action_events_raw,
        "action_events_text_raw": action_events_text_raw,
        "vp_meta": vp_meta,
    }


# ═══════════════════════════════════════════════════════════════
#  工具
# ═══════════════════════════════════════════════════════════════

def _stitch(subviews: List[np.ndarray]) -> np.ndarray:
    h, w = subviews[0].shape[:2]
    pano = np.zeros((h, w * len(subviews), 3), dtype=np.uint8)
    for i, sv in enumerate(subviews):
        pano[:, i * w : (i + 1) * w, :] = sv
    return pano


def _downsample_pano(pano: np.ndarray, out_h: int = 384) -> np.ndarray:
    """
    渲染使用 256x640 子视图（全景 768x640），随后降采样到 768x384。
    """
    h, w = pano.shape[:2]
    if h == out_h:
        return pano
    return cv2.resize(pano, (w, out_h), interpolation=cv2.INTER_AREA)


def _aug_dict(aug: AugParams) -> Dict:
    return {
        "dropout": aug.dropout,
        "alpha": aug.alpha,
        "color": list(aug.color),
        "width_factor": aug.width_factor,
        "force_mode": aug.force_mode,
    }


# ═══════════════════════════════════════════════════════════════
#  保存
# ═══════════════════════════════════════════════════════════════

def save_episode(
    out_dir: str,
    sample: EpisodeSample,
    result: Dict,
    frame_stride: int,
    sample_mode: str,
):
    ep_dir = os.path.join(out_dir, f"episode_{sample.episode_id}")
    os.makedirs(ep_dir, exist_ok=True)

    frame_paths = []
    for idx, img in enumerate(result["frames"]):
        fp = os.path.join(ep_dir, f"frame_{idx:04d}.jpg")
        img.save(fp, quality=90)
        frame_paths.append(fp)

    record = {
        "episode_id": sample.episode_id,
        "trajectory_id": sample.trajectory_id,
        "scene_id": sample.scene_id,
        "instruction": sample.instruction,
        "start_position": sample.start_position,
        "start_rotation": sample.start_rotation,
        "goal_position": sample.goal_position,
        "path_points": result["path_points"],
        "actions": result["actions"],
        "actions_text": [ACTION_ID_TO_NAME.get(a, str(a)) for a in result["actions"]],
        "action_events": result["action_events"],
        "action_events_text": result["action_events_text"],
        "action_events_raw": result["action_events_raw"],
        "action_events_text_raw": result["action_events_text_raw"],
        "frames": frame_paths,
        "frame_stride": frame_stride,
        "actions_per_frame": 1 if sample_mode == "event" else frame_stride,
        "sample_mode": sample_mode,
        "visual_prompt_meta": result["vp_meta"],
    }
    with open(os.path.join(ep_dir, "sample.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="R2R-CE event-driven rendering with Visual Prompt")
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--scenes_root", default="PATH/TO/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--sample_mode", choices=["stride", "event"], default="event")
    parser.add_argument("--goal_radius", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--forward_step", type=float, default=0.25)
    parser.add_argument("--turn_angle", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=640,
                        help="子视图渲染分辨率默认 256x640，最终全景降采样到 768x384")
    parser.add_argument("--camera_height", type=float, default=1.2)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--panorama", action="store_true")
    parser.add_argument("--panorama_hfov", type=float, default=90.0)
    parser.add_argument("--panorama_step", type=float, default=90.0)
    parser.add_argument("--max_parts", type=int, default=3)
    parser.add_argument("--small_fwd_m", type=float, default=0.5)
    parser.add_argument("--small_turn_deg", type=float, default=30.0)
    parser.add_argument("--split_forward_threshold_m", type=float, default=6.0)

    # ── Visual Prompt 专属参数 ──
    parser.add_argument("--vp_enabled", action="store_true", default=True)
    parser.add_argument("--vp_dropout_prob", type=float, default=0.15)
    parser.add_argument("--vp_alpha", type=float, default=0.55)
    parser.add_argument("--vp_mixed_mode_prob", type=float, default=0.10)
    parser.add_argument("--vp_seed", type=int, default=42)
    parser.add_argument("--no_vp", action="store_true", help="禁用 visual prompt（仅渲染原始帧 + depth）")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    vp_cfg = VisualPromptConfig(
        prompt_dropout_prob=args.vp_dropout_prob,
        path_alpha_base=args.vp_alpha,
        mixed_mode_prob=args.vp_mixed_mode_prob,
    )
    rng = np.random.RandomState(args.vp_seed)
    enable_vp = args.vp_enabled and not args.no_vp

    samples = load_r2r_vlnce(args.train_json)
    if args.max_episodes:
        samples = samples[: args.max_episodes]

    by_scene: Dict[str, List[EpisodeSample]] = defaultdict(list)
    for s in samples:
        by_scene[s.scene_id].append(s)

    total_eps = sum(len(v) for v in by_scene.values())
    logging.info("总 episode 数: %s, visual_prompt=%s", total_eps, enable_vp)

    progress = tqdm(total=total_eps, desc="渲染进度", unit="ep")
    for scene_id, eps in by_scene.items():
        scene_path = os.path.join(args.scenes_root, scene_id)
        if not os.path.exists(scene_path):
            logging.warning("跳过：场景不存在 %s", scene_path)
            progress.update(len(eps))
            continue

        hfov = args.panorama_hfov if args.panorama else args.hfov
        panorama_step = args.panorama_step if args.panorama else args.hfov

        logging.info("加载场景: %s", scene_path)
        sim = build_sim_with_depth(
            scene_path,
            args.width,
            args.height,
            args.forward_step,
            args.turn_angle,
            hfov,
            args.camera_height,
        )

        for ep in eps:
            ep_dir = os.path.join(args.output_dir, f"episode_{ep.episode_id}")
            sample_path = os.path.join(ep_dir, "sample.json")
            if os.path.exists(sample_path):
                logging.info("跳过 episode %s：已存在", ep.episode_id)
                progress.update(1)
                continue

            result = render_episode_with_vp(
                sim,
                ep,
                args.goal_radius,
                args.max_steps,
                args.forward_step,
                args.turn_angle,
                panorama_step,
                hfov,
                args.split_forward_threshold_m,
                args.small_fwd_m,
                args.small_turn_deg,
                args.max_parts,
                vp_cfg,
                args.camera_height,
                rng,
                enable_vp,
            )

            if result is None or not result["frames"]:
                logging.warning("跳过：无有效路径 episode %s", ep.episode_id)
                progress.update(1)
                continue

            save_episode(args.output_dir, ep, result, args.frame_stride, args.sample_mode)
            logging.info("保存 episode %s: %s 帧", ep.episode_id, len(result["frames"]))
            progress.update(1)

        sim.close()

    progress.close()
    logging.info("完成")


if __name__ == "__main__":
    main()
