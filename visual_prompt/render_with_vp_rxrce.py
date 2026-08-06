#!/usr/bin/env python3
"""
RXR-CE 事件驱动渲染 + Adaptive World-Grounded Visual Prompting。

说明：
  - 数据加载、英文指令筛选、reference_path 经过逻辑与原始 RXR 渲染脚本一致
  - 视觉提示渲染逻辑复用 visual_prompt/overlay.py
  - 与 R2RCE visual 渲染入口分离，便于并行执行
"""
import argparse
import logging
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import habitat_sim
from habitat_sim.nav.greedy_geodesic_follower import GreedyGeodesicFollower
from habitat_sim.utils.common import quat_from_coeffs

_BASE = os.path.dirname(os.path.abspath(__file__))
_SRC_CANDIDATES = [
    os.path.join(_BASE, "..", "rendering"),
]
for _src in _SRC_CANDIDATES:
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

from nig_render_rxr_dataset import (  # noqa: E402
    ACTION_ID_TO_NAME,
    EpisodeSample,
    _normalize_action_name,
    aggregate_actions_to_segments,
    format_segment_text,
    load_rxr_vlnce,
    segments_to_events,
)

_VP = os.path.join(_BASE, "..")
if _VP not in sys.path:
    sys.path.insert(0, _VP)

try:
    from visual_prompt.augmentation import AugParams, sample_augmentation  # noqa: E402
    from visual_prompt.config import VisualPromptConfig  # noqa: E402
    from visual_prompt.overlay import overlay_panorama  # noqa: E402
    from visual_prompt.render_with_vp import (  # noqa: E402
        _aug_dict,
        _downsample_pano,
        _stitch,
        build_sim_with_depth,
        capture_subviews,
    )
    from visual_prompt.ribbon import determine_turn_direction  # noqa: E402
except ImportError:
    from augmentation import AugParams, sample_augmentation  # noqa: E402
    from config import VisualPromptConfig  # noqa: E402
    from overlay import overlay_panorama  # noqa: E402
    from render_with_vp import (  # noqa: E402
        _aug_dict,
        _downsample_pano,
        _stitch,
        build_sim_with_depth,
        capture_subviews,
    )
    from ribbon import determine_turn_direction  # noqa: E402


def _action_to_id(action) -> int:
    if isinstance(action, str):
        if action == "move_forward":
            return 1
        if action == "turn_left":
            return 2
        if action == "turn_right":
            return 3
        return 0
    if action is None:
        return 0
    return int(action)


def walk_reference_trajectory(
    sim: habitat_sim.Simulator,
    sample: EpisodeSample,
    goal_radius: float,
    max_steps: int,
) -> Tuple[List[int], List[Dict]]:
    """
    与原始 RXR 渲染逻辑一致：必须依次经过 reference_path waypoint。
    记录每一步结束后的 position / rotation。
    """
    agent = sim.get_agent(0)
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(sample.start_position, dtype=np.float32)
    if sample.start_rotation:
        agent_state.rotation = quat_from_coeffs(sample.start_rotation)
    agent.set_state(agent_state)

    follower = GreedyGeodesicFollower(sim.pathfinder, agent, goal_radius=goal_radius)

    def distance_to_target(target: List[float]) -> float:
        current = agent.get_state().position.tolist()
        try:
            dist = sim.pathfinder.geodesic_distance(current, target)
        except Exception:
            dist = None
        if dist is None or math.isinf(dist) or math.isnan(dist):
            dx = current[0] - target[0]
            dy = current[1] - target[1]
            dz = current[2] - target[2]
            return math.sqrt(dx * dx + dy * dy + dz * dz)
        return float(dist)

    actions: List[int] = []
    step_states: List[Dict] = []
    step = 0
    all_reached = True

    waypoints = list(sample.reference_path)
    if sample.goal_position:
        if not waypoints or waypoints[-1] != sample.goal_position:
            waypoints.append(sample.goal_position)

    for target in waypoints:
        while step < max_steps:
            try:
                action = follower.next_action_along(target)
            except Exception as exc:
                logging.warning("episode %s 路径跟随失败，跳过: %s", sample.episode_id, exc)
                all_reached = False
                step = max_steps
                break

            action_name = _normalize_action_name(action)
            if action is None or action_name == "stop" or action == 0:
                dist = distance_to_target(target)
                if dist > goal_radius:
                    logging.warning(
                        "episode %s 未到达reference点(距离%.2f > %.2f), 终止",
                        sample.episode_id,
                        dist,
                        goal_radius,
                    )
                    all_reached = False
                break

            actions.append(_action_to_id(action))
            sim.step(action)
            step += 1

            st = agent.get_state()
            step_states.append(
                {
                    "position": np.array(st.position, dtype=np.float64),
                    "rotation": st.rotation,
                }
            )

        if step >= max_steps or not all_reached:
            all_reached = False
            break

    if all_reached:
        actions.append(0)

    if not all_reached:
        return [], []
    if not actions or (len(actions) == 1 and actions[0] == 0):
        return [], []
    return actions, step_states


def render_episode_with_vp_rxrce(
    sim: habitat_sim.Simulator,
    sample: EpisodeSample,
    goal_radius: float,
    max_steps: int,
    forward_step: float,
    turn_angle: float,
    panorama_step: float,
    hfov: float,
    max_event_actions: int,
    vp_cfg: VisualPromptConfig,
    camera_height: float,
    rng: Optional[np.random.RandomState] = None,
    enable_vp: bool = True,
) -> Optional[Dict]:
    actions, step_states = walk_reference_trajectory(sim, sample, goal_radius, max_steps)
    if not actions:
        return None

    segments = aggregate_actions_to_segments(actions, forward_step, turn_angle)
    if not segments:
        return None

    action_events, action_events_text, event_end_steps = segments_to_events(
        segments, forward_step, turn_angle, max_parts=max_event_actions
    )
    if not event_end_steps:
        return None

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

    def _render_frame_at(agent_state, event_idx: int, frame_idx: int) -> Tuple[Image.Image, Dict]:
        agent.set_state(agent_state)
        rgb_list, depth_list, cam_pos, rot_list = capture_subviews(sim, panorama_step, camera_height)

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
            rgb_list,
            depth_list,
            cam_pos,
            rot_list,
            hfov,
            event_idx,
            cur_pos,
            step_states,
            event_end_steps,
            action_events,
            turn_dir,
            goal_pos if is_near_end else None,
            is_near_end,
            vp_cfg,
            aug,
        )
        pano = _downsample_pano(pano)
        meta = {"mode": modes, "augmentation": _aug_dict(aug)}
        return Image.fromarray(pano), meta

    frame0, meta0 = _render_frame_at(initial_state, event_idx=0, frame_idx=0)
    frames.append(frame0)
    path_points.append(sample.start_position)
    vp_meta.append(meta0)

    for i, end_step in enumerate(event_end_steps):
        if end_step >= len(step_states):
            continue
        boundary_state = habitat_sim.AgentState()
        boundary_state.position = step_states[end_step]["position"].astype(np.float32)
        boundary_state.rotation = step_states[end_step]["rotation"]

        next_event_idx = i + 1
        frame_i, meta_i = _render_frame_at(
            boundary_state, event_idx=next_event_idx, frame_idx=next_event_idx
        )
        frames.append(frame_i)
        path_points.append(step_states[end_step]["position"].tolist())
        vp_meta.append(meta_i)

    action_events_raw = [{"action": seg.action, "count": seg.count} for seg in segments]
    action_events_text_raw = [
        format_segment_text(seg.action, seg.count, forward_step, turn_angle) for seg in segments
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


def save_episode_rxrce(
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
        "reference_path": sample.reference_path,
        "path_points": result["path_points"],
        "actions": result["actions"],
        "actions_text": [ACTION_ID_TO_NAME.get(a, str(a)) for a in result["actions"]],
        "action_events": result["action_events"],
        "action_events_text": result["action_events_text"],
        "action_events_raw": result["action_events_raw"],
        "action_events_text_raw": result["action_events_text_raw"],
        "frames": frame_paths,
        "frame_stride": frame_stride,
        "sample_mode": sample_mode,
        "visual_prompt_meta": result["vp_meta"],
    }
    with open(os.path.join(ep_dir, "sample.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(record, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="RXR-CE event-driven rendering with Visual Prompt")
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--scenes_root", default="PATH/TO/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--sample_mode", choices=["stride", "event"], default="event")
    parser.add_argument("--goal_radius", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--forward_step", type=float, default=0.25)
    parser.add_argument("--turn_angle", type=float, default=30.0)
    parser.add_argument("--max_event_actions", type=int, default=3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--camera_height", type=float, default=1.2)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--panorama", action="store_true")
    parser.add_argument("--panorama_hfov", type=float, default=90.0)
    parser.add_argument("--panorama_step", type=float, default=90.0)
    parser.add_argument("--vp_enabled", action="store_true", default=True)
    parser.add_argument("--vp_dropout_prob", type=float, default=0.0)
    parser.add_argument("--vp_alpha", type=float, default=0.55)
    parser.add_argument("--vp_mixed_mode_prob", type=float, default=0.0)
    parser.add_argument("--vp_seed", type=int, default=42)
    parser.add_argument("--no_vp", action="store_true")
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

    samples = load_rxr_vlnce(args.train_json)
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

            result = render_episode_with_vp_rxrce(
                sim,
                ep,
                args.goal_radius,
                args.max_steps,
                args.forward_step,
                args.turn_angle,
                panorama_step,
                hfov,
                args.max_event_actions,
                vp_cfg,
                args.camera_height,
                rng,
                enable_vp,
            )
            if result is None or not result["frames"]:
                logging.warning("跳过：无有效路径 episode %s", ep.episode_id)
                progress.update(1)
                continue

            save_episode_rxrce(args.output_dir, ep, result, args.frame_stride, args.sample_mode)
            logging.info("保存 episode %s: %s 帧", ep.episode_id, len(result["frames"]))
            progress.update(1)

        sim.close()

    progress.close()
    logging.info("完成")


if __name__ == "__main__":
    main()
