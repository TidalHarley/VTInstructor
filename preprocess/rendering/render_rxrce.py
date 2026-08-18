#!/usr/bin/env python3
"""
Render RXR-CE (RxR_VLNCE_v0) samples using Habitat-Sim.
Select English instructions and force the path to pass through reference_path.
"""
import argparse
import gzip
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import habitat_sim
from habitat_sim.agent import ActionSpec, ActuationSpec
from habitat_sim.nav.greedy_geodesic_follower import GreedyGeodesicFollower
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs


ACTION_ID_TO_NAME = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
}
ACTION_NAME_TO_ID = {v: k for k, v in ACTION_ID_TO_NAME.items()}


@dataclass
class Segment:
    """聚合后的段：连续相同动作合并为一个段"""
    action: str  # 'forward' / 'left' / 'right'
    count: int   # 连续相同动作的步数
    end_step: int  # 该段结束时的step索引（用于定位采帧位置）


def _normalize_action_name(action) -> str:
    if isinstance(action, str):
        if action == "move_forward":
            return "forward"
        if action == "turn_left":
            return "left"
        if action == "turn_right":
            return "right"
        if action == "stop":
            return "stop"
        return str(action)
    if isinstance(action, (int, np.integer)):
        return ACTION_ID_TO_NAME.get(int(action), str(action))
    return str(action)


def _format_value(value: float) -> str:
    rounded = round(float(value), 2)
    if abs(rounded - round(rounded)) < 1e-6:
        return str(int(round(rounded)))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def format_segment_text(action_name: str, count: int, forward_step: float, turn_angle: float) -> str:
    """格式化单个段的文本描述"""
    if action_name == "forward":
        distance = count * forward_step
        return f"go forward for {_format_value(distance)}m"
    if action_name == "left":
        angle = count * turn_angle
        return f"turn left for {_format_value(angle)} degrees"
    if action_name == "right":
        angle = count * turn_angle
        return f"turn right for {_format_value(angle)} degrees"
    return action_name


def _is_small_segment(seg: Segment, forward_step: float, turn_angle: float) -> bool:
    """判断段是否是小段（可以和其他小段组合）
    小段：直行≤0.75m 或 旋转≤30°
    大段：直行>0.75m 或 旋转>30° → 必须单独成事件
    """
    if seg.action == "forward":
        distance = seg.count * forward_step
        return distance <= 0.75 + 1e-6
    if seg.action in ("left", "right"):
        angle = seg.count * turn_angle
        return angle <= 30.0 + 1e-6
    return False


def aggregate_actions_to_segments(actions: List[int], forward_step: float, turn_angle: float) -> List[Segment]:
    """第一步：把连续相同动作聚合成段
    例如：[1,1,1,1,1,1,1,3,3] → [Segment(forward,7), Segment(right,2)]
    """
    if not actions:
        return []
    
    segments: List[Segment] = []
    prev_action_name: Optional[str] = None
    run_count = 0
    step_idx = 0
    
    for act in actions:
        action_name = _normalize_action_name(act)
        if action_name == "stop":
            # stop不计入段
            if prev_action_name and run_count > 0:
                segments.append(Segment(prev_action_name, run_count, step_idx - 1))
            break
        
        if prev_action_name is None:
            prev_action_name = action_name
            run_count = 1
        elif action_name == prev_action_name:
            run_count += 1
        else:
            # 动作切换，保存之前的段
            segments.append(Segment(prev_action_name, run_count, step_idx - 1))
            prev_action_name = action_name
            run_count = 1
        step_idx += 1
    
    if prev_action_name and run_count > 0:
        segments.append(Segment(prev_action_name, run_count, step_idx - 1))
    
    return segments


def segments_to_events(
    segments: List[Segment],
    forward_step: float,
    turn_angle: float,
    max_parts: int = 3,
) -> Tuple[List[Dict], List[str], List[int]]:
    """第二步：基于聚合后的段组成事件
    规则：
    - 大段（>0.75m 或 >30°）单独成事件
    - 小段（≤0.75m 或 ≤30°）最多3个组合成一个事件
    
    返回：
    - events: 事件列表
    - texts: 事件文本列表
    - event_end_steps: 每个事件结束时的step索引（用于采帧）
    """
    events: List[Dict] = []
    texts: List[str] = []
    event_end_steps: List[int] = []
    
    i = 0
    while i < len(segments):
        seg = segments[i]
        is_large = not _is_small_segment(seg, forward_step, turn_angle)
        
        if is_large:
            # 大段单独成事件
            events.append({"action": seg.action, "count": seg.count})
            texts.append(format_segment_text(seg.action, seg.count, forward_step, turn_angle))
            event_end_steps.append(seg.end_step)
            i += 1
        else:
            # 小段，尝试和后续小段组合（最多max_parts个）
            group: List[Segment] = [seg]
            i += 1
            while i < len(segments) and len(group) < max_parts:
                next_seg = segments[i]
                if not _is_small_segment(next_seg, forward_step, turn_angle):
                    break
                group.append(next_seg)
                i += 1
            
            if len(group) == 1:
                g = group[0]
                events.append({"action": g.action, "count": g.count})
                texts.append(format_segment_text(g.action, g.count, forward_step, turn_angle))
                event_end_steps.append(g.end_step)
            else:
                # 多个小段组合成combo
                parts = [{"action": g.action, "count": g.count} for g in group]
                events.append({"action": "combo", "count": len(group), "parts": parts})
                texts.append(" / ".join(
                    format_segment_text(g.action, g.count, forward_step, turn_angle)
                    for g in group
                ))
                event_end_steps.append(group[-1].end_step)
    
    return events, texts, event_end_steps


@dataclass
class EpisodeSample:
    episode_id: str
    trajectory_id: str
    scene_id: str
    instruction: str
    start_position: List[float]
    start_rotation: List[float]
    goal_position: List[float]
    reference_path: List[List[float]]


def _extract_position(item) -> Optional[List[float]]:
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return [float(item[0]), float(item[1]), float(item[2])]
    if isinstance(item, dict):
        for key in ("position", "location", "pos"):
            if key in item:
                value = item[key]
                if isinstance(value, (list, tuple)) and len(value) >= 3:
                    return [float(value[0]), float(value[1]), float(value[2])]
    return None


def _extract_goal_position(ep: dict) -> Optional[List[float]]:
    if "goal_position" in ep:
        return _extract_position(ep["goal_position"])
    if "goals" in ep and ep["goals"]:
        goal = ep["goals"][0]
        if isinstance(goal, dict) and "position" in goal:
            return _extract_position(goal["position"])
    if "goal" in ep:
        return _extract_position(ep["goal"])
    return None


def _extract_start_rotation(ep: dict) -> List[float]:
    if "start_rotation" in ep:
        rot = ep["start_rotation"]
        if isinstance(rot, (list, tuple)) and len(rot) == 4:
            return [float(x) for x in rot]
    if "start_heading" in ep:
        heading = float(ep["start_heading"])
        quat = quat_from_angle_axis(heading, np.array([0.0, 1.0, 0.0]))
        return [float(x) for x in quat]
    return []


def _extract_instruction(ep: dict) -> Optional[Tuple[str, str]]:
    instr = ep.get("instruction")
    if isinstance(instr, dict):
        text = instr.get("instruction_text", "").strip()
        lang = instr.get("language", "")
        return text, lang
    if isinstance(instr, list):
        for item in instr:
            if not isinstance(item, dict):
                continue
            text = item.get("instruction_text", "").strip()
            lang = item.get("language", "")
            if text:
                return text, lang
    return None


def load_rxr_vlnce(train_json: str) -> List[EpisodeSample]:
    if train_json.endswith(".gz"):
        with gzip.open(train_json, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(train_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    episodes = data.get("episodes", [])

    samples = []
    for ep in episodes:
        instr = _extract_instruction(ep)
        if not instr:
            continue
        instr_text, instr_lang = instr
        if not instr_lang.lower().startswith("en"):
            continue

        start_pos = _extract_position(ep.get("start_position") or ep.get("start"))
        if not start_pos:
            continue
        start_rot = _extract_start_rotation(ep)

        ref_path_raw = ep.get("reference_path") or ep.get("reference_path_positions") or []
        reference_path = []
        for item in ref_path_raw:
            pos = _extract_position(item)
            if pos:
                reference_path.append(pos)
        if not reference_path:
            continue

        goal_pos = _extract_goal_position(ep)
        if not goal_pos:
            goal_pos = reference_path[-1]

        samples.append(
            EpisodeSample(
                episode_id=str(ep.get("episode_id")),
                trajectory_id=str(ep.get("trajectory_id")),
                scene_id=str(ep.get("scene_id")),
                instruction=instr_text,
                start_position=start_pos,
                start_rotation=start_rot,
                goal_position=goal_pos,
                reference_path=reference_path,
            )
        )
    return samples


def build_sim(
    scene_path: str,
    width: int,
    height: int,
    forward_step: float,
    turn_angle: float,
    hfov: float,
    camera_height: float,
) -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "rgb"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [height, width]
    rgb_sensor_spec.hfov = hfov
    rgb_sensor_spec.position = [0.0, camera_height, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_spec]
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
        raise RuntimeError(f"NavMesh未加载: {navmesh_path}")
    return sim


def render_episode(
    sim: habitat_sim.Simulator,
    sample: EpisodeSample,
    frame_stride: int,
    goal_radius: float,
    max_steps: int,
    sample_mode: str,
    forward_step: float,
    turn_angle: float,
    panorama: bool,
    panorama_step: float,
    max_event_actions: int,
) -> Tuple[
    List[Image.Image],
    List[int],
    List[List[float]],
    List[Dict],
    List[str],
    List[Dict],
    List[str],
]:
    """渲染一个episode (RxR版本，需要经过reference_path的所有waypoint)
    
    event模式下的逻辑：
    1. 先完整仿真，记录所有动作和每步的位置
    2. 把连续相同动作聚合成段
    3. 把段组合成事件
    4. 根据事件边界采集帧（起点+每个事件结束后）
    """
    
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(sample.start_position, dtype=np.float32)
    if sample.start_rotation:
        agent_state.rotation = quat_from_coeffs(sample.start_rotation)
    sim.get_agent(0).set_state(agent_state)

    follower = GreedyGeodesicFollower(sim.pathfinder, sim.get_agent(0), goal_radius=goal_radius)

    def capture_panorama(step_deg: float) -> Image.Image:
        agent = sim.get_agent(0)
        state = agent.get_state()
        base_rot = state.rotation
        yaw_delta = math.radians(step_deg)
        angles = [yaw_delta, 0.0, -yaw_delta]
        imgs: List[Image.Image] = []
        for a in angles:
            rot = quat_from_angle_axis(a, np.array([0.0, 1.0, 0.0])) * base_rot
            agent.set_state(habitat_sim.AgentState(position=state.position, rotation=rot))
            obs = sim.get_sensor_observations()
            rgb = obs["rgb"]
            img = Image.fromarray(rgb)
            if img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(img)
        agent.set_state(habitat_sim.AgentState(position=state.position, rotation=base_rot))
        width = imgs[0].width
        height = imgs[0].height
        pano = Image.new("RGB", (width * 3, height))
        for i, im in enumerate(imgs):
            pano.paste(im, (i * width, 0))
        return pano

    def capture_frame() -> Image.Image:
        if panorama:
            return capture_panorama(panorama_step)
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"]
        img = Image.fromarray(rgb)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def distance_to_target(target: List[float]) -> float:
        current = sim.get_agent(0).get_state().position.tolist()
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

    # ========== 第一阶段：完整仿真，记录动作和位置 ==========
    actions: List[int] = []
    step_positions: List[List[float]] = []  # 每步结束后的位置
    step_frames: List[Image.Image] = []     # 每步结束后的帧（用于event模式采样）
    
    start_frame = capture_frame()
    start_pos = sim.get_agent(0).get_state().position.tolist()
    
    step = 0
    all_reached = True
    
    # RxR需要经过所有waypoint
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
                step = max_steps
                all_reached = False
                break
            
            action_name = _normalize_action_name(action)
            if action is None or action_name == "stop" or action == 0:
                # 到达当前waypoint，检查距离
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
            
            if isinstance(action, str):
                if action == "move_forward":
                    actions.append(1)
                elif action == "turn_left":
                    actions.append(2)
                elif action == "turn_right":
                    actions.append(3)
                else:
                    actions.append(0)
            else:
                actions.append(int(action))
            
            sim.step(action)
            step += 1
            
            step_positions.append(sim.get_agent(0).get_state().position.tolist())
            if sample_mode == "event":
                step_frames.append(capture_frame())
        
        if step >= max_steps or not all_reached:
            all_reached = False
            break
    
    if all_reached:
        actions.append(0)
    
    if not all_reached:
        return [], [], [], [], [], [], []
    
    if not actions or (len(actions) == 1 and actions[0] == 0):
        return [], [], [], [], [], [], []
    
    # ========== 第二阶段：根据模式处理 ==========
    if sample_mode == "stride":
        # stride模式：固定间隔采样
        frames: List[Image.Image] = []
        path_points: List[List[float]] = []
        
        # 重新仿真采样
        agent_state.position = np.array(sample.start_position, dtype=np.float32)
        if sample.start_rotation:
            agent_state.rotation = quat_from_coeffs(sample.start_rotation)
        sim.get_agent(0).set_state(agent_state)
        
        for i, act in enumerate(actions):
            if act == 0:
                break
            if i % frame_stride == 0:
                frames.append(capture_frame())
                path_points.append(sim.get_agent(0).get_state().position.tolist())
            
            action_str = {1: "move_forward", 2: "turn_left", 3: "turn_right"}.get(act, "stop")
            sim.step(action_str)
        
        return frames, actions, path_points, [], [], [], []
    
    # ========== event模式 ==========
    # 第一步：聚合连续相同动作成段
    segments = aggregate_actions_to_segments(actions, forward_step, turn_angle)
    
    if not segments:
        return [], [], [], [], [], [], []
    
    # 第二步：把段组合成事件
    action_events, action_events_text, event_end_steps = segments_to_events(
        segments, forward_step, turn_angle, max_parts=max_event_actions
    )
    
    # 第三步：根据事件边界采集帧
    # 起点帧 + 每个事件结束后的帧
    frames: List[Image.Image] = [start_frame]
    path_points: List[List[float]] = [start_pos]
    
    for end_step in event_end_steps:
        if end_step < len(step_frames):
            frames.append(step_frames[end_step])
        if end_step < len(step_positions):
            path_points.append(step_positions[end_step])
    
    # raw事件就是聚合后的段（未组合）
    segments_raw = [{"action": seg.action, "count": seg.count} for seg in segments]
    segments_text_raw = [
        format_segment_text(seg.action, seg.count, forward_step, turn_angle)
        for seg in segments
    ]
    
    return frames, actions, path_points, action_events, action_events_text, segments_raw, segments_text_raw


def save_episode(
    out_dir: str,
    sample: EpisodeSample,
    frames: List[Image.Image],
    actions: List[int],
    path_points: List[List[float]],
    frame_stride: int,
    sample_mode: str,
    action_events: List[Dict],
    action_events_text: List[str],
    action_events_raw: List[Dict],
    action_events_text_raw: List[str],
):
    ep_dir = os.path.join(out_dir, f"episode_{sample.episode_id}")
    os.makedirs(ep_dir, exist_ok=True)

    frame_paths = []
    for idx, img in enumerate(frames):
        frame_path = os.path.join(ep_dir, f"frame_{idx:04d}.jpg")
        img.save(frame_path, quality=90)
        frame_paths.append(frame_path)

    record = {
        "episode_id": sample.episode_id,
        "trajectory_id": sample.trajectory_id,
        "scene_id": sample.scene_id,
        "instruction": sample.instruction,
        "start_position": sample.start_position,
        "start_rotation": sample.start_rotation,
        "goal_position": sample.goal_position,
        "reference_path": sample.reference_path,
        "path_points": path_points,
        "actions": actions,
        "actions_text": [ACTION_ID_TO_NAME.get(a, str(a)) for a in actions],
        "action_events": action_events,
        "action_events_text": action_events_text,
        "action_events_raw": action_events_raw,
        "action_events_text_raw": action_events_text_raw,
        "frames": frame_paths,
        "frame_stride": frame_stride,
        "sample_mode": sample_mode,
    }

    with open(os.path.join(ep_dir, "sample.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", default="PATH/TO/RxR_VLNCE_v0/train/train_guide.json")
    parser.add_argument("--scenes_root", default="PATH/TO/data")
    parser.add_argument("--output_dir", default="PATH/TO/outputs/nig_samples_rxr_train")
    parser.add_argument("--log_path", default="PATH/TO/outputs/nig_render_rxr.log")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument(
        "--sample_mode",
        choices=["stride", "event"],
        default="event",
        help="stride: 固定间隔采样; event: 按事件边界采样",
    )
    parser.add_argument("--goal_radius", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--forward_step", type=float, default=0.25)
    parser.add_argument("--turn_angle", type=float, default=30.0)
    parser.add_argument("--max_event_actions", type=int, default=5, help="事件内最多段数(仅event模式)")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--camera_height", type=float, default=0.75)
    parser.add_argument("--panorama", action="store_true", help="采样270度全景(左/中/右三张拼接)")
    parser.add_argument("--panorama_hfov", type=float, default=90.0, help="全景模式下单张视角水平FOV")
    parser.add_argument("--panorama_step", type=float, default=90.0, help="全景模式下左右旋转角度")
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

    samples = load_rxr_vlnce(args.train_json)
    if args.max_episodes:
        samples = samples[: args.max_episodes]

    by_scene: Dict[str, List[EpisodeSample]] = defaultdict(list)
    for s in samples:
        by_scene[s.scene_id].append(s)

    total_eps = sum(len(v) for v in by_scene.values())
    logging.info("总episode数: %s", total_eps)

    progress = tqdm(total=total_eps, desc="渲染进度", unit="ep")
    for scene_id, eps in by_scene.items():
        scene_path = os.path.join(args.scenes_root, scene_id)
        if not os.path.exists(scene_path):
            logging.warning("跳过：场景不存在 %s", scene_path)
            progress.update(len(eps))
            continue

        logging.info("加载场景: %s", scene_path)
        panorama_step = args.panorama_step
        if args.panorama:
            hfov = args.panorama_hfov
            panorama_step = args.panorama_step or args.panorama_hfov
        else:
            hfov = args.hfov
        sim = build_sim(
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
                logging.info("跳过 episode %s：已存在 %s", ep.episode_id, sample_path)
                progress.update(1)
                continue
            (
                frames,
                actions,
                path_points,
                action_events,
                action_events_text,
                action_events_raw,
                action_events_text_raw,
            ) = render_episode(
                sim,
                ep,
                args.frame_stride,
                args.goal_radius,
                args.max_steps,
                args.sample_mode,
                args.forward_step,
                args.turn_angle,
                args.panorama,
                panorama_step,
                args.max_event_actions,
            )
            if not frames:
                logging.warning("跳过：无有效路径 episode %s", ep.episode_id)
                progress.update(1)
                continue
            save_episode(
                args.output_dir,
                ep,
                frames,
                actions,
                path_points,
                args.frame_stride,
                args.sample_mode,
                action_events,
                action_events_text,
                action_events_raw,
                action_events_text_raw,
            )
            logging.info("保存 episode %s: %s 帧, %s 事件", ep.episode_id, len(frames), len(action_events))
            progress.update(1)

        sim.close()

    progress.close()
    logging.info("完成")


if __name__ == "__main__":
    main()
