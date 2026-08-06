#!/usr/bin/env python3
"""
R2R-CE 事件驱动渲染（panorama 版本，detail 划分）：
1) 连续动作先聚合为段；
2) forward 段若距离 > 6m，则二分为两个 forward 事件；
3) 小事件（forward<=0.5m 或 turn<=30deg）最多 3 段合并为一个 combo 事件；
4) 支持 panorama 输出（左/中/右拼接，尺寸 256x256x3）。
"""
import argparse
import gzip
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def _to_action_id(action) -> int:
    if isinstance(action, str):
        if action == "move_forward":
            return 1
        if action == "turn_left":
            return 2
        if action == "turn_right":
            return 3
        return 0
    try:
        return int(action)
    except Exception:
        return 0


def _format_value(value: float) -> str:
    rounded = round(float(value), 2)
    if abs(rounded - round(rounded)) < 1e-6:
        return str(int(round(rounded)))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def format_event_action(action_name: str, count: int, forward_step: float, turn_angle: float) -> str:
    if action_name == "forward":
        return f"go forward for {_format_value(count * forward_step)}m"
    if action_name == "left":
        return f"turn left for {_format_value(count * turn_angle)} degrees"
    if action_name == "right":
        return f"turn right for {_format_value(count * turn_angle)} degrees"
    return action_name


@dataclass
class EpisodeSample:
    episode_id: int
    trajectory_id: int
    scene_id: str
    instruction: str
    start_position: List[float]
    start_rotation: List[float]
    goal_position: List[float]


@dataclass
class Segment:
    action: str
    count: int
    end_step: int


def load_r2r_vlnce(split_json_gz: str) -> List[EpisodeSample]:
    with gzip.open(split_json_gz, "rt", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data["episodes"]
    samples: List[EpisodeSample] = []
    for ep in episodes:
        instr = ep["instruction"]["instruction_text"].strip()
        goals = ep.get("goals", [])
        if not goals:
            continue
        samples.append(
            EpisodeSample(
                episode_id=ep["episode_id"],
                trajectory_id=ep["trajectory_id"],
                scene_id=ep["scene_id"],
                instruction=instr,
                start_position=ep["start_position"],
                start_rotation=ep["start_rotation"],
                goal_position=goals[0]["position"],
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
    camera_height: float = 0.75,
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


def _is_small_segment(action_name: str, count: int, forward_step: float, turn_angle: float, small_fwd_m: float, small_turn_deg: float) -> bool:
    if action_name == "forward":
        return (count * forward_step) <= float(small_fwd_m) + 1e-6
    if action_name in ("left", "right"):
        return (count * turn_angle) <= float(small_turn_deg) + 1e-6
    return False


def _split_long_forward_segments(
    segments: List[Segment],
    forward_step: float,
    split_forward_threshold_m: float,
) -> List[Segment]:
    result: List[Segment] = []
    for seg in segments:
        if seg.action != "forward":
            result.append(seg)
            continue
        distance = seg.count * forward_step
        if distance <= float(split_forward_threshold_m) + 1e-6 or seg.count <= 1:
            result.append(seg)
            continue
        first_count = seg.count // 2
        second_count = seg.count - first_count
        first_end = seg.end_step - second_count
        result.append(Segment(action="forward", count=first_count, end_step=first_end))
        result.append(Segment(action="forward", count=second_count, end_step=seg.end_step))
    return result


def _segments_to_events(
    segments: List[Segment],
    forward_step: float,
    turn_angle: float,
    max_parts: int,
    small_fwd_m: float,
    small_turn_deg: float,
) -> Tuple[List[Dict[str, int]], List[str], List[int]]:
    events: List[Dict[str, int]] = []
    texts: List[str] = []
    end_steps: List[int] = []

    i = 0
    while i < len(segments):
        seg = segments[i]
        if not _is_small_segment(seg.action, seg.count, forward_step, turn_angle, small_fwd_m, small_turn_deg):
            events.append({"action": seg.action, "count": seg.count})
            texts.append(format_event_action(seg.action, seg.count, forward_step, turn_angle))
            end_steps.append(seg.end_step)
            i += 1
            continue

        group: List[Segment] = [seg]
        i += 1
        while i < len(segments) and len(group) < max_parts:
            nxt = segments[i]
            if not _is_small_segment(nxt.action, nxt.count, forward_step, turn_angle, small_fwd_m, small_turn_deg):
                break
            group.append(nxt)
            i += 1

        if len(group) == 1:
            g = group[0]
            events.append({"action": g.action, "count": g.count})
            texts.append(format_event_action(g.action, g.count, forward_step, turn_angle))
            end_steps.append(g.end_step)
        else:
            parts = [{"action": g.action, "count": g.count} for g in group]
            events.append({"action": "combo", "count": len(group), "parts": parts})
            texts.append(
                " / ".join(
                    format_event_action(g.action, g.count, forward_step, turn_angle) for g in group
                )
            )
            end_steps.append(group[-1].end_step)

    return events, texts, end_steps


def _build_segments_from_actions(actions: List[int]) -> List[Segment]:
    segments: List[Segment] = []
    run_action = None
    run_count = 0
    for step_idx, act in enumerate(actions):
        name = _normalize_action_name(act)
        if name == "stop":
            break
        if run_action is None:
            run_action = name
            run_count = 1
            continue
        if name == run_action:
            run_count += 1
            continue
        segments.append(Segment(action=run_action, count=run_count, end_step=step_idx - 1))
        run_action = name
        run_count = 1
    if run_action is not None and run_count > 0:
        end_step = min(len([a for a in actions if a != 0]) - 1, len(actions) - 1)
        segments.append(Segment(action=run_action, count=run_count, end_step=end_step))
    return segments


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
    split_forward_threshold_m: float,
    small_fwd_m: float,
    small_turn_deg: float,
    max_parts: int,
) -> Tuple[
    List[Image.Image],
    List[int],
    List[List[float]],
    List[Dict[str, int]],
    List[str],
    List[Dict[str, int]],
    List[str],
]:
    frames: List[Image.Image] = []
    actions: List[int] = []
    path_points: List[List[float]] = []

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

    if sample_mode == "event":
        frames.append(capture_frame())
        path_points.append(sim.get_agent(0).get_state().position.tolist())

    step_frames: List[Image.Image] = []
    step_positions: List[List[float]] = []

    step = 0
    while step < max_steps:
        try:
            action = follower.next_action_along(sample.goal_position)
        except Exception as exc:
            logging.warning("episode %s 路径跟随失败，跳过: %s", sample.episode_id, exc)
            break

        action_name = _normalize_action_name(action)
        if action is None or action_name == "stop" or action == 0:
            actions.append(0)
            break

        action_id = _to_action_id(action)
        actions.append(action_id)
        sim.step(action)
        step_frames.append(capture_frame())
        step_positions.append(sim.get_agent(0).get_state().position.tolist())
        step += 1

    if sample_mode == "stride":
        if not actions:
            return frames, actions, path_points, [], [], [], []
        stride_frames: List[Image.Image] = [capture_frame()]
        stride_points: List[List[float]] = [sim.get_agent(0).get_state().position.tolist()]
        return stride_frames, actions, stride_points, [], [], [], []

    raw_segments = _build_segments_from_actions(actions)
    if not raw_segments:
        return frames, actions, path_points, [], [], [], []

    split_segments = _split_long_forward_segments(raw_segments, forward_step, split_forward_threshold_m)
    action_events, action_events_text, event_end_steps = _segments_to_events(
        split_segments,
        forward_step,
        turn_angle,
        max_parts=max_parts,
        small_fwd_m=small_fwd_m,
        small_turn_deg=small_turn_deg,
    )

    for end_step in event_end_steps:
        if 0 <= end_step < len(step_frames):
            frames.append(step_frames[end_step])
            path_points.append(step_positions[end_step])

    action_events_raw = [{"action": s.action, "count": s.count} for s in split_segments]
    action_events_text_raw = [
        format_event_action(s.action, s.count, forward_step, turn_angle) for s in split_segments
    ]

    return frames, actions, path_points, action_events, action_events_text, action_events_raw, action_events_text_raw


def save_episode(
    out_dir: str,
    sample: EpisodeSample,
    frames: List[Image.Image],
    actions: List[int],
    path_points: List[List[float]],
    frame_stride: int,
    sample_mode: str,
    action_events: List[Dict[str, int]],
    action_events_text: List[str],
    action_events_raw: List[Dict[str, int]],
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
        "path_points": path_points,
        "actions": actions,
        "actions_text": [ACTION_ID_TO_NAME.get(a, str(a)) for a in actions],
        "action_events": action_events,
        "action_events_text": action_events_text,
        "action_events_raw": action_events_raw,
        "action_events_text_raw": action_events_text_raw,
        "frames": frame_paths,
        "frame_stride": frame_stride,
        "actions_per_frame": 1 if sample_mode == "event" else frame_stride,
        "sample_mode": sample_mode,
    }
    with open(os.path.join(ep_dir, "sample.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--camera_height", type=float, default=0.75)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--panorama", action="store_true")
    parser.add_argument("--panorama_hfov", type=float, default=90.0)
    parser.add_argument("--panorama_step", type=float, default=90.0)
    parser.add_argument("--max_parts", type=int, default=5, help="小事件最多合并段数")
    parser.add_argument("--small_fwd_m", type=float, default=0.5, help="小前进事件阈值(米)")
    parser.add_argument("--small_turn_deg", type=float, default=30.0, help="小转向事件阈值(度)")
    parser.add_argument("--split_forward_threshold_m", type=float, default=6.0, help="前进事件二分阈值(米)")
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

    samples = load_r2r_vlnce(args.train_json)
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

        panorama_step = args.panorama_step
        if args.panorama:
            hfov = args.panorama_hfov
            panorama_step = args.panorama_step or args.panorama_hfov
        else:
            hfov = args.hfov

        logging.info("加载场景: %s", scene_path)
        sim = build_sim(scene_path, args.width, args.height, args.forward_step, args.turn_angle, hfov, args.camera_height)

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
                args.split_forward_threshold_m,
                args.small_fwd_m,
                args.small_turn_deg,
                args.max_parts,
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
            logging.info("保存 episode %s: %s 帧", ep.episode_id, len(frames))
            progress.update(1)

        sim.close()

    progress.close()
    logging.info("完成")


if __name__ == "__main__":
    main()
