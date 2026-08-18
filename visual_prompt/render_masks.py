#!/usr/bin/env python3
"""Stage 2: paint VTP overlays and 3-channel masks onto Stage-1 RGB keyframes.

Reads existing episode_*/sample.json (written by preprocess/rendering),
re-walks the trajectory in Habitat for poses + depth, then writes
frame_*_overlay.jpg and frame_*_vpmask.png in place.

    python visual_prompt/render_masks.py \
        --data_dir data/R2RCE_visual/r2rce_train_visual \
        --scenes_root /path/to/scenes_root
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import habitat_sim
import numpy as np
from PIL import Image
from tqdm import tqdm
from habitat_sim.agent import ActionSpec, ActuationSpec
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from augmentation import AugParams
from config import VisualPromptConfig
from ribbon import determine_turn_direction
from vp_overlay_with_mask import overlay_panorama_with_mask


ACTION_TO_SIM = {
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
}


def build_sim(scene_path: str, width: int, height: int, forward_step: float,
              turn_angle: float, hfov: float, camera_height: float):
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

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    navmesh_path = os.path.splitext(scene_path)[0] + ".navmesh"
    if os.path.exists(navmesh_path):
        sim.pathfinder.load_nav_mesh(navmesh_path)
    if not sim.pathfinder.is_loaded:
        raise RuntimeError(f"NavMesh not loaded: {navmesh_path}")
    return sim


def capture_subviews(sim, panorama_step_deg: float, camera_height: float):
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
        agent.set_state(habitat_sim.AgentState(position=state.position, rotation=rot))
        obs = sim.get_sensor_observations()
        rgb = np.asarray(obs["rgb"])
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        depth = np.asarray(obs["depth"], dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        rgb_list.append(rgb.copy())
        depth_list.append(depth.copy())
        rot_list.append(rot)

    agent.set_state(habitat_sim.AgentState(position=state.position, rotation=base_rot))
    return rgb_list, depth_list, cam_pos, rot_list


def event_step_count(event: dict) -> int:
    if event.get("action") == "combo":
        return sum(int(p.get("count", 0)) for p in event.get("parts", []))
    return int(event.get("count", 0))


def derive_event_end_steps(action_events: List[dict]) -> List[int]:
    steps = []
    cur = 0
    for ev in action_events:
        cur += event_step_count(ev)
        if cur > 0:
            steps.append(cur - 1)
    return steps


def replay_actions(sim, sample: dict) -> List[dict]:
    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array(sample["start_position"], dtype=np.float32)
    if sample.get("start_rotation"):
        state.rotation = quat_from_coeffs(sample["start_rotation"])
    agent.set_state(state)

    step_states = []
    for action_id in sample.get("actions", []):
        action_id = int(action_id)
        if action_id == 0:
            break
        action_name = ACTION_TO_SIM.get(action_id)
        if action_name is None:
            break
        sim.step(action_name)
        st = agent.get_state()
        step_states.append({
            "position": np.array(st.position, dtype=np.float64),
            "rotation": st.rotation,
        })
    return step_states


def get_clean_frame_paths(sample: dict) -> List[str]:
    """Return the original RGB frames, preserving them across overlay reruns."""
    frames = sample.get("clean_frames") or sample.get("rgb_frames") or sample.get("frames", [])
    clean = []
    for fp in frames:
        if fp.endswith("_overlay.jpg"):
            candidate = fp.replace("_overlay.jpg", ".jpg")
            clean.append(candidate)
        else:
            clean.append(fp)
    return clean


def render_prompts_for_sample(sim, sample: dict, width: int, height: int,
                              camera_height: float, hfov: float,
                              panorama_step: float):
    action_events = sample.get("action_events", [])
    if not action_events:
        return []

    step_states = replay_actions(sim, sample)
    event_end_steps = derive_event_end_steps(action_events)
    goal_pos = np.array(sample["goal_position"], dtype=np.float64)
    vp_cfg = VisualPromptConfig()
    aug = AugParams()
    agent = sim.get_agent(0)

    initial_state = habitat_sim.AgentState()
    initial_state.position = np.array(sample["start_position"], dtype=np.float32)
    if sample.get("start_rotation"):
        initial_state.rotation = quat_from_coeffs(sample["start_rotation"])

    clean_frames = get_clean_frame_paths(sample)
    frame_count = len(clean_frames)
    n_frames = len(event_end_steps) + 1

    def render_at(agent_state, event_idx: int, frame_idx: int):
        agent.set_state(agent_state)
        rgb_list, depth_list, cam_pos, rot_list = capture_subviews(
            sim, panorama_step, camera_height)
        cur_pos = np.array(agent_state.position, dtype=np.float64)
        turn_dir = determine_turn_direction(event_idx, action_events) \
            if event_idx < len(action_events) else "straight"
        is_near_end = (frame_idx >= n_frames - 5) and (frame_idx < n_frames - 1)
        pano_overlay, _, pano_mask = overlay_panorama_with_mask(
            rgb_list, depth_list, cam_pos, rot_list, hfov,
            event_idx, cur_pos, step_states, event_end_steps,
            action_events, turn_dir,
            goal_pos if is_near_end else None, is_near_end,
            vp_cfg, aug,
        )
        return pano_overlay, pano_mask

    overlays, masks = [], []
    overlay, mask = render_at(initial_state, 0, 0)
    overlays.append(overlay)
    masks.append(mask)
    for i, end_step in enumerate(event_end_steps):
        if end_step >= len(step_states):
            continue
        st = habitat_sim.AgentState()
        st.position = step_states[end_step]["position"].astype(np.float32)
        st.rotation = step_states[end_step]["rotation"]
        overlay, mask = render_at(st, i + 1, i + 1)
        overlays.append(overlay)
        masks.append(mask)

    if frame_count:
        if len(masks) > frame_count:
            overlays = overlays[:frame_count]
            masks = masks[:frame_count]
        elif len(masks) < frame_count:
            zeros = np.zeros((height, width * 3, 3), dtype=np.float32)
            for i in range(len(masks), frame_count):
                clean_path = clean_frames[i]
                if os.path.exists(clean_path):
                    overlays.append(np.asarray(Image.open(clean_path).convert("RGB")))
                else:
                    overlays.append(np.zeros((height, width * 3, 3), dtype=np.uint8))
                masks.append(zeros.copy())
    return overlays, masks


def save_visual_prompts(ep_dir: str, overlays: List[np.ndarray],
                        masks: List[np.ndarray]) -> Tuple[List[str], List[str]]:
    overlay_paths, mask_paths = [], []
    for idx, (overlay, mask) in enumerate(zip(overlays, masks)):
        overlay_fp = os.path.join(ep_dir, f"frame_{idx:04d}_overlay.jpg")
        Image.fromarray(np.asarray(overlay, dtype=np.uint8), mode="RGB").save(
            overlay_fp, quality=95)
        overlay_paths.append(overlay_fp)

        m = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
        if m.ndim == 2:
            m = np.stack([m, m, m], axis=-1)
        fp = os.path.join(ep_dir, f"frame_{idx:04d}_vpmask.png")
        Image.fromarray(m, mode="RGB").save(fp)
        mask_paths.append(fp)
    return overlay_paths, mask_paths


def load_samples(data_dir: str, max_episodes: int = 0) -> List[str]:
    files = []
    for name in os.listdir(data_dir):
        if name.startswith("episode_"):
            p = os.path.join(data_dir, name, "sample.json")
            if os.path.exists(p):
                files.append(p)
    files = sorted(files, key=lambda p: os.path.basename(os.path.dirname(p)))
    return files[:max_episodes] if max_episodes else files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--scenes_root", required=True)
    ap.add_argument("--log_path", default="")
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--camera_height", type=float, default=0.75)
    ap.add_argument("--forward_step", type=float, default=0.25)
    ap.add_argument("--turn_angle", type=float, default=30.0)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--panorama_step", type=float, default=90.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.log_path:
        os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log_path, mode="w", encoding="utf-8")
            if args.log_path else logging.StreamHandler(),
            logging.StreamHandler(),
        ] if args.log_path else [logging.StreamHandler()],
    )

    sample_files = load_samples(args.data_dir, args.max_episodes)
    by_scene: Dict[str, List[str]] = defaultdict(list)
    for fp in sample_files:
        with open(fp, "r", encoding="utf-8") as f:
            sample = json.load(f)
        by_scene[sample["scene_id"]].append(fp)

    progress = tqdm(total=len(sample_files), desc="VP overlays/masks from samples", unit="ep")
    for scene_id, files in by_scene.items():
        scene_path = os.path.join(args.scenes_root, scene_id)
        if not os.path.exists(scene_path):
            logging.warning("Skipping missing scene: %s", scene_path)
            progress.update(len(files))
            continue
        sim = build_sim(scene_path, args.width, args.height, args.forward_step,
                        args.turn_angle, args.hfov, args.camera_height)
        for sample_path in files:
            ep_dir = os.path.dirname(sample_path)
            first_mask = os.path.join(ep_dir, "frame_0000_vpmask.png")
            with open(sample_path, "r", encoding="utf-8") as f:
                sample = json.load(f)
            first_overlay = os.path.join(ep_dir, "frame_0000_overlay.jpg")
            if (os.path.exists(first_overlay) and os.path.exists(first_mask)
                    and not args.overwrite):
                progress.update(1)
                continue
            clean_frames = get_clean_frame_paths(sample)
            overlays, masks = render_prompts_for_sample(
                sim, sample, args.width, args.height,
                args.camera_height, args.hfov, args.panorama_step)
            if not masks:
                logging.warning("No overlays/masks for %s", sample_path)
                progress.update(1)
                continue
            overlay_paths, mask_paths = save_visual_prompts(ep_dir, overlays, masks)
            sample["clean_frames"] = clean_frames
            sample["overlay_frames"] = overlay_paths
            sample["frames"] = overlay_paths
            sample["vp_masks"] = mask_paths
            sample["vp_mask_schema"] = "C0=ribbon,C1=arrow,C2=endpoint"
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)
            progress.update(1)
        sim.close()
    progress.close()
    logging.info("Done")


if __name__ == "__main__":
    main()
