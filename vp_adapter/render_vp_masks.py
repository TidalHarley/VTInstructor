#!/usr/bin/env python3
"""
VP Mask Renderer — generates 3-channel semantic masks for existing episodes.

Re-walks trajectories in habitat-sim and runs the VP overlay computation,
but instead of re-saving frames (they already exist), only saves the VP
semantic masks as PNG files alongside existing frames.

Usage:
  python VP-adapter/render_vp_masks.py \
      --train_json <r2rce_train.json.gz> \
      --data_dir /path/to/existing/rendered/episodes \
      --output_dir /path/to/existing/rendered/episodes \
      --panorama --sample_mode event

When --output_dir equals --data_dir, masks are written into existing
episode folders (frame_0000_vpmask.png, etc.) and sample.json is updated
with a "vp_masks" field.
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

# ── Import habitat-sim ──
import habitat_sim
from habitat_sim.agent import ActionSpec, ActuationSpec
from habitat_sim.nav.greedy_geodesic_follower import GreedyGeodesicFollower
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs

# ── Import from original rendering code ──
_BASE = os.path.dirname(os.path.abspath(__file__))
_SRC_CANDIDATES = [
    os.path.join(_BASE, "..", "rendering"),
]
for _src in _SRC_CANDIDATES:
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

_VP_ROOT = os.path.join(_BASE, "..")
if _VP_ROOT not in sys.path:
    sys.path.insert(0, _VP_ROOT)

from nig_render_dataset_r2rce_detail import (
    ACTION_ID_TO_NAME, EpisodeSample,
    _build_segments_from_actions, _normalize_action_name,
    _split_long_forward_segments, _segments_to_events,
    _to_action_id, format_event_action, load_r2r_vlnce,
)
_VP_LOCAL = os.path.join(_VP_ROOT, "visual_prompt")
if os.path.isdir(_VP_LOCAL) and _VP_LOCAL not in sys.path:
    sys.path.insert(0, _VP_LOCAL)
from augmentation import AugParams, sample_augmentation
from config import VisualPromptConfig
from ribbon import determine_turn_direction

# ── Our modified overlay ──
from vp_overlay_with_mask import overlay_panorama_with_mask


def build_sim_with_depth(scene_path, width, height, forward_step,
                          turn_angle, hfov, camera_height):
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
        raise RuntimeError(f"NavMesh not loaded: {navmesh_path}")
    return sim


def capture_subviews(sim, panorama_step_deg, camera_height):
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
        rgb_list.append(rgb.copy())

        depth = np.asarray(obs["depth"], dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth_list.append(depth.copy())
        rot_list.append(rot)

    agent.set_state(habitat_sim.AgentState(position=state.position, rotation=base_rot))
    return rgb_list, depth_list, cam_pos, rot_list


def walk_trajectory(sim, sample, goal_radius, max_steps):
    agent = sim.get_agent(0)
    initial_state = habitat_sim.AgentState()
    initial_state.position = np.array(sample.start_position, dtype=np.float32)
    if sample.start_rotation:
        initial_state.rotation = quat_from_coeffs(sample.start_rotation)
    agent.set_state(initial_state)

    follower = GreedyGeodesicFollower(sim.pathfinder, agent, goal_radius=goal_radius)
    actions, step_states = [], []
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
        step_states.append({
            "position": np.array(st.position, dtype=np.float64),
            "rotation": st.rotation,
        })
    return actions, step_states


def _downsample(img, out_h=384):
    h, w = img.shape[:2]
    if h == out_h:
        return img
    return cv2.resize(img, (w, out_h), interpolation=cv2.INTER_AREA)


def _downsample_mask(mask, out_h=384):
    h, w = mask.shape[:2]
    if h == out_h:
        return mask
    resized = cv2.resize(mask, (w, out_h), interpolation=cv2.INTER_NEAREST)
    return np.where(resized > 0.5, 1.0, 0.0).astype(np.float32)


def render_masks_for_episode(
    sim, sample, goal_radius, max_steps,
    forward_step, turn_angle, panorama_step, hfov,
    split_forward_threshold_m, small_fwd_m, small_turn_deg, max_parts,
    vp_cfg, camera_height, rng, enable_vp,
):
    actions, step_states = walk_trajectory(sim, sample, goal_radius, max_steps)
    raw_segments = _build_segments_from_actions(actions)
    if not raw_segments:
        return None

    split_segments = _split_long_forward_segments(
        raw_segments, forward_step, split_forward_threshold_m)
    action_events, action_events_text, event_end_steps = _segments_to_events(
        split_segments, forward_step, turn_angle,
        max_parts=max_parts, small_fwd_m=small_fwd_m, small_turn_deg=small_turn_deg)
    if not event_end_steps:
        return None

    agent = sim.get_agent(0)
    initial_state = habitat_sim.AgentState()
    initial_state.position = np.array(sample.start_position, dtype=np.float32)
    if sample.start_rotation:
        initial_state.rotation = quat_from_coeffs(sample.start_rotation)

    masks = []
    n_frames = len(event_end_steps) + 1
    goal_pos = np.array(sample.goal_position, dtype=np.float64)

    def _render_mask_at(agent_state, event_idx, frame_idx):
        agent.set_state(agent_state)
        rgb_list, depth_list, cam_pos, rot_list = capture_subviews(
            sim, panorama_step, camera_height)

        h, w = rgb_list[0].shape[:2]
        if not enable_vp or event_idx >= len(action_events):
            return np.zeros((h, w * len(rgb_list), 3), dtype=np.float32)

        aug = sample_augmentation(vp_cfg, rng) if rng is not None else AugParams()
        if aug.dropout:
            return np.zeros((h, w * len(rgb_list), 3), dtype=np.float32)

        cur_pos = np.array(agent_state.position, dtype=np.float64)
        turn_dir = determine_turn_direction(event_idx, action_events)
        is_near_end = (frame_idx >= n_frames - 5) and (frame_idx < n_frames - 1)

        _, _, pano_mask = overlay_panorama_with_mask(
            rgb_list, depth_list, cam_pos, rot_list, hfov,
            event_idx, cur_pos, step_states, event_end_steps,
            action_events, turn_dir,
            goal_pos if is_near_end else None, is_near_end,
            vp_cfg, aug,
        )
        return pano_mask

    # Frame 0
    mask0 = _render_mask_at(initial_state, event_idx=0, frame_idx=0)
    masks.append(_downsample_mask(mask0))

    # Event boundary frames
    for i, end_step in enumerate(event_end_steps):
        if end_step >= len(step_states):
            continue
        boundary_state = habitat_sim.AgentState()
        boundary_state.position = step_states[end_step]["position"].astype(np.float32)
        boundary_state.rotation = step_states[end_step]["rotation"]
        next_event_idx = i + 1
        mask_i = _render_mask_at(boundary_state, event_idx=next_event_idx,
                                  frame_idx=next_event_idx)
        masks.append(_downsample_mask(mask_i))

    return masks


def save_masks(ep_dir, masks):
    """Save VP semantic masks as compact 3-channel binary PNGs.

    The mask is the direct input to the VPEncoder: a 3-channel binary
    semantic mask stored as an RGB PNG where R=C0 (ribbon), G=C1 (arrow),
    B=C2 (endpoint), each pixel 0 or 255.  PNG compresses binary masks to
    a few KB, so no bulky .npy is written.  Loaded back via PIL → /255 it
    yields exactly (H, W, 3) in {0, 1} with the same channel order.
    """
    mask_paths = []
    for idx, mask in enumerate(masks):
        m = np.clip(mask, 0.0, 1.0)
        if m.ndim == 2:
            m = np.stack([m, m, m], axis=-1)
        m = (m > 0.5).astype(np.uint8) * 255  # 二值化 0/255, C0=R,C1=G,C2=B
        fp = os.path.join(ep_dir, f"frame_{idx:04d}_vpmask.png")
        Image.fromarray(m, mode="RGB").save(fp)
        mask_paths.append(fp)
    return mask_paths


def main():
    parser = argparse.ArgumentParser(description="Generate VP semantic masks")
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--data_dir", required=True,
                        help="Existing rendered episode directory (for reference)")
    parser.add_argument("--output_dir", default="",
                        help="Where to save masks (defaults to data_dir)")
    parser.add_argument("--scenes_root", default="PATH/TO/data")
    parser.add_argument("--log_path", default="logs/render_vp_masks.log")
    parser.add_argument("--max_episodes", type=int, default=0)
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
    parser.add_argument("--max_parts", type=int, default=5)
    parser.add_argument("--small_fwd_m", type=float, default=0.5)
    parser.add_argument("--small_turn_deg", type=float, default=30.0)
    parser.add_argument("--split_forward_threshold_m", type=float, default=6.0)
    parser.add_argument("--sample_mode", choices=["stride", "event"], default="event")
    parser.add_argument("--vp_dropout_prob", type=float, default=0.15)
    parser.add_argument("--vp_alpha", type=float, default=0.55)
    parser.add_argument("--vp_mixed_mode_prob", type=float, default=0.10)
    parser.add_argument("--vp_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing masks instead of skipping")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
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

    samples = load_r2r_vlnce(args.train_json)
    if args.max_episodes:
        samples = samples[:args.max_episodes]

    by_scene: Dict[str, List[EpisodeSample]] = defaultdict(list)
    for s in samples:
        by_scene[s.scene_id].append(s)

    total_eps = sum(len(v) for v in by_scene.values())
    logging.info("Total episodes: %s", total_eps)

    progress = tqdm(total=total_eps, desc="Rendering VP masks", unit="ep")
    for scene_id, eps in by_scene.items():
        scene_path = os.path.join(args.scenes_root, scene_id)
        if not os.path.exists(scene_path):
            logging.warning("Skipping missing scene: %s", scene_path)
            progress.update(len(eps))
            continue

        hfov = args.panorama_hfov if args.panorama else args.hfov
        panorama_step = args.panorama_step if args.panorama else args.hfov

        sim = build_sim_with_depth(
            scene_path, args.width, args.height,
            args.forward_step, args.turn_angle, hfov, args.camera_height)

        for ep in eps:
            ep_dir = os.path.join(output_dir, f"episode_{ep.episode_id}")
            sample_path = os.path.join(ep_dir, "sample.json")

            if not args.overwrite and os.path.exists(os.path.join(ep_dir, "frame_0000_vpmask.png")):
                progress.update(1)
                continue

            if not os.path.exists(sample_path):
                logging.warning("No sample.json for episode %s", ep.episode_id)
                progress.update(1)
                continue

            masks = render_masks_for_episode(
                sim, ep, args.goal_radius, args.max_steps,
                args.forward_step, args.turn_angle, panorama_step, hfov,
                args.split_forward_threshold_m, args.small_fwd_m,
                args.small_turn_deg, args.max_parts,
                vp_cfg, args.camera_height, rng, True,
            )

            if masks is None or not masks:
                logging.warning("No masks for episode %s", ep.episode_id)
                progress.update(1)
                continue

            os.makedirs(ep_dir, exist_ok=True)
            mask_paths = save_masks(ep_dir, masks)

            # Update sample.json with vp_masks field
            with open(sample_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            record["vp_masks"] = mask_paths
            record["vp_mask_schema"] = "C0=ribbon,C1=arrow,C2=endpoint"
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            logging.info("Saved %d masks for episode %s", len(masks), ep.episode_id)
            progress.update(1)

        sim.close()

    progress.close()
    logging.info("Done")


if __name__ == "__main__":
    main()
