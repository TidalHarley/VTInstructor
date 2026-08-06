#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
渲染单条轨迹，验证 vp_overlay_with_mask：
  - 输出每帧的 overlaid RGB（ribbon/arrow/endpoint 叠加在全景上）
  - 输出每帧的 3 通道 0/1 mask（.npy）及彩色可视化（ribbon=蓝, arrow=绿, endpoint=红）
  - 输出一张 RGB+mask 上下拼接的总览图

用法：
  python test_render_one_traj_mask.py \
      --sample_json .../episode_1/sample.json \
      --scenes_root PATH/TO/scenes_root \
      --out_dir .../outputs/vp_mask_demo
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

_BASE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_BASE)               # VT-Instructor
sys.path.insert(0, _BASE)                    # visual_prompt
sys.path.insert(0, os.path.join(_PROJ, "rendering"))
sys.path.insert(0, os.path.join(_PROJ, "vp_adapter"))

import json

from render_vp_masks import build_sim_with_depth, capture_subviews, walk_trajectory
from nig_render_dataset_r2rce_detail import (
    EpisodeSample, _build_segments_from_actions,
    _split_long_forward_segments, _segments_to_events,
)
from ribbon import determine_turn_direction
from config import VisualPromptConfig
from augmentation import AugParams
from vp_overlay_with_mask import overlay_panorama_with_mask


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """(H,W,3) 0/1 -> RGB 可视化: ribbon=蓝, arrow=绿, endpoint=红。"""
    h, w = mask.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:, :, 2] = (mask[:, :, 0] * 255).astype(np.uint8)  # ribbon  -> Blue
    vis[:, :, 1] = (mask[:, :, 1] * 255).astype(np.uint8)  # arrow   -> Green
    vis[:, :, 0] = (mask[:, :, 2] * 255).astype(np.uint8)  # endpoint-> Red
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_json", required=True)
    ap.add_argument("--scenes_root", default="PATH/TO/scenes_root")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--camera_height", type=float, default=0.75)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--panorama_step", type=float, default=90.0)
    ap.add_argument("--forward_step", type=float, default=0.25)
    ap.add_argument("--turn_angle", type=float, default=30.0)
    ap.add_argument("--goal_radius", type=float, default=0.5)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--split_forward_threshold_m", type=float, default=6.0)
    ap.add_argument("--small_fwd_m", type=float, default=0.5)
    ap.add_argument("--small_turn_deg", type=float, default=30.0)
    ap.add_argument("--max_parts", type=int, default=5)
    args = ap.parse_args()

    with open(args.sample_json, "r", encoding="utf-8") as f:
        rec = json.load(f)

    sample = EpisodeSample(
        episode_id=rec["episode_id"],
        trajectory_id=rec.get("trajectory_id", -1),
        scene_id=rec["scene_id"],
        instruction=rec.get("instruction", ""),
        start_position=rec["start_position"],
        start_rotation=rec["start_rotation"],
        goal_position=rec["goal_position"],
    )
    print(f"[INFO] episode {sample.episode_id} | scene {sample.scene_id}")
    print(f"[INFO] instruction: {sample.instruction}")

    scene_path = os.path.join(args.scenes_root, sample.scene_id)
    if not os.path.exists(scene_path):
        print(f"[FATAL] scene not found: {scene_path}")
        sys.exit(1)

    sim = build_sim_with_depth(
        scene_path, args.width, args.height,
        args.forward_step, args.turn_angle, args.hfov, args.camera_height)

    import habitat_sim
    from habitat_sim.utils.common import quat_from_coeffs

    # 走轨迹 + 还原 event
    actions, step_states = walk_trajectory(sim, sample, args.goal_radius, args.max_steps)
    raw_segments = _build_segments_from_actions(actions)
    split_segments = _split_long_forward_segments(
        raw_segments, args.forward_step, args.split_forward_threshold_m)
    action_events, action_events_text, event_end_steps = _segments_to_events(
        split_segments, args.forward_step, args.turn_angle,
        max_parts=args.max_parts, small_fwd_m=args.small_fwd_m,
        small_turn_deg=args.small_turn_deg)
    print(f"[INFO] steps={len(step_states)} events={len(event_end_steps)}")
    for i, t in enumerate(action_events_text):
        print(f"    event {i}: {t}")

    vp_cfg = VisualPromptConfig()
    aug = AugParams()  # 不做随机增强 / dropout
    goal_pos = np.array(sample.goal_position, dtype=np.float64)
    n_frames = len(event_end_steps) + 1

    ep_out = os.path.join(args.out_dir, f"episode_{sample.episode_id}")
    os.makedirs(ep_out, exist_ok=True)

    def render_at(agent_state, event_idx, frame_idx):
        sim.get_agent(0).set_state(agent_state)
        rgb_list, depth_list, cam_pos, rot_list = capture_subviews(
            sim, args.panorama_step, args.camera_height)
        cur_pos = np.array(agent_state.position, dtype=np.float64)
        turn_dir = determine_turn_direction(event_idx, action_events) \
            if event_idx < len(action_events) else "straight"
        is_near_end = (frame_idx >= n_frames - 5) and (frame_idx < n_frames - 1)
        pano_rgb, modes, pano_mask = overlay_panorama_with_mask(
            rgb_list, depth_list, cam_pos, rot_list, args.hfov,
            event_idx, cur_pos, step_states, event_end_steps,
            action_events, turn_dir,
            goal_pos if is_near_end else None, is_near_end,
            vp_cfg, aug,
        )
        return pano_rgb, pano_mask, modes

    # frame 0
    init_state = habitat_sim.AgentState()
    init_state.position = np.array(sample.start_position, dtype=np.float32)
    if sample.start_rotation:
        init_state.rotation = quat_from_coeffs(sample.start_rotation)

    frame_specs = [(init_state, 0, 0)]
    for i, end_step in enumerate(event_end_steps):
        if end_step >= len(step_states):
            continue
        st = habitat_sim.AgentState()
        st.position = step_states[end_step]["position"].astype(np.float32)
        st.rotation = step_states[end_step]["rotation"]
        frame_specs.append((st, i + 1, i + 1))

    overview_rows = []
    for state, ev_idx, fr_idx in frame_specs:
        pano_rgb, pano_mask, modes = render_at(state, ev_idx, fr_idx)
        mask_vis = colorize_mask(pano_mask)

        Image.fromarray(pano_rgb).save(
            os.path.join(ep_out, f"frame_{fr_idx:04d}_overlay.jpg"), quality=92)
        # 规范二值 mask（直接给 VTMod）：R=ribbon, G=arrow, B=endpoint，0/255
        bin_mask = ((pano_mask > 0.5).astype(np.uint8) * 255)
        Image.fromarray(bin_mask, mode="RGB").save(
            os.path.join(ep_out, f"frame_{fr_idx:04d}_vpmask.png"))

        nonzero = [int((pano_mask[:, :, c] > 0).sum()) for c in range(3)]
        print(f"  frame {fr_idx} ev={ev_idx} mode={modes[1]:>18} "
              f"mask px [ribbon,arrow,end]={nonzero}")

        # 总览：每帧 = RGB 叠加 | mask 可视化（左右拼）
        row = np.concatenate([pano_rgb, mask_vis], axis=1)
        overview_rows.append(row)

    if overview_rows:
        max_w = max(r.shape[1] for r in overview_rows)
        padded = [np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0))) for r in overview_rows]
        overview = np.concatenate(padded, axis=0)
        ov_path = os.path.join(ep_out, "overview.png")
        Image.fromarray(overview).save(ov_path)
        print(f"[OK] overview saved: {ov_path}")

    sim.close()
    print(f"[DONE] outputs in {ep_out}")


if __name__ == "__main__":
    main()
