"""
VP-Adapter aware dataset and collator.

Extends the original PanoramaFilteredDataset to also load VP semantic masks.
The VPCollator produces batches that include both standard Qwen3-VL inputs
(pixel_values, image_grid_thw, input_ids, labels) and VP mask tensors.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset

import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.dataset import (
    PanoramaFilteredDataset, PanoramaCollator,
    get_system_prompt, load_sample_files, safe_load_json,
    group_actions, is_panorama_format, load_filtered_episodes,
    R2RCE_SYSTEM_PROMPT, RXRCE_SYSTEM_PROMPT,
)


def load_samples(root_dir: str) -> List[str]:
    """Sorted `episode_*/sample.json` paths. Alias used by eval and generate."""
    return load_sample_files(root_dir)


def shard_files_by_key(files: List[str], num_shards: int, shard_idx: int) -> List[str]:
    """Partition sample files by trajectory_id (fallback: episode_id / path hash)."""
    if num_shards <= 1:
        return files
    out: List[str] = []
    for fp in files:
        sample = safe_load_json(fp)
        if sample is None:
            continue
        if sample.get("trajectory_id") is not None:
            key = int(sample["trajectory_id"])
        elif sample.get("episode_id") is not None:
            key = int(sample["episode_id"])
        else:
            key = abs(hash(os.path.basename(fp)))
        if (key % num_shards) == shard_idx:
            out.append(fp)
    return out


_ZERO_MASK_NOTIFIED: set = set()


def notify_missing_vp_mask(context: str, episode_dir: str,
                           level: str = "WARN", hint: str = "") -> None:
    """Report (once per context) that a frame fell back to an all-zero VP mask.

    Running without masks is never fatal — the model simply generates without
    the visual trajectory prompt. It is a mistake during training, where it
    means Stage 2 was skipped, but a legitimate mode at inference time, so the
    caller picks the severity.
    """
    if context in _ZERO_MASK_NOTIFIED:
        return
    _ZERO_MASK_NOTIFIED.add(context)
    message = (f"[VP][{level}] {context}: no vp_masks for these frames, using "
               f"all-zero masks. First occurrence: {episode_dir}.")
    if hint:
        message += f" {hint}"
    print(message, flush=True)


def _load_vp_mask(mask_path: str, episode_dir: str) -> Optional[np.ndarray]:
    """Load a VP semantic mask, return (H, W, 3) float32 in [0, 1].

    Prefers .npy (raw 3-channel semantic) over .png (blue visualization).
    """
    if not mask_path:
        return None

    npy_path = mask_path.replace("_vpmask.png", "_vpmask.npy")
    for candidate in [npy_path, os.path.join(episode_dir, os.path.basename(npy_path))]:
        if os.path.exists(candidate):
            arr = np.load(candidate).astype(np.float32)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            return arr

    p = mask_path if os.path.exists(mask_path) else os.path.join(
        episode_dir, os.path.basename(mask_path))
    if not os.path.exists(p):
        return None
    img = Image.open(p)
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


class VPDataset(Dataset):
    """
    Dataset that loads trajectory frames + VP semantic masks.

    Falls back gracefully: if a VP mask is missing for a frame, an all-zero
    mask is created (equivalent to VP dropout).
    """

    def __init__(
        self,
        root_dir: str,
        filtered_json: str = "",
        max_frames: int = 32,
        max_samples: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        dataset_type: str = "r2rce",
    ):
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.dataset_type = dataset_type
        self.system_prompt = get_system_prompt(dataset_type)
        self.use_filter = bool(filtered_json)

        self.filtered_episodes: Dict[int, str] = {}
        if self.use_filter:
            self.filtered_episodes = load_filtered_episodes(filtered_json)

        all_files = load_sample_files(root_dir)
        valid = []
        for p in all_files:
            s = safe_load_json(p)
            if s is None or not is_panorama_format(s):
                continue
            if self.use_filter:
                ep_id = s.get("episode_id")
                if ep_id is None or int(ep_id) not in self.filtered_episodes:
                    continue
            valid.append(p)

        self.sample_files = valid
        if shuffle:
            random.Random(seed).shuffle(self.sample_files)
        if max_samples > 0:
            self.sample_files = self.sample_files[:max_samples]

        mask_count = 0
        for p in self.sample_files[:min(100, len(self.sample_files))]:
            s = safe_load_json(p)
            if s and s.get("vp_masks"):
                mask_count += 1
        probed = min(100, len(self.sample_files))
        print(f"[VPDataset:{dataset_type}] {len(self.sample_files)} samples "
              f"(mask coverage in first 100: {mask_count}/{probed})")
        if probed > 0 and mask_count == 0:
            print(
                f"[VPDataset:{dataset_type}][WARN] none of the first {probed} episodes "
                f"declares vp_masks in sample.json. Training would run with the "
                f"VP-Adapter fully disabled. Did you run Stage 2 "
                f"(visual_prompt/render_masks.py) on {root_dir}?",
                flush=True,
            )

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample_path = self.sample_files[idx]
        sample = safe_load_json(sample_path)
        if sample is None:
            raise ValueError(f"Failed to load: {sample_path}")

        if "action_events_text" in sample:
            grouped_actions = sample["action_events_text"]
        else:
            actions_text = sample.get("actions_text") or [
                str(a) for a in sample.get("actions", [])]
            group_size = int(sample.get("actions_per_frame")
                             or sample.get("frame_stride") or 2)
            grouped_actions = group_actions(actions_text, group_size)

        num_frames = min(len(sample["frames"]),
                         len(grouped_actions) + 1, self.max_frames)
        frames = sample["frames"][:num_frames]
        actions_list = grouped_actions[:max(0, num_frames - 1)]

        episode_dir = os.path.dirname(sample_path)

        # Load images (robust to truncated/corrupt files)
        images = []
        _bad = False
        for fp in frames:
            p = fp if os.path.exists(fp) else os.path.join(
                episode_dir, os.path.basename(fp))
            try:
                _img = Image.open(p)
                _img.load()
                images.append(_img.convert("RGB"))
            except Exception:
                _bad = True
                break
        if _bad:
            return self.__getitem__((idx + 1) % len(self.sample_files))

        vp_mask_paths = sample.get("vp_masks", [])
        vp_masks = []
        for i in range(num_frames):
            if i < len(vp_mask_paths) and vp_mask_paths[i]:
                m = _load_vp_mask(vp_mask_paths[i], episode_dir)
                if m is not None:
                    vp_masks.append(m)
                    continue
            # Fallback: all-zero mask matching frame size
            notify_missing_vp_mask(
                f"VPDataset:{self.dataset_type}", episode_dir, level="WARN",
                hint="Training this way leaves the VP-Adapter with no signal — "
                     "run visual_prompt/render_masks.py on this split.")
            img_arr = np.array(images[i])
            h, w = img_arr.shape[:2]
            vp_masks.append(np.zeros((h, w, 3), dtype=np.float32))

        ep_id = int(sample["episode_id"])
        if self.use_filter:
            instruction = self.filtered_episodes.get(
                ep_id, sample.get("instruction", ""))
        else:
            instruction = sample.get("instruction", "")

        return {
            "images": images,
            "vp_masks": vp_masks,
            "actions_list": actions_list,
            "system_prompt": self.system_prompt,
            "instruction": instruction,
            "episode_id": sample.get("episode_id"),
            "trajectory_id": sample.get("trajectory_id"),
            "dataset_type": self.dataset_type,
        }


class VPCollator:
    """
    Collator that extends PanoramaCollator with VP mask handling.

    Produces a batch dict with standard Qwen3-VL keys plus:
      - "vp_masks": list of tensors, each (3, H_i, W_i) float32 in [0, 1]
                    One tensor per image across all items in the batch.
    """

    def __init__(self, processor, max_length: Optional[int] = None):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: list) -> dict:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        all_vp_masks: List[torch.Tensor] = []

        for item in batch:
            images = item["images"]
            actions_list = item["actions_list"]
            system_prompt = item["system_prompt"]
            instruction = item["instruction"]
            vp_masks = item["vp_masks"]

            user_content = [{"type": "text", "text": system_prompt}]
            assert len(images) == len(actions_list) + 1
            user_content.append({"type": "image", "image": images[0]})
            for i, action in enumerate(actions_list):
                user_content.append({"type": "text", "text": f"[Action {i+1}: {action}]"})
                user_content.append({"type": "image", "image": images[i + 1]})

            msg_user = [{"role": "user", "content": user_content}]
            msg_full = msg_user + [
                {"role": "assistant",
                 "content": [{"type": "text", "text": instruction}]}
            ]

            full = self.processor.apply_chat_template(
                msg_full, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors="pt",
            )
            prompt_only = self.processor.apply_chat_template(
                msg_user, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )

            ids = full["input_ids"][0]
            mask = full["attention_mask"][0]
            lbl = ids.clone()
            prompt_len = prompt_only["input_ids"].shape[1]

            if self.max_length and ids.size(0) > self.max_length:
                if prompt_len >= self.max_length:
                    raise ValueError(f"Prompt too long: {prompt_len} >= {self.max_length}")
                ids = ids[:self.max_length]
                mask = mask[:self.max_length]
                lbl = lbl[:self.max_length]

            lbl[:prompt_len] = -100

            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            labels_list.append(lbl)

            if "pixel_values" in full:
                pv = full["pixel_values"]
                if pv.dim() == 3:
                    pv = pv[0]
                pixel_values_list.append(pv)
            if "image_grid_thw" in full:
                grid = full["image_grid_thw"]
                if grid.dim() == 3:
                    grid = grid[0]
                image_grid_thw_list.append(grid)

            for m in vp_masks:
                t = torch.from_numpy(m).permute(2, 0, 1).float()
                all_vp_masks.append(t)

        max_len = max(x.size(0) for x in input_ids_list)
        pad_id = self.processor.tokenizer.pad_token_id

        def pad_seq(seq, pad_value):
            if seq.size(0) < max_len:
                pad = torch.full((max_len - seq.size(0),), pad_value,
                                 dtype=seq.dtype)
                return torch.cat([seq, pad], dim=0)
            return seq

        batch_out = {
            "input_ids": torch.stack([pad_seq(x, pad_id) for x in input_ids_list]),
            "attention_mask": torch.stack([pad_seq(x, 0) for x in attention_mask_list]),
            "labels": torch.stack([pad_seq(x, -100) for x in labels_list]),
        }

        if pixel_values_list:
            batch_out["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            batch_out["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

        # VP masks are variable-size so kept as a list (not stacked)
        batch_out["vp_masks"] = all_vp_masks

        return batch_out
