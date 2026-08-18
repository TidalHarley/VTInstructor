#!/usr/bin/env python3
"""Shared prompts, action grouping, and panorama dataset for VTInstructor.

Imported by SFT, VP-GRPO, evaluation, and instruction generation. Keep the
system prompts here so training and inference never drift apart.
"""
import json
import os
import random
import torch
from typing import List, Optional
from PIL import Image
from torch.utils.data import Dataset


# 数据集专用 System Prompt（训练和推理必须一致）
R2RCE_SYSTEM_PROMPT = '''You are an R2R-style indoor navigation instruction writer.

Task:
- Given a first-person trajectory with action snippets and frame observations, write one concise instruction that matches the full path.
- The images already contain grounded navigation cues rendered on the scene, such as floor-following ribbons, turn indicators, and goal markers.
- Use these visual cues together with landmarks and action snippets to recover the path faithfully, but do not mention the cues themselves in the final instruction.

Strict constraints:
1) Use 25-40 words.
2) Mention 2-3 concrete landmarks.
3) Explicitly say "go up the stairs" or "go down the stairs" when stairs are involved.
4) End with a precise stop location next to a visible object.
5) Avoid loops, filler words, and repeated phrases.
6) Do not mention ribbons, arrows, colored lines, overlays, prompts, or markers.

Output one final instruction only.'''

RXRCE_SYSTEM_PROMPT = '''You are an RXR-style multilingual route narrator in English.

Goal:
- Describe the shown route as a step-by-step path grounded in scene details and action sequence.
- The images already contain grounded navigation cues rendered on the scene, such as floor-following ribbons, turn indicators, and goal markers.
- Use these visual cues to better infer motion and turning structure, but never mention the cues themselves in the final instruction.

Requirements:
1) Prefer richer spatial detail than R2R (about 35-60 words).
2) Use explicit orientation and transition cues (e.g., slight right, pass through, continue along).
3) Mention distinctive visual anchors, not generic placeholders.
4) Keep temporal order faithful to the trajectory.
5) End with a concrete waiting point.
6) No repetitive template sentences.
7) Do not mention ribbons, arrows, colored lines, overlays, prompts, or markers.

Return exactly one route instruction.'''

PANORAMA_SYSTEM_PROMPT = R2RCE_SYSTEM_PROMPT


def get_system_prompt(dataset_type: str = "r2rce") -> str:
    t = (dataset_type or "r2rce").lower()
    if t == "rxrce":
        return RXRCE_SYSTEM_PROMPT
    return R2RCE_SYSTEM_PROMPT


def infer_dataset_type_from_path(path: str) -> str:
    p = (path or "").lower()
    if "rxr" in p:
        return "rxrce"
    return "r2rce"


def load_sample_files(root_dir: str) -> List[str]:
    episodes = []
    for name in os.listdir(root_dir):
        if not name.startswith("episode_"):
            continue
        sample_path = os.path.join(root_dir, name, "sample.json")
        if os.path.exists(sample_path):
            episodes.append(sample_path)
    return sorted(episodes)


def safe_load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        return json.loads(line)
        except Exception:
            return None
    except Exception:
        return None


def group_actions(actions_text: List[str], group_size: int) -> List[str]:
    if group_size <= 0:
        group_size = 1
    grouped = []
    for i in range(0, len(actions_text), group_size):
        chunk = actions_text[i : i + group_size]
        grouped.append(" / ".join(chunk))
    return grouped


def is_panorama_format(sample: dict) -> bool:
    frames = sample.get("frames", [])
    if not frames:
        return False
    return isinstance(frames[0], str)


def load_filtered_episodes(filtered_json: str) -> dict:
    """
    加载过滤结果

    Returns:
        {episode_id (int): instruction (str)}
    """
    with open(filtered_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 支持两种格式：
    #   格式 1（apply_threshold.py 输出）: {"metadata": {...}, "episodes": {"1": {"instruction":...}, ...}}
    #   格式 2（简化）: {"1": {"instruction":...}, ...}
    if "episodes" in data and isinstance(data["episodes"], dict):
        episodes_dict = data["episodes"]
    else:
        episodes_dict = data

    result = {}
    for ep_id_str, entry in episodes_dict.items():
        if ep_id_str == "metadata":
            continue
        ep_id = int(ep_id_str)
        if isinstance(entry, dict):
            inst = entry.get("instruction", "")
        elif isinstance(entry, str):
            inst = entry
        else:
            continue
        if inst.strip():
            result[ep_id] = inst.strip()

    return result


class PanoramaFilteredDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        filtered_json: str = "",
        max_frames: int = 30,
        max_samples: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        validate_json: bool = True,
        dataset_type: str = "r2rce",
    ):
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.dataset_type = dataset_type
        self.system_prompt = get_system_prompt(dataset_type)
        self.use_filter = bool(filtered_json)

        self.filtered_episodes = {}
        if self.use_filter:
            self.filtered_episodes = load_filtered_episodes(filtered_json)
            print(f"[FilteredDataset:{self.dataset_type}] Loaded {len(self.filtered_episodes)} filtered episodes")
        else:
            print(f"[FilteredDataset:{self.dataset_type}] No filtered_json, using original instructions")

        all_files = load_sample_files(root_dir)

        valid_samples = []
        invalid_count = 0
        non_panorama_count = 0
        filtered_out_count = 0

        for p in all_files:
            if validate_json:
                s = safe_load_json(p)
                if s is None:
                    invalid_count += 1
                    continue
                if not is_panorama_format(s):
                    non_panorama_count += 1
                    continue

                if self.use_filter:
                    ep_id = s.get("episode_id")
                    if ep_id is None or int(ep_id) not in self.filtered_episodes:
                        filtered_out_count += 1
                        continue

            valid_samples.append(p)

        self.sample_files = valid_samples

        if invalid_count > 0:
            print(f"[FilteredDataset:{self.dataset_type}] Skipped {invalid_count} invalid JSON files")
        if non_panorama_count > 0:
            print(f"[FilteredDataset:{self.dataset_type}] Skipped {non_panorama_count} non-panorama samples")
        if self.use_filter and filtered_out_count > 0:
            print(f"[FilteredDataset:{self.dataset_type}] Filtered out {filtered_out_count} low-quality samples")

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.sample_files)

        if max_samples > 0:
            self.sample_files = self.sample_files[:max_samples]

        print(f"[FilteredDataset:{self.dataset_type}] Final dataset: {len(self.sample_files)} samples")

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
            actions_text = sample.get("actions_text") or [str(a) for a in sample.get("actions", [])]
            group_size = int(sample.get("actions_per_frame") or sample.get("frame_stride") or 2)
            grouped_actions = group_actions(actions_text, group_size)

        num_frames = min(len(sample["frames"]), len(grouped_actions) + 1, self.max_frames)
        frames = sample["frames"][:num_frames]
        actions_list = grouped_actions[: max(0, num_frames - 1)]

        images = [Image.open(p).convert("RGB") for p in frames]

        # r2rce 使用 filtered 指令；rxrce 等未过滤时使用原始指令
        ep_id = int(sample["episode_id"])
        if self.use_filter:
            instruction = self.filtered_episodes.get(ep_id, sample.get("instruction", ""))
        else:
            instruction = sample.get("instruction", "")

        return {
            "images": images,
            "actions_list": actions_list,
            "system_prompt": self.system_prompt,
            "instruction": instruction,
            "episode_id": sample.get("episode_id"),
            "trajectory_id": sample.get("trajectory_id"),
            "dataset_type": self.dataset_type,
        }


class PanoramaCollator:
    """
    Panorama 数据整理器（与原版完全相同）
    """

    def __init__(self, processor, max_length: Optional[int] = None):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        image_grid_thw_list = []

        for item in batch:
            images = item["images"]
            actions_list = item["actions_list"]
            system_prompt = item["system_prompt"]
            instruction = item["instruction"]

            user_content = [{"type": "text", "text": system_prompt}]

            assert len(images) == len(actions_list) + 1, (
                f"Misaligned: len(images)={len(images)} != len(actions)+1={len(actions_list)+1}"
            )

            user_content.append({"type": "image", "image": images[0]})

            for i, action in enumerate(actions_list):
                user_content.append({"type": "text", "text": f"[Action {i+1}: {action}]"})
                user_content.append({"type": "image", "image": images[i + 1]})

            msg_user = [{"role": "user", "content": user_content}]
            msg_full = msg_user + [{"role": "assistant", "content": [{"type": "text", "text": instruction}]}]

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
                ids = ids[: self.max_length]
                mask = mask[: self.max_length]
                lbl = lbl[: self.max_length]

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

        max_len = max(x.size(0) for x in input_ids_list)
        pad_id = self.processor.tokenizer.pad_token_id

        def pad_seq(seq, pad_value):
            if seq.size(0) < max_len:
                pad = torch.full((max_len - seq.size(0),), pad_value, dtype=seq.dtype)
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

        return batch_out


def build_inference_prompt(
    images: List[Image.Image],
    actions_list: List[str],
    system_prompt: Optional[str] = None,
) -> List[dict]:
    """构建推理用的 user_content（与训练时格式完全一致）"""
    user_content = [{"type": "text", "text": system_prompt or PANORAMA_SYSTEM_PROMPT}]

    if len(images) != len(actions_list) + 1:
        n = min(len(images) - 1, len(actions_list))
        images = images[:n + 1]
        actions_list = actions_list[:n]

    user_content.append({"type": "image", "image": images[0]})

    for i, action in enumerate(actions_list):
        user_content.append({"type": "text", "text": f"[Action {i+1}: {action}]"})
        user_content.append({"type": "image", "image": images[i + 1]})

    return user_content
