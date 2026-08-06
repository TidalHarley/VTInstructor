#!/usr/bin/env python3
"""
VP-Adapter aware evaluation script for NIG.

Loads a VP-Adapter checkpoint, attaches VP modules, and performs
inference with VP feature injection during generation.

Usage:
    python VP-adapter/nig_eval_vp.py \
        --model_dir /path/to/sft_checkpoint \
        --data_dir /path/to/eval_data \
        --eval_mode r2r_multiref
"""
import argparse
import gzip
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_VP_DIR = os.path.dirname(os.path.abspath(__file__))
if _VP_DIR not in sys.path:
    sys.path.insert(0, _VP_DIR)

_SRC_CANDIDATES = [
    os.path.join(_VP_DIR, "..", "common"),
]
for _src in _SRC_CANDIDATES:
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

from vp_config import VPAdapterConfig
from vp_model_wrapper import attach_vp_adapter, load_vp_modules, set_vp_features, clear_vp_features
from vp_dataset import _load_vp_mask

from nig_dataset_panorama_merged import (
    build_inference_prompt,
    get_system_prompt,
    group_actions,
    infer_dataset_type_from_path,
    safe_load_json,
)


def load_samples(root_dir: str) -> List[str]:
    files = []
    for name in os.listdir(root_dir):
        if not name.startswith("episode_"):
            continue
        p = os.path.join(root_dir, name, "sample.json")
        if os.path.exists(p):
            files.append(p)
    return sorted(files)


def shard_files_by_key(files: List[str], num_shards: int, shard_idx: int) -> List[str]:
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


def normalize_text_for_metrics(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def build_prompt_from_sample(sample: dict, max_frames: int):
    import os as _os
    if _os.environ.get("EVAL_FRAME_SOURCE", "overlay") == "clean" and sample.get("clean_frames"):
        sample = dict(sample); sample["frames"] = sample["clean_frames"]
    if "action_events_text" in sample:
        actions_list = sample["action_events_text"]
    else:
        group_size = int(sample.get("actions_per_frame") or sample.get("frame_stride") or 2)
        actions_text = sample.get("actions_text") or [str(a) for a in sample.get("actions", [])]
        actions_list = group_actions(actions_text, group_size)

    num_frames = min(len(sample["frames"]), len(actions_list) + 1, max_frames)
    frames = sample["frames"][:num_frames]
    actions_list = actions_list[: max(0, num_frames - 1)]
    episode_dir = os.path.dirname(sample.get("_path", ""))
    images = []
    for fp in frames:
        p = fp if os.path.exists(fp) else os.path.join(episode_dir, os.path.basename(fp))
        images.append(Image.open(p).convert("RGB"))

    vp_mask_paths = [] if _os.environ.get("EVAL_ZERO_VP", "0") == "1" else sample.get("vp_masks", [])
    vp_masks = []
    for i in range(num_frames):
        if i < len(vp_mask_paths) and vp_mask_paths[i]:
            m = _load_vp_mask(vp_mask_paths[i], episode_dir)
            if m is not None:
                vp_masks.append(m)
                continue
        img_arr = np.array(images[i])
        h, w = img_arr.shape[:2]
        vp_masks.append(np.zeros((h, w, 3), dtype=np.float32))

    return images, actions_list, sample.get("instruction", ""), vp_masks


def generate_with_vp(
    model,
    processor,
    vp_encoder,
    images: List[Image.Image],
    actions_list: List[str],
    vp_masks: List[np.ndarray],
    max_new_tokens: int = 150,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
) -> str:
    user_content = build_inference_prompt(images, actions_list, system_prompt=system_prompt)
    messages = [{"role": "user", "content": user_content}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )

    model_device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    grid_thw = inputs.get("image_grid_thw")
    if grid_thw is not None and vp_encoder is not None and vp_masks:
        vp_device = next(vp_encoder.parameters()).device
        mask_tensors = [
            torch.from_numpy(m).permute(2, 0, 1).float()
            for m in vp_masks
        ]
        with torch.no_grad():
            vp_features = vp_encoder(mask_tensors, grid_thw.to(vp_device))
        set_vp_features(model, vp_features)

    with torch.no_grad():
        if temperature > 0:
            gen_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature, top_p=0.9,
            )
        else:
            gen_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )

    clear_vp_features(model)

    gen_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()


def build_r2r_path_id_index(r2r_json_path: str) -> Dict[int, List[str]]:
    if not os.path.exists(r2r_json_path):
        return {}
    with open(r2r_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx = {}
    for item in data:
        path_id = item.get("path_id")
        insts = item.get("instructions")
        if path_id is not None and isinstance(insts, list):
            insts_clean = [normalize_text_for_metrics(x) for x in insts if normalize_text_for_metrics(x)]
            if insts_clean:
                idx[int(path_id)] = insts_clean
    return idx


def load_vlnce_episode_info(vlnce_json_gz: str) -> Dict[int, Dict]:
    if not os.path.exists(vlnce_json_gz):
        return {}
    with gzip.open(vlnce_json_gz, "rt", encoding="utf-8") as f:
        data = json.load(f)
    episode_info = {}
    for ep in data.get("episodes", []):
        ep_id = ep.get("episode_id")
        if ep_id is not None:
            episode_info[int(ep_id)] = {
                "trajectory_id": ep.get("trajectory_id"),
                "instruction": normalize_text_for_metrics(
                    ep.get("instruction", {}).get("instruction_text", "")
                ),
            }
    return episode_info


def eval_r2r_multiref(model, processor, vp_encoder, files, max_frames, max_new_tokens,
                       r2r_path_idx, vlnce_episode_info, temperature=0.0, system_prompt=None):
    traj_to_files: Dict[int, List[str]] = defaultdict(list)
    for fp in files:
        sample = safe_load_json(fp)
        if sample is None:
            continue
        traj_id = sample.get("trajectory_id")
        if traj_id is None:
            ep_id = sample.get("episode_id")
            if ep_id is not None and int(ep_id) in vlnce_episode_info:
                traj_id = vlnce_episode_info[int(ep_id)].get("trajectory_id")
        if traj_id is not None:
            traj_to_files[int(traj_id)].append(fp)

    print(f"[EVAL] {len(traj_to_files)} unique trajectories")
    records, preds, refs_all = [], [], []
    found, not_found = 0, 0

    for traj_id in tqdm(sorted(traj_to_files.keys()), desc="Evaluating", unit="traj"):
        fp = traj_to_files[traj_id][0]
        sample = safe_load_json(fp)
        if sample is None:
            continue
        sample["_path"] = fp

        images, actions_list, instruction, vp_masks = build_prompt_from_sample(sample, max_frames)
        pred = normalize_text_for_metrics(
            generate_with_vp(model, processor, vp_encoder, images, actions_list,
                             vp_masks, max_new_tokens, temperature, system_prompt)
        )

        refs = r2r_path_idx.get(traj_id)
        if refs and len(refs) >= 1:
            found += 1
        else:
            not_found += 1
            refs = [normalize_text_for_metrics(instruction)] if instruction else [""]

        preds.append(pred)
        refs_all.append(refs)
        records.append({
            "trajectory_id": traj_id,
            "episode_id": sample.get("episode_id"),
            "num_episodes_in_traj": len(traj_to_files[traj_id]),
            "prediction": pred,
            "references": refs,
            "num_refs": len(refs),
            "ref_source": "r2r_lookup" if traj_id in r2r_path_idx else "single_ref",
        })

    print(f"[EVAL] R2R lookup success: {found}, fallback: {not_found}")
    return records, preds, refs_all


def eval_rxr_multiref(model, processor, vp_encoder, files, max_frames, max_new_tokens,
                       temperature=0.0, system_prompt=None):
    traj_to_files: Dict[int, List[str]] = defaultdict(list)
    for fp in files:
        sample = safe_load_json(fp)
        if sample is None:
            continue
        traj_id = sample.get("trajectory_id")
        if traj_id is not None:
            traj_to_files[int(traj_id)].append(fp)

    print(f"[EVAL] {len(traj_to_files)} unique trajectories (rxr_multiref)")
    records, preds, refs_all = [], [], []

    for traj_id in tqdm(sorted(traj_to_files.keys()), desc="Evaluating", unit="traj"):
        traj_files = traj_to_files[traj_id]
        first_sample = safe_load_json(traj_files[0])
        if first_sample is None:
            continue
        first_sample["_path"] = traj_files[0]

        images, actions_list, instruction, vp_masks = build_prompt_from_sample(first_sample, max_frames)
        pred = normalize_text_for_metrics(
            generate_with_vp(model, processor, vp_encoder, images, actions_list,
                             vp_masks, max_new_tokens, temperature, system_prompt)
        )

        refs_raw = []
        for tf in traj_files:
            s = safe_load_json(tf)
            if s is None:
                continue
            ins = normalize_text_for_metrics(s.get("instruction", ""))
            if ins:
                refs_raw.append(ins)
        seen = set()
        refs = []
        for r in refs_raw:
            if r not in seen:
                seen.add(r)
                refs.append(r)
        if not refs:
            refs = [normalize_text_for_metrics(instruction)] if instruction else [""]

        preds.append(pred)
        refs_all.append(refs)
        records.append({
            "trajectory_id": traj_id,
            "episode_id": first_sample.get("episode_id"),
            "num_episodes_in_traj": len(traj_files),
            "prediction": pred,
            "references": refs,
            "num_refs": len(refs),
            "ref_source": "rxr_traj_multiref",
        })

    return records, preds, refs_all


def eval_single_ref(model, processor, vp_encoder, files, max_frames, max_new_tokens,
                     temperature=0.0, system_prompt=None):
    records, preds, refs_all = [], [], []
    for fp in tqdm(files, desc="Evaluating", unit="sample"):
        sample = safe_load_json(fp)
        if sample is None:
            continue
        sample["_path"] = fp

        images, actions_list, instruction, vp_masks = build_prompt_from_sample(sample, max_frames)
        pred = normalize_text_for_metrics(
            generate_with_vp(model, processor, vp_encoder, images, actions_list,
                             vp_masks, max_new_tokens, temperature, system_prompt)
        )

        refs = [normalize_text_for_metrics(instruction)] if instruction else [""]
        preds.append(pred)
        refs_all.append(refs)
        records.append({
            "episode_id": sample.get("episode_id"),
            "trajectory_id": sample.get("trajectory_id"),
            "prediction": pred,
            "reference": instruction,
        })

    return records, preds, refs_all


def main():
    parser = argparse.ArgumentParser(description="Evaluate NIG with VP-Adapter")

    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--processor_dir", default=(
        "PATH/TO/models_cache/models--Qwen--Qwen3-VL-8B-Instruct"
        "/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"))
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_json", default="")

    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dataset_type", choices=["auto", "r2rce", "rxrce"], default="auto")
    parser.add_argument("--eval_mode", choices=["r2r_multiref", "rxr_multiref", "single_ref"],
                        default="r2r_multiref")

    parser.add_argument("--r2r_val_json",
                        default="PATH/TO/R2R/R2R/data/R2R_val_unseen.json")
    parser.add_argument("--vlnce_val_json_gz",
                        default="PATH/TO/dataset/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz")

    # VP-Adapter config (must match training — v2.0 defaults)
    parser.add_argument("--vp_dim", type=int, default=384)
    parser.add_argument("--adapter_layers", default="7")
    parser.add_argument("--adapter_num_heads", type=int, default=4)

    args = parser.parse_args()

    if not args.out_json:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_json = f"PATH/TO/outputs/nig_eval_vp_{ts}.json"

    print("=" * 60)
    print("NIG Evaluation with VP-Adapter")
    print("=" * 60)
    print(f"Model dir:      {args.model_dir}")
    print(f"Processor dir:  {args.processor_dir}")
    print(f"Data dir:       {args.data_dir}")
    print(f"Output:         {args.out_json}")
    print(f"Eval mode:      {args.eval_mode}")
    print(f"Temperature:    {args.temperature}")
    print(f"Shard:          {args.shard_idx}/{args.num_shards}")
    print("=" * 60)

    dataset_type = args.dataset_type if args.dataset_type != "auto" else infer_dataset_type_from_path(args.data_dir)
    system_prompt = get_system_prompt(dataset_type)
    print(f"[INFO] Resolved dataset_type: {dataset_type}")

    adapter_layers = tuple(int(x) for x in args.adapter_layers.split(","))
    vp_cfg = VPAdapterConfig(
        vp_dim=args.vp_dim,
        adapter_layers=adapter_layers,
        adapter_num_heads=args.adapter_num_heads,
        freeze_vit_backbone=True,
    )

    print("[INFO] Loading processor ...")
    processor = AutoProcessor.from_pretrained(
        args.processor_dir, trust_remote_code=True, local_files_only=True)

    print(f"[INFO] Loading model from {args.model_dir} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True, local_files_only=True,
        device_map="auto",
    )

    print("[INFO] Attaching VP-Adapter ...")
    vp_encoder, _ = attach_vp_adapter(model, vp_cfg)
    load_vp_modules(vp_encoder, model, args.model_dir)
    model.eval()
    vp_encoder.eval()

    files = load_samples(args.data_dir)
    if not files:
        raise SystemExit(f"No samples found in {args.data_dir}")
    print(f"[INFO] Found {len(files)} samples")

    rng = random.Random(args.seed)
    rng.shuffle(files)
    if args.num_samples > 0:
        files = files[:args.num_samples]
    if args.num_shards > 1:
        files = shard_files_by_key(files, args.num_shards, args.shard_idx)
        print(f"[INFO] Sharded samples: {len(files)}")
        if not files:
            os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
            with open(args.out_json, "w") as f:
                json.dump([], f)
            return

    if args.eval_mode == "r2r_multiref":
        r2r_path_idx = build_r2r_path_id_index(args.r2r_val_json)
        vlnce_info = load_vlnce_episode_info(args.vlnce_val_json_gz)
        records, preds, refs = eval_r2r_multiref(
            model, processor, vp_encoder, files,
            args.max_frames, args.max_new_tokens,
            r2r_path_idx, vlnce_info,
            temperature=args.temperature, system_prompt=system_prompt,
        )
    elif args.eval_mode == "rxr_multiref":
        records, preds, refs = eval_rxr_multiref(
            model, processor, vp_encoder, files,
            args.max_frames, args.max_new_tokens,
            temperature=args.temperature, system_prompt=system_prompt,
        )
    else:
        records, preds, refs = eval_single_ref(
            model, processor, vp_encoder, files,
            args.max_frames, args.max_new_tokens,
            temperature=args.temperature, system_prompt=system_prompt,
        )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved {len(records)} predictions to {args.out_json}")
    pred_lens = [len(r["prediction"].split()) for r in records]
    avg_len = sum(pred_lens) / len(pred_lens) if pred_lens else 0
    print(f"[INFO] Average prediction length: {avg_len:.1f} words")


if __name__ == "__main__":
    main()
