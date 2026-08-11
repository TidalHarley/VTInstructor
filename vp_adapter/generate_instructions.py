#!/usr/bin/env python3
"""
Standalone instruction generation for data augmentation.

This is *not* a benchmark script: it computes no metrics, needs no reference
instructions, and never touches R2R_val_unseen.json. It takes rendered
trajectories, runs a VTInstructor checkpoint over them, and writes the
generated navigation instructions out.

The intended use is augmenting VLN training data: point it at any rendered
split (including `train`), collect the generated instructions, and mix them
into a follower's training set.

VTP rendering is optional. The model generates from RGB frames alone, since
the VTP context is already learned into the weights during training; supplying
Stage-2 vp_masks additionally drives the VP-Adapter and yields more accurate
instructions. `--vp_mode auto` (the default) picks whichever the split
supports, so the same command works either way.

`--dataset_type` selects the output style: `r2rce` for R2R-style instructions,
`rxrce` for the longer, more spatially detailed RxR-style narration.

Examples
--------
# R2R-style instructions from RGB renders only
python vp_adapter/generate_instructions.py \
    --model_dir  /path/to/checkpoint \
    --processor_dir /path/to/Qwen3-VL-8B-Instruct \
    --data_dir   /path/to/rendered_split \
    --dataset_type r2rce \
    --out_json   outputs/augment/r2r_style.json

# RxR-style narration, and also emit a drop-in VLN-CE split
python vp_adapter/generate_instructions.py \
    --model_dir  /path/to/checkpoint \
    --data_dir   /path/to/rendered_split \
    --dataset_type rxrce \
    --out_json   outputs/augment/rxr_style.json \
    --vlnce_template /path/to/RxR_VLNCE_v0/train/train_guide.json.gz \
    --out_vlnce_gz   outputs/augment/train_generated.json.gz
"""
import argparse
import copy
import datetime
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
for _src in [os.path.join(_VP_DIR, "..", "common")]:
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

from vp_config import VPAdapterConfig
from vp_model_wrapper import attach_vp_adapter, load_vp_modules
from vp_dataset import _load_vp_mask
from nig_eval_vp import generate_with_vp, load_samples, shard_files_by_key
from nig_dataset_panorama_merged import (
    get_system_prompt,
    group_actions,
    infer_dataset_type_from_path,
    safe_load_json,
)

VP_ENCODER_FILENAME = "vp_encoder.pt"
VP_ADAPTERS_FILENAME = "vp_adapters.pt"


def clean_text(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def resolve_frames(sample: dict, frame_source: str) -> List[str]:
    """Pick which rendered frames to feed the model.

    `frames` is rewritten in place by Stage 2 to point at the overlays, so on a
    Stage-2 split `auto` means overlays and on a Stage-1 split it means the
    plain RGB keyframes.
    """
    if frame_source == "clean":
        return sample.get("clean_frames") or sample.get("frames", [])
    if frame_source == "overlay":
        return sample.get("overlay_frames") or sample.get("frames", [])
    return sample.get("frames", [])


def build_inputs(
    sample: dict,
    sample_path: str,
    max_frames: int,
    frame_source: str,
    use_vp: bool,
) -> Tuple[List[Image.Image], List[str], List[np.ndarray]]:
    if "action_events_text" in sample:
        actions_list = sample["action_events_text"]
    else:
        group_size = int(sample.get("actions_per_frame") or sample.get("frame_stride") or 2)
        actions_text = sample.get("actions_text") or [str(a) for a in sample.get("actions", [])]
        actions_list = group_actions(actions_text, group_size)

    frame_paths = resolve_frames(sample, frame_source)
    num_frames = min(len(frame_paths), len(actions_list) + 1, max_frames)
    actions_list = actions_list[: max(0, num_frames - 1)]

    episode_dir = os.path.dirname(sample_path)
    images = []
    for fp in frame_paths[:num_frames]:
        p = fp if os.path.exists(fp) else os.path.join(episode_dir, os.path.basename(fp))
        images.append(Image.open(p).convert("RGB"))

    # An empty mask list makes generate_with_vp skip VP injection entirely, so
    # the vision tower runs unmodified rather than being fed a blank prompt.
    if not use_vp:
        return images, actions_list, []

    mask_paths = sample.get("vp_masks") or []
    vp_masks: List[np.ndarray] = []
    for i in range(num_frames):
        mask = None
        if i < len(mask_paths) and mask_paths[i]:
            mask = _load_vp_mask(mask_paths[i], episode_dir)
        if mask is None:
            h, w = np.array(images[i]).shape[:2]
            mask = np.zeros((h, w, 3), dtype=np.float32)
        vp_masks.append(mask)
    return images, actions_list, vp_masks


def probe_vp_masks(files: List[str], limit: int = 100) -> Tuple[int, int]:
    probed = min(limit, len(files))
    with_masks = sum(
        1 for p in files[:probed] if (safe_load_json(p) or {}).get("vp_masks")
    )
    return with_masks, probed


def group_units(files: List[str], granularity: str) -> List[Tuple[Optional[int], List[str]]]:
    """Return the list of generation units: one per trajectory, or per episode.

    Episodes that share a trajectory_id are different human annotations of the
    same path, so generating once per trajectory avoids paying for identical
    renders several times.
    """
    if granularity == "episode":
        return [(None, [fp]) for fp in files]

    by_traj: Dict[int, List[str]] = defaultdict(list)
    loose: List[Tuple[Optional[int], List[str]]] = []
    for fp in files:
        sample = safe_load_json(fp)
        if sample is None:
            continue
        traj_id = sample.get("trajectory_id")
        if traj_id is None:
            loose.append((None, [fp]))
        else:
            by_traj[int(traj_id)].append(fp)
    units: List[Tuple[Optional[int], List[str]]] = [
        (traj_id, sorted(by_traj[traj_id])) for traj_id in sorted(by_traj)
    ]
    units.extend(loose)
    return units


def write_vlnce_split(
    template_gz: str,
    out_gz: str,
    instruction_by_episode: Dict[int, str],
) -> None:
    """Emit a drop-in VLN-CE split with instructions replaced by generated ones.

    Episode geometry is copied verbatim from the template; only the instruction
    text changes, so the result can be loaded by any VLN-CE dataloader.
    """
    with gzip.open(template_gz, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    kept, dropped_tokens = [], 0
    for ep in episodes:
        ep_id = ep.get("episode_id")
        if ep_id is None:
            continue
        generated = instruction_by_episode.get(int(ep_id))
        if not generated:
            continue
        new_ep = copy.deepcopy(ep)
        instruction = dict(new_ep.get("instruction") or {})
        instruction["instruction_text"] = generated
        # Stale token ids would no longer match the new text; let the consumer
        # re-tokenise instead of shipping a silently wrong field.
        if instruction.pop("instruction_tokens", None) is not None:
            dropped_tokens += 1
        new_ep["instruction"] = instruction
        kept.append(new_ep)

    out = dict(data)
    out["episodes"] = kept

    os.makedirs(os.path.dirname(out_gz) or ".", exist_ok=True)
    with gzip.open(out_gz, "wt", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"[AUG] Wrote VLN-CE split: {out_gz}")
    print(f"[AUG]   {len(kept)}/{len(episodes)} episodes carry a generated instruction")
    if dropped_tokens:
        print(f"[AUG]   dropped stale instruction_tokens on {dropped_tokens} episodes "
              f"(re-tokenise downstream)")


def merge_shards(shard_dir: str, out_json: str,
                 vlnce_template: str = "", out_vlnce_gz: str = "") -> None:
    """Concatenate per-GPU shard outputs back into a single records file."""
    shards = sorted(
        os.path.join(shard_dir, n) for n in os.listdir(shard_dir)
        if n.startswith("shard_") and n.endswith(".json")
    )
    if not shards:
        raise SystemExit(f"No shard_*.json found in {shard_dir}")

    records: List[dict] = []
    for path in shards:
        with open(path, "r", encoding="utf-8") as f:
            records.extend(json.load(f))

    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[AUG] Merged {len(shards)} shards -> {len(records)} records: {out_json}")

    if out_vlnce_gz:
        instruction_by_episode: Dict[int, str] = {}
        for r in records:
            for ep_id in r.get("episode_ids", []):
                instruction_by_episode[int(ep_id)] = r["generated_instruction"]
        write_vlnce_split(vlnce_template, out_vlnce_gz, instruction_by_episode)


def main():
    parser = argparse.ArgumentParser(
        description="Generate navigation instructions for data augmentation "
                    "(no scoring, no references required)")

    parser.add_argument("--model_dir", required=True,
                        help="VTInstructor checkpoint (SFT or VP-GRPO)")
    parser.add_argument("--processor_dir", default="",
                        help="Qwen3-VL processor; defaults to --model_dir")
    parser.add_argument("--data_dir", required=True,
                        help="Rendered split containing episode_*/sample.json")
    parser.add_argument("--out_json", default="",
                        help="Output records JSON (auto-named if omitted)")

    parser.add_argument("--vp_mode", choices=["auto", "on", "off"], default="auto",
                        help="auto: use vp_masks when the split has them; "
                             "on: require them; off: never use them")
    parser.add_argument("--frame_source", choices=["auto", "clean", "overlay"], default="auto",
                        help="auto: whatever the render pipeline last wrote to `frames`")
    parser.add_argument("--granularity", choices=["trajectory", "episode"], default="trajectory",
                        help="one generation per trajectory (default) or per episode")

    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help=">0 samples; use e.g. 1.0 to generate diverse augmentations")
    parser.add_argument("--dataset_type", choices=["auto", "r2rce", "rxrce"], default="auto")

    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vlnce_template", default="",
                        help="Original VLN-CE split (.json.gz) to clone geometry from")
    parser.add_argument("--out_vlnce_gz", default="",
                        help="Write a drop-in VLN-CE split with generated instructions")
    parser.add_argument("--merge_shards", default="",
                        help="Merge shard_*.json from this directory into --out_json "
                             "and exit; no model is loaded")

    # Must match training
    parser.add_argument("--vp_dim", type=int, default=384)
    parser.add_argument("--adapter_layers", default="7")
    parser.add_argument("--adapter_num_heads", type=int, default=4)

    args = parser.parse_args()

    if args.out_vlnce_gz and not args.vlnce_template:
        parser.error("--out_vlnce_gz requires --vlnce_template")
    if args.merge_shards and not args.out_json:
        parser.error("--merge_shards requires --out_json")

    if args.merge_shards:
        merge_shards(args.merge_shards, args.out_json,
                     args.vlnce_template, args.out_vlnce_gz)
        return

    if not args.out_json:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_json = os.path.join(
            _VP_DIR, "..", "outputs", "augment", f"generated_{ts}.json")

    processor_dir = args.processor_dir or args.model_dir
    dataset_type = (args.dataset_type if args.dataset_type != "auto"
                    else infer_dataset_type_from_path(args.data_dir))
    system_prompt = get_system_prompt(dataset_type)

    print("=" * 64)
    print("VTInstructor — instruction generation for data augmentation")
    print("=" * 64)
    print(f"Model:        {args.model_dir}")
    print(f"Processor:    {processor_dir}")
    print(f"Data:         {args.data_dir}")
    print(f"Output:       {args.out_json}")
    print(f"Dataset type: {dataset_type}")
    print(f"Granularity:  {args.granularity}")
    print(f"Temperature:  {args.temperature}")
    print(f"Shard:        {args.shard_idx}/{args.num_shards}")
    print("=" * 64)

    files = load_samples(args.data_dir)
    if not files:
        raise SystemExit(f"No episode_*/sample.json found in {args.data_dir}")
    print(f"[AUG] Found {len(files)} episodes")

    # ── Decide whether the visual trajectory prompt is available ──
    with_masks, probed = probe_vp_masks(files)
    has_masks = with_masks > 0
    if args.vp_mode == "on":
        if not has_masks:
            raise SystemExit(
                f"--vp_mode on, but none of the first {probed} episodes in "
                f"{args.data_dir} declares vp_masks. Run Stage 2 "
                f"(vp_adapter/render_vp_masks_from_samples.py) first, or use "
                f"--vp_mode auto/off.")
        use_vp = True
    elif args.vp_mode == "off":
        use_vp = False
    else:
        use_vp = has_masks

    print(f"[AUG] vp_mask coverage in first {probed}: {with_masks}/{probed}")
    if use_vp:
        print("[AUG] Visual trajectory prompting: ON (VP-Adapter driven by vp_masks)")
    else:
        print("[AUG] Visual trajectory prompting: OFF — generating from RGB frames "
              "only. This is supported: the VTP context is already learned into the "
              "weights. Rendering vp_masks (Stage 2) gives more accurate instructions.")

    print("[AUG] Loading processor ...")
    processor = AutoProcessor.from_pretrained(
        processor_dir, trust_remote_code=True, local_files_only=True)

    print(f"[AUG] Loading model from {args.model_dir} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True, local_files_only=True,
        device_map="auto",
    )

    vp_encoder = None
    if use_vp:
        adapter_layers = tuple(int(x) for x in args.adapter_layers.split(","))
        vp_cfg = VPAdapterConfig(
            vp_dim=args.vp_dim,
            adapter_layers=adapter_layers,
            adapter_num_heads=args.adapter_num_heads,
            freeze_vit_backbone=True,
        )
        print("[AUG] Attaching VP-Adapter ...")
        vp_encoder, _ = attach_vp_adapter(model, vp_cfg)
        if not os.path.exists(os.path.join(args.model_dir, VP_ENCODER_FILENAME)):
            print(f"[AUG][WARN] {VP_ENCODER_FILENAME} not found in {args.model_dir}; "
                  f"the adapter would run with random weights. Falling back to "
                  f"--vp_mode off.")
            use_vp, vp_encoder = False, None
        else:
            load_vp_modules(vp_encoder, model, args.model_dir)
            vp_encoder.eval()
    model.eval()

    rng = random.Random(args.seed)
    rng.shuffle(files)
    if args.num_samples > 0:
        files = files[:args.num_samples]
    if args.num_shards > 1:
        files = shard_files_by_key(files, args.num_shards, args.shard_idx)
        print(f"[AUG] Shard holds {len(files)} episodes")

    units = group_units(files, args.granularity)
    print(f"[AUG] {len(units)} generation units ({args.granularity})")

    records = []
    instruction_by_episode: Dict[int, str] = {}

    for traj_id, unit_files in tqdm(units, desc="Generating", unit="unit"):
        sample_path = unit_files[0]
        sample = safe_load_json(sample_path)
        if sample is None:
            continue

        images, actions_list, vp_masks = build_inputs(
            sample, sample_path, args.max_frames, args.frame_source, use_vp)
        if not images:
            continue

        generated = clean_text(generate_with_vp(
            model, processor, vp_encoder, images, actions_list, vp_masks,
            args.max_new_tokens, args.temperature, system_prompt))

        episode_ids = []
        for fp in unit_files:
            s = safe_load_json(fp)
            if s is not None and s.get("episode_id") is not None:
                episode_ids.append(int(s["episode_id"]))
        for ep_id in episode_ids:
            instruction_by_episode[ep_id] = generated

        records.append({
            "dataset": dataset_type,
            "trajectory_id": traj_id if traj_id is not None else sample.get("trajectory_id"),
            "episode_ids": episode_ids,
            "scene_id": sample.get("scene_id"),
            "sample_path": sample_path,
            "num_frames": len(images),
            "frame_source": args.frame_source,
            "vp_prompted": use_vp,
            "generated_instruction": generated,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    lengths = [len(r["generated_instruction"].split()) for r in records]
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    print(f"[AUG] Saved {len(records)} generated instructions to {args.out_json}")
    print(f"[AUG] Covered {len(instruction_by_episode)} episodes, "
          f"average length {avg_len:.1f} words")

    if args.out_vlnce_gz:
        write_vlnce_split(args.vlnce_template, args.out_vlnce_gz, instruction_by_episode)


if __name__ == "__main__":
    main()
