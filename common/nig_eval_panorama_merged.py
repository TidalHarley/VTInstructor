#!/usr/bin/env python3
"""
Panorama 格式的 NIG 评测脚本（融合标签训练后的模型评测）

注意：评测时使用原始 R2R 多参考进行评分，NOT 融合标签。
      融合标签仅用于训练，评测标准与原版完全一致。

使用方法：
    python nig_eval_panorama_merged.py --model_dir /path/to/model --data_dir /path/to/eval_data
"""
import argparse
import gzip
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from nig_dataset_panorama_merged import (
    build_inference_prompt,
    get_system_prompt,
    group_actions,
    infer_dataset_type_from_path,
    safe_load_json,
)


def load_samples(root_dir: str) -> List[str]:
    """加载所有 episode 的 sample.json 路径"""
    files = []
    for name in os.listdir(root_dir):
        if not name.startswith("episode_"):
            continue
        p = os.path.join(root_dir, name, "sample.json")
        if os.path.exists(p):
            files.append(p)
    return sorted(files)


def shard_files_by_key(files: List[str], num_shards: int, shard_idx: int) -> List[str]:
    """
    对样本文件做稳定分片：
    - 优先按 trajectory_id 分片，避免同 trajectory 跨卡重复
    - 否则按 episode_id 分片
    - 再否则按文件名哈希分片
    """
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
    """
    评测文本标准化（对齐 C-Instructor/PTBTokenizer 预期）：
    - 将换行替换为空格（C-Instructor 原本只处理 \\n）
    - 额外处理 \\r，避免 CRLF 导致逐条错位
    - 压缩多余空白
    """
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def build_prompt_from_sample(sample: dict, max_frames: int) -> Tuple[List[Image.Image], List[str], str]:
    """
    从 sample 构建推理用的数据

    Returns:
        (images, actions_list, instruction)
    """
    # 处理动作
    if "action_events_text" in sample:
        actions_list = sample["action_events_text"]
    else:
        group_size = int(sample.get("actions_per_frame") or sample.get("frame_stride") or 2)
        actions_text = sample.get("actions_text") or [str(a) for a in sample.get("actions", [])]
        actions_list = group_actions(actions_text, group_size)

    # 截取帧和动作
    num_frames = min(len(sample["frames"]), len(actions_list) + 1, max_frames)
    frames = sample["frames"][:num_frames]
    actions_list = actions_list[: max(0, num_frames - 1)]

    # 加载图像
    images = [Image.open(p).convert("RGB") for p in frames]

    return images, actions_list, sample.get("instruction", "")


def generate_prediction(
    model,
    processor,
    images: List[Image.Image],
    actions_list: List[str],
    max_new_tokens: int = 150,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
) -> str:
    """
    生成预测（使用与训练完全相同的格式）

    - 使用 PANORAMA_SYSTEM_PROMPT
    - 使用 build_inference_prompt 构建 user_content
    - temperature=0 → Greedy decoding；temperature>0 → 温度采样
    """
    # 构建 user_content（与训练时格式完全一致）
    user_content = build_inference_prompt(images, actions_list, system_prompt=system_prompt)

    # 构建消息
    messages = [{"role": "user", "content": user_content}]

    # Tokenize
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        if temperature > 0:
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )
        else:
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    # 只取生成的部分
    gen_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()


def build_r2r_path_id_index(r2r_json_path: str) -> Dict[int, List[str]]:
    """构建 R2R path_id -> instructions 映射"""
    if not os.path.exists(r2r_json_path):
        return {}

    with open(r2r_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    idx = {}
    for item in data:
        path_id = item.get("path_id")
        insts = item.get("instructions")
        if path_id is not None and isinstance(insts, list):
            insts_clean = [
                normalize_text_for_metrics(x)
                for x in insts
                if normalize_text_for_metrics(x)
            ]
            if insts_clean:
                idx[int(path_id)] = insts_clean
    return idx


def load_vlnce_episode_info(vlnce_json_gz: str) -> Dict[int, Dict]:
    """加载 VLNCE episode 信息"""
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


def eval_r2r_multiref(
    model,
    processor,
    files: List[str],
    max_frames: int,
    max_new_tokens: int,
    r2r_path_idx: Dict[int, List[str]],
    vlnce_episode_info: Dict[int, Dict],
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
) -> Tuple[List[Dict], List[str], List[List[str]]]:
    """
    R2R multiref 评测模式

    按 trajectory_id 分组，每个 trajectory 只评测一次，参考 R2R 的多条 instructions
    """
    # 按 trajectory_id 分组
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
    print(f"[EVAL] R2R index: {len(r2r_path_idx)} path_ids")

    records = []
    preds = []
    refs_per_sample = []

    found_in_r2r = 0
    not_found_in_r2r = 0

    for traj_id in tqdm(sorted(traj_to_files.keys()), desc="Evaluating", unit="traj"):
        fp = traj_to_files[traj_id][0]
        sample = safe_load_json(fp)
        if sample is None:
            continue

        images, actions_list, instruction = build_prompt_from_sample(sample, max_frames)
        pred = normalize_text_for_metrics(
            generate_prediction(
                model,
                processor,
                images,
                actions_list,
                max_new_tokens,
                temperature,
                system_prompt=system_prompt,
            )
        )

        # 获取参考 instructions（原始 R2R 多参考）
        refs = r2r_path_idx.get(traj_id)
        if refs and len(refs) >= 1:
            found_in_r2r += 1
        else:
            not_found_in_r2r += 1
            refs = [normalize_text_for_metrics(instruction)] if instruction else [""]

        preds.append(pred)
        refs_per_sample.append(refs)

        records.append({
            "trajectory_id": traj_id,
            "episode_id": sample.get("episode_id"),
            "num_episodes_in_traj": len(traj_to_files[traj_id]),
            "prediction": pred,
            "references": refs,
            "num_refs": len(refs),
            "ref_source": "r2r_lookup" if traj_id in r2r_path_idx else "single_ref",
        })

    print(f"[EVAL] R2R lookup success: {found_in_r2r}, fallback: {not_found_in_r2r}")

    return records, preds, refs_per_sample


def eval_single_ref(
    model,
    processor,
    files: List[str],
    max_frames: int,
    max_new_tokens: int,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
) -> Tuple[List[Dict], List[str], List[List[str]]]:
    """
    单条参考评测模式
    """
    records = []
    preds = []
    refs_per_sample = []

    for fp in tqdm(files, desc="Evaluating", unit="sample"):
        sample = safe_load_json(fp)
        if sample is None:
            continue

        images, actions_list, instruction = build_prompt_from_sample(sample, max_frames)
        pred = normalize_text_for_metrics(
            generate_prediction(
                model,
                processor,
                images,
                actions_list,
                max_new_tokens,
                temperature,
                system_prompt=system_prompt,
            )
        )

        refs = [normalize_text_for_metrics(instruction)] if instruction else [""]

        preds.append(pred)
        refs_per_sample.append(refs)

        records.append({
            "episode_id": sample.get("episode_id"),
            "trajectory_id": sample.get("trajectory_id"),
            "prediction": pred,
            "reference": instruction,
        })

    return records, preds, refs_per_sample


def eval_rxr_multiref(
    model,
    processor,
    files: List[str],
    max_frames: int,
    max_new_tokens: int,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
) -> Tuple[List[Dict], List[str], List[List[str]]]:
    """
    RXR 多参考评测模式
    - 按 trajectory_id 分组
    - 对每个 trajectory 只生成一次预测
    - references 来自同 trajectory 下所有 episode 的 instruction
    """
    traj_to_files: Dict[int, List[str]] = defaultdict(list)
    for fp in files:
        sample = safe_load_json(fp)
        if sample is None:
            continue
        traj_id = sample.get("trajectory_id")
        if traj_id is not None:
            traj_to_files[int(traj_id)].append(fp)

    print(f"[EVAL] {len(traj_to_files)} unique trajectories (rxr_multiref)")

    records = []
    preds = []
    refs_per_sample = []

    for traj_id in tqdm(sorted(traj_to_files.keys()), desc="Evaluating", unit="traj"):
        traj_files = traj_to_files[traj_id]
        first_sample = safe_load_json(traj_files[0])
        if first_sample is None:
            continue

        # 生成一次预测（trajectory 代表样本）
        images, actions_list, instruction = build_prompt_from_sample(first_sample, max_frames)
        pred = normalize_text_for_metrics(
            generate_prediction(
                model,
                processor,
                images,
                actions_list,
                max_new_tokens,
                temperature,
                system_prompt=system_prompt,
            )
        )

        # 从同 trajectory 下聚合多参考
        refs_raw = []
        for tf in traj_files:
            s = safe_load_json(tf)
            if s is None:
                continue
            ins = normalize_text_for_metrics(s.get("instruction", ""))
            if ins:
                refs_raw.append(ins)

        # 去重并保持顺序
        seen = set()
        refs = []
        for r in refs_raw:
            if r not in seen:
                seen.add(r)
                refs.append(r)
        if not refs:
            refs = [normalize_text_for_metrics(instruction)] if instruction else [""]

        preds.append(pred)
        refs_per_sample.append(refs)
        records.append({
            "trajectory_id": traj_id,
            "episode_id": first_sample.get("episode_id"),
            "num_episodes_in_traj": len(traj_files),
            "prediction": pred,
            "references": refs,
            "num_refs": len(refs),
            "ref_source": "rxr_traj_multiref",
        })

    return records, preds, refs_per_sample


def main():
    parser = argparse.ArgumentParser(description="Evaluate NIG model (merged-label trained) on Panorama data")

    # 模型路径
    parser.add_argument(
        "--model_dir",
        required=True,
        help="微调后的模型目录（checkpoint）"
    )
    parser.add_argument(
        "--processor_dir",
        default="PATH/TO/models_cache/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        help="Processor 目录"
    )

    # 数据路径
    parser.add_argument(
        "--data_dir",
        default="PATH/TO/outputs/R2RCE_visual/r2rce_valunseen_visual",
        help="评测数据目录"
    )

    # 输出
    parser.add_argument(
        "--out_json",
        default="",
        help="输出 JSON 文件路径（默认自动生成）"
    )

    # 评测参数
    parser.add_argument("--max_frames", type=int, default=30, help="最大帧数")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="最大生成 token 数")
    parser.add_argument("--num_samples", type=int, default=0, help="评测样本数（0=全部）")
    parser.add_argument("--num_shards", type=int, default=1, help="分片总数（多卡并行推理用）")
    parser.add_argument("--shard_idx", type=int, default=0, help="当前分片下标，从0开始")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="生成温度(0=greedy>0 为温度采样，如 0.8）")
    parser.add_argument(
        "--dataset_type",
        choices=["auto", "r2rce", "rxrce"],
        default="auto",
        help="选择推理 prompt；auto 根据 data_dir 自动判断",
    )

    # 评测模式
    parser.add_argument(
        "--eval_mode",
        choices=["r2r_multiref", "rxr_multiref", "single_ref"],
        default="r2r_multiref",
        help="评测模式"
    )

    # R2R 参考文件
    parser.add_argument(
        "--r2r_val_json",
        default="PATH/TO/R2R/R2R/data/R2R_val_unseen.json",
        help="R2R val_unseen JSON 路径"
    )
    parser.add_argument(
        "--vlnce_val_json_gz",
        default="PATH/TO/dataset/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz",
        help="VLNCE val_unseen JSON.gz 路径"
    )

    args = parser.parse_args()

    # 自动生成输出文件名
    if not args.out_json:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_json = f"PATH/TO/outputs/nig_eval_panorama_merged_visual_{ts}.json"

    print("=" * 60)
    print("Panorama NIG Evaluation With Visual Prompt (Merged-Label Model)")
    print("=" * 60)
    print(f"Model dir:      {args.model_dir}")
    print(f"Processor dir:  {args.processor_dir}")
    print(f"Data dir:       {args.data_dir}")
    print(f"Output:         {args.out_json}")
    print(f"Eval mode:      {args.eval_mode}")
    print(f"Max frames:     {args.max_frames}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature:    {args.temperature} ({'greedy' if args.temperature == 0 else 'sampling'})")
    print(f"Dataset type:   {args.dataset_type}")
    print(f"Shard:          {args.shard_idx}/{args.num_shards}")
    print("=" * 60)

    dataset_type = args.dataset_type if args.dataset_type != "auto" else infer_dataset_type_from_path(args.data_dir)
    system_prompt = get_system_prompt(dataset_type)
    print(f"[INFO] Resolved dataset_type: {dataset_type}")

    # 加载模型
    print(f"[INFO] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.processor_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"[INFO] Loading model from {args.model_dir}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )

    # 加载数据
    files = load_samples(args.data_dir)
    if not files:
        raise SystemExit(f"No samples found in {args.data_dir}")

    print(f"[INFO] Found {len(files)} samples")

    # 打乱并截取
    rng = random.Random(args.seed)
    rng.shuffle(files)
    if args.num_samples > 0:
        files = files[:args.num_samples]
        print(f"[INFO] Using {len(files)} samples")
    if args.num_shards > 1:
        if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
            raise SystemExit(f"Invalid shard_idx={args.shard_idx}, num_shards={args.num_shards}")
        files = shard_files_by_key(files, args.num_shards, args.shard_idx)
        print(f"[INFO] Sharded samples: {len(files)}")
        if not files:
            print("[WARN] Empty shard, writing empty output and exit.")
            os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return

    # 评测
    if args.eval_mode == "r2r_multiref":
        r2r_path_idx = build_r2r_path_id_index(args.r2r_val_json)
        vlnce_episode_info = load_vlnce_episode_info(args.vlnce_val_json_gz)

        records, preds, refs = eval_r2r_multiref(
            model, processor, files,
            args.max_frames, args.max_new_tokens,
            r2r_path_idx, vlnce_episode_info,
            temperature=args.temperature,
            system_prompt=system_prompt,
        )
    elif args.eval_mode == "rxr_multiref":
        records, preds, refs = eval_rxr_multiref(
            model, processor, files,
            args.max_frames, args.max_new_tokens,
            temperature=args.temperature,
            system_prompt=system_prompt,
        )
    else:
        records, preds, refs = eval_single_ref(
            model, processor, files,
            args.max_frames, args.max_new_tokens,
            temperature=args.temperature,
            system_prompt=system_prompt,
        )

    # 保存结果
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved {len(records)} predictions to {args.out_json}")

    # 统计
    pred_lens = [len(r["prediction"].split()) for r in records]
    avg_len = sum(pred_lens) / len(pred_lens) if pred_lens else 0
    print(f"[INFO] Average prediction length: {avg_len:.1f} words")


if __name__ == "__main__":
    main()
