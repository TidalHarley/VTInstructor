#!/usr/bin/env python3
"""Collect R2R-CE train instructions from rendered episode_*/sample.json.

Writes a list for score_instructions.py. Default input is the Stage-1
R2R-CE train render directory.

    python preprocess/filtering/collect_instructions.py \
        --train_dir data/R2RCE_visual/r2rce_train_visual \
        --output preprocess/filtering/to_score.json

Output:
    [{"episode_id": 1, "trajectory_id": 4, "instruction": "..."}, ...]
"""

import os
import sys
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Collect instructions from R2R-CE train renders")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    default_train_dir = os.path.join(project_dir, "data", "R2RCE_visual", "r2rce_train_visual")

    parser.add_argument(
        "--train_dir",
        default=default_train_dir,
        help="Render output with episode_*/sample.json",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(script_dir, "to_score.json"),
        help="JSON list of instructions to score",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.train_dir):
        print(f"[FATAL] train 目录不存在: {args.train_dir}")
        sys.exit(1)

    episodes = sorted(
        d for d in os.listdir(args.train_dir)
        if d.startswith("episode_") and os.path.isdir(os.path.join(args.train_dir, d))
    )
    print(f"[INFO] 在 {args.train_dir} 发现 {len(episodes)} 个 episode 目录")

    items = []
    missing = 0
    empty_instr = 0
    for ep in episodes:
        sample_path = os.path.join(args.train_dir, ep, "sample.json")
        if not os.path.exists(sample_path):
            missing += 1
            continue
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"[WARN] 读取失败 {sample_path}: {e}")
            missing += 1
            continue

        instruction = (d.get("instruction") or "").strip()
        if not instruction:
            empty_instr += 1
            continue

        items.append({
            "episode_id": d.get("episode_id"),
            "trajectory_id": d.get("trajectory_id"),
            "instruction": instruction,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[OK] 写出 {len(items)} 条待打分指令 -> {args.output}")
    if missing:
        print(f"[WARN] 缺少/无法读取 sample.json: {missing} 个")
    if empty_instr:
        print(f"[WARN] instruction 为空: {empty_instr} 个")


if __name__ == "__main__":
    main()
