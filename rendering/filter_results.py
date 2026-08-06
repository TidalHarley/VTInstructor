#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据打分结果过滤低质量指令

读取 scored_instructions.json，按阈值过滤，输出 filtered_ult.json。
训练时只使用 filtered_ult.json 中保留的 episode。

使用方法：
    # 默认阈值 >= 5
    python filter_results.py

    # 自定义阈值
    python filter_results.py --threshold 6

    # 指定输入输出
    python filter_results.py --input scored.json --output filtered.json --threshold 5

输出格式 (filtered_ult.json):
    {
        "metadata": {
            "threshold": 5,
            "total": 10819,
            "kept": 9500,
            "removed": 1319
        },
        "episodes": {
            "1": {"instruction": "...", "score": 8, "trajectory_id": 4},
            "2": {"instruction": "...", "score": 7, "trajectory_id": 4},
            ...
        }
    }
"""

import os
import sys
import json
import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Filter instructions by quality score")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--input",
        default=os.path.join(script_dir, "scored_instructions.json"),
        help="输入：打分结果 JSON"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(script_dir, "filtered_ult.json"),
        help="输出：过滤后的 JSON"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="最低分数阈值（score >= threshold 的保留，默认 5）"
    )

    args = parser.parse_args()

    # 读取打分结果
    if not os.path.exists(args.input):
        print(f"[FATAL] 输入文件不存在: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        scored: list = json.load(f)

    print(f"[INFO] 读取 {len(scored)} 条打分结果")

    # 统计
    scores = [it["score"] for it in scored if it.get("score") is not None]
    unscored = [it for it in scored if it.get("score") is None]
    dist = Counter(scores)

    print(f"[INFO] 已打分: {len(scores)}, 未打分: {len(unscored)}")
    print(f"\n  分数分布:")
    for s in range(1, 11):
        cnt = dist.get(s, 0)
        marker = " <<<< 过滤线" if s == args.threshold else ""
        bar = "█" * (cnt // 20)
        print(f"    {s:2d}: {cnt:5d} {bar}{marker}")
    if scores:
        print(f"  平均分: {sum(scores)/len(scores):.2f}")

    # 过滤
    kept = {}
    removed_count = 0
    for it in scored:
        ep_id = it["episode_id"]
        score = it.get("score")

        if score is not None and score >= args.threshold:
            kept[str(ep_id)] = {
                "instruction": it["instruction"],
                "score": score,
                "trajectory_id": it.get("trajectory_id"),
            }
        else:
            removed_count += 1

    total = len(scored)
    kept_count = len(kept)

    print(f"\n[FILTER] threshold >= {args.threshold}")
    print(f"  保留: {kept_count} ({100*kept_count/total:.1f}%)")
    print(f"  过滤: {removed_count} ({100*removed_count/total:.1f}%)")

    # 按 trajectory 统计：有多少 trajectory 至少保留了 1 条
    traj_kept = set()
    traj_all = set()
    for it in scored:
        tid = it.get("trajectory_id")
        if tid is not None:
            traj_all.add(tid)
    for v in kept.values():
        tid = v.get("trajectory_id")
        if tid is not None:
            traj_kept.add(tid)
    print(f"  涉及 trajectory: {len(traj_kept)} / {len(traj_all)} 保留了至少 1 条指令")

    # 输出
    output_data = {
        "metadata": {
            "threshold": args.threshold,
            "total": total,
            "kept": kept_count,
            "removed": removed_count,
            "source": args.input,
        },
        "episodes": kept,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] 输出: {args.output}")

    # 被过滤的示例
    removed_items = [it for it in scored if it.get("score") is not None and it["score"] < args.threshold]
    if removed_items:
        removed_items.sort(key=lambda x: x["score"])
        print(f"\n  被过滤的低分示例:")
        for it in removed_items[:8]:
            print(f"    score={it['score']} ep={it['episode_id']}: \"{it['instruction'][:90]}\"")


if __name__ == "__main__":
    main()
