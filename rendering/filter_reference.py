#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen API 指令质量打分脚本（纯文本）

功能：
    读取 to_score.json，批量调用 Qwen（阿里云 DashScope，OpenAI 兼容接口）
    为每条 navigation instruction 打分（1-10）。
    打分依据：是否引用 landmark、是否过短/信息不足、语言是否流畅。
    输出 scored_instructions.json。主要用于过滤掉质量太差的指令。

使用方法：
    # 正式运行（默认读取同目录 api0key 文件中的 key）
    python filter_reference.py

    # 显式指定 key
    python filter_reference.py --api_key YOUR_KEY

    # 断点续传
    python filter_reference.py --resume

    # 试跑（不调 API）
    python filter_reference.py --dry_run

    # 只跑前 100 条
    python filter_reference.py --max_items 100

输入格式 (to_score.json):
    [
        {"episode_id": 1, "trajectory_id": 4, "instruction": "..."},
        ...
    ]

输出格式 (scored_instructions.json):
    [
        {"episode_id": 1, "trajectory_id": 4, "instruction": "...", "score": 8},
        ...
    ]
"""

import os
import sys
import json
import time
import argparse
import http.client
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Qwen API 调用（DashScope OpenAI 兼容接口）
# ============================================================================

DASHSCOPE_HOST = "dashscope.aliyuncs.com"
DASHSCOPE_PATH = "/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen3-max"


def _load_default_api_key() -> str:
    """默认从同目录的 api0key 文件读取 key（其次环境变量）。"""
    env_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if env_key:
        return env_key.strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(script_dir, "api0key"),
        os.path.join(os.path.dirname(script_dir), "api0key"),
    ):
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                key = f.read().strip()
            if key:
                return key
    return ""


SCORING_SYSTEM_PROMPT = """You are an expert at evaluating navigation instruction quality. Score each instruction from 1 to 10.

Scoring criteria:
- Landmarks: Mentions concrete visual landmarks (furniture, rooms, objects, architectural features such as door, stairs, kitchen, sofa). Instructions with NO landmarks should score low.
- Length / Completeness: Should not be too short or vague. Very short instructions (< 8 words) or ones missing route/direction info should score low.
- Fluency: Natural, grammatical English prose. Not choppy fragments or "go to X / go to Y" lists.

Score guide:
  1-3: Very poor. Fragments, "go to X / go to Y" lists, unintelligible, too short, no landmarks, or missing route info.
  4-5: Below average. Very short (< 10 words), vague, or no concrete landmarks.
  6-7: Average. Functional directions but generic, few landmarks.
  8-10: Good to excellent. Fluent, concrete landmarks, clear route with a stop point.

IMPORTANT: Return ONLY a JSON array of integer scores, one per instruction, in the same order.
Example: [7, 3, 8, 5, 9]"""


def qwen_chat_completions(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str = DEFAULT_MODEL,
    host: str = DASHSCOPE_HOST,
    path: str = DASHSCOPE_PATH,
    timeout_s: int = 120,
    temperature: float = 0.1,
    max_tokens: int = 512,
    max_retries: int = 5,
    backoff_base: float = 2.0,
) -> Tuple[str, Dict[str, Any]]:
    """OpenAI-compatible Chat Completions call via Aliyun DashScope (Qwen)."""
    payload_obj = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            conn = http.client.HTTPSConnection(host, timeout=timeout_s)
            conn.request("POST", path, body=payload, headers=headers)
            res = conn.getresponse()
            data = res.read()
            conn.close()

            if res.status != 200:
                try:
                    err_json = json.loads(data.decode("utf-8", errors="replace"))
                except Exception:
                    err_json = {"raw": data.decode("utf-8", errors="replace")}
                raise RuntimeError(f"HTTP {res.status} {res.reason}: {err_json}")

            resp = json.loads(data.decode("utf-8"))
            text = resp["choices"][0]["message"]["content"]
            return text, resp

        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                break
            wait = backoff_base ** attempt
            print(f"    [RETRY {attempt+1}/{max_retries}] {e}, waiting {wait:.1f}s...")
            time.sleep(wait)

    raise RuntimeError(f"Request failed after {max_retries} retries: {last_err}")


def build_scoring_prompt(batch: List[Dict]) -> str:
    """构建批量打分的 user 消息"""
    lines = []
    for i, item in enumerate(batch):
        lines.append(f'{i+1}. [ep={item["episode_id"]}] "{item["instruction"]}"')
    return (
        f"Score the following {len(batch)} navigation instructions (1-10 each):\n\n"
        + "\n".join(lines)
        + "\n\nReturn ONLY a JSON array of integer scores:"
    )


def parse_scores(text: str, expected_count: int) -> Optional[List[int]]:
    """从 LLM 返回文本中解析分数数组"""
    text = text.strip()

    # 尝试直接 JSON parse
    try:
        scores = json.loads(text)
        if isinstance(scores, list) and len(scores) == expected_count:
            return [int(s) for s in scores]
    except (json.JSONDecodeError, ValueError):
        pass

    # 尝试提取 [...] 部分
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            scores = json.loads(text[start:end + 1])
            if isinstance(scores, list) and len(scores) == expected_count:
                return [int(s) for s in scores]
        except (json.JSONDecodeError, ValueError):
            pass

    # 尝试逐行提取数字
    import re
    nums = re.findall(r'\b(\d{1,2})\b', text)
    nums = [int(n) for n in nums if 1 <= int(n) <= 10]
    if len(nums) == expected_count:
        return nums

    return None


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score navigation instructions using GPT API (batch mode)"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--input",
        default=os.path.join(script_dir, "to_score.json"),
        help="输入：待打分指令 JSON"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(script_dir, "scored_instructions.json"),
        help="输出：含分数的 JSON"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="API 模型名称（Qwen 文本模型）")
    parser.add_argument("--batch_size", type=int, default=20, help="每次 API 调用打分的条数")
    parser.add_argument("--sleep", type=float, default=0.1, help="每次请求间隔（秒）")
    parser.add_argument("--api_key", default=None, help="API Key")
    parser.add_argument("--resume", action="store_true", help="断点续传")
    parser.add_argument("--save_every", type=int, default=200, help="每打分多少条保存一次")
    parser.add_argument("--max_items", type=int, default=0, help="最多处理多少条（0=全部）")
    parser.add_argument("--dry_run", action="store_true", help="仅检查数据")

    args = parser.parse_args()

    # ---- 读取输入 ----
    if not os.path.exists(args.input):
        print(f"[FATAL] 输入文件不存在: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        all_items: List[Dict] = json.load(f)

    print(f"[INFO] 读取 {len(all_items)} 条指令")

    if args.max_items > 0:
        all_items = all_items[:args.max_items]
        print(f"[INFO] 限制处理前 {args.max_items} 条")

    if args.dry_run:
        print("\n[DRY RUN] 数据预览:")
        lens = [len(it["instruction"].split()) for it in all_items]
        print(f"  词数 - 平均: {sum(lens)/len(lens):.1f}, 最短: {min(lens)}, 最长: {max(lens)}")
        short = [it for it in all_items if len(it["instruction"].split()) < 8]
        print(f"  极短指令（<8词）: {len(short)} 条")
        for it in short[:5]:
            print(f"    ep={it['episode_id']}: \"{it['instruction']}\"")
        print(f"\n  批量打分示例 (batch_size={args.batch_size}):")
        prompt = build_scoring_prompt(all_items[:min(5, len(all_items))])
        print(prompt[:500])
        return

    # ---- API Key ----
    api_key = args.api_key or _load_default_api_key()
    if not api_key:
        print("[FATAL] 请提供 API Key：--api_key YOUR_KEY、export DASHSCOPE_API_KEY=YOUR_KEY，或在同目录放置 api0key 文件")
        sys.exit(1)

    # ---- Resume：加载已有结果 ----
    scored_map: Dict[int, int] = {}  # episode_id -> score
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for it in existing:
            if "score" in it and it["score"] is not None:
                scored_map[it["episode_id"]] = it["score"]
        print(f"[RESUME] 已有 {len(scored_map)} 条打分结果")

    # 过滤出未打分的
    pending = [it for it in all_items if it["episode_id"] not in scored_map]
    print(f"[INFO] 待打分: {len(pending)} / {len(all_items)}")

    # ---- 批量打分 ----
    total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
    scored_count = 0
    error_count = 0

    for batch_idx in range(total_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(pending))
        batch = pending[start:end]

        user_prompt = build_scoring_prompt(batch)
        messages = [
            {"role": "system", "content": SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            text, raw = qwen_chat_completions(
                messages=messages,
                api_key=api_key,
                model=args.model,
            )

            scores = parse_scores(text, len(batch))

            if scores is not None:
                for item, score in zip(batch, scores):
                    scored_map[item["episode_id"]] = max(1, min(10, score))
                scored_count += len(batch)
            else:
                # 解析失败，逐条标记为需重试
                print(f"    [WARN] batch {batch_idx+1}: 无法解析返回值，跳过 {len(batch)} 条")
                print(f"    返回内容: {text[:200]}")
                error_count += len(batch)

        except Exception as e:
            print(f"    [ERROR] batch {batch_idx+1}: {e}")
            error_count += len(batch)

        # 进度
        processed = start + len(batch)
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
            print(
                f"[{processed}/{len(pending)}] "
                f"batches={batch_idx+1}/{total_batches} "
                f"scored={scored_count} errors={error_count}"
            )

        # 定期保存
        if processed % args.save_every < args.batch_size or (batch_idx + 1) == total_batches:
            _save_results(all_items, scored_map, args.output)

        if args.sleep > 0 and batch_idx < total_batches - 1:
            time.sleep(args.sleep)

    # ---- 最终保存 ----
    _save_results(all_items, scored_map, args.output)

    # ---- 统计 ----
    scored_values = [v for v in scored_map.values()]
    print(f"\n{'='*60}")
    print(f"打分完成!")
    print(f"  已打分: {len(scored_map)} / {len(all_items)}")
    print(f"  失败:   {error_count}")
    print(f"  输出:   {args.output}")
    if scored_values:
        from collections import Counter
        dist = Counter(scored_values)
        print(f"\n  分数分布:")
        for s in range(1, 11):
            cnt = dist.get(s, 0)
            bar = "█" * (cnt // 20)
            print(f"    {s:2d}: {cnt:5d} {bar}")
        avg = sum(scored_values) / len(scored_values)
        print(f"  平均分: {avg:.2f}")
    print(f"{'='*60}")


def _save_results(all_items: List[Dict], scored_map: Dict[int, int], output_path: str):
    """保存打分结果"""
    results = []
    for it in all_items:
        entry = {
            "episode_id": it["episode_id"],
            "trajectory_id": it["trajectory_id"],
            "instruction": it["instruction"],
            "score": scored_map.get(it["episode_id"]),
        }
        results.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
