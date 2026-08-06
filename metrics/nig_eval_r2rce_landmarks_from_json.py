#!/usr/bin/env python3
"""Evaluate R2RCE predictions from a JSON file (no inference)."""
import argparse
import json
import os
import shutil
import subprocess
import ssl
import tempfile
import urllib.request
from typing import Any, Dict, List
from zipfile import ZipFile


def normalize_text_for_metrics(text: Any) -> str:
    """
    将\r, \n和多余空格替换为空格
    一定要修改\r, 否则会有对齐的问题
    """
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def ensure_spice_corenlp():
    # 确保SPICE使用的CoreNLP版本为3.6.0，确保SPICE可以正常进行评测
    try:
        import pycocoevalcap.spice as spice_pkg
    except Exception:
        return

    spice_dir = os.path.dirname(spice_pkg.__file__)
    lib_dir = os.path.join(spice_dir, "lib")
    jar_base = "stanford-corenlp-3.6.0"
    jar_path = os.path.join(lib_dir, f"{jar_base}.jar")
    models_path = os.path.join(lib_dir, f"{jar_base}-models.jar")
    if os.path.exists(jar_path) and os.path.exists(models_path):
        return

    os.makedirs(lib_dir, exist_ok=True)
    urls = [
        "https://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip",
        "http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip",
        "https://downloads.sourceforge.net/project/stanford-corenlp.mirror/stanford-corenlp-full-2015-12-09.zip",
    ]
    zip_prefix = "stanford-corenlp-full-2015-12-09"
    members = [
        f"{zip_prefix}/{jar_base}.jar",
        f"{zip_prefix}/{jar_base}-models.jar",
    ]

    last_error = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        tmp.write(chunk)
                    tmp_path = tmp.name

            with ZipFile(tmp_path) as zf:
                for member in members:
                    with zf.open(member) as src, open(
                        os.path.join(lib_dir, os.path.basename(member)), "wb"
                    ) as dst:
                        dst.write(src.read())
            os.remove(tmp_path)
            return
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise RuntimeError(f"CoreNLP download failed: {last_error}")


def ensure_ptb_corenlp_3_4_1():
    # 确保 PTBTokenizer 依赖的 stanford-corenlp-3.4.1.jar 可用。
    import pycocoevalcap.tokenizer.ptbtokenizer as ptb_mod
    tok_dir = os.path.dirname(os.path.abspath(ptb_mod.__file__))
    jar_name = "stanford-corenlp-3.4.1.jar"
    jar_path = os.path.join(tok_dir, jar_name)

    if shutil.which("java") is None:
        raise RuntimeError("java not found in PATH. Please install JRE/JDK (java).")

    if os.path.exists(jar_path):
        return

    url = "https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.4.1/stanford-corenlp-3.4.1.jar"
    print(f"[PTBTokenizer] downloading {jar_name} to {jar_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
        data = resp.read()
    with open(jar_path, "wb") as f:
        f.write(data)

    try:
        subprocess.check_call(
            ["java", "-cp", jar_name, "edu.stanford.nlp.process.PTBTokenizer", "-h"],
            cwd=tok_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def compute_spice_batched(
    gts_tok: Dict,
    res_tok: Dict,
    batch_size: int = 100,
    java_xmx: str = "64G",
    max_caption_words: int = 100,
) -> float:
    # 把 SPICE 分批计算，避免一次性跑 613 条导致内存/时间爆炸
    if set(gts_tok.keys()) != set(res_tok.keys()):
        raise ValueError("SPICE: gts/res keys mismatch")

    all_ids = sorted(gts_tok.keys())
    n_samples = len(all_ids)
    if n_samples == 0:
        return 0.0

    all_scores = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    print(f"[SPICE] 分批评测: {n_samples} 样本, {n_batches} 批, 每批 {batch_size}, Java -Xmx{java_xmx}")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_ids = all_ids[start_idx:end_idx]

        batch_gts_remapped = {}
        batch_res_remapped = {}
        for new_id, orig_id in enumerate(batch_ids):
            batch_gts_remapped[new_id] = gts_tok[orig_id]
            batch_res_remapped[new_id] = res_tok[orig_id]

        try:
            batch_score = _compute_spice_single_batch(
                batch_gts_remapped, batch_res_remapped, java_xmx, max_caption_words
            )
            all_scores.extend(batch_score)
            print(f"[SPICE] 批次 {batch_idx+1}/{n_batches} 完成, 样本 {start_idx}-{end_idx-1}")
        except Exception as e:
            print(f"[SPICE] 批次 {batch_idx+1}/{n_batches} 失败: {e}")
            all_scores.extend([float("nan")] * len(batch_ids))

    import numpy as np
    valid_scores = [s for s in all_scores if not np.isnan(s)]
    if valid_scores:
        return float(np.mean(valid_scores))
    return 0.0


def _truncate_caption(text: str, max_words: int = 100) -> str:
    # prediction 最多100个单词，reference也最多100个单词
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def _compute_spice_single_batch(
    gts: Dict, res: Dict, java_xmx: str = "64G", max_caption_words: int = 100
) -> List[float]:
    import numpy as np

    ensure_spice_corenlp()

    import pycocoevalcap.spice as spice_pkg
    spice_dir = os.path.dirname(spice_pkg.__file__)

    SPICE_JAR = "spice-1.0.jar"
    user_cache_dir = os.path.expanduser("~/.cache/spice_cache")
    os.makedirs(user_cache_dir, exist_ok=True)
    user_temp_dir = os.path.expanduser("~/.cache/spice_tmp")
    os.makedirs(user_temp_dir, exist_ok=True)

    imgIds = sorted(gts.keys())
    input_data = []
    for id in imgIds:
        hypo = res[id]
        ref = gts[id]
        if isinstance(hypo, list):
            hypo = hypo[0] if hypo else ""
        if isinstance(ref, list):
            ref = ref if ref else [""]

        hypo = _truncate_caption(hypo, max_caption_words)
        if isinstance(ref, list):
            ref = [_truncate_caption(r, max_caption_words) for r in ref]
        else:
            ref = [_truncate_caption(ref, max_caption_words)]

        input_data.append({"image_id": id, "test": hypo, "refs": ref})

    in_file = tempfile.NamedTemporaryFile(delete=False, dir=user_temp_dir, mode="w+", suffix=".json")
    json.dump(input_data, in_file, indent=2)
    in_file.close()

    out_file = tempfile.NamedTemporaryFile(delete=False, dir=user_temp_dir, suffix=".json")
    out_file.close()

    spice_cmd = [
        "java", "-jar", f"-Xmx{java_xmx}", SPICE_JAR, in_file.name,
        "-cache", user_cache_dir,
        "-out", out_file.name,
        "-subset",
        "-silent",
    ]
    subprocess.check_call(spice_cmd, cwd=spice_dir)

    with open(out_file.name) as data_file:
        results = json.load(data_file)

    os.remove(in_file.name)
    os.remove(out_file.name)

    scores = []
    for item in results:
        try:
            score = float(item["scores"]["All"]["f"])
        except Exception:
            score = float("nan")
        scores.append(score)

    return scores


def coco_eval(
    preds: List[str],
    refs_per_sample: List[List[str]],
    spice_batch_size: int = 100,
    spice_java_xmx: str = "64G",
    max_caption_words: int = 100,
    skip_spice: bool = False,
    skip_meteor: bool = False,
) -> Dict[str, Any]:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    ensure_ptb_corenlp_3_4_1()

    gts = {i: [{"caption": r} for r in refs_per_sample[i]] for i in range(len(refs_per_sample))}
    res = {i: [{"caption": preds[i]}] for i in range(len(preds))}

    tokenizer = PTBTokenizer()
    gts_tok = tokenizer.tokenize(gts)
    res_tok = tokenizer.tokenize(res)

    out: Dict[str, Any] = {}

    bleu = Bleu(4)
    bleu_score, _ = bleu.compute_score(gts_tok, res_tok)
    out["BLEU-1"] = float(bleu_score[0])
    out["BLEU-2"] = float(bleu_score[1])
    out["BLEU-3"] = float(bleu_score[2])
    out["BLEU-4"] = float(bleu_score[3])

    if skip_meteor:
        out["METEOR"] = None
        out["METEOR_error"] = "skipped"
    else:
        from pycocoevalcap.meteor.meteor import Meteor
        meteor = Meteor()
        m_score, _ = meteor.compute_score(gts_tok, res_tok)
        out["METEOR"] = float(m_score)

    rouge = Rouge()
    r_score, _ = rouge.compute_score(gts_tok, res_tok)
    out["ROUGE-L"] = float(r_score)

    cider = Cider()
    c_score, _ = cider.compute_score(gts_tok, res_tok)
    out["CIDEr"] = float(c_score)

    if skip_spice:
        out["SPICE"] = None
        out["SPICE_error"] = "skipped"
    else:
        try:
            ensure_spice_corenlp()
            s_score = compute_spice_batched(
                gts_tok,
                res_tok,
                batch_size=spice_batch_size,
                java_xmx=spice_java_xmx,
                max_caption_words=max_caption_words,
            )
            out["SPICE"] = float(s_score)
        except Exception as e:
            out["SPICE"] = None
            out["SPICE_error"] = str(e)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True, help="Prediction json from inference")
    ap.add_argument("--out_json", default="", help="Optional metrics output json")
    ap.add_argument("--print_ref_stats", action="store_true")
    ap.add_argument("--skip_spice", action="store_true")
    ap.add_argument("--skip_meteor", action="store_true")
    ap.add_argument("--spice_batch_size", type=int, default=100)
    ap.add_argument("--spice_java_xmx", default="64G")
    ap.add_argument("--max_caption_words", type=int, default=100)
    args = ap.parse_args()

    with open(args.pred_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = []
    refs_per_sample = []
    for item in data:
        pred = item.get("prediction", "")
        preds.append(normalize_text_for_metrics(pred))

        if "references" in item and isinstance(item["references"], list):
            refs = item["references"]
        elif "reference" in item:
            refs = [item.get("reference", "")]
        else:
            refs = [""]
        refs_per_sample.append([normalize_text_for_metrics(r) for r in refs])

    metrics = coco_eval(
        preds,
        refs_per_sample,
        spice_batch_size=args.spice_batch_size,
        spice_java_xmx=args.spice_java_xmx,
        max_caption_words=args.max_caption_words,
        skip_spice=args.skip_spice,
        skip_meteor=args.skip_meteor,
    )
    metrics["num_evaluated"] = len(preds)

    if args.print_ref_stats:
        num_len3 = sum(1 for refs in refs_per_sample if len(refs) == 3)
        ratio = (num_len3 / len(preds)) if preds else 0.0
        metrics["num_with_3refs"] = num_len3
        metrics["pct_with_3refs"] = ratio

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
