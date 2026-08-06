#!/usr/bin/env python3
"""
GRPO reward computation using pycocoevalcap scoring algorithms.

Matches the evaluation pipeline (nig_eval_r2rce_landmarks_from_json.py):
  - Tokenization: PTBTokenizer (identical to eval), simple fallback if unavailable
  - BLEU-1/4: pycocoevalcap.bleu.bleu.Bleu  (per-sample scores)
  - ROUGE-L:  pycocoevalcap.rouge.rouge.Rouge (per-sample scores)
  - CIDEr-D:  pycocoevalcap.cider.cider_scorer.CiderScorer
              with **corpus-level IDF** pre-computed from all training references
  - METEOR:   pycocoevalcap.meteor.meteor.Meteor (Java-based, same as eval)

CIDEr-D note:  Standard CIDEr computes IDF from the evaluation batch, which is
meaningless for per-sample RL reward (batch = G copies of the same refs).
We pre-compute IDF from the entire training corpus and inject it via a
CiderScorer subclass, keeping all other logic (TF-IDF weighting, min-clip
cosine, Gaussian length penalty, per-n-gram-level average, x10) identical.
"""

import io
import math
import os
import re
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure conda env bin dir is in PATH so Java can be found by subprocess
# ---------------------------------------------------------------------------
_python_bin_dir = os.path.dirname(os.path.realpath(sys.executable))
if _python_bin_dir and _python_bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _python_bin_dir + os.pathsep + os.environ.get("PATH", "")

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider_scorer import CiderScorer

# PTBTokenizer (same Java-based tokenizer used by eval script)
_HAS_PTB = False
_PTB_WANTS_DICT = False
try:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    _HAS_PTB = True
    import inspect as _inspect
    _ptb_src = _inspect.getsource(PTBTokenizer.tokenize)
    _PTB_WANTS_DICT = "['caption']" in _ptb_src or '["caption"]' in _ptb_src
    del _inspect, _ptb_src
except Exception:
    pass

# METEOR (requires Java + meteor-1.5.jar)
_HAS_METEOR = False
try:
    from pycocoevalcap.meteor.meteor import Meteor
    _test_m = Meteor()
    _HAS_METEOR = True
    del _test_m
except Exception:
    pass


# ---------------------------------------------------------------------------
# Fallback tokenisation (used only when PTBTokenizer is unavailable)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _normalize(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


# ---------------------------------------------------------------------------
# CIDEr-D with corpus-level IDF
# ---------------------------------------------------------------------------

class CorpusCiderScorer(CiderScorer):
    """
    CiderScorer subclass that uses pre-computed corpus-level IDF
    instead of computing IDF from the (tiny) evaluation batch.
    """

    def __init__(self, corpus_doc_freq: dict, corpus_ref_len: float,
                 n: int = 4, sigma: float = 6.0):
        super().__init__(n=n, sigma=sigma)
        self._corpus_doc_freq = corpus_doc_freq
        self._corpus_ref_len = corpus_ref_len

    def compute_doc_freq(self):
        df = defaultdict(float)
        df.update(self._corpus_doc_freq)
        self.document_frequency = df

    def compute_score(self, option=None, verbose=0):
        self.compute_doc_freq()
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)

    def compute_cider(self):
        def counts2vec(cnts):
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                df = np.log(max(1.0, self.document_frequency[ngram]))
                n = len(ngram) - 1
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                norm[n] += pow(vec[n][ngram], 2)
                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                for (ngram, count) in vec_hyp[n].items():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n] * norm_ref[n])
                assert not math.isnan(val[n])
                val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        self.ref_len = self._corpus_ref_len

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)
        return scores


# ---------------------------------------------------------------------------
# Simple METEOR fallback (exact unigram matching + fragmentation penalty)
# ---------------------------------------------------------------------------

def _meteor_simple(hyp_str: str, ref_strs: List[str]) -> float:
    hyp = hyp_str.split()
    if not hyp:
        return 0.0
    best = 0.0
    for ref_str in ref_strs:
        ref = ref_str.split()
        if not ref:
            continue
        matched_h: set = set()
        matched_r: set = set()
        for i, h in enumerate(hyp):
            for j, r in enumerate(ref):
                if h == r and i not in matched_h and j not in matched_r:
                    matched_h.add(i)
                    matched_r.add(j)
                    break
        m = len(matched_h)
        if m == 0:
            continue
        p = m / len(hyp)
        r = m / len(ref)
        f_mean = (p * r) / (0.1 * p + 0.9 * r) if (p + r) > 0 else 0.0
        sorted_h = sorted(matched_h)
        chunks = 1
        for k in range(1, len(sorted_h)):
            if sorted_h[k] != sorted_h[k - 1] + 1:
                chunks += 1
        penalty = 0.5 * ((chunks / m) ** 3)
        best = max(best, f_mean * (1 - penalty))
    return best


# ---------------------------------------------------------------------------
# Length penalty
# ---------------------------------------------------------------------------

def length_penalty(
    num_words: int, target_min: int = 25, target_max: int = 40,
) -> float:
    if target_min <= num_words <= target_max:
        return 0.0
    if num_words < target_min:
        return -0.3 * (target_min - num_words) / target_min
    excess = num_words - target_max
    return -0.3 * min(excess / target_max, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "BLEU-1": 0.2,
    "BLEU-4": 0.2,
    "CIDEr": 0.2,
    "METEOR": 0.2,
    "ROUGE-L": 0.2,
}


class RewardComputer:
    """
    Per-sample GRPO reward computation using pycocoevalcap metrics.
    Uses PTBTokenizer (same as eval script) when available.

    Usage::

        rc = RewardComputer()
        rc.build_cider_idf(multiref_map)   # {traj_id: [ref1, ref2, ...]}
        rewards, metrics = rc.compute_rewards(completions, refs, weights)
    """

    def __init__(self):
        self._meteor = None
        if _HAS_METEOR:
            try:
                self._meteor = Meteor()
            except Exception:
                pass

        self._ptb = None
        if _HAS_PTB:
            try:
                self._ptb = PTBTokenizer()
            except Exception:
                pass

        self._corpus_doc_freq: Optional[dict] = None
        self._corpus_ref_len: Optional[float] = None

        tag = "[RewardComputer]"
        if self._ptb is not None:
            print(f"{tag} PTBTokenizer ready (identical to eval script)")
        else:
            print(f"{tag} PTBTokenizer unavailable, using simple tokenizer")
        if self._meteor is not None:
            print(f"{tag} Java METEOR ready (identical to eval script)")
        else:
            print(f"{tag} Java METEOR unavailable, using simplified fallback")

    # ---- internal tokenisation helpers ----

    @staticmethod
    def _fmt_ptb(text_dict: Dict[int, List[str]]) -> Dict:
        """Format for PTBTokenizer: auto-adapt to dict or string input."""
        if _PTB_WANTS_DICT:
            return {k: [{"caption": s} for s in v] for k, v in text_dict.items()}
        return text_dict

    def _tok(self, text_dict: Dict[int, List[str]]) -> Dict[int, List[str]]:
        """Tokenize {id: [str, ...]} -> {id: [str, ...]}"""
        if self._ptb is not None:
            return self._ptb.tokenize(self._fmt_ptb(text_dict))
        return {k: [_tokenize(s) for s in v] for k, v in text_dict.items()}

    # ---- CIDEr IDF pre-computation ----

    def build_cider_idf(self, multiref_map: Dict[int, List[str]]) -> None:
        """
        Pre-compute corpus-level IDF for CIDEr-D.

        Each trajectory in *multiref_map* is treated as one document
        (with possibly multiple reference captions), exactly matching the
        convention used in pycocoevalcap's standard CIDEr evaluation.
        """
        refs_raw = {tid: list(refs) for tid, refs in multiref_map.items()}
        refs_tok = self._tok(refs_raw)

        scorer = CiderScorer(n=4, sigma=6.0)
        for tid in refs_tok:
            scorer += ("dummy", refs_tok[tid])
        scorer.compute_doc_freq()
        self._corpus_doc_freq = dict(scorer.document_frequency)
        self._corpus_ref_len = np.log(float(max(len(scorer.crefs), 1)))
        print(
            f"[RewardComputer] CIDEr-D IDF built: "
            f"{len(self._corpus_doc_freq)} n-grams, "
            f"ref_len={self._corpus_ref_len:.4f} "
            f"({len(multiref_map)} trajectories)"
        )

    # ---- Batch reward computation ----

    def compute_rewards(
        self,
        completions: List[str],
        references: List[str],
        weights: Dict[str, float],
        apply_length_penalty: bool = False,
        target_min_words: int = 25,
        target_max_words: int = 40,
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """
        Compute per-sample rewards for *G* completions against *references*.

        Args:
            completions: G generated strings.
            references:  list of reference strings (multi-ref supported).
            weights:     metric name -> weight.

        Returns:
            (list[float] of G rewards, list[dict] of G metric dicts)
        """
        G = len(completions)
        if G == 0:
            return [], []

        gts_raw = {i: list(references) for i in range(G)}
        res_raw = {i: [completions[i]] for i in range(G)}

        gts_tok = self._tok(gts_raw)
        res_tok = self._tok(res_raw)

        # ---- BLEU ----
        with redirect_stdout(io.StringIO()):
            _, bleu_per = Bleu(4).compute_score(gts_tok, res_tok)

        # ---- ROUGE-L ----
        _, rouge_per = Rouge().compute_score(gts_tok, res_tok)

        # ---- METEOR ----
        if self._meteor is not None:
            try:
                _, meteor_per = self._meteor.compute_score(gts_tok, res_tok)
            except Exception:
                meteor_per = [
                    _meteor_simple(res_tok[i][0], gts_tok[i])
                    for i in range(G)
                ]
        else:
            meteor_per = [
                _meteor_simple(res_tok[i][0], gts_tok[i])
                for i in range(G)
            ]

        # ---- CIDEr-D ----
        if self._corpus_doc_freq is not None:
            cider = CorpusCiderScorer(
                self._corpus_doc_freq, self._corpus_ref_len, n=4, sigma=6.0,
            )
        else:
            cider = CiderScorer(n=4, sigma=6.0)
        for i in range(G):
            cider += (res_tok[i][0], gts_tok[i])
        _, cider_per = cider.compute_score()

        # ---- Assemble ----
        all_rewards: List[float] = []
        all_metrics: List[Dict[str, float]] = []
        for i in range(G):
            metrics = {
                "BLEU-1": float(bleu_per[0][i]),
                "BLEU-4": float(bleu_per[3][i]),
                "CIDEr": float(cider_per[i]),
                "METEOR": float(meteor_per[i]),
                "ROUGE-L": float(rouge_per[i]),
            }
            total = sum(weights.get(k, 0.0) * v for k, v in metrics.items())

            if apply_length_penalty:
                lp = length_penalty(len(completions[i].split()),
                                    target_min_words, target_max_words)
                metrics["len_penalty"] = lp
                total += lp

            all_rewards.append(total)
            all_metrics.append(metrics)

        return all_rewards, all_metrics
