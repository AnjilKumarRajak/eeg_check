
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wolpaw_bits_per_selection(N: int, acc: float) -> float:
    """B = log2 N + P log2 P + (1-P) log2((1-P)/(N-1)); 0 at chance and below."""
    if N < 2:
        return 0.0
    P = min(max(acc, 1e-12), 1 - 1e-12)
    if P <= 1.0 / N:
        return 0.0
    B = math.log2(N) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (N - 1))
    return max(0.0, B)


def nykopp_bits_per_selection(confusion: np.ndarray) -> float:
    C = np.asarray(confusion, dtype=np.float64)
    n = C.sum()
    if n <= 0:
        return 0.0
    P = C / n
    px = P.sum(axis=1, keepdims=True)
    py = P.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(P > 0, P / (px @ py), 1.0)
        I = float(np.nansum(np.where(P > 0, P * np.log2(ratio), 0.0)))
    return max(0.0, I)


def hit_confusion(per_pool_hit: list, N: int) -> np.ndarray:
    hits = float(np.sum(per_pool_hit))
    misses = float(len(per_pool_hit)) - hits
    C = np.zeros((N, N))
    np.fill_diagonal(C, hits / N)
    if N > 1:
        off = misses / (N * (N - 1))
        C += off
        np.fill_diagonal(C, hits / N)   # restore diagonal after the += broadcast
    return C


@dataclass
class ITRRow:
    N: int
    accuracy: float
    wolpaw_bits: float
    nykopp_bits: float
    denominators_s: dict            # name -> seconds per selection
    itr_bits_per_min: dict          # (formula, denominator) -> bits/min

    def describe(self) -> str:
        lines = [f"  N={self.N}  acc={self.accuracy:.3f}  "
                 f"wolpaw={self.wolpaw_bits:.3f} b/sel  nykopp={self.nykopp_bits:.3f} b/sel"]
        for k, v in sorted(self.itr_bits_per_min.items()):
            lines.append(f"    {k[0]:>7} / {k[1]:<8} {v:7.2f} bits/min")
        return "\n".join(lines)


def itr_row(N: int, per_pool_hit: list, denominators_s: dict) -> ITRRow:
    acc = float(np.mean(per_pool_hit)) if per_pool_hit else 0.0
    wb = wolpaw_bits_per_selection(N, acc)
    nb = (nykopp_bits_per_selection(hit_confusion(per_pool_hit, N))
          if acc > 1.0 / max(1, N) else 0.0)
    itr = {}
    for name, sec in denominators_s.items():
        if sec and sec > 0:
            itr[("wolpaw", name)] = 60.0 * wb / sec
            itr[("nykopp", name)] = 60.0 * nb / sec
    return ITRRow(N=N, accuracy=acc, wolpaw_bits=wb, nykopp_bits=nb,
                  denominators_s=denominators_s, itr_bits_per_min=itr)


def reading_time_denominators(covariates=None, texts: list | None = None,
                              n_words_by_text: dict | None = None,
                              trial_overhead_s: float = 2.0,
                              wpm: float = 240.0) -> dict:

    out = {}
    if covariates is not None and texts:
        per_text = []
        for t in texts:
            th = None
            # covariates keyed by (subject, text_hash, word_idx); aggregate over any subject
            from .covariates import text_hash as _th
            th = _th(t)
            # sum TRT over words PER SUBJECT (one reading), then the median reading
            # across subjects. Summing over all subjects (the old code) made T1 ~12-18x
            # too long and every T1/T2 ITR ~12-18x too small.
            per_subj: dict = {}
            for (subj, h, wi), r in covariates.rows.items():
                if h == th and np.isfinite(r["trt"]):
                    per_subj[subj] = per_subj.get(subj, 0.0) + float(r["trt"])
            if per_subj:
                per_text.append(float(np.median(list(per_subj.values()))) / 500.0)  # samples@500Hz -> s
        if per_text:
            t1 = float(np.median(per_text))
            out["T1_reading"] = t1
            out["T2_trial"] = t1 + trial_overhead_s
    if n_words_by_text and texts:
        words = [n_words_by_text[t] for t in texts if t in n_words_by_text]
        if words:
            out["T3_nominal"] = float(np.median(words)) / (wpm / 60.0)
    return out
