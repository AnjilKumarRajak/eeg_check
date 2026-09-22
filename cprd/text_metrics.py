"""Paper-grade text metrics, computed identically on every arm.

Families (7): BLEU-1..4 (sacrebleu, corpus), chrF, TER, METEOR (nltk), BERTScore-F1,
ROUGE-1/2/L-F, WER/CER (jiwer). Every metric ships with its own collapse floor because
the harness runs it on real/zeroed/gamma_zero arms alike — a metric without its floor
is not reportable.

Robustness contract: each metric is computed in its own try/except; unavailable
backends yield None (recorded as such), never a crash and never a silent hand-rolled
substitute. `light=True` (smoke/CPU) skips the heavy model-based metrics (BERTScore,
METEOR's wordnet path) so the pipeline stays runnable anywhere.
"""
from __future__ import annotations

import os
from typing import Optional


def _try(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def _sacre(hyps, refs):
    from sacrebleu.metrics import BLEU, CHRF, TER
    out = {}
    for n in (1, 2, 3, 4):
        out[f"bleu{n}"] = float(BLEU(max_ngram_order=n, effective_order=True)
                                .corpus_score(hyps, [refs]).score)
    out["chrf"] = float(CHRF().corpus_score(hyps, [refs]).score)
    out["ter"] = float(TER().corpus_score(hyps, [refs]).score)
    return out


def _rouge(hyps, refs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for h, r in zip(hyps, refs):
        s = sc.score(r, h)
        for k in agg:
            agg[k].append(s[k].fmeasure)
    return {k: float(sum(v) / max(1, len(v))) for k, v in agg.items()}


def _meteor(hyps, refs):
    import nltk
    from nltk.translate.meteor_score import meteor_score
    # wordnet is fetched in run_pipeline setup; this fallback keeps headless boxes alive
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    scores = [meteor_score([r.split()], h.split()) for h, r in zip(hyps, refs)]
    return {"meteor": float(sum(scores) / max(1, len(scores)))}


def _bertscore(hyps, refs, device: str = "cpu"):
    from bert_score import score as bs
    P, R, F = bs(hyps, refs, lang="en", rescale_with_baseline=False,
                 device=device, verbose=False)
    return {"bertscore_p": float(P.mean()), "bertscore_r": float(R.mean()),
            "bertscore_f1": float(F.mean())}


def _wer_cer(hyps, refs):
    import jiwer
    return {"wer": float(jiwer.wer(refs, hyps)), "cer": float(jiwer.cer(refs, hyps))}


def compute_all(hyps: list, refs: list, light: bool = False,
                device: str = "cpu") -> dict:
    """All families for one (hypotheses, references) pair. None-safe, never raises."""
    if not hyps or len(hyps) != len(refs):
        return {"_error": f"bad inputs: {len(hyps)} hyps vs {len(refs)} refs"}
    out = {"n": len(hyps)}
    for name, fn, extra in (("sacrebleu", _sacre, ()), ("rouge", _rouge, ()),
                            ("wer_cer", _wer_cer, ())):
        r = _try(fn, hyps, refs, *extra)
        if isinstance(r, dict) and "_error" in r:
            out[f"{name}_unavailable"] = r["_error"]
        else:
            out.update(r or {})
    if not light:
        for name, fn, extra in (("meteor", _meteor, ()),
                                ("bertscore", _bertscore, (device,))):
            r = _try(fn, hyps, refs, *extra)
            if isinstance(r, dict) and "_error" in r:
                out[f"{name}_unavailable"] = r["_error"]
            else:
                out.update(r or {})
    # collapse nested error dicts into flags
    clean = {}
    for k, v in out.items():
        if isinstance(v, dict) and "_error" in v:
            clean[f"{k}_unavailable"] = v["_error"]
        else:
            clean[k] = v
    return clean


PAPER_METRICS = ["bleu1", "bleu2", "bleu3", "bleu4", "chrf", "ter",
                 "meteor", "bertscore_f1", "rouge1", "rouge2", "rougeL",
                 "wer", "cer"]
