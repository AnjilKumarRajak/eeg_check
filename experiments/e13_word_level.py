#!/usr/bin/env python3
"""E13: word-level view of the E2 bound (reviewer item: word evidence is repeated over GPT-2 sub-tokens).

Same re-analysis machinery as E10 (stored per-token permutation draws, no new training or test access). Reports the bound at three units: per token (headline), per WORD (sum of the sub-token bits of a word, i.e. one unit per piece of evidence), and split by word length in tokens (one-token words vs multi-token words; first sub-token vs continuation tokens). Text-clustered bootstrap CIs.

E10 header follows.

Re-analysis of the E2 measurement (no new test access, no new training): the E2 stage
stored, for every observed token of the validation measurement half and each null, the
running log-sum-exp of the per-token log-likelihood ratio over its M permutation draws
(eval_ckpt/<tag>_<null>_lsetok.npy). Re-running the real forward pass in the same order
gives T_t(real); the per-token InfoNCE bound is then

    I_t = T_t - log( (e^{T_t} + sum_m e^{T_t^{(m)}}) / (M+1) )

and can be aggregated over any subset of tokens: by ZuCo task (SR / NR / TSR / NR-2.0),
by subject, and by decile of the prior's surprisal -log p0(w_t) (reviewer items 14, 29, 30).
Uncertainty: bootstrap over unique-text clusters within each subset.

    python experiments/e10_breakdown.py <E2 args...>   (same flags as the E2 stage)
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (apply_determinism, base_parser, base_record, build_prior,  # noqa: E402
                    ledger_for, load_split_sentences, prepare_batches, train_model)
from cprd.audit import Record                                              # noqa: E402
from cprd.data import selection_measurement_split                          # noqa: E402
from cprd.evaluate import _forward_tokens                                  # noqa: E402
from cprd.prereg import write_gate_artifact                                # noqa: E402

LN2 = math.log(2.0)


def cluster_bootstrap(vals, clusters, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    if len(uniq) < 2:
        return (float("nan"), float("nan"))
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[c] for c in pick])
        means[b] = vals[sel].mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    ap = base_parser("E13: word-level / one-token-vs-multi-token view of the E2 bound")
    args = ap.parse_args()
    apply_determinism(args)
    tr, _ = load_split_sentences(args, "train")
    va, fp = load_split_sentences(args, "val")
    prior = build_prior(args)
    model = train_model(prior, tr, va, args)
    _, va_meas = selection_measurement_split(va)
    batches, lp0 = prepare_batches(va_meas, prior, args.device, window=args.window, batch_size=args.batch_size)
    model.eval()
    dI, obs, texts, sent_id = _forward_tokens(model, batches, lp0)
    T = dI[obs].astype(np.float64)
    tag = f"e2_channel_{model.ckpt_key[:16]}_s{args.seed}"
    ck = os.path.join(args.runs_dir, "eval_ckpt")
    lse = {}
    for nm in ("sentence_derangement", "temporal_roll"):
        arr = np.load(os.path.join(ck, f"{tag}_{nm}_lsetok.npy"))
        assert len(arr) == len(T), f"{nm}: {len(arr)} stored tokens vs {len(T)} observed now"
        lse[nm] = arr
    M = args.n_perm
    I_tok = {nm: (T - (np.logaddexp(T, lse[nm]) - np.log(M + 1))) / LN2 for nm in lse}

    # token -> (global word id, position in word, word length in tokens, text); same order as the batches
    wid, pos, wlen, txt = [], [], [], []
    gid = 0
    for bi, b in enumerate(batches):
        ids = b["token_ids"].cpu()
        for i in range(ids.shape[0]):
            s = va_meas[bi * args.batch_size + i]
            n = int(b["valid"][i].sum())
            wi = s.word_index[:n] if s.word_index is not None else np.arange(n)
            full = np.bincount(s.word_index) if s.word_index is not None else np.ones(n, dtype=int)
            seen = {}
            for t in range(n):
                w = int(wi[t]); p = seen.get(w, 0); seen[w] = p + 1
                wid.append(gid + w); pos.append(p); wlen.append(int(full[w])); txt.append(s.text)
            gid += (int(s.word_index.max()) + 1) if s.word_index is not None else n
    wid = np.array(wid)[obs]; pos = np.array(pos)[obs]; wlen = np.array(wlen)[obs]; txt = np.array(txt)[obs]
    assert len(wid) == len(T)
    out = {}
    for nm, v in I_tok.items():
        uw, inv = np.unique(wid, return_inverse=True)
        wbits = np.bincount(inv, weights=v)                 # bits per word (sum over its observed sub-tokens)
        wtext = np.array([txt[np.flatnonzero(inv == k)[0]] for k in range(len(uw))]) if len(uw) < 1 else None
        first_idx = np.zeros(len(uw), dtype=int); first_idx[inv[::-1]] = np.arange(len(inv))[::-1]
        wtext = txt[first_idx]; wl = wlen[first_idx]; wtok = np.bincount(inv)
        def rec(mask_tok, mask_word, label):
            d = {"n_tokens": int(mask_tok.sum()), "n_words": int(mask_word.sum())}
            lo, hi = cluster_bootstrap(v[mask_tok], txt[mask_tok], seed=args.seed)
            d["bits_per_token"] = float(v[mask_tok].mean()); d["bits_per_token_ci95"] = [lo, hi]
            lo, hi = cluster_bootstrap(wbits[mask_word], wtext[mask_word], seed=args.seed)
            d["bits_per_word"] = float(wbits[mask_word].mean()); d["bits_per_word_ci95"] = [lo, hi]
            d["mean_tokens_per_word"] = float(wtok[mask_word].mean())
            if nm == "sentence_derangement":
                print(f"  {label:<32} words={d['n_words']:>6} tokens={d['n_tokens']:>6}  "
                      f"bits/token={d['bits_per_token']:+.4f}  bits/word={d['bits_per_word']:+.4f} "
                      f"[{lo:+.4f},{hi:+.4f}]  tok/word={d['mean_tokens_per_word']:.2f}", flush=True)
            return d
        allt = np.ones(len(T), dtype=bool); allw = np.ones(len(uw), dtype=bool)
        r = {"all": rec(allt, allw, "all words"),
             "one_token_words": rec(wlen == 1, wl == 1, "one-token words"),
             "multi_token_words": rec(wlen > 1, wl > 1, "multi-token words")}
        for k, (lo_, hi_) in {"first_subtoken": (0, 0), "continuation_subtokens": (1, 999)}.items():
            m = (pos >= lo_) & (pos <= hi_) & (wlen > 1)
            lo, hi = cluster_bootstrap(v[m], txt[m], seed=args.seed)
            r[k] = {"n_tokens": int(m.sum()), "bits_per_token": float(v[m].mean()), "bits_per_token_ci95": [lo, hi]}
            if nm == "sentence_derangement":
                print(f"  {k + ' (multi-token words)':<32} tokens={int(m.sum()):>6}  bits/token={v[m].mean():+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)
        out[nm] = r
    detail = {"by_null": out, "n_perm": M, "evidence": args.evidence, "window": args.window,
              "source": "re-analysis of E2 per-token draws (no new training or test access)"}
    write_gate_artifact(args.runs_dir, "wordlevel", True, detail)
    print("gate_wordlevel written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
