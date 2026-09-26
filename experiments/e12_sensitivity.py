#!/usr/bin/env python3
"""E12: evaluation-time sensitivity of the channel bound (eval only; no new training, no test access).

Re-scores the trained E2 checkpoint on the measurement half under (i) different evaluation gains
gamma_eval (the trained checkpoint is evaluated with a FIXED gain; the default is softplus(c)=0.127
because the gate parameters are never trained), (ii) rescaled evidence norm ||u_t||, and reports for
each setting: the per-token InfoNCE bound (+ text-clustered CI, exact permutation p), the bound as a
function of the number of negatives M' <= M (tightness/ceiling), and the SENTENCE-LEVEL InfoNCE
bound, whose score is S(y) = sum_t T_t (a proper conditional density ratio q(w|e)/p0(w)), with the
same M negatives -> a consolidated bits/sentence lower bound (<= log2(M+1)).

    python experiments/e12_sensitivity.py <E2 args...> --m-draws 300 --gamma-grid 0.5,1,2 --uscale-grid 0.5,2
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (apply_determinism, base_parser, build_prior, load_split_sentences,  # noqa: E402
                    prepare_batches, train_model)
from cprd.data import selection_measurement_split                                   # noqa: E402
from cprd.evaluate import _forward_tokens                                          # noqa: E402
from cprd.prereg import write_gate_artifact                                        # noqa: E402

LN2 = math.log(2.0)


def boot(vals, clusters, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=vals, minlength=len(uniq))
    cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        out[b] = sums[pick].sum() / cnt[pick].sum()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = base_parser("E12: gamma_eval / norm / negatives / sentence-level sensitivity")
    ap.add_argument("--m-draws", type=int, default=300)
    ap.add_argument("--gamma-grid", default="0.5,1,2")
    ap.add_argument("--uscale-grid", default="0.5,2")
    ap.add_argument("--m-subsets", default="10,30,100")
    ap.add_argument("--tag", default="sensitivity")
    ap.add_argument("--half", default="measurement", choices=["measurement", "selection"],
                    help="which validation half to score (selection = used ONLY to choose gamma_eval, never reported as a result)")
    args = ap.parse_args()
    apply_determinism(args)
    tr, _ = load_split_sentences(args, "train")
    va, fp = load_split_sentences(args, "val")
    prior = build_prior(args)
    model = train_model(prior, tr, va, args)
    va_sel_h, va_meas = selection_measurement_split(va)
    if args.half == "selection":
        va_meas = va_sel_h
    batches, lp0 = prepare_batches(va_meas, prior, args.device, window=args.window, batch_size=args.batch_size)
    model.eval()
    M = args.m_draws
    subsets = [int(x) for x in args.m_subsets.split(",") if x] + [M]
    orig_mode, orig_const = model.tilt.gamma_mode, model.tilt.gamma_const
    settings = [("default", None)] + [("gamma", float(g)) for g in args.gamma_grid.split(",") if g] \
        + [("uscale", float(s)) for s in args.uscale_grid.split(",") if s]
    orig_ev = model.evidence
    results = {}
    for kind, val in settings:
        model.tilt.gamma_mode, model.tilt.gamma_const = orig_mode, orig_const
        model.evidence = orig_ev
        label = "default" if kind == "default" else f"{kind}={val:g}"
        if kind == "gamma":
            model.tilt.gamma_mode, model.tilt.gamma_const = "constant", val
        elif kind == "uscale":
            model.evidence = (lambda s: (lambda *a, **k: s * orig_ev(*a, **k)))(val)
        t0 = time.time()
        dI, obs, texts, sid = _forward_tokens(model, batches, lp0)
        _, inv = np.unique(sid, return_inverse=True)
        n_s = inv.max() + 1
        S_real = np.bincount(inv, weights=dI.astype(np.float64), minlength=n_s)
        sent_text = np.array([texts[np.flatnonzero(inv == i)[0]] for i in range(n_s)])
        T_real = dI[obs].astype(np.float64)
        T_null = np.empty((M, int(obs.sum())), dtype=np.float32)
        S_null = np.empty((M, n_s), dtype=np.float64)
        for k in range(M):
            d, _, _, _ = _forward_tokens(model, batches, lp0, null="sentence_derangement", seed=args.seed + 1000 * k)
            T_null[k] = d[obs]
            S_null[k] = np.bincount(inv, weights=d.astype(np.float64), minlength=n_s)
            if (k + 1) % 50 == 0:
                print(f"  [{label}] draw {k + 1}/{M}  ({time.time() - t0:.0f}s)", flush=True)

        def nce_tok(mp):
            lse = np.logaddexp.reduce(T_null[:mp].astype(np.float64), axis=0)
            return T_real - (np.logaddexp(T_real, lse) - np.log(mp + 1))

        I_tok = nce_tok(M)
        lo, hi = boot(I_tok, texts[obs])
        draws = T_null.astype(np.float64).mean(1)
        p = float((1 + np.sum(draws >= T_real.mean())) / (1 + M))
        by_m = {str(mp): float(nce_tok(mp).mean() / LN2) for mp in subsets}
        lse_s = np.logaddexp.reduce(S_null, axis=0)
        I_sent = S_real - (np.logaddexp(S_real, lse_s) - np.log(M + 1))
        slo, shi = boot(I_sent, sent_text)
        mean_obs = float(obs.sum() / n_s)
        results[label] = {
            "bits_per_token": float(I_tok.mean() / LN2), "ci95": [lo / LN2, hi / LN2], "p_derangement": p,
            "bits_per_token_by_num_negatives": by_m,
            "sentence_level_bits": float(I_sent.mean() / LN2), "sentence_level_ci95": [slo / LN2, shi / LN2],
            "sentence_level_ceiling_bits": float(np.log2(M + 1)),
            "sum_of_token_bounds_bits_per_sentence": float(I_tok.mean() / LN2 * mean_obs),
            "mean_observed_tokens": mean_obs, "n_sentences": int(n_s), "n_observed_tokens": int(obs.sum()),
            "mean_u_norm_note": "see model config s_max",
        }
        print(f"== {label}: token {results[label]['bits_per_token']:+.4f} bits [{lo / LN2:+.4f},{hi / LN2:+.4f}] p={p:.4f} | "
              f"sentence-level {results[label]['sentence_level_bits']:+.3f} bits [{slo / LN2:+.3f},{shi / LN2:+.3f}] "
              f"(ceiling {np.log2(M + 1):.2f}) | by M: {by_m}", flush=True)
    model.tilt.gamma_mode, model.tilt.gamma_const = orig_mode, orig_const
    model.evidence = orig_ev
    default_gamma = float(torch.nn.functional.softplus(model.tilt.c).item())
    write_gate_artifact(args.runs_dir, args.tag, True, {
        "settings": results, "default_eval_gamma": default_gamma, "gamma_mode_after_training": orig_mode,
        "m_draws": M, "evidence": args.evidence, "split_fingerprint": fp,
        "source": "eval-only re-scoring of the E2 checkpoint on the measurement half"})
    print("gate_%s written" % args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
