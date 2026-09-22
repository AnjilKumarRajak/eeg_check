
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
    ap = base_parser("E10: per-task / per-subject / per-surprisal breakdown of the E2 bound")
    ap.add_argument("--n-decile", type=int, default=10)
    args = ap.parse_args()
    apply_determinism(args)

    tr, _ = load_split_sentences(args, "train")
    va, fp = load_split_sentences(args, "val")
    prior = build_prior(args)
    model = train_model(prior, tr, va, args)          # reuses the E2 checkpoint (same key)
    _, va_meas = selection_measurement_split(va)
    batches, lp0 = prepare_batches(va_meas, prior, args.device,
                                   window=args.window, batch_size=args.batch_size)
    model.eval()
    dI, obs, texts, sent_id = _forward_tokens(model, batches, lp0)      # all tokens, in order
    T = dI[obs].astype(np.float64)
    tag = f"e2_channel_{model.ckpt_key[:16]}_s{args.seed}"
    ck = os.path.join(args.runs_dir, "eval_ckpt")
    lse = {}
    for nm in ("sentence_derangement", "temporal_roll"):
        p = os.path.join(ck, f"{tag}_{nm}_lsetok.npy")
        arr = np.load(p)
        assert len(arr) == len(T), f"{nm}: {len(arr)} stored tokens vs {len(T)} observed now"
        lse[nm] = arr
    M = args.n_perm
    I_tok = {nm: (T - (np.logaddexp(T, lse[nm]) - np.log(M + 1))) / LN2 for nm in lse}

    # token -> sentence metadata (same order as the batches: batch-major, then sentence)
    meta_task, meta_subj, meta_text, surpr = [], [], [], []
    dev = next(model.parameters()).device
    for bi, b in enumerate(batches):
        lp = lp0[bi] if (lp0 is not None and lp0[bi] is not None) else prior.log_probs(b["token_ids"].to(dev)).cpu()
        ids = b["token_ids"].cpu()
        for i in range(ids.shape[0]):
            s = va_meas[bi * args.batch_size + i]
            n = int(b["valid"][i].sum())
            g = lp[i, torch.arange(n), ids[i, :n]].numpy()
            for t in range(n):
                meta_task.append(s.task); meta_subj.append(s.subject_id); meta_text.append(s.text)
                surpr.append(-float(g[t]))
    meta_task = np.array(meta_task)[obs]; meta_subj = np.array(meta_subj)[obs]
    meta_text = np.array(meta_text)[obs]; surpr = np.array(surpr)[obs]
    assert len(meta_task) == len(T)

    def agg(mask, label):
        out = {"n_tokens": int(mask.sum()), "n_texts": int(len(np.unique(meta_text[mask])))}
        for nm, v in I_tok.items():
            lo, hi = cluster_bootstrap(v[mask], meta_text[mask], seed=args.seed)
            out[nm] = {"bits_per_token": float(v[mask].mean()), "ci95": [lo, hi]}
        print(f"  {label:<28} n={out['n_tokens']:>6} texts={out['n_texts']:>3}  "
              f"der={out['sentence_derangement']['bits_per_token']:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)
        return out

    print("== overall (headline bound with text-clustered bootstrap CI)")
    overall = agg(np.ones(len(T), dtype=bool), "all measured tokens")
    print("== by task")
    by_task = {t: agg(meta_task == t, f"task {t}") for t in sorted(set(meta_task))}
    print("== by subject")
    by_subj = {s: agg(meta_subj == s, f"subject {s}") for s in sorted(set(meta_subj))}
    print("== by prior-surprisal decile (-log p0(w_t), nats)")
    edges = np.quantile(surpr, np.linspace(0, 1, args.n_decile + 1))
    by_dec = {}
    for k in range(args.n_decile):
        m = (surpr >= edges[k]) & (surpr <= edges[k + 1]) if k == args.n_decile - 1 else (surpr >= edges[k]) & (surpr < edges[k + 1])
        d = agg(m, f"decile {k + 1} [{edges[k]:.1f},{edges[k + 1]:.1f})")
        d["surprisal_range_nats"] = [float(edges[k]), float(edges[k + 1])]
        d["mean_surprisal_nats"] = float(surpr[m].mean())
        by_dec[str(k + 1)] = d
    # is the bound a monotone function of surprisal?  Spearman over deciles
    from scipy.stats import spearmanr, pearsonr
    xs = [by_dec[str(k + 1)]["mean_surprisal_nats"] for k in range(args.n_decile)]
    ys = [by_dec[str(k + 1)]["sentence_derangement"]["bits_per_token"] for k in range(args.n_decile)]
    rho, prho = spearmanr(xs, ys)
    r_tok, p_tok = pearsonr(surpr, I_tok["sentence_derangement"])
    print(f"  decile-level Spearman rho={rho:+.3f} (p={prho:.3f}); token-level Pearson r={r_tok:+.4f} (p={p_tok:.2e})")

    detail = {"overall_bits_per_token": {nm: float(v.mean()) for nm, v in I_tok.items()},
              "overall_with_ci": overall,
              "by_task": by_task, "by_subject": by_subj, "by_surprisal_decile": by_dec,
              "surprisal_relation": {"spearman_rho_deciles": float(rho), "p": float(prho),
                                     "pearson_r_tokens": float(r_tok), "p_tokens": float(p_tok)},
              "n_perm": M, "evidence": args.evidence, "source": "re-analysis of E2 per-token draws (no new training or test access)"}
    write_gate_artifact(args.runs_dir, "breakdown", True, detail)
    led = ledger_for(args)
    base = base_record(args, split_fp=f"{fp}:measurement_half", experiment="e10_breakdown", gates=["build", "estimator", "channel"])
    for t, d in by_task.items():
        led.append(Record(metric="dI_hat_nce_bits_per_token_by_task", value=d["sentence_derangement"]["bits_per_token"],
                          ci_lo=d["sentence_derangement"]["ci95"][0], ci_hi=d["sentence_derangement"]["ci95"][1],
                          n_clusters=d["n_texts"], arm=f"task={t}", **base))
    for s_, d in by_subj.items():
        led.append(Record(metric="dI_hat_nce_bits_per_token_by_subject", value=d["sentence_derangement"]["bits_per_token"],
                          ci_lo=d["sentence_derangement"]["ci95"][0], ci_hi=d["sentence_derangement"]["ci95"][1],
                          n_clusters=d["n_texts"], arm=f"subject={s_}", **base))
    for k, d in by_dec.items():
        led.append(Record(metric="dI_hat_nce_bits_per_token_by_surprisal_decile", value=d["sentence_derangement"]["bits_per_token"],
                          ci_lo=d["sentence_derangement"]["ci95"][0], ci_hi=d["sentence_derangement"]["ci95"][1],
                          n_clusters=d["n_texts"], arm=f"decile={k}", **base))
    print("gate_breakdown written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
