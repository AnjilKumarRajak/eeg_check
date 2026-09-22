
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (read_gate_optional, apply_determinism, base_parser, base_record, build_prior, ledger_for,     # noqa: E402
                    load_split_sentences, train_model)
from cprd.abstain import (cheap_confidence, confidence_auc,                # noqa: E402
                          partial_confidence_auc, risk_coverage)
from cprd.audit import Record, sha256_of, code_git_rev                     # noqa: E402
from cprd.covariates import CovariateTable, text_hash                      # noqa: E402
from cprd.data import dataset_fingerprint                                  # noqa: E402
from cprd.itr import itr_row, reading_time_denominators                    # noqa: E402
from cprd.pools import build_corpus_pools, mean_prior_logprob, pool_validity_gate  # noqa: E402
from cprd.prereg import (GuardedTestAccess, check_gate_artifact,           # noqa: E402
                         write_gate_artifact, write_prereg)
from cprd.selector import (fano_bits, prepare_eeg_lookup, run_selection,   # noqa: E402
                           score_candidate)
from cprd.stats import tost_equivalence                                    # noqa: E402
from cprd.text_metrics import compute_all                                  # noqa: E402

LN2 = math.log(2.0)


def transformed_lookup(lookup: dict, kind: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    texts = list(lookup.keys())
    if kind == "zeroed":
        return {t: (torch.zeros_like(w), o, p, ob) for t, (w, o, p, ob) in lookup.items()}
    if kind == "derangement":
        perm = rng.permutation(len(texts))
        for i in range(len(texts)):               # no fixed points
            if perm[i] == i:
                j = (i + 1) % len(texts)
                perm[i], perm[j] = perm[j], perm[i]
        return {texts[i]: lookup[texts[int(perm[i])]] for i in range(len(texts))}
    if kind == "gaussian_matched":
        obs_vals = torch.cat([w[o] for (w, o, p, ob) in lookup.values() if bool(o.any())])
        mu, sd = obs_vals.mean(0), obs_vals.std(0).clamp(min=1e-8)
        out = {}
        for t, (w, o, p, ob) in lookup.items():
            # stable across processes (builtin hash() is salted by PYTHONHASHSEED)
            th = int(sha256_of(t)[:8], 16) % 100000
            g = torch.Generator().manual_seed(seed + th)
            out[t] = (mu + sd * torch.randn(w.shape, generator=g), o, p, ob)
        return out
    if kind == "amplitude_only":
        out = {}
        for t, (w, o, p, ob) in lookup.items():
            mean_vec = w[o].mean(0) if bool(o.any()) else torch.zeros(w.shape[-1])
            out[t] = (mean_vec.expand_as(w).contiguous(), o, p, ob)
        return out
    if kind == "position_only":
        out = {}
        for t, (w, o, p, ob) in lookup.items():
            T, W, F = w.shape
            pos = torch.arange(T, dtype=w.dtype)
            enc = torch.zeros(T, F, dtype=w.dtype)
            half = min(F, 64) // 2
            for k in range(half):
                enc[:, 2 * k] = torch.sin(pos / (10000 ** (k / max(1, half))))
                enc[:, 2 * k + 1] = torch.cos(pos / (10000 ** (k / max(1, half))))
            out[t] = (enc.unsqueeze(1).expand(T, W, F).contiguous(), o, p, ob)
        return out
    raise ValueError(kind)


def main():
    ap = base_parser("system evaluation: guarded matched-N + full arm battery")
    ap.add_argument("--trial-overhead-s", type=float, default=2.0)
    ap.add_argument("--attr-beta", type=float, default=0.0,
                    help="attribute-fusion weight from e3 val tuning (0 = off/ablation)")
    ap.add_argument("--tost-margin", type=float, default=0.02,
                    help="H5 equivalence margin (pre-registered +-2pp)")
    ap.add_argument("--light-metrics", action="store_true",
                    help="skip BERTScore/METEOR (smoke/CPU)")
    args = ap.parse_args()
    apply_determinism(args)
    check_gate_artifact(args.runs_dir, "build")
    read_gate_optional(args.runs_dir, "estimator")
    gate_ch = check_gate_artifact(args.runs_dir, "channel")
    gate_sel = check_gate_artifact(args.runs_dir, "selection")
    n_star = int(gate_sel["detail"]["matched_N_prediction"])
    bits_sent = float(gate_ch["detail"]["bits_per_sentence"])
    bits_sel_val = gate_sel["detail"].get("bits_realized_val")   # honest predictor if present
    n_star_meets = gate_sel["detail"].get("matched_N_meets_target")
    sfx = "" if args.seed == 0 else f"_seed{args.seed}"

    # ---- freeze prereg BEFORE test access ---------------------------------
    prereg_dir = os.path.join(args.runs_dir, "prereg" if not sfx else f"prereg/seed{args.seed}")
    data_fps = {s: dataset_fingerprint(os.path.abspath(
        os.path.join(args.data_dir, f"zuco2_{s}.h5"))) for s in ("train", "val", "test")}
    dev_snapshot = []
    dev_log = os.path.join(prereg_dir, "deviations.log")
    if os.path.exists(dev_log):
        dev_snapshot = open(dev_log).read().splitlines()
    prereg_sha = write_prereg(prereg_dir, {
        "config_sha": sha256_of(vars(args)),
        "split_fingerprints": data_fps,
        "prior_ids": [f"{args.prior}:{args.prior_model}"],
        "matched_N_prediction": n_star,
        "matched_N_meets_target": n_star_meets,
        "bits_per_sentence_val_dIxTokens": bits_sent,
        "bits_per_sentence_val_selection": bits_sel_val,
        "deviations_at_freeze": dev_snapshot,
        "code_rev": code_git_rev(),
        "seeds": [args.seed], "n_perm": args.n_perm, "n_boot": args.n_boot,
    })
    print(f"prereg frozen (sha {prereg_sha}); N* = {n_star}")

    # ---- train on train, select on val (never test) ------------------------
    tr, _ = load_split_sentences(args, "train")
    va, _ = load_split_sentences(args, "val")
    prior = build_prior(args)
    model = train_model(prior, tr, va, args)

    attr_head = None
    if args.attr_beta != 0.0:
        from cprd.attr_head import AttributeHead
        attr_head = AttributeHead().fit(tr)
        print(f"attribute head fused with beta={args.attr_beta} "
              f"(classes: {attr_head.classes_})")

    # ---- the ONE guarded test session --------------------------------------
    guard = GuardedTestAccess(os.path.abspath(
        os.path.join(args.data_dir, "zuco2_test.h5")), prereg_dir)
    te = guard.load(caller="e4_system", config_sha=sha256_of(vars(args)),
                    evidence=args.evidence)
    if args.limit:
        te = te[: args.limit]
    lookup = prepare_eeg_lookup(te, window=args.window)
    uniq = sorted({s.text for s in te})
    N = min(n_star, len(uniq))
    if N < n_star:
        print(f"  note: N*={n_star} capped to {N} by test-split unique texts")
    task_of = {s.text: s.task for s in te}

    # built exactly like the E3 validation pools (log-prob matched with a real prior),
    # and attacked with the same no-brain battery including the prior rules
    attack_prior = prior if args.prior != "tiny" else None
    pools = build_corpus_pools(te, N=N, seed=args.seed, prior=attack_prior, device=args.device)
    pgate = pool_validity_gate(pools, prior=attack_prior, device=args.device)
    print(f"  pool gate: attack={pgate['attack_acc']:.3f} chance={pgate['chance']:.3f} "
          f"passed={pgate['passed']}")

    # ---- scoring pass over all arms (single implementation) ----------------
    def score_pool_with(look, p_, gamma_override=None):
        ttext = p_.candidates[p_.true_idx].text
        w, wo, wp, ob = look[ttext]
        s = np.array([score_candidate(model, prior, torch.from_numpy(c.token_ids),
                                      w, wo, wp, ob, args.device,
                                      gamma_override=gamma_override)
                      for c in p_.candidates])
        if attr_head is not None and args.attr_beta != 0.0:
            true_sent = next(x for x in te if x.text == ttext)
            a = np.array([attr_head.log_prob_of(true_sent, task_of.get(c.text, ""))
                          for c in p_.candidates])
            s = s + args.attr_beta * a
        return s

    ARMS = ["real", "zeroed", "gamma_zero", "derangement",
            "gaussian_matched", "amplitude_only", "position_only"]
    arm_results, tost = {}, {}
    real_scores = []                              # kept for abstention + metrics
    zero_anchor_spread = 0.0                      # max |s_max - s_min| at gamma=0
    rng = np.random.default_rng(args.seed)
    for arm in ARMS:
        if arm in ("real", "gamma_zero"):
            look = lookup
        else:
            look = transformed_lookup(lookup, arm, seed=args.seed)
        hits, per_pool, choices = 0, [], []
        for p_ in pools:
            if arm == "gamma_zero":
                s = score_pool_with(look, p_, gamma_override=0.0)
                zero_anchor_spread = max(zero_anchor_spread, float(np.ptp(s)))
            else:
                s = score_pool_with(look, p_)
                if arm == "real":
                    real_scores.append(s)
            # float64 scores (selector.score_candidate); gaps below 1e-9 nats are rounding
            # noise and are broken uniformly at random, never by that noise
            if arm == "gamma_zero":
                best = np.arange(len(s))          # tie asserted below (spread < 1e-6)
            else:
                best = np.flatnonzero(np.isclose(s, s.max(), rtol=0.0, atol=1e-9))
            choice = int(rng.choice(best))
            hit = int(choice == p_.true_idx)
            hits += hit
            per_pool.append(hit)
            choices.append(choice)
        acc = hits / max(1, len(pools))
        arm_results[arm] = {"acc": acc, "per_pool": per_pool, "choices": choices,
                            "bits": fano_bits(acc, N)}
        if arm != "real":
            tost[arm] = tost_equivalence(hits, len(pools), 1.0 / N, margin=args.tost_margin)
        print(f"  arm={arm:<17} acc={acc:.4f}  bits={arm_results[arm]['bits']:.3f}"
              + ("" if arm == "real" else
                 f"  TOST-collapse={'YES' if tost[arm]['equivalent'] else 'no'}"), flush=True)
    if zero_anchor_spread > 1e-6:
        raise SystemExit(f"ZERO ANCHOR VIOLATED: gamma=0 scores differ across candidates "
                         f"by {zero_anchor_spread:.3e} (> 1e-6). The partition function "
                         f"is not exact or the tilt leaks at gamma=0. Refusing to report.")
    print(f"  zero anchor on test pools: max score spread at gamma=0 = {zero_anchor_spread:.2e}",
          flush=True)

    realized = arm_results["real"]["bits"]
    predictor = bits_sent
    ratio = realized / predictor if predictor > 0 else float("inf")
    matched = predictor > 0 and (1 / 1.5) <= max(ratio, 1e-9) <= 1.5
    print(f"  matched-N (H2): predicted {predictor:.4f} bits from E2 channel, realized "
          f"{realized:.4f} at N={N} -> {'WITHIN 1.5x' if matched else 'OUTSIDE 1.5x'}"
          + (f"  [val-selection realized {bits_sel_val:.4f}, informational]" if bits_sel_val else ""))

    # ---- ITR ----------------------------------------------------------------
    cov = None
    cov_csv = os.path.join(args.data_dir, "covariates.csv")
    if os.path.exists(cov_csv):
        cov = CovariateTable.load(cov_csv)
    n_words = {t: len(t.split()) for t in uniq}
    denoms = reading_time_denominators(cov, uniq, n_words,
                                       trial_overhead_s=args.trial_overhead_s)
    if not denoms:
        denoms = {"T3_nominal": float(np.median([len(t.split()) for t in uniq])) / 4.0}
    row = itr_row(N, arm_results["real"]["per_pool"], denoms)
    print(row.describe())

    # ---- abstention (real plogp + gaze covariates) --------------------------
    margins, correct, lens, plogp, clusters, gaze_cov = [], [], [], [], [], []
    logp_cache = {}
    for p_, s in zip(pools, real_scores):
        top = int(np.argmax(s))
        srt = np.sort(s)[::-1]
        margins.append(float(srt[0] - srt[1]) if len(srt) > 1 else 0.0)
        correct.append(int(top == p_.true_idx))
        cand = p_.candidates[top]
        lens.append(len(cand.token_ids))
        if cand.text not in logp_cache:
            logp_cache[cand.text] = mean_prior_logprob(prior, cand, args.device)
        plogp.append(logp_cache[cand.text])
        clusters.append(p_.candidates[p_.true_idx].text)
        if cov is not None:
            th = text_hash(p_.candidates[p_.true_idx].text)
            vecs = [cov.vector(sub, th, wi) for (sub, h, wi) in
                    [(k[0], k[1], k[2]) for k in cov.rows if k[1] == th][:40]]
            gaze_cov.append(np.nanmean(np.stack(vecs), axis=0) if vecs
                            else np.full(5, np.nan))
    margins = np.asarray(margins); correct = np.asarray(correct)
    lens = np.asarray(lens, float); plogp = np.asarray(plogp, float)
    covmat = np.column_stack([lens, plogp] +
                             ([np.nan_to_num(np.stack(gaze_cov))] if gaze_cov else []))
    rc = risk_coverage(correct, margins, clusters=np.asarray(clusters))
    auc = confidence_auc(correct.astype(bool), margins)
    pauc = partial_confidence_auc(correct, margins, covmat)
    cheap = cheap_confidence(lens, plogp)
    rc_cheap = risk_coverage(correct, cheap, clusters=np.asarray(clusters))
    dominates = rc.aurc <= rc_cheap.aurc + 1e-9
    print(rc.describe())
    print(f"  confidence AUC={auc:.3f}  partial AUC (len+logp+gaze removed)={pauc:.3f}  "
          f"dominates cheap baseline: {dominates}")
    with open(os.path.join(args.runs_dir, f"e4_margins{sfx}.json"), "w") as fh:
        json.dump({"margin": margins.tolist(), "plogp": plogp.tolist(),
                   "lens": lens.tolist()}, fh, indent=2)
    with open(os.path.join(args.runs_dir, f"e4_risk_coverage{sfx}.json"), "w") as fh:
        json.dump({"coverages": rc.coverages.tolist(), "risks": rc.risks.tolist(),
                   "aurc": rc.aurc, "full_risk": rc.full_risk,
                   "risks_cheap": rc_cheap.risks.tolist(),
                   "aurc_cheap": rc_cheap.aurc}, fh, indent=2)

    refs = [p_.candidates[p_.true_idx].text for p_ in pools]
    def texts_from_choices(choices):
        return [p_.candidates[c].text for p_, c in zip(pools, choices)]
    r_rand = np.random.default_rng(args.seed + 7)
    random_choices = [int(r_rand.integers(0, len(p_.candidates))) for p_ in pools]
    metric_arms = {}
    for arm in ("real", "zeroed", "gamma_zero"):
        metric_arms[arm] = compute_all(texts_from_choices(arm_results[arm]["choices"]),
                                       refs, light=args.light_metrics, device=args.device)
    metric_arms["random_pick"] = compute_all(texts_from_choices(random_choices), refs,
                                             light=args.light_metrics, device=args.device)
    hyps_real = texts_from_choices(arm_results["real"]["choices"])
    hz = texts_from_choices(arm_results["zeroed"]["choices"])
    with open(os.path.join(args.runs_dir, f"e4_paper_metrics{sfx}.json"), "w") as fh:
        json.dump(metric_arms, fh, indent=2)
    with open(os.path.join(args.runs_dir, f"e4_selected_texts{sfx}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["reference", "hypothesis_real", "hypothesis_zeroed", "correct"])
        for r, h, z, c in zip(refs, hyps_real, hz, correct):
            w.writerow([r, h, z, int(c)])
    bleu_sel = metric_arms["real"].get("bleu4")
    print(f"  paper metrics (real arm): "
          + "  ".join(f"{k}={metric_arms['real'][k]:.2f}"
                      for k in ("bleu1", "bleu4", "chrf", "rouge1")
                      if isinstance(metric_arms['real'].get(k), float)))

    # ---- ledger + gate -------------------------------------------------------
    led = ledger_for(args)
    base = base_record(args, split_fp=data_fps["test"], experiment="e4_system",
                       gates=["build", "estimator", "channel", "selection",
                              "prior_sanity" if args.prior != "tiny" else "build"])
    base["prereg_sha"] = prereg_sha
    if pgate["passed"]:
        base["gates_passed"] = base["gates_passed"] + ["pool_validity"]
    for arm, r in arm_results.items():
        led.append(Record(metric=f"selection_acc_Nstar{N}", value=r["acc"],
                          n_sentences=len(pools), arm=arm,
                          pool_hash=pools[0].pool_hash if pools else "", **base))
    led.append(Record(metric="bits_realized_per_sentence", value=realized,
                      arm="real", **base))
    for (formula, name), v in row.itr_bits_per_min.items():
        led.append(Record(metric=f"itr_{formula}_{name}", value=v, arm="real", **base))
    led.append(Record(metric="aurc", value=rc.aurc, arm="real", **base))
    led.append(Record(metric="confidence_auc", value=auc, arm="real", **base))
    led.append(Record(metric="partial_confidence_auc", value=pauc, arm="real", **base))
    for arm, m in metric_arms.items():
        for k in ("bleu1", "bleu4", "chrf", "meteor", "bertscore_f1",
                  "rouge1", "rougeL", "wer"):
            if isinstance(m.get(k), (int, float)):
                led.append(Record(metric=f"selectedtext_{k}", value=float(m[k]),
                                  arm=arm, **base))

    passed = pgate["passed"]
    write_gate_artifact(args.runs_dir, f"system{sfx}", passed, {
        "seed": args.seed, "pool_gate": pgate,
        "matched_N_meets_target": n_star_meets,
        "tost_detail": tost,
        "abstention_correct_rule": "argmax of real scores (first max)",
        "N_star": N, "predicted_bits": predictor,
        "predictor_source": "e2_channel_dI_x_tokens",
        "bits_realized_val_selection_informational": bits_sel_val,
        "realized_bits": realized, "matched_within_1p5x": bool(matched),
        "arms": {a: {"acc": r["acc"], "bits": r["bits"]} for a, r in arm_results.items()},
        "tost_collapse": {a: t["equivalent"] for a, t in tost.items()},
        "itr": {f"{k[0]}_{k[1]}": v for k, v in row.itr_bits_per_min.items()},
        "aurc": rc.aurc, "confidence_auc": auc, "partial_auc": pauc,
        "dominates_cheap_confidence": bool(dominates),
        "bleu_selected": bleu_sel, "attr_beta": args.attr_beta,
        "zero_anchor_max_spread": zero_anchor_spread,
        "text_metrics_note": "selected-text metrics; a function of selection accuracy, not generation",
        "prereg_sha": prereg_sha,
    })
    from cprd.error_analysis import analyze
    ea = analyze(args.runs_dir, task_of=task_of) if not sfx else {"_error": "seed replicate"}
    if "_error" not in ea:
        print(f"  error analysis: acc={ea['accuracy']:.3f}  "
              f"taxonomy={ea.get('failure_taxonomy')}  "
              f"wrong-pick overlap median={ea.get('wrong_overlap_median')}")

    print(led.render_table("e4_system",
                           required_gates=("build", "channel", "selection")))
    print(f"gate_system{sfx} written (passed={passed})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
