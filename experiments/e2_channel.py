#!/usr/bin/env python3
"""E2 (Phase 2): the channel measurement. Needs gate_build + gate_estimator.
Writes gate_channel with bits/sentence — the number that sizes everything downstream.

    python experiments/e2_channel.py --data-dir runs/data --prior causal_lm \
        --prior-model gpt2-large --epochs 15 --n-perm 10000
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (read_gate_optional, apply_determinism, base_parser, base_record, build_prior,  # noqa: E402
                    ledger_for, load_split_sentences, prepare_batches,
                    model_config_from_args, prior_gate_or_die,
                    train_config_from_args, train_model)
from cprd.audit import Record                                              # noqa: E402
from cprd.evaluate import evaluate, zero_gamma_check                       # noqa: E402
from cprd.model import ModelConfig, PriorResidualModel                     # noqa: E402
from cprd.data import selection_measurement_split                          # noqa: E402
from cprd.prereg import check_gate_artifact, write_gate_artifact           # noqa: E402
from cprd.train import TrainConfig, config_hash                            # noqa: E402

LN2 = math.log(2.0)


def main():
    parser = base_parser("channel measurement (dI_hat, dual nulls, clustered)")
    parser.add_argument("--resume-eval", action="store_true",
                        help="load the completed E2 training checkpoint and resume only the final evaluation")
    args = parser.parse_args()
    apply_determinism(args)
    check_gate_artifact(args.runs_dir, "build")
    # E2 runs whether or not E1 passed (run_campaign.sh relies on this): a failed or
    # missing estimator gate is carried as a LABEL (records omit 'estimator';
    # reading_vs_floor says UNCALIBRATED), never as a crash that loses the measurement.
    estimator_gate = read_gate_optional(args.runs_dir, "estimator")
    if (estimator_gate.get("detail", {}).get("mode") == "RELAXED_SMOKE"
            and args.prior != "tiny"):
        # a production prior must never ride a smoke-mode estimator gate;
        # tiny-prior runs ARE the smoke, so they may.
        raise RuntimeError(
            "gate_estimator is RELAXED_SMOKE; run strict E1 before production E2"
        )

    tr, _ = load_split_sentences(args, "train")
    va, _ = load_split_sentences(args, "val")
    prior = build_prior(args)
    gates = ["build", "estimator"]      # collect_green_gates keeps only what is green
    if args.prior != "tiny":
        prior_gate_or_die(prior, va, args.device)
        gates.append("prior_sanity")

    va_sel, va_meas = selection_measurement_split(va)
    print(f"  val split: {len(va_sel)} sentences for checkpoint selection, "
          f"{len(va_meas)} held out for the measurement", flush=True)
    if args.resume_eval:
        mcfg = model_config_from_args(args)
        tcfg = train_config_from_args(args)
        from common import train_model_key
        chash = train_model_key(args)
        ckpt = os.path.join(args.runs_dir, "ckpt", f"phi_{chash}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"--resume-eval requested but checkpoint is missing: {ckpt}")
        model = PriorResidualModel(prior, mcfg).to(torch.device(args.device))
        state = torch.load(ckpt, map_location=torch.device(args.device))
        model.load_state_dict(state["phi"])
        model.ckpt_key = chash
        print(f"loaded E2 checkpoint for evaluation resume: {ckpt}", flush=True)
    else:
        model = train_model(prior, tr, va, args)

    # VALIDATION split only here; the test split stays behind the prereg wrapper (e4/e5)
    # measured on the held-out MEASUREMENT half (never the half used for selection)
    batches, lp0 = prepare_batches(va_meas, prior, args.device,
                                   window=args.window, batch_size=args.batch_size)
    res = evaluate(model, batches, lp0,
                   nulls=("sentence_derangement", "temporal_roll"),
                   n_perm=args.n_perm, n_boot=args.n_boot,
                   seed=args.seed, progress=True,
                   checkpoint_dir=os.path.join(args.runs_dir, "eval_ckpt"), checkpoint_every=20,
                   # tied to the exact model: a retrained model never inherits old draws
                   checkpoint_tag=f"e2_channel_{model.ckpt_key[:16]}_s{args.seed}")
    zg = zero_gamma_check(model, batches, lp0)
    print(res.describe())
    print(f"  zero-gamma check: {zg:.2e}")

    # bits/sentence from the observed-token rate x mean observed tokens per sentence
    mean_obs_tokens = res.n_observed / max(1, res.n_sentences)
    stat = res.dI_hat_nce if args.estimand == "nce" else res.dI_hat
    bits_tok = stat["sentence_derangement"] / LN2
    bits_tok_roll = stat["temporal_roll"] / LN2
    bits_sent = bits_tok * mean_obs_tokens
    print(f"  bits/token (derangement null): {bits_tok:+.4f}   (roll): {bits_tok_roll:+.4f}")
    print(f"  bits/sentence (derangement):   {bits_sent:+.3f}")

    both_null_pass = (res.p_value["sentence_derangement"] < 0.05
                      and res.p_value["temporal_roll"] < 0.05
                      and bits_tok > 0 and bits_tok_roll > 0)
    passed = zg < 1e-4 and np.isfinite(bits_sent)
    detail = {"bits_per_token_derangement": bits_tok,
              "bits_per_token_roll": bits_tok_roll,
              "bits_per_sentence": bits_sent,
              "mean_observed_tokens": mean_obs_tokens,
              "p_derangement": res.p_value["sentence_derangement"],
              "p_roll": res.p_value["temporal_roll"],
              "H1_primary_endpoint_pass": bool(both_null_pass),
              "zero_gamma": zg, "n_clusters": res.n_clusters,
              "bits_per_token_DV_derangement": (res.dI_hat_dv or {}).get("sentence_derangement", float("nan")) / LN2,
              "bits_per_token_DV_roll": (res.dI_hat_dv or {}).get("temporal_roll", float("nan")) / LN2,
              "estimand": args.estimand, "free_tilt_rank": args.free_tilt_rank,
              "bits_per_token_meannull_derangement": res.dI_hat["sentence_derangement"] / LN2,
              "bits_per_token_nce_derangement": (res.dI_hat_nce or {}).get("sentence_derangement", float("nan")) / LN2,
              "bits_per_token_nce_roll": (res.dI_hat_nce or {}).get("temporal_roll", float("nan")) / LN2,
              "estimator_note": "dI_hat = real - mean(null) is the pre-registered quantity; it is NOT a lower bound on I (docs/FINDINGS_20260917.md). The DV value real - log mean exp(null) IS a lower bound.",
              "ci95_bits_per_token_derangement": [
                  (res.ci95_dI_hat or {}).get("sentence_derangement", (float("nan"),) * 2)[0] / LN2,
                  (res.ci95_dI_hat or {}).get("sentence_derangement", (float("nan"),) * 2)[1] / LN2],
              "n_sentences_measured": res.n_sentences,
              "measurement_split": "val measurement half (text-hash); selection half used for checkpoint selection",
              "estimator_gate_passed": bool(estimator_gate.get("passed", False)),
              "null_note": "cross-sentence null rejects same-text donors"}
    # Compare the reading to the estimator's demonstrated resolution floor (E1). A
    # channel below the floor is not "measured at X"; it is "below what this
    # instrument can resolve". Both the number and this verdict go in the gate.
    # re-read: E1 may have been re-judged (more seeds) while this stage was running
    estimator_gate = read_gate_optional(args.runs_dir, "estimator") or estimator_gate
    est_detail = (estimator_gate.get("detail") or {})
    floor = est_detail.get("resolution_floor_bits_per_token")
    detail["estimator_resolution_floor_bits_per_token"] = floor
    detail["estimator_mode"] = est_detail.get("mode")
    if floor is None:
        detail["reading_vs_floor"] = "UNCALIBRATED (estimator recovered no injected capacity)"
    elif est_detail.get("gate_mode") == "lower_bound":
        # a validated LOWER BOUND: the reading is compared with what the bound returns at
        # the smallest detected capacity (its recovered value), not with the capacity itself
        tb = est_detail.get("tightness_by_b") or {}
        rec_floor = float(floor) * float(tb.get(str(floor), tb.get(str(float(floor)), 0.0)) or 0.0)
        detail["reading_vs_floor"] = (f"validated lower bound (E1 {est_detail.get('mode')}, gate_mode=lower_bound; "
                                      f"tightness {tb}); reading {bits_tok:.4f} vs bound at detection floor "
                                      f"b={floor}: {rec_floor:.4f} bits/tok -> "
                                      + ("above" if bits_tok >= rec_floor else "below") + " the detection floor")
    elif bits_tok < float(floor):          # signed: a negative reading is never "above floor"
        detail["reading_vs_floor"] = f"BELOW FLOOR ({bits_tok:.5f} < {floor} bits/tok): report as 'below the instrument's resolution', not as a measurement"
    else:
        detail["reading_vs_floor"] = f"above floor ({bits_tok:.5f} >= {floor} bits/tok)"
    print(f"  calibration: {detail['reading_vs_floor']}", flush=True)
    write_gate_artifact(args.runs_dir, "channel", passed, detail)

    led = ledger_for(args)
    _, fp = load_split_sentences(args, "val")
    fp = f"{fp}:measurement_half"
    base = base_record(args, split_fp=fp, experiment="e2_channel", gates=gates)
    ci_hat = (res.ci95_dI_hat or {}).get("sentence_derangement", (None, None))
    led.append(Record(metric="dI_hat_bits_per_token", value=bits_tok,
                      ci_lo=(ci_hat[0] / LN2 if ci_hat[0] is not None else None),
                      ci_hi=(ci_hat[1] / LN2 if ci_hat[1] is not None else None),
                      p=res.p_value["sentence_derangement"],
                      n_clusters=res.n_clusters, n_sentences=res.n_sentences,
                      arm="real_vs_derangement", **base))
    led.append(Record(metric="dI_hat_bits_per_token", value=bits_tok_roll,
                      p=res.p_value["temporal_roll"], n_clusters=res.n_clusters,
                      arm="real_vs_temporal_roll", **base))
    led.append(Record(metric="dI_hat_DV_bits_per_token",
                      value=(res.dI_hat_dv or {}).get("sentence_derangement", float("nan")) / LN2,
                      n_clusters=res.n_clusters, arm="real_vs_derangement_DV", **base))
    led.append(Record(metric="bits_per_sentence", value=bits_sent,
                      n_clusters=res.n_clusters, arm="derangement_null", **base))
    print(f"gate_channel written (passed={passed}; H1={both_null_pass})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
