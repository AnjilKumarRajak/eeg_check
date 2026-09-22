#!/usr/bin/env python3
"""E1 (Phase 1): estimator validation on channels of KNOWN capacity. Writes gate_estimator.

Design (post-audit):
  * PARITY: the synthetic model/training is EXACTLY what E2 uses (same ModelConfig
    defaults, same objective — no contrastive asymmetry), so passing E1 validates the
    procedure that produces the real numbers.
  * SEEDS: every sweep point runs a fixed set of seeds; ALL attempts are ledgered
    (metric 'e1_attempt'); the judged value is the MEDIAN — no best-of-k.
  * PARTIAL PASS: the gate passes if the null point is clean and at least the smallest
    non-zero capacity is recovered; `detail.validated_range_bits_per_token` records how
    far validation reaches, `detail.partial_pass` marks it, and downstream ledger
    records carry 'estimator_partial'. Full pass requires the whole sweep + calibration.
    => you never need to hand-edit this gate to proceed; a partial pass unblocks E2+
    with honest labeling, and `experiments/override_gate.py` exists for anything else.
  * RESUME: per-point training uses resumable train-state files; the sweep itself
    checkpoints to runs/e1_sweep_state.json so an interrupted E1 continues where it was.

    smoke:  python experiments/e1_estimator.py --prior tiny --sweep 0,1 --seeds 0 \
                --n-null 6 --n-sent 300 --epochs 25 --lr 3e-3 --relaxed
    full:   python experiments/e1_estimator.py --prior causal_lm --prior-model gpt2-large \
                --sweep 0,0.05,0.1,0.25,0.5,1,2 --seeds 0,1,2 --n-sent 2000 \
                --n-null 100 --epochs 50 --lr 1e-3 --n-perm 1000
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
                    ledger_for, objective_label, train_config_from_args)
from cprd.audit import Record                                              # noqa: E402
from cprd.evaluate import evaluate                                         # noqa: E402
from cprd.model import ModelConfig, PriorResidualModel                     # noqa: E402
from cprd.prereg import write_gate_artifact                                # noqa: E402
from cprd.synth import (ValidationOutcome, judge, ks_uniform, make_channel,  # noqa: E402
                        make_synthetic_sentences)
from cprd.train import TrainConfig, train                                  # noqa: E402

LN2 = math.log(2.0)


def embedding_aligned_clusters(prior, n_clusters: int, seed: int) -> np.ndarray:
    """Assign each vocab token to a cluster by its position in the prior's OWN output
    embedding geometry — the injected code is then expressible by the rank-r tilt,
    making E1 a fair test of the production parameterization rather than of an
    arbitrary token%K code the tilt cannot represent."""
    with torch.no_grad():
        omega = prior.omega.detach().float().cpu()
        g = torch.Generator().manual_seed(seed)
        cent = omega[torch.randperm(omega.shape[0], generator=g)[:n_clusters]]
        assign = (omega @ cent.t()).argmax(dim=1).numpy()
    # guarantee non-empty clusters
    for k in range(n_clusters):
        if not (assign == k).any():
            assign[k % len(assign)] = k
    return assign.astype(np.int64)


@torch.no_grad()
def sample_prior_sentences(prior, n: int, seed: int, device, min_len: int = 6,
                           max_len: int = 14, batch: int = 64) -> list:
    """Ancestral samples from the frozen prior (top-k 0, temperature 1): the synthetic
    token marginal IS the prior, as in reading. Lengths uniform in [min_len, max_len]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)                        # multinomial draws (device generator)
    prior.model.to(device)
    out = []
    bos = prior._bos
    while len(out) < n:
        B = min(batch, n - len(out))
        L = int(torch.randint(min_len, max_len + 1, (1,), generator=g))
        ids = torch.full((B, 1), bos, dtype=torch.long, device=device)
        for _ in range(L):
            lp = prior.model(input_ids=ids).logits[:, -1].float().log_softmax(-1)
            nxt = torch.multinomial(lp.exp(), 1, generator=None)
            ids = torch.cat([ids, nxt], 1)
        for row in ids[:, 1:].cpu().numpy():
            out.append(row.astype(np.int64))
    return out


def balanced_embedding_clusters(prior, n_clusters: int, seed: int, token_seqs: list) -> np.ndarray:
    """Clusters = contiguous bands of the projection of Omega onto a random direction,
    with band edges at equal EMPIRICAL PRIOR MASS (from the sampled token stream), so
    the cluster prior is uniform (H(cluster) = log2 K) and every cluster is a slab of
    the embedding space (representable by a rank-1 nonlinear tilt)."""
    with torch.no_grad():
        omega = prior.omega.detach().float().cpu()
        g = torch.Generator().manual_seed(seed)
        d = torch.randn(omega.shape[1], generator=g)
        proj = (omega @ d).numpy()
    counts = np.bincount(np.concatenate(token_seqs), minlength=omega.shape[0]).astype(np.float64) + 1e-3
    order = np.argsort(proj)
    cum = np.cumsum(counts[order]) / counts.sum()
    assign = np.zeros(omega.shape[0], dtype=np.int64)
    assign[order] = np.minimum((cum * n_clusters).astype(np.int64), n_clusters - 1)
    for k in range(n_clusters):                     # never an empty cluster
        if not (assign == k).any():
            assign[order[k]] = k
    return assign


def run_point(b: float, seed: int, args, prior, train_it: bool = True):
    """Train + evaluate one (capacity, seed) point with the E2-parity procedure.

    Parity with E2 (2026-09 fixes): the model is TRAINED at every sweep point
    including b=0 (an untrained b=0 point never exercises the train-and-select
    procedure whose zero point is being validated), checkpoint selection uses the
    selection half of the synthetic val set, and dI_hat/p are measured on the
    disjoint measurement half -- exactly as E2 does on ZuCo."""
    ch = make_channel(bits=b, seed=seed, n_mc=200_000 if not args.limit else 20_000)
    tok_clusters = None
    n = args.n_sent
    n_total = n + max(80, n // 3)
    token_seqs = None
    if args.synth_tokens == "prior":
        token_seqs = sample_prior_sentences(prior, n_total, seed, args.device)
        if ch.n_clusters > 1:
            tok_clusters = balanced_embedding_clusters(prior, ch.n_clusters, seed, token_seqs)
            cl = np.concatenate([tok_clusters[t] for t in token_seqs])
            print(f"    prior-matched synth: cluster mass {np.bincount(cl, minlength=ch.n_clusters) / len(cl)}", flush=True)
    elif ch.n_clusters > 1:
        tok_clusters = embedding_aligned_clusters(prior, ch.n_clusters, seed)
    sents = make_synthetic_sentences(ch, n_total, prior.vocab_size,
                                     missing_frac=0.2, seed=seed,
                                     token_clusters=tok_clusters, token_sequences=token_seqs)
    tr, va = sents[:n], sents[n:]
    from cprd.data import selection_measurement_split
    va_sel, va_meas = selection_measurement_split(va)
    # PARITY: identical model + training config to common.train_model / E2
    from common import model_config_from_args
    mcfg = model_config_from_args(args)          # E2 parity, incl. centring
    tcfg = train_config_from_args(args, seed=seed, n_perm_val=20)   # E2 parity
    model = PriorResidualModel(prior, mcfg)
    if train_it:
        train(model, tr, va_sel, mcfg, tcfg,
              os.path.join(args.runs_dir, "ckpt_synth"),
              state_tag=f"e1_{config_key(args)[:16]}_b{b}_s{seed}")
    # train() is the only thing that moves the model onto args.device, and it is
    # SKIPPED at b=0 (n_clusters==1). Without this the null/calibration points
    # evaluate on CPU -- same arithmetic, ~10x the wall-time. Placement only.
    model.to(args.device)
    if getattr(args, "eval_gamma", None) is not None:
        model.tilt.gamma_mode, model.tilt.gamma_const = "constant", float(args.eval_gamma)
    from common import prepare_batches
    batches, lp0 = prepare_batches(va_meas, prior, args.device, window=args.window,
                                   batch_size=args.batch_size)
    res = evaluate(model, batches, lp0,
                   nulls=("sentence_derangement", "temporal_roll"),
                   n_perm=args.n_perm, seed=seed)
    stat = res.dI_hat_nce if args.estimand == "nce" else res.dI_hat
    print(f"    [b={b} seed={seed}] mean-null={res.dI_hat['sentence_derangement']/LN2:+.3f}  "
          f"InfoNCE={res.dI_hat_nce['sentence_derangement']/LN2:+.3f}  DV={res.dI_hat_dv['sentence_derangement']/LN2:+.3f} bits/tok  "
          f"p={res.p_value['sentence_derangement']:.4f}", flush=True)
    if getattr(args, "extra_eval_gamma", None) is not None:
        base_gm, base_gc = model.tilt.gamma_mode, model.tilt.gamma_const
        model.tilt.gamma_mode, model.tilt.gamma_const = "constant", float(args.extra_eval_gamma)
        res2 = evaluate(model, batches, lp0, nulls=("sentence_derangement", "temporal_roll"),
                        n_perm=args.n_perm, seed=seed)
        stat2 = res2.dI_hat_nce if args.estimand == "nce" else res2.dI_hat
        with open(os.path.join(args.runs_dir, "e1_extra_eval.jsonl"), "a") as fh:
            fh.write(json.dumps({"b": ch.true_bits_per_token, "seed": seed, "gamma_default_bits": stat["sentence_derangement"] / LN2,
                                 "p_default": res.p_value["sentence_derangement"], "eval_gamma": float(args.extra_eval_gamma),
                                 "bits": stat2["sentence_derangement"] / LN2, "p": res2.p_value["sentence_derangement"],
                                 "dv_bits": res2.dI_hat_dv["sentence_derangement"] / LN2}) + "\n")
        print(f"    [b={b} seed={seed}] eval-gamma {args.extra_eval_gamma}: InfoNCE={stat2['sentence_derangement']/LN2:+.3f} p={res2.p_value['sentence_derangement']:.4f}", flush=True)
        model.tilt.gamma_mode, model.tilt.gamma_const = base_gm, base_gc
    return (ch.true_bits_per_token,
            stat["sentence_derangement"] / LN2,
            res.p_value["sentence_derangement"])


def config_key(args) -> str:
    """Everything that determines an E1 result. The sweep state and per-point training
    state are keyed by it, so a re-run with a different config never reuses stale
    (true, recovered, p) values (the Sept audit found a copied pre-fix gate)."""
    from cprd.audit import sha256_of
    tc = train_config_from_args(args, seed=0, n_perm_val=20).to_dict()
    from common import model_config_from_args
    return sha256_of([tc, model_config_from_args(args).to_dict(), args.n_sent, args.n_perm, args.limit, args.estimand, args.synth_tokens, args.gate_mode,
                      f"{args.prior}:{args.prior_model}", "e1:v2-selmeas-trainnull"])


def load_state(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {"points": {}, "null_ps": []}


def save_state(path, st):
    with open(path, "w") as fh:
        json.dump(st, fh, indent=2)


def main():
    ap = base_parser("estimator validation on known-capacity synthetic channels")
    ap.add_argument("--sweep", default="0,0.5,1,2")
    ap.add_argument("--seeds", default="0,1,2",
                    help="ALL run and ledgered; judged value is the MEDIAN")
    ap.add_argument("--n-sent", type=int, default=300)
    ap.add_argument("--n-null", type=int, default=20,
                    help="independent b=0 datasets for p-calibration")
    ap.add_argument("--train-null", action="store_true")
    ap.add_argument("--gate-mode", default="point", choices=["point", "lower_bound"],
                    help="point = pre-registered rule (recover >= 50%% of b, p<0.05); lower_bound = the "
                         "rule for a variational LOWER BOUND (InfoNCE/DV estimands): clean zero point, "
                         "never above truth, monotone in b, detected (p<0.05) at every b > 0; tightness "
                         "(recovered/true) is reported per capacity")
    ap.add_argument("--synth-tokens", default="prior", choices=["prior", "uniform"],
                    help="prior = sentences sampled from the frozen prior with prior-mass-balanced "
                         "embedding clusters (2026-09-17, the deployment regime); uniform = the "
                         "original uniform-random-token design")
    ap.add_argument("--skip-calibration", action="store_true",
                    help="defer the KS leg (recorded; full pass then impossible, "
                         "partial pass still available)")
    ap.add_argument("--relaxed", action="store_true",
                    help="SMOKE ONLY (recovery floor 15%% of truth, p<0.05; stamped "
                         "RELAXED_SMOKE; never a scientific gate)")
    args = ap.parse_args()
    apply_determinism(args)
    if args.n_perm > 200 and args.prior == "tiny":
        args.n_perm = 200
    seeds = [int(x) for x in args.seeds.split(",")]

    prior = build_prior(args)
    led = ledger_for(args)
    state_path = os.path.join(args.runs_dir, f"e1_sweep_state_{config_key(args)[:16]}.json")
    os.makedirs(args.runs_dir, exist_ok=True)
    st = load_state(state_path)

    sweep_bs = [float(x) for x in args.sweep.split(",")]
    table, outcomes = [], []
    for b in sweep_bs:
        keyb = f"{b}"
        st["points"].setdefault(keyb, {})
        per_seed = []
        for sd in seeds:
            keys = f"{sd}"
            if keys in st["points"][keyb]:
                bt, rec, p = st["points"][keyb][keys]
            else:
                bt, rec, p = run_point(b, args.seed + sd * 101, args, prior)
                st["points"][keyb][keys] = [bt, rec, p]
                save_state(state_path, st)
            per_seed.append((bt, rec, p))
            base = base_record(args, split_fp="synthetic",
                               experiment="e1_estimator", gates=[])
            led.append(Record(metric="e1_attempt", value=rec, p=p,
                              arm=f"b={b}/seed={sd}", **base))
        bt = float(np.median([x[0] for x in per_seed]))
        if args.relaxed:
            # RELAXED_SMOKE judges best-of-seeds: it verifies the PLUMBING can recover
            # injected bits; seed-robustness is the strict gate's job, not smoke's.
            best = int(np.argmax([x[1] for x in per_seed]))
            rec = float(per_seed[best][1])
            p = float(per_seed[best][2])
        else:
            # STRICT judges the median over seeds — no best-of-k cherry-picking.
            rec = float(np.median([x[1] for x in per_seed]))
            p = float(np.median([x[2] for x in per_seed]))

        if args.relaxed and bt > 0:
            o = ValidationOutcome(b_nominal=b, b_true=bt, b_recovered=rec, p_value=p,
                                  passed_recovery=(rec >= 0.15 * bt and p < 0.05),
                                  passed_never_overshoot=(rec <= bt + 0.05))
        elif args.gate_mode == "lower_bound" and bt > 0:
            # a bound must be detected and must not exceed the truth; tightness is reported
            o = ValidationOutcome(b_nominal=b, b_true=bt, b_recovered=rec, p_value=p,
                                  passed_recovery=(rec > 0 and p < 0.05),
                                  passed_never_overshoot=(rec <= bt + 0.05))
        else:
            o = judge(bt, rec, p)
        outcomes.append(o)
        table.append({"b_nominal": b, "b_true": bt, "recovered_median": rec, "p": p,
                      "tightness": (rec / bt) if bt > 0 else None,
                      "n_seeds": len(seeds), "recovery_ok": o.passed_recovery,
                      "no_overshoot": o.passed_never_overshoot,
                      "attempts": per_seed})
        print(f"  b={b:<5} true={bt:.3f}  recovered(median/{len(seeds)} seeds)="
              f"{rec:+.3f} bits/tok  p={p:.4f}  {'OK' if o.ok else 'FAIL'}", flush=True)

    # ---- calibration ----
    if args.skip_calibration:
        null_ps, ks_p, calib_status = st.get("null_ps", []), None, "deferred"
    else:
        null_ps = st.get("null_ps", [])
        for k in range(len(null_ps), args.n_null):
            _, _, p = run_point(0.0, 10_000 + k, args, prior, train_it=args.train_null)
            null_ps.append(p)
            st["null_ps"] = null_ps
            save_state(state_path, st)
        ks_p = ks_uniform(np.asarray(null_ps))
        calib_status = "done"
    # a non-finite KS p (e.g. --n-null 0/1) is NOT a passed calibration
    calib_ok = (calib_status == "done"
                and ((np.isfinite(ks_p) and ks_p > 0.05) or args.relaxed))
    if calib_status == "done":
        print(f"  KS-uniform p = {ks_p:.4f}  ({'OK' if calib_ok else 'FAIL'})", flush=True)
    else:
        print("  calibration: DEFERRED (full pass unavailable; partial pass possible)",
              flush=True)

    # ---- verdict: full / partial / fail (no hand-editing ever needed) ----
    zero_pts = [o for o, b in zip(outcomes, sweep_bs) if b <= 0]
    nz = sorted([(b, o) for b, o in zip(sweep_bs, outcomes) if b > 0])
    zero_ok = all(o.ok for o in zero_pts) if zero_pts else True
    no_overshoot_all = all(o.passed_never_overshoot for o in outcomes)
    # 2026-09-16: the old rule took the largest capacity passing CONTIGUOUSLY from zero,
    # which (with a vacuous small-b rule) let "validated 0-0.05" certify an estimator
    # that recovered nothing. The honest semantics are a RESOLUTION FLOOR: the smallest
    # injected capacity the estimator demonstrably recovers (>=50%, p<0.05). A channel
    # reading below that floor is "below the instrument's resolution", not "measured".
    # lower-bound mode adds MONOTONICITY: the recovered value must not decrease with b
    if args.gate_mode == "lower_bound" and len(nz) >= 2:
        recs = [o.b_recovered for _, o in nz]
        if any(recs[i + 1] < recs[i] - 0.02 for i in range(len(recs) - 1)):
            print("  lower-bound gate: recovered value is NOT monotone in injected capacity -> FAIL", flush=True)
            nz = [(b, ValidationOutcome(b_nominal=o.b_nominal, b_true=o.b_true, b_recovered=o.b_recovered,
                                        p_value=o.p_value, passed_recovery=False,
                                        passed_never_overshoot=o.passed_never_overshoot)) for b, o in nz]
    validated = sorted(b for b, o in nz if o.ok)
    # the floor must be MONOTONE: the smallest b such that every larger injected
    # capacity is also recovered (one lucky small-b pass with larger failures is not
    # a resolution the instrument has)
    resolution_floor = None
    for b_, o_ in sorted(nz, reverse=True):
        if not o_.ok:
            break
        resolution_floor = b_
    validated_max = validated[-1] if validated else 0.0
    sweep_ok_full = zero_ok and all(o.ok for _, o in nz)
    full_pass = sweep_ok_full and calib_ok
    partial_pass = (not full_pass) and zero_ok and no_overshoot_all and resolution_floor is not None
    passed = full_pass or partial_pass

    detail = {"sweep": table, "null_p_values": null_ps, "ks_p": ks_p,
              "calibration_status": calib_status,
              "sweep_ok_full": sweep_ok_full,
              "partial_pass": bool(partial_pass),
              "validated_bits_per_token": validated,
              "resolution_floor_bits_per_token": resolution_floor,
              "validated_max_bits_per_token": validated_max,
              "rule": "recovered >= 0.5*b AND p<0.05 per point; median over seeds; floor = smallest b with all larger b also passing",
              "calibration_model": ("trained (--train-null)" if args.train_null else
                                    "untrained: checks permutation-test calibration only; the trained b=0 sweep points check the train-and-select zero point"),
              "config_key": config_key(args), "estimand": args.estimand, "synth_tokens": args.synth_tokens,
              "gate_mode": args.gate_mode,
              "tightness_by_b": {str(r["b_nominal"]): r["tightness"] for r in table if r["tightness"] is not None},
              "objective": objective_label(args), "free_tilt_rank": args.free_tilt_rank,
              "seeds": seeds, "policy": ("best-of-seeds (RELAXED_SMOKE plumbing check)" if args.relaxed else "median over seeds") + "; all attempts ledgered",
              "mode": "RELAXED_SMOKE" if args.relaxed else "strict"}
    write_gate_artifact(args.runs_dir, "estimator", passed, detail)

    base = base_record(args, split_fp="synthetic", experiment="e1_estimator",
                       gates=(["estimator"] if passed else []))
    for row in table:
        led.append(Record(metric="recovered_bits_per_token",
                          value=row["recovered_median"], p=row["p"],
                          arm=f"b={row['b_nominal']}", **base))
    reason = ("not monotone in injected capacity" if (args.gate_mode == "lower_bound" and len(nz) >= 2
              and any(nz[i + 1][1].b_recovered < nz[i][1].b_recovered - 0.02 for i in range(len(nz) - 1)))
              else "recovers no injected capacity")
    kind = "FULL" if full_pass else ("PARTIAL (resolution floor %s bits/tok; validated %s)"
                                     % (resolution_floor, validated)
                                     if partial_pass else f"FAILED ({reason})")
    print(f"gate_estimator written: {kind} (passed={passed})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
