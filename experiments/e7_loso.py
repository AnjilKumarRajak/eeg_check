#!/usr/bin/env python3
"""E7: leave-one-subject-out generalization (uses the real subject ids in the build).

Folds are disjoint on BOTH axes by construction: texts (train texts never appear in
val — guaranteed by the text-clustered split) AND subjects (the held-out subject's
readings are removed from training). Selection is evaluated on the held-out subject's
VAL-split readings only — the test split is never touched here.

Reportable either way: graceful degradation across subjects, or an honest failure
("within-subject calibration required", the norm for working BCIs).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (model_config_from_args, train_config_from_args, apply_determinism, base_parser, base_record, build_prior, ledger_for,     # noqa: E402
                    load_split_sentences)
from cprd.audit import Record                                              # noqa: E402
from cprd.model import ModelConfig, PriorResidualModel                     # noqa: E402
from cprd.pools import build_corpus_pools, pool_validity_gate              # noqa: E402
from cprd.prereg import check_gate_artifact, write_gate_artifact           # noqa: E402
from cprd.selector import fano_bits, prepare_eeg_lookup, run_selection     # noqa: E402
from cprd.train import TrainConfig, train                                  # noqa: E402


def main():
    ap = base_parser("leave-one-subject-out generalization")
    ap.add_argument("--n-pool", type=int, default=32)
    ap.add_argument("--max-subjects", type=int, default=0,
                    help="cap folds (0 = all subjects present)")
    ap.add_argument("--subject-prefix", default="",
                    help="only hold out subjects whose id starts with this (e.g. 'Z' = ZuCo 1.0)")
    ap.add_argument("--n-list", default="4,32", help="pool sizes evaluated per fold")
    ap.add_argument("--skip-subjects", default="", help="comma list of held-out subjects already done (resume)")
    args = ap.parse_args()
    apply_determinism(args)
    check_gate_artifact(args.runs_dir, "build")
    check_gate_artifact(args.runs_dir, "channel")

    tr, _ = load_split_sentences(args, "train")
    va, fp = load_split_sentences(args, "val")
    subjects = sorted({s.subject_id for s in va if s.subject_id not in ("", "UNKNOWN")})
    if not subjects:
        print("no real subject ids in this build; LOSO impossible — recording skip")
        write_gate_artifact(args.runs_dir, "loso", True,
                            {"skipped": "no subject ids"})
        return 0
    if args.subject_prefix:
        subjects = [s for s in subjects if s.startswith(args.subject_prefix)]
    if args.skip_subjects:
        subjects = [s for s in subjects if s not in set(args.skip_subjects.split(","))]
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
    import torch
    from cprd.audit import sha256_of
    n_list = [int(x) for x in args.n_list.split(",")]

    prior = build_prior(args)
    led = ledger_for(args)
    rows = []
    for subj in subjects:
        tr_fold = [s for s in tr if s.subject_id != subj]
        va_in = [s for s in va if s.subject_id != subj]      # early stopping pool
        va_out = [s for s in va if s.subject_id == subj]     # the held-out subject
        if len(va_out) < args.n_pool:
            print(f"  {subj}: only {len(va_out)} val readings; skipping fold")
            continue
        assert not ({s.subject_id for s in tr_fold} & {subj}), "subject leak"

        mcfg = model_config_from_args(args)
        tcfg = train_config_from_args(args)
        model = PriorResidualModel(prior, mcfg)
        key = sha256_of([mcfg.to_dict(), tcfg.to_dict(), f"{args.prior}:{args.prior_model}"])[:16]
        train(model, tr_fold, va_in, mcfg, tcfg,
              os.path.join(args.runs_dir, "ckpt_loso"),
              state_tag=f"loso_{key}_{subj}")
        lookup = prepare_eeg_lookup(va_out, window=args.window)
        zlook = {t: (torch.zeros_like(w), o, p, ob) for t, (w, o, p, ob) in lookup.items()}
        base = base_record(args, split_fp=fp, experiment="e7_loso", gates=["build", "channel"])
        for n_pool in n_list:
            N = min(n_pool, len({s_.text for s_ in va_out}))
            pools = build_corpus_pools(va_out, N=N, seed=args.seed)
            pgate = pool_validity_gate(pools, prior=None)
            res = run_selection(model, prior, pools, lookup, device=args.device, seed=args.seed)
            res0 = run_selection(model, prior, pools, zlook, device=args.device, seed=args.seed)
            rows.append({"subject": subj, "n_pools": res.n_pools, "N": res.N,
                         "acc": res.accuracy, "acc_zeroed": res0.accuracy, "bits": res.bits_realized,
                         "chance": 1.0 / max(1, res.N), "pool_gate": pgate["passed"]})
            print(f"  LOSO {subj} N={res.N}: acc={res.accuracy:.3f} zeroed={res0.accuracy:.3f} "
                  f"(chance {1.0/max(1,res.N):.3f}) bits={res.bits_realized:.2f}", flush=True)
            led.append(Record(metric=f"loso_selection_acc_N{res.N}", value=res.accuracy,
                              n_sentences=res.n_pools, arm=f"heldout={subj}", **base))
            led.append(Record(metric=f"loso_selection_acc_N{res.N}", value=res0.accuracy,
                              n_sentences=res.n_pools, arm=f"heldout={subj}/zeroed", **base))

    accs = [r["acc"] for r in rows]
    above = sum(1 for r in rows if r["acc"] > 2 * r["chance"])
    by_n = {}
    for r in rows:
        by_n.setdefault(str(r["N"]), []).append(r)
    summary = {n: {"mean_acc": float(np.mean([x["acc"] for x in v])), "sd_acc": float(np.std([x["acc"] for x in v])),
                   "mean_acc_zeroed": float(np.mean([x["acc_zeroed"] for x in v])), "chance": v[0]["chance"],
                   "n_subjects": len(v), "subjects_above_2x_chance": sum(1 for x in v if x["acc"] > 2 * x["chance"])}
               for n, v in by_n.items()}
    detail = {"rows": rows, "summary_by_N": summary,
              "mean_acc": float(np.mean(accs)) if accs else None,
              "subjects_above_2x_chance": above, "n_folds": len(rows),
              "evidence": args.evidence, "objective": getattr(args, "objective", "raw")}
    write_gate_artifact(args.runs_dir, "loso", len(rows) > 0, detail)
    print(f"LOSO: {above}/{len(rows)} held-out subjects above 2x chance "
          f"(mean acc {np.mean(accs):.3f})" if rows else "LOSO: no folds ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())
