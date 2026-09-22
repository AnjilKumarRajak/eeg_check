#!/usr/bin/env python3
"""E8: scaling curve — realized bits and selection accuracy vs training-data fraction.

COFETT's third evidence criterion: predictable improvement with data. Fractions
subsample TRAIN by unique text (fixed seed); dev/val untouched; test never touched.
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
from cprd.selector import prepare_eeg_lookup, run_selection                # noqa: E402
from cprd.train import TrainConfig, train                                  # noqa: E402


def main():
    ap = base_parser("scaling: bits/accuracy vs training fraction")
    ap.add_argument("--fractions", default="0.25,0.5,1.0")
    ap.add_argument("--n-pool", type=int, default=64)
    args = ap.parse_args()
    apply_determinism(args)
    check_gate_artifact(args.runs_dir, "build")
    check_gate_artifact(args.runs_dir, "channel")

    tr, _ = load_split_sentences(args, "train")
    va, fp = load_split_sentences(args, "val")
    prior = build_prior(args)
    lookup = prepare_eeg_lookup(va, window=args.window)
    uniq_tr = sorted({s.text for s in tr})
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(uniq_tr))

    led = ledger_for(args)
    rows = []
    for frac in [float(x) for x in args.fractions.split(",")]:
        keep = {uniq_tr[int(i)] for i in order[: max(1, int(frac * len(uniq_tr)))]}
        tr_sub = [s for s in tr if s.text in keep]
        mcfg = model_config_from_args(args)
        tcfg = train_config_from_args(args)
        model = PriorResidualModel(prior, mcfg)
        train(model, tr_sub, va, mcfg, tcfg,
              os.path.join(args.runs_dir, "ckpt_scaling"),
              state_tag=f"scale_{frac}")
        N = min(args.n_pool, len({s.text for s in va}))
        pools = build_corpus_pools(va, N=N, seed=args.seed)
        pool_validity_gate(pools, prior=None)
        res = run_selection(model, prior, pools, lookup, device=args.device,
                            seed=args.seed)
        rows.append({"fraction": frac, "n_train_texts": len(keep),
                     "accuracy": res.accuracy, "bits_realized": res.bits_realized,
                     "N": N})
        print(f"  frac={frac:<5} texts={len(keep):<5} acc={res.accuracy:.3f} "
              f"bits={res.bits_realized:.2f}", flush=True)
        base = base_record(args, split_fp=fp, experiment="e8_scaling",
                           gates=["build", "channel"])
        led.append(Record(metric=f"scaling_selection_acc_N{N}", value=res.accuracy,
                          arm=f"frac={frac}", n_sentences=res.n_pools, **base))

    mono = all(rows[i]["bits_realized"] <= rows[i + 1]["bits_realized"] + 0.15
               for i in range(len(rows) - 1))
    write_gate_artifact(args.runs_dir, "scaling", len(rows) > 0,
                        {"rows": rows, "roughly_monotone": bool(mono)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
