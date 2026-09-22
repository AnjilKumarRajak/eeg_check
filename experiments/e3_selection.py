
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (apply_determinism, base_parser, base_record, build_prior, ledger_for,     # noqa: E402
                    load_split_sentences, train_model)
from cprd.audit import Record                                              # noqa: E402
from cprd.nulls import apply_text_cluster_null                             # noqa: E402
from cprd.pools import build_corpus_pools, pool_validity_gate              # noqa: E402
from cprd.prereg import check_gate_artifact, write_gate_artifact           # noqa: E402
from cprd.nulls import _derangement                                       # noqa: E402
from cprd.selector import (fano_bits, n_star_meets_target, predict_n_star,  # noqa: E402
                           prepare_eeg_lookup, run_selection)


def main():
    ap = base_parser("selection sweep + matched-N prediction")
    ap.add_argument("--n-grid", default="2,4,8,16,32")
    args = ap.parse_args()
    apply_determinism(args)
    check_gate_artifact(args.runs_dir, "build")
    gate_ch = check_gate_artifact(args.runs_dir, "channel")
    bits_sent = float(gate_ch["detail"]["bits_per_sentence"])

    tr, _ = load_split_sentences(args, "train")
    va, fp = load_split_sentences(args, "val")
    prior = build_prior(args)
    model = train_model(prior, tr, va, args)
    lookup = prepare_eeg_lookup(va, window=args.window)

    n_grid = [int(x) for x in args.n_grid.split(",")]
    uniq_texts = sorted({s.text for s in va})
    n_grid = [n for n in n_grid if n <= len(uniq_texts)]

    led = ledger_for(args)
    sweep, pools_ok = [], True
    attack_prior = prior if args.prior != "tiny" else None
    for N in n_grid:
        pools = build_corpus_pools(va, N=N, seed=args.seed, prior=attack_prior,
                                   device=args.device)
        gate = pool_validity_gate(pools, prior=attack_prior, device=args.device)
        gate_pre_rebuild = None
        if not gate["passed"]:
            gate_pre_rebuild = dict(gate)          # keep the failing attack on record
            print(f"  N={N}: POOL VALIDITY FAILED (attack {gate['attack_acc']:.3f} "
                  f"vs chance {gate['chance']:.3f}) — rebuilding once with new seed")
            pools = build_corpus_pools(va, N=N, seed=args.seed + 777, prior=attack_prior,
                                       device=args.device)
            gate = pool_validity_gate(pools, prior=attack_prior, device=args.device)
        # a single unrepaired N fails the whole stage; a repaired one never
        # erases earlier failures (audit finding 7.7)
        pools_ok = pools_ok and gate["passed"]

        arms = {}
        arms["real"] = run_selection(model, prior, pools, lookup,
                                     device=args.device, seed=args.seed)
        arms["gamma_zero"] = run_selection(model, prior, pools, lookup,
                                           device=args.device, gamma_zero=True,
                                           seed=args.seed)
        # zeroed-input arm: same pools, EEG zeroed in the lookup
        zlook = {t: (torch.zeros_like(w), o, p, ob)
                 for t, (w, o, p, ob) in lookup.items()}
        arms["zeroed"] = run_selection(model, prior, pools, zlook,
                                       device=args.device, seed=args.seed)
        # cross-text EEG: donor lookup deranged by text
        texts = list(lookup.keys())
        rng = np.random.default_rng(args.seed)
        perm = _derangement(len(texts), rng)       # no text keeps its own EEG
        dlook = {texts[i]: lookup[texts[int(perm[i])]] for i in range(len(texts))}
        arms["derangement"] = run_selection(model, prior, pools, dlook,
                                            device=args.device, seed=args.seed)

        row = {"N": N, "pool_gate": gate, "pool_gate_pre_rebuild": gate_pre_rebuild}
        for name, r in arms.items():
            row[name] = {"acc": r.accuracy, "bits": r.bits_realized}
        sweep.append(row)
        print(f"  N={N:<4} real={arms['real'].accuracy:.3f} "
              f"(bits {arms['real'].bits_realized:.2f})  "
              f"g0={arms['gamma_zero'].accuracy:.3f}  "
              f"zero={arms['zeroed'].accuracy:.3f}  "
              f"derang={arms['derangement'].accuracy:.3f}", flush=True)

        base = base_record(args, split_fp=fp, experiment="e3_selection",
                           gates=["build", "channel"] + (["pool_validity"] if gate["passed"] else []))
        for name, r in arms.items():
            led.append(Record(metric=f"selection_acc_N{N}", value=r.accuracy,
                              n_sentences=r.n_pools, arm=name,
                              pool_hash=pools[0].pool_hash if pools else "", **base))

    import numpy as _np
    bits_realized_val = float(_np.median(
        [row["real"]["bits"] for row in sweep if "real" in row])) if sweep else 0.0
    n_star = predict_n_star(bits_sent)
    n_star_ok = n_star_meets_target(bits_sent, n_star)
    if not n_star_ok:
        print(f"  NOTE: {bits_sent:.4f} bits/sentence predicts < 50% accuracy at EVERY N; "
              f"N*={n_star} is a placeholder (recorded as matched_N_meets_target=false)")
    print(f"\nmatched-N prediction from measured {bits_sent:.2f} bits/sentence: "
          f"N* = {n_star}  (FROZEN into prereg; test-split one-shot happens in e4)")

    passed = pools_ok and len(sweep) > 0
    write_gate_artifact(args.runs_dir, "selection", passed,
                        {"sweep": sweep, "matched_N_prediction": n_star,
                         "matched_N_meets_target": bool(n_star_ok),
                         "note": "pools over the full val split; the model's checkpoint was selected on the val selection half",
                         "bits_per_sentence_input": bits_sent,
                         "bits_realized_val": bits_realized_val})
    print(f"gate_selection written (passed={passed})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
