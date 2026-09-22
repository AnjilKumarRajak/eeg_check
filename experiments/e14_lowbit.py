#!/usr/bin/env python3
"""E14: low-bit calibration of the estimator with the evaluation gain chosen on the selection half.

For each known capacity b (bits/token) and seed: build the prior-matched synthetic channel exactly as E1,
train with the E2-parity procedure (re-using a finished E1 training state when one exists), then
  1. score the SELECTION half at each gamma in --gamma-grid (derangement null, few draws) and pick the
     gamma with the largest InfoNCE bound  -> never reported as a result;
  2. score the disjoint MEASUREMENT half at the default gain and at the picked gain with the full
     permutation protocol (both nulls).
Output: <runs-dir>/e14_lowbit.jsonl (one row per b, seed) and gate_lowbit.json (summary: mean recovered,
tightness, detection power, false-positive rate at b=0, never-exceeds-truth check).
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (apply_determinism, base_parser, build_prior, model_config_from_args,   # noqa: E402
                    prepare_batches, train_config_from_args)
import e1_estimator as E1                                                                  # noqa: E402
from cprd.data import selection_measurement_split                                          # noqa: E402
from cprd.evaluate import evaluate                                                         # noqa: E402
from cprd.model import PriorResidualModel                                                  # noqa: E402
from cprd.prereg import write_gate_artifact                                                # noqa: E402
from cprd.synth import make_channel, make_synthetic_sentences                              # noqa: E402
from cprd.train import train                                                               # noqa: E402
LN2 = math.log(2.0)


def main():
    ap = base_parser("E14 low-bit calibration with selection-half gain choice")
    ap.add_argument("--sweep", default="0,0.05,0.1,0.25,0.5")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n-sent", type=int, default=2000)
    ap.add_argument("--gamma-grid", default="0.127,0.25,0.5")
    ap.add_argument("--n-perm-select", type=int, default=40)
    ap.add_argument("--reuse-ckpt-dir", default="", help="ckpt_synth dir of a finished E1 run with the same config")
    # fields config_key() reads, fixed to the E1 power-run values so finished training states are re-used
    ap.add_argument("--synth-tokens", default="prior"); ap.add_argument("--gate-mode", default="lower_bound")
    args = ap.parse_args()
    apply_determinism(args)
    prior = build_prior(args)
    os.makedirs(args.runs_dir, exist_ok=True)
    out_path = os.path.join(args.runs_dir, "e14_lowbit.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            r = json.loads(l); done.add((r["b_nominal"], r["seed_index"]))
    grid = [float(g) for g in args.gamma_grid.split(",")]
    key = E1.config_key(args)[:16]
    for b in [float(x) for x in args.sweep.split(",")]:
        for sd in [int(x) for x in args.seeds.split(",")]:
            if (b, sd) in done:
                continue
            seed = args.seed + sd * 101
            ch = make_channel(bits=b, seed=seed, n_mc=200_000)
            n = args.n_sent; n_total = n + max(80, n // 3)
            toks = E1.sample_prior_sentences(prior, n_total, seed, args.device)
            cl = E1.balanced_embedding_clusters(prior, ch.n_clusters, seed, toks) if ch.n_clusters > 1 else None
            sents = make_synthetic_sentences(ch, n_total, prior.vocab_size, missing_frac=0.2, seed=seed,
                                             token_clusters=cl, token_sequences=toks)
            tr, va = sents[:n], sents[n:]
            va_sel, va_meas = selection_measurement_split(va)
            mcfg = model_config_from_args(args); tcfg = train_config_from_args(args, seed=seed, n_perm_val=20)
            model = PriorResidualModel(prior, mcfg)
            tag = f"e1_{key}_b{b}_s{seed}"
            reused = False
            for d in (args.reuse_ckpt_dir.split(",") + [os.path.join(args.runs_dir, "ckpt_synth")]):
                sp = os.path.join(d, f"train_state_{tag}.pt") if d else ""
                if sp and os.path.exists(sp):
                    st = torch.load(sp, map_location="cpu")
                    if (int(st["epoch"]) + 1 >= tcfg.epochs or int(st.get("patience_ctr", 0)) >= tcfg.patience) and st.get("best_state") is not None:
                        model.load_state_dict(st["best_state"]); reused = True
                        print(f"  [b={b} s={sd}] re-using finished training state {sp}", flush=True)
                        break
            if not reused:
                train(model, tr, va_sel, mcfg, tcfg, os.path.join(args.runs_dir, "ckpt_synth"), state_tag=tag)
            model.to(args.device); model.eval()
            bs, lps = prepare_batches(va_sel, prior, args.device, window=args.window, batch_size=args.batch_size)
            bm, lpm = prepare_batches(va_meas, prior, args.device, window=args.window, batch_size=args.batch_size)
            base_mode, base_const = model.tilt.gamma_mode, model.tilt.gamma_const
            sel = {}
            for g in grid:
                model.tilt.gamma_mode, model.tilt.gamma_const = "constant", g
                r = evaluate(model, bs, lps, nulls=("sentence_derangement",), n_perm=args.n_perm_select, n_boot=10, seed=seed)
                sel[g] = r.dI_hat_nce["sentence_derangement"] / LN2
            g_star = max(sel, key=sel.get)
            row = {"b_nominal": b, "b_true": ch.true_bits_per_token, "seed_index": sd, "seed": seed, "reused_training": reused,
                   "masking_rate_nominal": 0.2, "masking_rate_actual": float(1.0 - np.concatenate([x.observed for x in sents]).mean()), "sigma": ch.sigma, "n_clusters": ch.n_clusters, "selection_half_bits_by_gamma": {str(k): v for k, v in sel.items()}, "gamma_selected": g_star}
            for name, g in (("default", grid[0]), ("selected", g_star)):
                if name == "selected" and g == grid[0]:
                    row["selected"] = dict(row["default"]); continue
                model.tilt.gamma_mode, model.tilt.gamma_const = "constant", g
                r = evaluate(model, bm, lpm, nulls=("sentence_derangement", "temporal_roll"), n_perm=args.n_perm, n_boot=200, seed=seed)
                row[name] = {"gamma": g, "bits": r.dI_hat_nce["sentence_derangement"] / LN2, "bits_roll": r.dI_hat_nce["temporal_roll"] / LN2,
                             "dv_bits": r.dI_hat_dv["sentence_derangement"] / LN2,
                             "p": r.p_value["sentence_derangement"], "p_roll": r.p_value["temporal_roll"]}
            model.tilt.gamma_mode, model.tilt.gamma_const = base_mode, base_const
            with open(out_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"== b={b} seed={sd}: true={row['b_true']:.3f} | default g={grid[0]}: {row['default']['bits']:+.4f} p={row['default']['p']:.4f} | "
                  f"selected g={g_star}: {row['selected']['bits']:+.4f} p={row['selected']['p']:.4f}", flush=True)
            del model; torch.cuda.empty_cache()
    rows = [json.loads(l) for l in open(out_path)]
    summ = {}
    for b in sorted({r["b_nominal"] for r in rows}):
        R = [r for r in rows if r["b_nominal"] == b]
        s = {"n_seeds": len(R), "b_true_mean": float(np.mean([r["b_true"] for r in R]))}
        for name in ("default", "selected"):
            v = np.array([r[name]["bits"] for r in R]); p = np.array([max(r[name]["p"], r[name]["p_roll"]) for r in R])
            s[name] = {"mean_bits": float(v.mean()), "sd_bits": float(v.std()), "max_bits": float(v.max()),
                       "detected_both_nulls": int((p < 0.05).sum()), "power": float((p < 0.05).mean()),
                       "tightness": (float(v.mean() / s["b_true_mean"]) if s["b_true_mean"] > 0 else None),
                       "exceeds_truth": bool((v > np.array([r["b_true"] for r in R]) + 0.05).any())}
        s["gamma_selected"] = [r["gamma_selected"] for r in R]
        summ[str(b)] = s
    ok = all(not s[n]["exceeds_truth"] for s in summ.values() for n in ("default", "selected"))
    write_gate_artifact(args.runs_dir, "lowbit", ok, {"by_capacity": summ, "gamma_grid": grid, "n_perm": args.n_perm,
                        "n_perm_select": args.n_perm_select, "n_sent": args.n_sent,
                        "detection_rule": "p<0.05 under BOTH nulls (same as H1)"})
    print("gate_lowbit written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
