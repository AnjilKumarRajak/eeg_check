
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cprd.build import build                                    # noqa: E402
from cprd.data import assert_no_text_leakage, read_split        # noqa: E402
from cprd.prereg import write_gate_artifact                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle-root", default=os.path.expanduser("~/datasets/ZuCo"))
    ap.add_argument("--mat-root", default=os.path.expanduser("~/datasets/ZuCo"))
    ap.add_argument("--out", default="runs/data")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--prior-model", default="gpt2-large",
                    help="tokenizer source for the build")
    ap.add_argument("--tasks", default="task1-SR,task2-NR,task3-TSR,task2-NR-2.0")
    ap.add_argument("--split-kind", default="text", choices=["text", "instance"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-covariates", action="store_true")
    ap.add_argument("--eeg-key", default="GD", choices=["GD", "FFD", "TRT"],
                    help="which word-level EEG window supplies the 840-d features (robustness sweep)")
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic build: no pickles/mats/downloads needed")
    args = ap.parse_args()
    import cprd.build as _b
    _b.EEG_KEY = args.eeg_key

    if args.smoke:
        import h5py
        import numpy as np
        from cprd.synth import make_channel, make_synthetic_sentences
        os.makedirs(args.out, exist_ok=True)
        from cprd.build import GAZE_COLS, GAZE_DIM
        ch = make_channel(bits=1.0, seed=args.seed, n_mc=20_000)
        sents = make_synthetic_sentences(ch, 60, vocab_size=64,
                                         missing_frac=0.3, seed=args.seed)
        cuts = {"train": sents[:40], "val": sents[40:50], "test": sents[50:]}
        rng = np.random.default_rng(args.seed + 11)
        for split, ss in cuts.items():
            with h5py.File(os.path.join(args.out, f"zuco2_{split}.h5"), "w") as f:
                g0 = f.require_group("sentences")
                for i, s in enumerate(ss):
                    T = int(s.token_ids.shape[0])
                    gz = np.zeros((T, GAZE_DIM), dtype=np.float32)
                    gz[:, 0] = s.observed.astype(np.float32)
                    gz[:, 1:] = rng.normal(0.0, 1.0, size=(T, GAZE_DIM - 1)) * gz[:, :1]
                    g = g0.create_group(str(i))
                    g.create_dataset("token_ids", data=s.token_ids)
                    g.create_dataset("eeg_features", data=s.eeg)
                    g.create_dataset("nan_mask", data=~s.observed)
                    g.create_dataset("word_index", data=np.arange(T, dtype=np.int32))
                    g.create_dataset("gaze_features", data=gz)
                    g.attrs["gaze_cols"] = ",".join(GAZE_COLS)
                    g.attrs["gaze_source"] = "smoke"
                    g.attrs["text"] = s.text
                    g.attrs["task"] = s.task
                    g.attrs["subject_id"] = s.subject_id
        manifest = {"split_kind": "smoke", "counts": {k: len(v) for k, v in cuts.items()},
                    "gaze": {"source": "smoke", "dim": GAZE_DIM}}
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.prior_model)
        tasks = tuple(t.strip() for t in args.tasks.split(","))
        manifest = build(args.pickle_root, args.out, tok, tasks,
                         split_kind=args.split_kind, seed=args.seed)
        from cprd.build import GAZE_COLS, GAZE_DIM, fill_gaze_durations
        manifest["gaze"] = {"source": "pickle", "dim": GAZE_DIM, "cols": list(GAZE_COLS)}
        if not args.skip_covariates:
            from cprd.covariates import extract_all
            print("extracting gaze covariates from .mat (one read-only pass)...", flush=True)
            cov_csv = os.path.join(args.out, "covariates.csv")
            counts = extract_all(args.mat_root, cov_csv)
            manifest["covariate_rows"] = counts
            if sum(counts.values()) > 0:
                print("filling gaze durations into gaze_features from covariates...", flush=True)
                manifest["gaze"]["source"] = "pickle+mat"
                manifest["gaze"]["fill"] = fill_gaze_durations(args.out, cov_csv)
            else:
                print("  no .mat rows found; gaze channel is pickle-only (fixated, nfix)", flush=True)

    splits = {s: read_split(os.path.abspath(os.path.join(args.out, f"zuco2_{s}.h5")))
              for s in ("train", "val", "test")}
    # the gaze channel must be readable from the same files, with the same row counts
    for s_, sents_ in splits.items():
        gz = read_split(os.path.abspath(os.path.join(args.out, f"zuco2_{s_}.h5")),
                        evidence="gaze", limit=5)
        assert all(a.token_ids.shape[0] == b.eeg.shape[0] for a, b in zip(sents_[:5], gz)), \
            f"{s_}: gaze_features rows do not match token counts"
        assert all(bool(b.observed.all()) for b in gz), f"{s_}: gaze must be fully observed"
    print(f"gaze channel: readable in all splits, dim={gz[0].eeg.shape[1]}, "
          f"source={manifest.get('gaze', {}).get('source')}", flush=True)
    leak_ok = True
    if args.split_kind == "text" or args.smoke:
        try:
            assert_no_text_leakage(splits)
            print("leakage gate: PASS", flush=True)
        except AssertionError as e:
            leak_ok = False
            print(f"leakage gate: FAIL — {e}", flush=True)
    write_gate_artifact(args.runs_dir, "build",
                        passed=leak_ok, detail=manifest)
    print(f"gate_build written (passed={leak_ok})")
    return 0 if leak_ok else 1


if __name__ == "__main__":
    sys.exit(main())
