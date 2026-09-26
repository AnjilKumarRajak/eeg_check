#!/usr/bin/env python3
"""Adapter: COFETT per-character band-power (extract_features.py .npz) -> the HDF5
schema cprd/data.py::read_split expects, so the unmodified e2_channel.py runs on it.

EEG-only (no gaze exists): gaze_features written as zeros, evidence must be "eeg".
Tokenisation: each character is tokenised on its own with the prior's tokenizer
(Qwen2.5 by default; Chinese characters are almost always single tokens) and the
character's 840-d vector is replicated over its sub-tokens, exactly as
cprd/build.py replicates a word's vector. Punctuation characters (never
highlighted) are NaN / unobserved, like unfixated words in ZuCo.

--phase reading : exact 0.4 s/character alignment (the controlled stage)
--phase recall  : nominal uniform alignment inside the silent-recall interval
Text unit = the sentence string; split by text (no overlap) with cprd's make_splits.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import h5py
import numpy as np

CPRD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (contains cprd/)
sys.path.insert(0, CPRD_ROOT)
from cprd.build import GAZE_COLS, GAZE_DIM, make_splits          # noqa: E402
from cprd.covariates import text_hash                             # noqa: E402
from cprd.prereg import write_gate_artifact                       # noqa: E402

FEAT_DIM = 840


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--prior-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--phase", default="reading", choices=["reading", "recall"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-kind", default="text", choices=["text", "repeat", "repeat16"],
                    help="text: text-disjoint (default). repeat: COFETT's own design -- the same sentences appear in "
                         "train and test; test = last repetition (para2 run-04), val = run-03, everything else train")
    ap.add_argument("--only-subject", default="", help="keep only this participant (e.g. sub-01): COFETT trains one model per participant")
    ap.add_argument("--spike-alpha", type=float, default=0.0,
                    help="positive control: add alpha x per-feature std x (random 6->840 map of a fixed random 6-d code of the character's token id) "
                         "to the observed EEG vectors. The code is a deterministic function of the target token, so it carries known information "
                         "about the text beyond the prior; COFETT has no gaze channel to borrow one from.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # repeat16: the COFETT benchmark scheme with all four sessions of the 252-sentence list (para2, 4 runs x 4 sessions = 16 repetitions
    # per participant): test = repetitions 15-16, validation = 13-14, training = repetitions 1-12 plus every para1 reading.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.prior_model)
    key = "feats_read" if args.phase == "reading" else "feats_recall"

    readings = []       # (subject, session, run, sentence, feats(n_chars, 840) with NaN rows)
    files = sorted(f for d_ in args.feat_dir.split(",") for f in glob.glob(os.path.join(d_, "*.npz")))
    if not files:
        sys.exit("no feature files")
    for f in files:
        z = np.load(f, allow_pickle=True)
        run = str(z["run"]); subj = run.split("_")[0]; ses = run.split("_")[1]
        F = z[key]; keep = z["keep_chars"]; sents = list(z["sentences"])
        assert F.shape[2] == FEAT_DIM, F.shape
        for i, s in enumerate(sents):
            n = len(s)
            if not keep[i].any():
                continue                          # timing mismatch -> skipped by extractor
            fe = F[i, :n]
            if args.phase == "recall":
                fe = np.where(keep[i, :n, None], fe, np.nan)
            if args.only_subject and subj != args.only_subject: continue
            readings.append((subj, ses, run, s, fe.astype(np.float32)))
    texts = [r[3] for r in readings]
    print(f"{len(readings)} readings from {len(files)} runs, {len(set(texts))} unique sentences", flush=True)

    if args.spike_alpha > 0:
        _rows = np.concatenate([fe[np.isfinite(fe).all(axis=1)] for *_, fe in readings], axis=0)
        SD = _rows.std(axis=0).astype(np.float32); del _rows
        _G = np.random.default_rng(1234).normal(size=(6, FEAT_DIM)).astype(np.float32) / np.sqrt(6.0)
        def _spike(tid):
            z = np.random.default_rng(int(tid) + 99991).normal(size=6).astype(np.float32)
            return args.spike_alpha * SD * (z @ _G)
    else:
        _spike = None
    # per-character-set standardisation is NOT applied here (features are used as
    # extracted, mirroring ZuCo 'as released'); log power already taken.
    if args.split_kind == "repeat16":
        def _rep16(run):
            if "task-para2" not in run: return "train"
            ses = int(run.split("_ses-")[1][:2]); rr = int(run.split("run-")[1][:2]); rep = (ses - 1) * 4 + rr
            return "test" if rep >= 15 else ("val" if rep >= 13 else "train")
        assign_of = lambda s_, run_: _rep16(run_)
    elif args.split_kind == "repeat":
        def _rep_split(run):
            if "task-para2" in run and run.endswith("run-04"): return "test"
            if "task-para2" in run and run.endswith("run-03"): return "val"
            return "train"
        assign_of = lambda s_, run_: _rep_split(run_)
    else:
        _a = make_splits(texts, seed=args.seed)
        assign_of = lambda s_, run_: _a[s_]
    h5 = {s: h5py.File(os.path.join(args.out, f"zuco2_{s}.h5"), "w") for s in ("train", "val", "test")}
    counters = {s: 0 for s in h5}; skipped = 0
    for subj, ses, run, s, fe in readings:
        ids, eeg, obs, widx = [], [], [], []
        for ci, ch in enumerate(s):
            t_ids = tok(ch, add_special_tokens=False)["input_ids"]
            if not t_ids:
                continue
            row = fe[ci]; ok = bool(np.isfinite(row).all())
            for t in t_ids:
                ids.append(t); widx.append(ci)
                eeg.append((row + _spike(t) if (_spike is not None and ok) else row) if ok else np.full(FEAT_DIM, np.nan, np.float32)); obs.append(ok)
        if len(ids) < 2:
            skipped += 1; continue
        split = assign_of(s, run)
        g = h5[split].require_group("sentences").create_group(str(counters[split]))
        g.create_dataset("token_ids", data=np.asarray(ids, dtype=np.int64))
        g.create_dataset("eeg_features", data=np.stack(eeg).astype(np.float32))
        g.create_dataset("nan_mask", data=~np.asarray(obs, dtype=bool))
        g.create_dataset("word_index", data=np.asarray(widx, dtype=np.int32))
        g.create_dataset("gaze_features", data=np.zeros((len(ids), GAZE_DIM), np.float32))
        g.attrs["gaze_cols"] = ",".join(GAZE_COLS); g.attrs["gaze_source"] = "none"
        g.attrs["text"] = s; g.attrs["text_hash"] = text_hash(s)
        g.attrs["task"] = f"cofett-{args.phase}"; g.attrs["subject_id"] = subj
        g.attrs["session"] = ses; g.attrs["run"] = run
        counters[split] += 1
    for f in h5.values():
        f.close()
    manifest = {"dataset": f"COFETT ds006317, phase={args.phase}", "spike_alpha": args.spike_alpha, "only_subject": args.only_subject, "split_kind": args.split_kind, "seed": args.seed,
                "counts": counters, "skipped": skipped, "n_unique_texts": len(set(texts)),
                "runs": [os.path.basename(f) for f in files], "prior_model": args.prior_model,
                "eeg": {"dim": FEAT_DIM, "source": "extract_features.py (8 bands x 105 ch, log Hilbert power per character)"},
                "gaze": {"source": "none"}}
    json.dump(manifest, open(os.path.join(args.out, "split_manifest.json"), "w"), indent=2)
    print(f"counts={counters} skipped={skipped}", flush=True)
    from cprd.data import assert_no_text_leakage, read_split
    splits = {s: read_split(os.path.abspath(os.path.join(args.out, f"zuco2_{s}.h5"))) for s in ("train", "val", "test")}
    ok = True
    if args.split_kind == "text":
        try:
            assert_no_text_leakage(splits); print("leakage gate: PASS", flush=True)
        except AssertionError as e:
            ok = False; print(f"leakage gate: FAIL - {e}", flush=True)
    else:
        manifest["note"] = f"{args.split_kind} split: sentences are shared across splits BY DESIGN (mirrors the COFETT protocol); this is not a text-disjoint evaluation"
        print("repeat split: leakage gate not applied (shared sentences by design)", flush=True)
    write_gate_artifact(args.runs_dir, "build", passed=ok, detail=manifest)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
