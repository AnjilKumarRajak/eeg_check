
from __future__ import annotations
import argparse, hashlib, json, os, sys
import h5py
import numpy as np
import pandas as pd

CPRD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (contains cprd/)
sys.path.insert(0, CPRD_ROOT)
from cprd.build import GAZE_COLS, GAZE_DIM, make_splits          # noqa: E402
from cprd.covariates import text_hash                             # noqa: E402
from cprd.prereg import write_gate_artifact                       # noqa: E402

COLS = ["participant_id", "TRIAL_INDEX", "IA_ID", "IA_LABEL",
        "IA_FIXATION_COUNT", "IA_FIRST_FIXATION_DURATION", "IA_FIRST_RUN_DWELL_TIME",
        "IA_DWELL_TIME", "IA_REGRESSION_PATH_DURATION", "IA_SKIP",
        "article_id", "paragraph_id", "difficulty_level", "practice_trial",
        "repeated_reading_trial"]


def load_rows(csv_path: str) -> pd.DataFrame:
    chunks = []
    for ch in pd.read_csv(csv_path, usecols=COLS, chunksize=500_000):
        ch = ch[(~ch.practice_trial) & (~ch.repeated_reading_trial.fillna(False))]
        chunks.append(ch)
    df = pd.concat(chunks, ignore_index=True)
    df["IA_LABEL"] = df["IA_LABEL"].astype(str)
    for c in ["IA_FIXATION_COUNT", "IA_FIRST_FIXATION_DURATION", "IA_FIRST_RUN_DWELL_TIME",
              "IA_DWELL_TIME", "IA_REGRESSION_PATH_DURATION"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["IA_ID"] = pd.to_numeric(df["IA_ID"], errors="coerce")
    df = df.dropna(subset=["IA_ID"])
    df["IA_ID"] = df["IA_ID"].astype(int)
    return df


def build_reading(g: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    g = g.sort_values("IA_ID")
    words = g["IA_LABEL"].tolist()
    gaze = np.zeros((len(g), GAZE_DIM), dtype=np.float32)
    nfix = g["IA_FIXATION_COUNT"].to_numpy(dtype=np.float64)
    gaze[:, 0] = (nfix > 0).astype(np.float32)
    gaze[:, 1] = np.log1p(np.clip(nfix, 0, None))
    gaze[:, 2] = np.log1p(np.clip(g["IA_FIRST_FIXATION_DURATION"].to_numpy(dtype=np.float64), 0, None))
    gaze[:, 3] = np.log1p(np.clip(g["IA_FIRST_RUN_DWELL_TIME"].to_numpy(dtype=np.float64), 0, None))
    gaze[:, 4] = np.log1p(np.clip(g["IA_DWELL_TIME"].to_numpy(dtype=np.float64), 0, None))
    gaze[:, 5] = np.log1p(np.clip(g["IA_REGRESSION_PATH_DURATION"].to_numpy(dtype=np.float64), 0, None))
    return words, gaze


def tokenize_reading(words, gaze, tokenizer):
    ids, gz, widx = [], [], []
    for wi, word in enumerate(words):
        if not isinstance(word, str) or not word.strip():
            continue
        piece = word if wi == 0 else " " + word
        widS = tokenizer(piece, add_special_tokens=False)["input_ids"]
        if not widS:
            continue
        for t in widS:
            ids.append(t); widx.append(wi); gz.append(gaze[wi])
    if not ids:
        return None
    return (np.asarray(ids, dtype=np.int64), np.stack(gz).astype(np.float32),
            np.asarray(widx, dtype=np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--prior-model", default="gpt2-large")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-participants", type=int, default=0, help="0 = all")
    ap.add_argument("--split-kind", default="text", choices=["text", "instance"],
                    help="text: text-disjoint (default). instance: readings assigned to train/val/test independently of text identity (leakage ablation)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.prior_model)

    print("loading IA report ...", flush=True)
    df = load_rows(args.csv)
    print(f"  {len(df)} word rows after dropping practice/repeated", flush=True)

    df["text_id"] = list(zip(df.article_id, df.paragraph_id, df.difficulty_level))
    parts = sorted(df.participant_id.unique())
    if args.max_participants:
        rng = np.random.default_rng(args.seed)
        parts = sorted(rng.choice(parts, size=min(args.max_participants, len(parts)), replace=False))
        df = df[df.participant_id.isin(parts)]
    print(f"  participants used: {len(parts)}", flush=True)

    readings = []   # (subject, text_id, text_str, words, gaze)
    for (subj, trial, tid), g in df.groupby(["participant_id", "TRIAL_INDEX", "text_id"], sort=False):
        words, gaze = build_reading(g)
        if len(words) < 3:
            continue
        readings.append((subj, tid, " ".join(words), words, gaze))
    print(f"  {len(readings)} readings, {len(set(t for _, t, *_ in readings))} unique texts", flush=True)

    texts = [t for _, t, *_ in readings]
    if args.split_kind == "instance":
        _rng = np.random.default_rng(args.seed + 11); _order = _rng.permutation(len(readings)); _n = len(readings)
        _a, _b = int(0.8 * _n), int(0.9 * _n); _lab = [""] * _n
        for _rank, _idx in enumerate(_order):
            _lab[_idx] = "train" if _rank < _a else ("val" if _rank < _b else "test")
        splits_assigned = _lab
    else:
        assign_by_text = make_splits([str(t) for t in texts], seed=args.seed)
        splits_assigned = [assign_by_text[str(t)] for t in texts]
    files = {s: h5py.File(os.path.join(args.out, f"zuco2_{s}.h5"), "w") for s in ("train", "val", "test")}
    counters = {s: 0 for s in files}
    skipped = 0
    for (subj, tid, text_str, words, gaze), split in zip(readings, splits_assigned):
        tok_out = tokenize_reading(words, gaze, tok)
        if tok_out is None:
            skipped += 1; continue
        ids, gz, widx = tok_out
        T = ids.shape[0]
        g = files[split].require_group("sentences").create_group(str(counters[split]))
        g.create_dataset("token_ids", data=ids)
        g.create_dataset("eeg_features", data=np.full((T, 840), np.nan, dtype=np.float32))
        g.create_dataset("nan_mask", data=np.ones(T, dtype=bool))     # EEG: fully unobserved (none exists)
        g.create_dataset("word_index", data=widx)
        g.create_dataset("gaze_features", data=gz)
        g.attrs["gaze_cols"] = ",".join(GAZE_COLS)
        g.attrs["gaze_source"] = "onestop_ia_report"
        g.attrs["text"] = text_str
        g.attrs["text_hash"] = text_hash(text_str)
        g.attrs["task"] = "onestop-ordinary"
        g.attrs["subject_id"] = subj
        g.attrs["source_text_id"] = str(tid)
        counters[split] += 1
    for f in files.values():
        f.close()

    manifest = {
        "dataset": "OneStop (ordinary reading, Adv+Ele as separate texts, no practice/repeat)",
        "split_kind": args.split_kind, "seed": args.seed, "counts": counters, "skipped": skipped,
        "n_unique_texts": len(set(str(t) for t in texts)), "n_participants": len(parts),
        "gaze": {"source": "onestop_ia_report", "dim": GAZE_DIM, "cols": list(GAZE_COLS)},
        "eeg": {"source": "none - OneStop has no EEG; eeg channel is fully unobserved by construction"},
    }
    with open(os.path.join(args.out, "split_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"counts={counters} skipped={skipped}", flush=True)

    from cprd.data import assert_no_text_leakage, read_split
    splits = {s: read_split(os.path.abspath(os.path.join(args.out, f"zuco2_{s}.h5"))) for s in ("train", "val", "test")}
    leak_ok = True
    if args.split_kind == "text":
        try:
            assert_no_text_leakage(splits)
            print("leakage gate: PASS", flush=True)
        except AssertionError as e:
            leak_ok = False
            print(f"leakage gate: FAIL - {e}", flush=True)
    else:
        print("instance split: text overlap across splits by design (leakage ablation)", flush=True)
    write_gate_artifact(args.runs_dir, "build", passed=leak_ok, detail=manifest)
    print(f"gate_build written (passed={leak_ok})")
    return 0 if leak_ok else 1


if __name__ == "__main__":
    sys.exit(main())
