"""Model-ready dataset builder: verified pickles -> HDF5 splits + manifests.

Derivation, not re-preprocessing. Inputs are the four Table-2-verified pickles
(~/datasets/ZuCo/<task>/pickle/). Outputs:

    <out>/zuco2_train.h5, zuco2_val.h5, zuco2_test.h5     (same schema read_split expects,
                                                           plus REAL subject_id/task attrs)
    <out>/split_manifest.json                              (both split kinds, hashed)

Two split kinds, both generated here so nothing ever touches the retired legacy files:
    text     -- global unique-text split across ALL tasks (the honest one)
    instance -- per-reading split (the literature's leaky convention, for dual-split rows)

Tokenization: per-word with correct prefix spaces under a caller-supplied tokenizer
(the prior's), each word's 840-d GD vector replicated across its subword tokens,
unfixated words carried as NaN rows with observed=False (1:1 alignment preserved).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle as pkl

import h5py
import numpy as np

from .covariates import text_hash

BANDS = ("_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2")
FEAT_DIM = 840

# Gaze evidence: the eye-movement record per word, as a SEPARATE evidence channel.
#   [fixated(0/1), log1p(nFixations), log1p(FFD), log1p(GD), log1p(TRT), log1p(GPT)]
# nFixations comes from the pickle; the four durations come from the .mat covariate
# table when it exists (e0 post-pass), else stay 0. Word length / pupil are NOT
# included: word length is a property of the text (would leak identity), pupil is
# sparse. Gaze is fully observed -- a skipped word is a real observation (nfix=0), so
# `observed` is all-True under --evidence gaze. This channel is never fed to the EEG
# model; it is measured on its own so the two can be compared under one estimator.
GAZE_DIM = 6
GAZE_COLS = ("fixated", "log1p_nfix", "log1p_ffd", "log1p_gd", "log1p_trt", "log1p_gpt")


def _word_nfix(wobj) -> float:
    try:
        v = np.atleast_1d(np.asarray(wobj.get("nFixations", 0), dtype=np.float64)).ravel()
        return float(v[0]) if v.size and np.isfinite(v[0]) else 0.0
    except Exception:
        return 0.0


EEG_KEY = "GD"        # which word-level EEG window: GD (gaze duration, default) | FFD | TRT


def _word_feats(wobj) -> np.ndarray | None:
    try:
        f = np.concatenate([np.atleast_1d(wobj["word_level_EEG"][EEG_KEY][EEG_KEY + b])
                            for b in BANDS]).astype(np.float32)
    except Exception:
        return None
    return f if f.shape[0] == FEAT_DIM else None


def iter_readings(pickle_root: str, tasks):
    """Yield (subject, task, sent_idx, words_all, fixated_feats: {word_idx: (840,)})."""
    for task in tasks:
        # Accept either the stock pickle or the *_wRaw one. The _wRaw build (used by
        # the Amrani baseline) is a strict superset: same schema plus a per-word
        # `rawEEG` key, so `word_level_EEG` -- all this function reads -- is identical.
        # Preferring the stock name keeps existing runs bit-identical; falling back to
        # _wRaw means ONE .mat download + ONE rebuild serves both baselines.
        cands = [os.path.join(pickle_root, task, "pickle", f"{task}-dataset.pickle"),
                 os.path.join(pickle_root, task, "pickle", f"{task}-dataset_wRaw.pickle")]
        p = next((c for c in cands if os.path.exists(c)), None)
        if p is None:
            continue
        with open(p, "rb") as fh:
            d = pkl.load(fh)
        for subject in sorted(d.keys()):
            for si, sent in enumerate(d[subject]):
                if not sent:
                    continue
                words_all = list(sent.get("word_tokens_all", []))
                if not words_all:
                    continue
                feats, nfix = {}, {}
                wi = 0
                for wobj in sent.get("word", []):
                    while wi < len(words_all) and words_all[wi] != wobj["content"]:
                        wi += 1
                    if wi >= len(words_all):
                        break
                    f = _word_feats(wobj)
                    if f is not None and not np.isnan(f).all():
                        feats[wi] = f
                    nfix[wi] = _word_nfix(wobj)
                    wi += 1
                yield subject, task, si, words_all, feats, nfix


def tokenize_reading(words_all, feats, tokenizer, nfix=None):
    """Per-word tokenization with prefix spaces; EEG replicated per subword; NaN rows
    where unfixated. Returns (token_ids, eeg, observed, word_index, gaze) or None if
    any word yields no tokens. `gaze` carries only the pickle-derived columns here
    (fixated, log1p nfix); durations are filled by the e0 post-pass from the .mat."""
    ids, eeg, obs, widx, gaze = [], [], [], [], []
    nfix = nfix or {}
    for wi, word in enumerate(words_all):
        piece = word if wi == 0 else " " + word
        widS = tokenizer(piece, add_special_tokens=False)["input_ids"]
        if not widS:
            return None
        vec = feats.get(wi)
        nf = float(nfix.get(wi, 0.0))
        gvec = np.zeros(GAZE_DIM, dtype=np.float32)
        gvec[0] = 1.0 if nf > 0 else 0.0
        gvec[1] = np.log1p(nf)
        for t in widS:
            ids.append(t); widx.append(wi); gaze.append(gvec)
            if vec is not None:
                eeg.append(vec); obs.append(True)
            else:
                eeg.append(np.full(FEAT_DIM, np.nan, dtype=np.float32)); obs.append(False)
    return (np.asarray(ids, dtype=np.int64),
            np.stack(eeg).astype(np.float32),
            np.asarray(obs, dtype=bool),
            np.asarray(widx, dtype=np.int32),
            np.stack(gaze).astype(np.float32))


def make_splits(all_texts: list[str], ratios=(0.8, 0.1, 0.1), seed: int = 0) -> dict:
    """text-hash based global unique-text split. Deterministic."""
    uniq = sorted(set(all_texts))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n = len(uniq)
    a, b = int(ratios[0] * n), int(ratios[0] * n) + int(ratios[1] * n)
    assign = {}
    for rank, idx in enumerate(order):
        split = "train" if rank < a else ("val" if rank < b else "test")
        assign[uniq[idx]] = split
    return assign


def build(pickle_root: str, out_dir: str, tokenizer, tasks,
          split_kind: str = "text", seed: int = 0, verbose: bool = True) -> dict:
    """Build the three HDF5 splits + manifest. Returns summary dict."""
    os.makedirs(out_dir, exist_ok=True)
    readings = list(iter_readings(pickle_root, tasks))
    if verbose:
        print(f"  {len(readings)} readings from {len(tasks)} tasks", flush=True)

    texts = [" ".join(w) for _, _, _, w, _, _ in readings]
    if split_kind == "text":
        assign_by_text = make_splits(texts, seed=seed)
        assign = [assign_by_text[t] for t in texts]
    elif split_kind == "instance":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(readings))
        n = len(readings)
        a, b = int(0.8 * n), int(0.9 * n)
        assign = [""] * n
        for rank, idx in enumerate(order):
            assign[idx] = "train" if rank < a else ("val" if rank < b else "test")
    else:
        raise ValueError(split_kind)

    files = {s: h5py.File(os.path.join(out_dir, f"zuco2_{s}.h5"), "w") for s in ("train", "val", "test")}
    counters = {s: 0 for s in files}
    skipped = 0
    for (subject, task, si, words_all, feats, nfix), split in zip(readings, assign):
        text = " ".join(words_all)
        tok = tokenize_reading(words_all, feats, tokenizer, nfix=nfix)
        if tok is None:
            skipped += 1
            continue
        ids, eeg, obs, widx, gaze = tok
        g = files[split].require_group("sentences").create_group(str(counters[split]))
        g.create_dataset("token_ids", data=ids)
        g.create_dataset("eeg_features", data=eeg)
        g.create_dataset("nan_mask", data=~obs)
        g.create_dataset("word_index", data=widx)
        g.create_dataset("gaze_features", data=gaze)
        g.attrs["gaze_cols"] = ",".join(GAZE_COLS)
        g.attrs["gaze_source"] = "pickle"
        g.attrs["text"] = text
        g.attrs["text_hash"] = text_hash(text)
        g.attrs["task"] = task
        g.attrs["subject_id"] = subject
        g.attrs["source_sent_idx"] = si
        counters[split] += 1
    for f in files.values():
        f.close()

    manifest = {
        "split_kind": split_kind, "seed": seed, "tasks": list(tasks),
        "counts": counters, "skipped": skipped,
        "n_unique_texts": len(set(texts)),
        "split_hash": hashlib.sha256(
            json.dumps(assign, sort_keys=False).encode()).hexdigest()[:16],
    }
    with open(os.path.join(out_dir, "split_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    if verbose:
        print(f"  counts={counters} skipped={skipped} "
              f"unique_texts={manifest['n_unique_texts']}", flush=True)
    return manifest


def fill_gaze_durations(out_dir: str, covariates_csv: str, verbose: bool = True) -> dict:
    """e0 post-pass: fill gaze columns 2..5 (FFD, GD, TRT, GPT; log1p, samples @500Hz)
    from the .mat covariate table, joined on (subject, text_hash, word_idx). Words the
    table lacks keep 0 in those columns. Returns per-split fill statistics."""
    from .covariates import CovariateTable
    cov = CovariateTable.load(covariates_csv)
    stats = {}
    for split in ("train", "val", "test"):
        p = os.path.join(out_dir, f"zuco2_{split}.h5")
        n_sent = n_hit = n_word = 0
        with h5py.File(p, "a") as f:
            g0 = f["sentences"]
            for k in g0.keys():
                g = g0[k]
                if "gaze_features" not in g or "word_index" not in g:
                    continue
                gz = np.asarray(g["gaze_features"][:], dtype=np.float32)
                widx = np.asarray(g["word_index"][:], dtype=np.int64)
                subj = str(g.attrs.get("subject_id", ""))
                th = str(g.attrs.get("text_hash", ""))
                task = str(g.attrs.get("task", ""))
                n_sent += 1
                for wi in np.unique(widx):
                    n_word += 1
                    # task-aware join: the same subject may read the same text in NR and TSR
                    r = (cov.rows_by_task or {}).get((subj, task, th, int(wi)))
                    if r is None and not cov.rows_by_task:
                        r = cov.rows.get((subj, th, int(wi)))
                    if r is None:
                        continue
                    n_hit += 1
                    vals = [r.get("ffd"), r.get("gd"), r.get("trt"), r.get("gpt")]
                    vals = [np.log1p(v) if (v is not None and np.isfinite(v) and v >= 0) else 0.0
                            for v in vals]
                    gz[widx == wi, 2:6] = np.asarray(vals, dtype=np.float32)
                g["gaze_features"][...] = gz
                g.attrs["gaze_source"] = "pickle+mat"
        stats[split] = {"sentences": n_sent, "words": n_word, "words_with_durations": n_hit,
                        "fill_frac": (n_hit / n_word) if n_word else 0.0}
        if verbose:
            print(f"  gaze durations {split}: {n_hit}/{n_word} words filled "
                  f"({100*stats[split]['fill_frac']:.1f}%)", flush=True)
    return stats
