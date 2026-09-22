"""ZuCo loading with missingness and metadata preserved end-to-end.

Measured properties of <PROJECT_ROOT>/zuco_*.h5:

    train 7511 / val 925 / test 931 sentences
    840 = 105 channels x 8 bands, float32
    43.5% of test tokens are FULLY NaN (unfixated words); nan_mask agrees with the
      all-NaN rows on 27074/27074 tokens -- there are no partial rows
    no BOS/EOS tokens; sentences end with id 4 ('.')
    zero sentence-text overlap across the three splits
    the test split is only 79 UNIQUE texts, read by up to 24 subjects
    subject_id is the literal string 'UNKNOWN' for all 9367 sentences

The last two facts are load-bearing and are surfaced, not hidden: `text` is the
clustering unit for every confidence interval, and subject-stratified analysis is
impossible on this file.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

FEAT_DIM = 840
EVIDENCE_KINDS = ("eeg", "gaze", "both")
EEG_DIM, GAZE_DIM_ = 840, 6
BOTH_DIM = EEG_DIM + GAZE_DIM_          # [eeg(840) | gaze(6)]; gaze col 0 = fixated flag


@dataclass
class Sentence:
    token_ids: np.ndarray          # (T,) int64
    eeg: np.ndarray                # (T, F) float32 -- the EVIDENCE array (840-d EEG, or
                                   #   6-d gaze when read with evidence="gaze"); NaN where unobserved
    observed: np.ndarray           # (T,) bool, True where the evidence is real
    text: str
    task: str
    subject_id: str
    index: int
    evidence: str = "eeg"          # which channel `eeg` holds
    word_index: Optional[np.ndarray] = None   # (T,) word id of each token (sub-token grouping)


@dataclass
class SplitStats:
    n_sentences: int
    n_tokens: int
    n_unique_texts: int
    frac_observed: float
    tasks: dict
    subjects: dict
    has_real_subject_ids: bool

    def describe(self) -> str:
        return (
            f"  sentences={self.n_sentences}  tokens={self.n_tokens}  "
            f"unique_texts={self.n_unique_texts}\n"
            f"  observed-EEG tokens={self.frac_observed*100:.1f}%  "
            f"tasks={self.tasks}\n"
            f"  real subject ids: {self.has_real_subject_ids} "
            f"({len(self.subjects)} distinct)"
        )


def read_split(h5_path: str, limit: Optional[int] = None,
               evidence: str = "eeg") -> list[Sentence]:
    """Read one split. Never mixes splits: one file in, one list out.

    evidence="eeg"  -> `eeg` is the 840-d band-power array, NaN/unobserved where unfixated.
    evidence="gaze" -> `eeg` is the 6-d gaze array (build.GAZE_COLS), fully observed:
                       a skipped word is a real observation of the eye-movement process.
    evidence="both" -> `eeg` is [eeg(840, NaN->0 where unfixated) | gaze(6)], fully
                       observed. The combined channel measures I(text; EEG, gaze); the
                       EEG-beyond-gaze increment is  I(both) - I(gaze), reported by report.py.
    Everything downstream (windows, nulls, verifier, estimator) reads only `eeg` and
    `observed`, so the two channels run through one identical pipeline."""
    if not os.path.isabs(h5_path):
        raise ValueError(f"use an absolute path, got {h5_path!r}")
    if evidence not in EVIDENCE_KINDS:
        raise ValueError(f"evidence must be one of {EVIDENCE_KINDS}, got {evidence!r}")
    out: list[Sentence] = []
    with h5py.File(h5_path, "r") as f:
        g = f["sentences"]
        keys = sorted(g.keys(), key=int)
        if limit:
            keys = keys[:limit]
        for k in keys:
            grp = g[k]
            if evidence in ("gaze", "both"):
                if "gaze_features" not in grp:
                    raise KeyError(f"{h5_path}: sentence {k} has no gaze_features -- "
                                   f"rebuild with the current e0_build.py")
                gz = np.asarray(grp["gaze_features"][:], dtype=np.float32)
                if evidence == "gaze":
                    arr = gz
                else:
                    ee = np.nan_to_num(np.asarray(grp["eeg_features"][:], dtype=np.float32), nan=0.0)
                    arr = np.concatenate([ee, gz], axis=1).astype(np.float32)
                obs = np.ones(arr.shape[0], dtype=bool)
            else:
                arr = np.asarray(grp["eeg_features"][:], dtype=np.float32)
                if "nan_mask" in grp:
                    obs = ~np.asarray(grp["nan_mask"][:], dtype=bool)
                else:
                    obs = ~np.isnan(arr).all(axis=1)
            out.append(Sentence(
                token_ids=np.asarray(grp["token_ids"][:], dtype=np.int64),
                eeg=arr,
                observed=obs,
                text=str(grp.attrs.get("text", "")),
                task=str(grp.attrs.get("task", "UNKNOWN")),
                subject_id=str(grp.attrs.get("subject_id", "UNKNOWN")),
                index=int(k),
                evidence=evidence,
                word_index=(np.asarray(grp["word_index"][:], dtype=np.int64) if "word_index" in grp else None),
            ))
    return out


def selection_measurement_split(sents: list[Sentence]) -> tuple[list[Sentence], list[Sentence]]:
    """Split a validation list into disjoint (selection, measurement) halves by TEXT.

    Checkpoint selection maximises val dI_hat over epochs; measuring dI_hat, its
    p-value and CI on the same sentences reports the maximum of noisy looks
    (winner's curse) and invalidates the permutation p-value. Selection therefore
    uses one half and every reported channel number the other. Assignment is a
    stable hash of the text, so every reading of a text lands in the same half and
    the split is identical across runs, drivers and evidence channels.
    """
    sel, meas = [], []
    for s in sents:
        h = int(hashlib.sha1(s.text.strip().lower().encode("utf-8")).hexdigest(), 16)
        (sel if h % 2 == 0 else meas).append(s)
    if not sel or not meas:                     # degenerate (e.g. a single text)
        k = max(1, len(sents) // 2)
        sel, meas = list(sents[:k]), list(sents[k:]) or list(sents[:k])
    return sel, meas


def split_stats(sents: list[Sentence]) -> SplitStats:
    from collections import Counter
    ntok = sum(len(s.token_ids) for s in sents)
    nobs = sum(int(s.observed.sum()) for s in sents)
    subs = Counter(s.subject_id for s in sents)
    return SplitStats(
        n_sentences=len(sents),
        n_tokens=ntok,
        n_unique_texts=len({s.text for s in sents}),
        frac_observed=nobs / max(1, ntok),
        tasks=dict(Counter(s.task for s in sents)),
        subjects=dict(subs),
        has_real_subject_ids=not (set(subs) <= {"UNKNOWN", ""}),
    )


def assert_no_text_leakage(splits: dict[str, list[Sentence]]) -> None:
    """Hard gate: no sentence text may appear in more than one split."""
    texts = {k: {s.text for s in v} for k, v in splits.items()}
    names = list(texts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = texts[a] & texts[b]
            if overlap:
                raise AssertionError(
                    f"split leakage: {len(overlap)} texts shared between "
                    f"{a} and {b}; e.g. {sorted(overlap)[0][:70]!r}"
                )


def recover_word_groups(eeg: np.ndarray, observed: np.ndarray) -> list[tuple[int, int]]:
    """Recover word boundaries from token-level EEG.

    Preprocessing replicated each word's 840-d vector across that word's subword
    tokens, so runs of identical consecutive rows reconstruct words. NaN rows never
    compare equal, so missing runs are grouped on the mask instead.

    Needed only for the causal-LM prior, which re-tokenizes from raw text.
    """
    T = eeg.shape[0]
    if T == 0:
        return []
    groups, start = [], 0
    for t in range(1, T + 1):
        if t == T:
            groups.append((start, t))
            break
        both_missing = (not observed[t]) and (not observed[t - 1])
        both_same = observed[t] and observed[t - 1] and np.array_equal(eeg[t], eeg[t - 1])
        if not (both_missing or both_same):
            groups.append((start, t))
            start = t
    return groups


def recover_word_spans(text: str, stored_ids: np.ndarray, stored_tok) -> Optional[list[tuple[int, int]]]:
    """Exact (start, end) spans into `stored_ids`, one per whitespace word.

    The HDF5's tokenization is corrupted in a *deterministic* way: each whitespace word
    was tokenized separately with no leading space and no special tokens, then
    concatenated (verified: 100.0% exact BPE match on 200 test sentences). Reproducing
    that scheme and matching run lengths recovers word boundaries exactly.

    Returns None if the reconstruction does not match, so the caller can refuse the
    sentence instead of guessing. Run-length grouping of identical EEG rows is NOT used:
    it recovers only ~6% of sentences, because consecutive unfixated words are all-NaN
    and collapse into one group.
    """
    spans, pos = [], 0
    stored = list(map(int, stored_ids))
    for w in text.split():
        ids = stored_tok(w, add_special_tokens=False)["input_ids"]
        n = len(ids)
        if n == 0 or stored[pos:pos + n] != list(ids):
            return None
        spans.append((pos, pos + n))
        pos += n
    return spans if pos == len(stored) else None


@dataclass
class RetokenizeStats:
    n_in: int = 0
    n_ok: int = 0
    n_refused: int = 0
    refusal_reasons: dict = field(default_factory=dict)

    def note(self, reason: str) -> None:
        self.n_refused += 1
        self.refusal_reasons[reason] = self.refusal_reasons.get(reason, 0) + 1

    def describe(self) -> str:
        pct = 100 * self.n_ok / max(1, self.n_in)
        s = f"  retokenized {self.n_ok}/{self.n_in} ({pct:.1f}%), refused {self.n_refused}"
        for r, c in sorted(self.refusal_reasons.items(), key=lambda x: -x[1]):
            s += f"\n    {c:>5}  {r}"
        return s


def retokenize_split(sents: list[Sentence], stored_tok, new_tok) -> tuple[list[Sentence], RetokenizeStats]:
    """Rebuild a split under a correct tokenization, keeping EEG aligned 1:1.

    Word EEG is taken from the first token of each recovered span (all tokens in a span
    carry the same replicated vector), then re-replicated across that word's new subword
    tokens. Sentences whose spans do not reconstruct exactly are refused and counted.
    """
    out: list[Sentence] = []
    st = RetokenizeStats()
    for s in sents:
        st.n_in += 1
        if not s.text.strip():
            st.note("empty text attribute")
            continue
        spans = recover_word_spans(s.text, s.token_ids, stored_tok)
        if spans is None:
            st.note("word spans did not reconstruct exactly")
            continue

        new_ids, new_eeg, new_obs = [], [], []
        for j, (a, _b) in enumerate(spans):
            word = s.text.split()[j]
            piece = word if j == 0 else " " + word
            wi = new_tok(piece, add_special_tokens=False)["input_ids"]
            if not wi:
                new_ids = None
                break
            new_ids.extend(wi)
            new_eeg.extend([s.eeg[a]] * len(wi))
            new_obs.extend([bool(s.observed[a])] * len(wi))
        if not new_ids:
            st.note("new tokenizer produced an empty word")
            continue

        out.append(Sentence(
            token_ids=np.asarray(new_ids, dtype=np.int64),
            eeg=np.stack(new_eeg).astype(np.float32),
            observed=np.asarray(new_obs, dtype=bool),
            text=s.text, task=s.task, subject_id=s.subject_id, index=s.index,
        ))
        st.n_ok += 1
    return out, st


class PRDDataset(Dataset):
    """Windows of word-level EEG aligned 1:1 with tokens, plus masks and metadata."""

    def __init__(self, sentences: list[Sentence], window: int = 1, max_len: int = 128):
        self.sents = sentences
        self.window = window
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sents)

    def __getitem__(self, i: int) -> dict:
        s = self.sents[i]
        T = min(len(s.token_ids), self.max_len)
        w = self.window
        W = 2 * w + 1

        eeg = np.nan_to_num(s.eeg[:T], nan=0.0)
        obs = s.observed[:T]
        Fd = int(s.eeg.shape[1])          # 840 for EEG, 6 for gaze

        win = np.zeros((T, W, Fd), dtype=np.float32)
        win_obs = np.zeros((T, W), dtype=bool)
        win_pad = np.ones((T, W), dtype=bool)
        for off in range(-w, w + 1):
            j = off + w
            lo, hi = max(0, -off), min(T, T - off)
            if lo >= hi:
                continue
            win[lo:hi, j] = eeg[lo + off:hi + off]
            win_obs[lo:hi, j] = obs[lo + off:hi + off]
            win_pad[lo:hi, j] = False

        return {
            "token_ids": torch.from_numpy(s.token_ids[:T].copy()),
            "window": torch.from_numpy(win),
            "win_observed": torch.from_numpy(win_obs),
            "win_pad": torch.from_numpy(win_pad),
            "observed": torch.from_numpy(obs.copy()),
            "text": s.text,
            "task": s.task,
            "subject_id": s.subject_id,
            "index": s.index,
        }


def collate(batch: list[dict]) -> dict:
    """Right-pad to the longest sentence. `valid` marks real tokens."""
    B = len(batch)
    T = max(b["token_ids"].shape[0] for b in batch)
    W = batch[0]["window"].shape[1]
    Fd = batch[0]["window"].shape[2]

    token_ids = torch.zeros(B, T, dtype=torch.long)
    window = torch.zeros(B, T, W, Fd)
    win_obs = torch.zeros(B, T, W, dtype=torch.bool)
    win_pad = torch.ones(B, T, W, dtype=torch.bool)
    observed = torch.zeros(B, T, dtype=torch.bool)
    valid = torch.zeros(B, T, dtype=torch.bool)

    for i, b in enumerate(batch):
        t = b["token_ids"].shape[0]
        token_ids[i, :t] = b["token_ids"]
        window[i, :t] = b["window"]
        win_obs[i, :t] = b["win_observed"]
        win_pad[i, :t] = b["win_pad"]
        observed[i, :t] = b["observed"]
        valid[i, :t] = True

    return {
        "token_ids": token_ids, "window": window, "win_observed": win_obs,
        "win_pad": win_pad, "observed": observed, "valid": valid,
        "text": [b["text"] for b in batch],
        "task": [b["task"] for b in batch],
        "subject_id": [b["subject_id"] for b in batch],
        "index": [b["index"] for b in batch],
    }


def dataset_fingerprint(h5_path: str) -> str:
    """Content-based id for a split, recorded alongside every result.

    Hashes the sorted set of sentence texts plus per-sentence token counts — a
    regenerated file with different content can never silently reuse a fingerprint.
    (The earlier filename+size hash could.)
    """
    h = hashlib.sha256()
    with h5py.File(h5_path, "r") as f:
        g = f["sentences"]
        entries = []
        for k in g.keys():
            txt = str(g[k].attrs.get("text", ""))
            n = int(g[k]["token_ids"].shape[0])
            entries.append(f"{txt}\x1f{n}")
        for e in sorted(entries):
            h.update(e.encode())
    return h.hexdigest()[:16]
