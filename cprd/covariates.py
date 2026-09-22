
from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass

import numpy as np

V1_TASKS = ("task1-SR", "task2-NR", "task3-TSR")
V2_TASKS = ("task2-NR-2.0",)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


def _scalar(v) -> float:
    a = np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel()
    return float(a[0]) if a.size and np.isfinite(a[0]) else float("nan")


def _fmt(x: float) -> str:
    return "" if not np.isfinite(x) else f"{x:.6g}"


def extract_v1_file(mat_path: str, task: str, writer, subject: str) -> int:
    import scipy.io as sio
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    sd = np.atleast_1d(m["sentenceData"])
    n = 0
    for si, s in enumerate(sd):
        if isinstance(getattr(s, "word", None), float) or not hasattr(s, "word"):
            continue
        words = np.atleast_1d(s.word)
        content = " ".join(str(w.content) for w in words)
        th = text_hash(content)
        for wi, w in enumerate(words):
            nf_raw = getattr(w, "nFixations", np.nan)
            nf = _scalar(nf_raw) if np.size(nf_raw) else 0.0
            writer.writerow([
                subject, task, si, th, wi, str(w.content),
                _fmt(nf if np.isfinite(nf) else 0.0),
                _fmt(_scalar(getattr(w, "FFD", np.nan))),
                _fmt(_scalar(getattr(w, "GD", np.nan))),
                _fmt(_scalar(getattr(w, "TRT", np.nan))),
                _fmt(_scalar(getattr(w, "GPT", np.nan))),
                _fmt(_scalar(getattr(w, "meanPupilSize", np.nan))),
                len(str(w.content)),
            ])
            n += 1
    return n


def extract_v2_file(mat_path: str, task: str, writer, subject: str) -> int:
    import h5py
    n = 0
    with h5py.File(mat_path, "r") as f:
        sd = f["sentenceData"]

        def deref(ref):
            return f[ref]

        def matlab_str(ref):
            arr = np.asarray(deref(ref)[()]).ravel()
            return "".join(chr(int(c)) for c in arr if int(c) > 0)

        content_refs = sd["content"]
        word_refs = sd["word"]
        n_sent = len(content_refs)
        for si in range(n_sent):
            try:
                wgrp = deref(word_refs[si][0])
                if "content" not in wgrp:
                    continue
                wc = wgrp["content"]
                n_words = len(wc)
                words = [matlab_str(wc[j][0]) for j in range(n_words)]
            except Exception:
                continue
            th = text_hash(" ".join(words))

            def field(name, j):
                if name not in wgrp:
                    return float("nan")
                try:
                    v = np.asarray(deref(wgrp[name][j][0])[()], dtype=np.float64).ravel()
                    return float(v[0]) if v.size and np.isfinite(v[0]) else float("nan")
                except Exception:
                    return float("nan")

            for wi, word in enumerate(words):
                nf = field("nFixations", wi)
                writer.writerow([
                    subject, task, si, th, wi, word,
                    _fmt(nf if np.isfinite(nf) else 0.0),
                    _fmt(field("FFD", wi)), _fmt(field("GD", wi)),
                    _fmt(field("TRT", wi)), _fmt(field("GPT", wi)),
                    _fmt(field("meanPupilSize", wi)), len(word),
                ])
                n += 1
    return n


def extract_all(mat_root: str, out_csv: str,
                tasks: tuple = V1_TASKS + V2_TASKS, verbose: bool = True) -> dict:
    """One pass over all .mat files -> covariates.csv. Returns per-task row counts."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    counts: dict = {}
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "task", "sent_idx", "text_hash", "word_idx", "word",
                    "nfix", "ffd", "gd", "trt", "gpt", "pupil", "word_len"])
        for task in tasks:
            d = os.path.join(mat_root, task, "Matlab_files")
            if not os.path.isdir(d):
                counts[task] = 0
                continue
            total = 0
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".mat"):
                    continue
                subject = fn.replace("results", "").split("_")[0]
                path = os.path.join(d, fn)
                try:
                    if task in V2_TASKS:
                        total += extract_v2_file(path, task, w, subject)
                    else:
                        total += extract_v1_file(path, task, w, subject)
                except Exception as e:  # a corrupt file must not silently vanish
                    if verbose:
                        print(f"  !! {task}/{fn}: {type(e).__name__}: {e}", flush=True)
            counts[task] = total
            if verbose:
                print(f"  {task}: {total} word rows", flush=True)
    return counts


@dataclass
class CovariateTable:
    rows: dict
    rows_by_task: dict = None

    NUMERIC = ("nfix", "ffd", "gd", "trt", "gpt", "pupil", "word_len")

    @classmethod
    def load(cls, csv_path: str) -> "CovariateTable":
        rows, by_task = {}, {}
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["subject"], r["text_hash"], int(r["word_idx"]))
                vals = {k: (float(r[k]) if r[k] != "" else float("nan"))
                        for k in cls.NUMERIC}
                rows[key] = vals
                by_task[(r["subject"], r["task"], r["text_hash"], int(r["word_idx"]))] = vals
        return cls(rows, by_task)

    def vector(self, subject: str, thash: str, word_idx: int) -> np.ndarray:
        r = self.rows.get((subject, thash, word_idx))
        if r is None:
            return np.full(5, np.nan)
        return np.array([
            np.log1p(r["nfix"]) if np.isfinite(r["nfix"]) else np.nan,
            np.log1p(r["gd"]) if np.isfinite(r["gd"]) else np.nan,
            np.log1p(r["trt"]) if np.isfinite(r["trt"]) else np.nan,
            r["word_len"],
            r["pupil"],
        ])
