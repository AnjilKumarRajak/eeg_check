#!/usr/bin/env python3
"""E5 (Phase 3): attribute decoding with the mandatory gaze triplet.

Every accuracy is reported three ways — gaze-only / EEG-only / EEG-residualized-on-gaze
— and only the residualized row may carry a neural claim. Labels available without
external files: corpus topic (SR movie reviews vs NR/TSR Wikipedia), the GLIM-style
zero-shot axis. Sentiment/relation labels can be supplied as a JSON {text_hash: label}.

Real-data driver (needs covariates.csv from e0). Functions are imported by tests.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import base_parser, base_record, ledger_for, load_split_sentences  # noqa: E402
from cprd.audit import Record                                                  # noqa: E402
from cprd.covariates import CovariateTable, text_hash                          # noqa: E402
from cprd.prereg import check_gate_artifact, write_gate_artifact               # noqa: E402


def sentence_features(sentences, cov: CovariateTable | None):
    """Per sentence: mean-over-observed EEG (840), gaze vector (5), text, label=task."""
    X_eeg, X_gaze, texts, labels, subjects = [], [], [], [], []
    for s in sentences:
        if not s.observed.any():
            continue
        X_eeg.append(np.nan_to_num(s.eeg[s.observed]).mean(axis=0))
        th = text_hash(s.text)
        if cov is not None:
            vecs = [cov.vector(s.subject_id, th, wi) for wi in range(len(s.text.split()))]
            g = np.nanmean(np.stack(vecs), axis=0) if vecs else np.full(5, np.nan)
        else:
            g = np.array([float(s.observed.sum()), float(len(s.token_ids)),
                          np.nan, np.nan, np.nan])
        X_gaze.append(np.nan_to_num(g))
        texts.append(s.text)
        labels.append(s.task)
        subjects.append(s.subject_id)
    return (np.stack(X_eeg), np.stack(X_gaze), np.array(texts),
            np.array(labels), np.array(subjects))


def residualize(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Remove the best linear gaze prediction from every EEG feature."""
    Cd = np.column_stack([np.ones(len(C)), C])
    beta, *_ = np.linalg.lstsq(Cd, X, rcond=None)
    return X - Cd @ beta


def probe(X, y, groups, seed=0, n_splits=5) -> float:
    """Balanced accuracy of a logistic probe, GroupKFold by unique text."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    preds = np.empty(len(y), dtype=object)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit(sc.transform(X[tr]), y[tr])
        preds[te] = clf.predict(sc.transform(X[te]))
    return float(balanced_accuracy_score(y, preds.astype(str)))


def label_permutation_null(X, y, groups, subjects, n_perm=200, seed=0):
    """Null accuracies with labels shuffled WITHIN subject (plain shuffling lets
    subject identity leak in as fake signal)."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_perm):
        y2 = y.copy()
        for s in np.unique(subjects):
            m = np.flatnonzero(subjects == s)
            y2[m] = y2[rng.permutation(m)]
        out.append(probe(X, y2, groups, seed=seed + k))
    return np.asarray(out)


def main():
    ap = base_parser("attribute decoding with the gaze triplet")
    ap.add_argument("--n-perm-labels", type=int, default=200)
    args = ap.parse_args()
    check_gate_artifact(args.runs_dir, "build")

    va, fp = load_split_sentences(args, "val")
    cov_csv = os.path.join(args.data_dir, "covariates.csv")
    cov = CovariateTable.load(cov_csv) if os.path.exists(cov_csv) else None
    if cov is None:
        print("  note: covariates.csv absent — gaze row uses count proxies only")

    X_eeg, X_gaze, texts, y, subjects = sentence_features(va, cov)
    if len(np.unique(y)) < 2:
        print("only one task label present in this split; attribute probe skipped")
        write_gate_artifact(args.runs_dir, "attributes", True,
                            {"skipped": "single-class split"})
        return 0

    rows = {
        "gaze_only": probe(X_gaze, y, texts, seed=args.seed),
        "eeg_only": probe(X_eeg, y, texts, seed=args.seed),
        "eeg_residualized": probe(residualize(X_eeg, X_gaze), y, texts, seed=args.seed),
    }
    null = label_permutation_null(residualize(X_eeg, X_gaze), y, texts, subjects,
                                  n_perm=args.n_perm_labels, seed=args.seed)
    p = float((1 + np.sum(null >= rows["eeg_residualized"])) / (1 + len(null)))
    for k, v in rows.items():
        print(f"  {k:<18} balanced acc = {v:.3f}")
    print(f"  residualized-vs-null p = {p:.4f}  (null mean {null.mean():.3f})")

    led = ledger_for(args)
    base = base_record(args, split_fp=fp, experiment="e5_attributes",
                       gates=["build"])
    for k, v in rows.items():
        led.append(Record(metric="task_topic_balanced_acc", value=v, arm=k,
                          p=p if k == "eeg_residualized" else None,
                          n_sentences=len(y), **base))
    write_gate_artifact(args.runs_dir, "attributes", True,
                        {"rows": rows, "p_residualized": p})
    return 0


if __name__ == "__main__":
    sys.exit(main())
