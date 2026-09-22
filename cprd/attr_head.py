
from __future__ import annotations

import numpy as np


class AttributeHead:
    def __init__(self):
        self.classes_ = None
        self.scaler = None
        self.clf = None

    @staticmethod
    def _feat(sent) -> np.ndarray | None:
        if not sent.observed.any():
            return None
        return np.nan_to_num(sent.eeg[sent.observed]).mean(axis=0)

    def fit(self, train_sentences, label_fn=None) -> "AttributeHead":
        """label_fn(sent)->str; default = task attribute (always available)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        label_fn = label_fn or (lambda s: s.task)
        X, y = [], []
        for s in train_sentences:
            f = self._feat(s)
            if f is not None:
                X.append(f)
                y.append(label_fn(s))
        X, y = np.stack(X), np.asarray(y)
        self.classes_ = sorted(set(y.tolist()))
        if len(self.classes_) < 2:
            self.clf = None                        # single-class corpus: head disabled
            return self
        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(max_iter=2000, C=0.1).fit(
            self.scaler.transform(X), y)
        return self

    def log_prob_of(self, eeg_sentence, attr_value: str) -> float:
        """log P(attr_value | EEG of the true reading); 0.0 when head is disabled."""
        if self.clf is None:
            return 0.0
        f = self._feat(eeg_sentence)
        if f is None or attr_value not in list(self.clf.classes_):
            return 0.0
        logp = self.clf.predict_log_proba(self.scaler.transform(f[None, :]))[0]
        return float(logp[list(self.clf.classes_).index(attr_value)])


def tune_beta(scores_dI: list, scores_attr: list, true_idx: list,
              betas=(0.0, 0.25, 0.5, 1.0, 2.0)) -> float:
    """Pick beta on VALIDATION pools: maximize top-1 accuracy of dI + beta*attr.

    scores_dI/scores_attr: list over pools of np.ndarray (n_candidates,).
    """
    best_beta, best_acc = 0.0, -1.0
    for b in betas:
        hits = 0
        for sd, sa, ti in zip(scores_dI, scores_attr, true_idx):
            s = np.asarray(sd) + b * np.asarray(sa)
            if int(np.argmax(s)) == ti:
                hits += 1
        acc = hits / max(1, len(true_idx))
        if acc > best_acc:
            best_acc, best_beta = acc, b
    return best_beta
