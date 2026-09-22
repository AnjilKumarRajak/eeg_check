
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskCoverage:
    coverages: np.ndarray
    risks: np.ndarray
    aurc: float
    full_risk: float

    def describe(self) -> str:
        pts = [f"{c:.0%}:{r:.3f}" for c, r in zip(self.coverages[::2], self.risks[::2])]
        return (f"  AURC={self.aurc:.4f}  full-coverage risk={self.full_risk:.3f}\n"
                f"  risk@coverage: {'  '.join(pts)}")


def risk_coverage(correct: np.ndarray, confidence: np.ndarray,
                  clusters: np.ndarray | None = None,
                  grid: np.ndarray | None = None) -> RiskCoverage:
    correct = np.asarray(correct, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if clusters is not None:
        uniq = np.unique(clusters)
        c_agg = np.array([correct[clusters == u].mean() for u in uniq])
        f_agg = np.array([confidence[clusters == u].mean() for u in uniq])
        correct, confidence = c_agg, f_agg
    order = np.argsort(-confidence)
    correct = correct[order]
    n = len(correct)
    if grid is None:
        grid = np.linspace(0.1, 1.0, 19)
    covs, risks = [], []
    for cov in grid:
        k = max(1, int(round(cov * n)))
        covs.append(k / n)
        risks.append(1.0 - float(correct[:k].mean()))
    covs, risks = np.asarray(covs), np.asarray(risks)
    _trapz = getattr(np, "trapezoid", None) or np.trapz          # numpy 1.x has no trapezoid
    aurc = float(_trapz(risks, covs) / (covs[-1] - covs[0])) if len(covs) > 1 else float("nan")
    return RiskCoverage(coverages=covs, risks=risks, aurc=aurc,
                        full_risk=1.0 - float(correct.mean()))


def confidence_auc(correct: np.ndarray, confidence: np.ndarray) -> float:
    """AUC of confidence for predicting correctness (Mann-Whitney)."""
    correct = np.asarray(correct, dtype=bool)
    pos, neg = confidence[correct], confidence[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([pos, neg]))      # midranks: ties count 1/2
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def partial_confidence_auc(correct: np.ndarray, confidence: np.ndarray,
                           covariates: np.ndarray) -> float:
    X = np.asarray(covariates, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    keep = np.isfinite(X).all(axis=1) & np.isfinite(confidence)
    X, y, c = X[keep], np.asarray(confidence)[keep], np.asarray(correct)[keep]
    if len(y) < 10:
        return float("nan")
    Xd = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    return confidence_auc(c, resid)


def cheap_confidence(lengths: np.ndarray, prior_logp: np.ndarray) -> np.ndarray:
    z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    return z(prior_logp) - 0.5 * z(lengths)


def ece(correct: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> float:
    conf = np.asarray(confidence, dtype=np.float64)
    conf = 1.0 / (1.0 + np.exp(-(conf - np.nanmean(conf)) / (np.nanstd(conf) + 1e-9)))
    order = np.argsort(conf)
    conf, corr = conf[order], np.asarray(correct, dtype=np.float64)[order]
    edges = np.array_split(np.arange(len(conf)), n_bins)
    total = 0.0
    for idx in edges:
        if len(idx) == 0:
            continue
        total += (len(idx) / len(conf)) * abs(conf[idx].mean() - corr[idx].mean())
    return float(total)
