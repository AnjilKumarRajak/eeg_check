"""Statistical helpers: TOST equivalence for collapse claims, Holm correction.

A control arm 'collapses to chance' only if BOTH one-sided tests reject at alpha:
observed accuracy is significantly above (chance - margin) AND significantly below
(chance + margin). Non-significance alone is never accepted as equivalence.
"""
from __future__ import annotations

import math

import numpy as np


def _binom_z(acc: float, p0: float, n: int) -> float:
    se = math.sqrt(max(p0 * (1 - p0) / max(1, n), 1e-12))
    return (acc - p0) / se


def tost_equivalence(hits: int, n: int, chance: float, margin: float = 0.02,
                     alpha: float = 0.05) -> dict:
    """TOST for 'accuracy == chance within +/-margin' on n Bernoulli trials.

    Returns dict(equivalent: bool, p_lower, p_upper). equivalent=True means the arm
    demonstrably sits inside [chance-margin, chance+margin].
    """
    from scipy import stats
    acc = hits / max(1, n)
    lo, hi = chance - margin, chance + margin
    # Exact one-sided binomial tests (the z-approximation is poor at small chance, e.g.
    # 1/128, and breaks when chance - margin <= 0).
    # H0a: p <= lo  vs  p > lo. If lo <= 0 the null is impossible for any accuracy -> reject.
    if lo <= 0:
        p_lower = 0.0
    else:
        p_lower = float(stats.binom.sf(hits - 1, n, lo))          # P(X >= hits | p=lo)
    # H0b: p >= hi  vs  p < hi.
    if hi >= 1:
        p_upper = 0.0
    else:
        p_upper = float(stats.binom.cdf(hits, n, hi))             # P(X <= hits | p=hi)
    # Power note: with n trials the test can only ever conclude equivalence if the
    # margin exceeds ~1.645*sqrt(chance*(1-chance)/n) on each side.
    min_margin = 1.645 * math.sqrt(max(chance * (1 - chance), 1e-12) / max(1, n))
    return {"acc": acc, "chance": chance, "margin": margin,
            "p_lower": p_lower, "p_upper": p_upper,
            "equivalent": bool(p_lower < alpha and p_upper < alpha),
            "powered": bool(margin > min_margin), "min_resolvable_margin": min_margin}


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni over a family; returns name -> (p, reject)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, blocked = {}, False
    for i, (name, p) in enumerate(items):
        thr = alpha / (m - i)
        reject = (p <= thr) and not blocked
        if not reject:
            blocked = True
        out[name] = {"p": p, "threshold": thr, "reject": reject}
    return out
