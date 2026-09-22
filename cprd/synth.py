
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .data import Sentence

LN2 = math.log(2.0)


@dataclass
class SynthChannel:
    bits: float                  # nominal b (log2 of n_clusters)
    sigma: float
    d_code: int
    n_clusters: int
    mus: np.ndarray              # (n_clusters, d_code)
    true_bits_per_token: float   # MC-exact I(E; cluster) in bits


def _mc_mutual_information(mus: np.ndarray, sigma: float, n_mc: int = 200_000,
                           seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    K, d = mus.shape
    c = rng.integers(0, K, size=n_mc)
    e = mus[c] + sigma * rng.standard_normal((n_mc, d))
    # log-densities up to a common constant (cancels in the difference)
    d2 = ((e[:, None, :] - mus[None, :, :]) ** 2).sum(-1)      # (n, K)
    log_comp = -d2 / (2 * sigma ** 2)                          # + const
    log_p_e_given_c = log_comp[np.arange(n_mc), c]
    log_p_e = torch.logsumexp(torch.from_numpy(log_comp), dim=1).numpy() - math.log(K)
    return float((log_p_e_given_c - log_p_e).mean() / LN2)


def make_channel(bits: float, sigma: float | None = None, d_code: int = 16,
                 feat_dim: int = 840, seed: int = 0, n_mc: int = 200_000) -> SynthChannel:
    rng = np.random.default_rng(seed)
    n_clusters = 1 if bits <= 0 else max(2, int(math.ceil(2 ** bits)))
    # well-separated but not orthogonal centroids; scale sets the SNR together with sigma
    mus = rng.standard_normal((n_clusters, d_code)) * 2.0
    if n_clusters == 1:
        sigma = 1.0 if sigma is None else sigma
        true = 0.0
    else:
        if sigma is None:
            lo, hi = (0.05, 20.0) if bits >= 0.05 else (0.05, 400.0)
            for _ in range(24):
                mid = (lo + hi) / 2
                est = _mc_mutual_information(mus, mid, n_mc=max(20_000, n_mc // 10), seed=seed + 1)
                if est > bits:
                    lo = mid
                else:
                    hi = mid
            sigma = hi
        true = _mc_mutual_information(mus, sigma, n_mc=n_mc, seed=seed + 1)
    return SynthChannel(bits=bits, sigma=sigma, d_code=d_code,
                        n_clusters=n_clusters, mus=mus, true_bits_per_token=true)


def make_synthetic_sentences(channel: SynthChannel, n_sentences: int, vocab_size: int,
                              sent_len_range=(6, 14), missing_frac: float = 0.0,
                              feat_dim: int = 840, seed: int = 0,
                              token_clusters: np.ndarray | None = None,
                              token_sequences: list | None = None) -> list[Sentence]:
    rng = np.random.default_rng(seed)
    pools = None
    if token_clusters is not None:
        if len(token_clusters) != vocab_size:
            raise ValueError("token_clusters length must match vocab_size")
        pools = [np.flatnonzero(token_clusters == k) for k in range(channel.n_clusters)]
        if any(len(p) == 0 for p in pools):
            raise ValueError("each synthetic cluster needs at least one token")
    out = []
    for i in range(n_sentences):
        T = int(rng.integers(*sent_len_range))
        if token_sequences is not None:
            tokens = np.asarray(token_sequences[i % len(token_sequences)], dtype=np.int64)
            T = int(tokens.shape[0])
            clusters = (token_clusters[tokens] if token_clusters is not None
                        else tokens % channel.n_clusters)
        elif pools is None:
            tokens = rng.integers(1, vocab_size, size=T).astype(np.int64)
            clusters = tokens % channel.n_clusters
        else:
            clusters = rng.integers(0, channel.n_clusters, size=T).astype(np.int64)
            tokens = np.asarray([rng.choice(pools[int(c)]) for c in clusters], dtype=np.int64)
        eeg = rng.standard_normal((T, feat_dim)).astype(np.float32)
        if channel.n_clusters > 1:
            eeg[:, :channel.d_code] = (
                channel.mus[clusters] + channel.sigma
                * rng.standard_normal((T, channel.d_code))).astype(np.float32)
        obs = rng.random(T) >= missing_frac
        eeg[~obs] = np.nan
        out.append(Sentence(
            token_ids=tokens, eeg=eeg, observed=obs,
            text=f"synthetic sentence {i} seed {seed}",
            task="synthetic", subject_id=f"SYN{i % 4}", index=i,
        ))
    return out


@dataclass
class ValidationOutcome:
    b_nominal: float
    b_true: float                 # MC-exact ground truth (bits/token)
    b_recovered: float            # dI_hat on observed tokens (bits/token)
    p_value: float
    passed_recovery: bool
    passed_never_overshoot: bool

    @property
    def ok(self) -> bool:
        return self.passed_recovery and self.passed_never_overshoot


def judge(b_true: float, recovered_bits: float, p_value: float,
          overshoot_tol: float = 0.05) -> ValidationOutcome:
    """Apply the pre-registered pass criteria for one sweep point."""
    if b_true <= 1e-9:
        rec_ok = abs(recovered_bits) <= 0.02 and p_value >= 0.05
        over_ok = True
    else:
        rec_ok = (recovered_bits >= 0.5 * b_true) and (p_value < 0.05)
        over_ok = recovered_bits <= b_true + max(overshoot_tol, 0.2 * b_true)
    return ValidationOutcome(
        b_nominal=b_true, b_true=b_true, b_recovered=recovered_bits,
        p_value=p_value, passed_recovery=rec_ok, passed_never_overshoot=over_ok,
    )


def ks_uniform(pvals: np.ndarray) -> float:
    """KS test p-value against Uniform(0,1) — the p-calibration gate."""
    from scipy import stats
    if len(pvals) < 5:
        return float("nan")
    return float(stats.kstest(pvals, "uniform").pvalue)
