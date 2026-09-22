"""Information gain with an EXACT partition function.

    p_phi(v|e,C) = p0(v|C) exp(gamma ell(v)) / Z        Z = sum_{v in V} p0(v|C) e^{gamma ell(v)}
    dI_t         = log p_phi(w_t|e,C) - log p0(w_t|C) = gamma_t ell_t(w_t) - log Z_t

Why Z is exact rather than top-K
--------------------------------
Truncating Z to a top-K support UNDER-estimates log Z, therefore OVER-estimates dI, so
dI stops being a lower bound on anything. It also destroys the zero point: at gamma = 0
a truncated estimator returns

    dI_t = -log( sum_{v in S} p0(v) )  >  0

rather than 0, so null-invariance -- the property the whole method rests on -- is simply
false in the code. On ZuCo with K=256 that floor is about +0.70 nats/token, which is the
same order as any plausible real effect.

Computing Z exactly costs one matvec Omega @ (B u_t), the same cost as the LM head that
already produced log p0. There is no reason to approximate it.
"""
from __future__ import annotations

from typing import Optional

import torch


def information_gain(
    log_p0: torch.Tensor,        # (B, V)  log p0(.|C_t), frozen
    omega: torch.Tensor,         # (V, d)  frozen output embeddings
    gold_ids: torch.Tensor,      # (B,)
    bu: torch.Tensor,            # (B, d)  = B u_t, the evidence direction in LM space
    gamma: torch.Tensor,         # (B,)    >= 0, already gated by missingness
) -> dict:
    """One timestep of dI, batched. Returns dI plus diagnostics.

    Shapes are checked rather than assumed: a silent broadcast here would corrupt every
    downstream number.
    """
    if log_p0.dim() != 2:
        raise ValueError(f"log_p0 must be (B,V), got {tuple(log_p0.shape)}")
    B, V = log_p0.shape
    if omega.shape[0] != V:
        raise ValueError(f"omega rows {omega.shape[0]} != vocab {V}")
    if bu.shape != (B, omega.shape[1]):
        raise ValueError(f"bu must be (B,{omega.shape[1]}), got {tuple(bu.shape)}")
    if gamma.shape != (B,) or gold_ids.shape != (B,):
        raise ValueError("gamma and gold_ids must be (B,)")

    # ell over the FULL vocabulary: (B,V) = (B,d) @ (d,V)
    ell = bu @ omega.transpose(0, 1)                      # (B, V)
    tilt = gamma.unsqueeze(-1) * ell                      # (B, V)

    log_Z = torch.logsumexp(log_p0 + tilt, dim=-1)        # (B,) exact
    idx = gold_ids.unsqueeze(-1)
    ell_gold = ell.gather(1, idx).squeeze(-1)             # (B,)
    dI = gamma * ell_gold - log_Z                         # (B,)

    return {
        "dI": dI,
        "gamma": gamma,
        "ell_gold": ell_gold,
        "log_Z": log_Z,
        "log_p0_gold": log_p0.gather(1, idx).squeeze(-1),
    }


def sentence_information_gain(
    log_p0: torch.Tensor,        # (B, T, V)
    omega: torch.Tensor,         # (V, d)
    token_ids: torch.Tensor,     # (B, T)
    bu: torch.Tensor,            # (B, T, d)
    gamma: torch.Tensor,         # (B, T)
    valid: torch.Tensor,         # (B, T) bool, True = real token (not padding)
    extra_ell: torch.Tensor | None = None,   # (B, T, V) additional tilt logits (free table)
) -> dict:
    """Vectorised over the whole sentence. No python loop over T.

    `valid` excludes padding only. Missing-EEG positions stay in the average with
    dI == 0, because gamma is already gated to 0 there -- they are genuine zero-evidence
    observations, not absent data.
    """
    B, T, V = log_p0.shape
    d = omega.shape[1]

    ell = torch.einsum("btd,vd->btv", bu, omega)          # (B,T,V) exact, full vocab
    if extra_ell is not None:
        ell = ell + extra_ell                              # still exact over the full vocab
    tilt = gamma.unsqueeze(-1) * ell
    log_Z = torch.logsumexp(log_p0 + tilt, dim=-1)        # (B,T)

    idx = token_ids.unsqueeze(-1)
    ell_gold = ell.gather(2, idx).squeeze(-1)             # (B,T)
    per_token = gamma * ell_gold - log_Z                  # (B,T)

    per_token = per_token.masked_fill(~valid, 0.0)
    n = valid.sum(dim=1).clamp(min=1)
    per_sentence = per_token.sum(dim=1) / n               # (B,) nats per TOKEN

    return {
        "per_token": per_token,                            # (B,T) nats/token
        "per_sentence_mean": per_sentence,                  # (B,)  nats/token
        "per_sentence_sum": per_token.sum(dim=1),           # (B,)  nats/sentence
        "gamma": gamma.masked_fill(~valid, 0.0),
        "log_p0_gold": log_p0.gather(2, idx).squeeze(-1).masked_fill(~valid, 0.0),
    }


def assert_upper_bound(per_token: torch.Tensor, log_p0_gold: torch.Tensor,
                       valid: torch.Tensor, tol: float = 1e-4) -> None:
    """dI_t <= -log p0(w_t) is an identity, since Z >= p0(w_t) e^{gamma ell(w_t)}.

    A violation means a shape/index bug or a truncated Z, not an interesting result.
    This is the tripwire that would have caught an impossible 730 nats/token instantly.
    """
    ceiling = -log_p0_gold
    bad = valid & (per_token > ceiling + tol)
    if bool(bad.any()):
        i = bad.nonzero()[0].tolist()
        raise AssertionError(
            f"dI exceeded its analytic ceiling at {tuple(i)}: "
            f"dI={float(per_token[tuple(i)]):.4f} > -log p0={float(ceiling[tuple(i)]):.4f}. "
            "Indicates a truncated partition function or an indexing bug."
        )
