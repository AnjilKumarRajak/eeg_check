"""Prior-residual selection: the likelihood-ratio verifier over candidate pools.

Score of candidate y given the recorded EEG e (from the TRUE reading):

    S(y) = sum_t [ gamma_t * ell_t(y_t) - log Z_t ]      (exact Z, full vocabulary)

which is log [ p_tilted(y | e) / p0(y) ] — the prior's own preference for y cancels by
construction. This is the repair for similarity-scored verification, which the fMRI
blind control exposed as prior-dominated (arXiv 2607.12079).

EEG-to-candidate alignment: position j of the candidate takes EEG word j of the true
reading, wrapped at the true reading's length — a documented approximation carried
from cprd.decode; exact for the true candidate, content-mismatched for distractors,
which is precisely what the score should detect.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .data import PRDDataset, collate
from .objective import sentence_information_gain
from .pools import Pool

LN2 = math.log(2.0)


@torch.no_grad()
def score_candidate(model, prior, cand_tokens: torch.Tensor,
                    window: torch.Tensor, win_obs: torch.Tensor,
                    win_pad: torch.Tensor, observed: torch.Tensor,
                    device="cpu", gamma_override: float | None = None) -> float:
    """Sum_t dI_t for one candidate against one EEG recording (wrapped alignment).

    gamma_override: if given, the gain gate is replaced by this constant. With 0.0
    every term is exactly gamma*ell - log Z = 0 - log(1) = 0, so all candidates tie --
    this is the structural gamma=0 arm, and E4 asserts the tie rather than assuming it."""
    Tc = int(cand_tokens.shape[0])
    Te = int(window.shape[0])
    idx = torch.arange(Tc) % max(1, Te)
    w = window[idx].unsqueeze(0).to(device)
    wo = win_obs[idx].unsqueeze(0).to(device)
    wp = win_pad[idx].unsqueeze(0).to(device)
    ob = observed[idx].unsqueeze(0).to(device)
    toks = cand_tokens.unsqueeze(0).to(device)

    log_p0 = prior.log_probs(toks)
    model.eval()
    u = model.evidence(w, wo, wp)                    # centred with the running mean
    bu = model.tilt.bu(u)
    p = log_p0.exp()
    H = -(p * log_p0.clamp_min(-40.0)).sum(dim=-1)
    gamma = model.tilt.gamma(H, u, ob)
    if gamma_override is not None:
        gamma = torch.full_like(gamma, float(gamma_override))
    valid = torch.ones_like(toks, dtype=torch.bool)
    # Candidate scores are SUMS over ~20-60 tokens of (gamma*ell - log Z). In float32 a
    # 50k-vocab log Z carries ~1e-6 rounding per token, so candidate scores would differ
    # by ~1e-5 at gamma=0 (breaking the asserted tie AFTER the one-shot test access) and
    # tiny real-score gaps would be rounding noise. Scoring is therefore done in float64
    # with p0 renormalised in float64 (a <1e-6 relative change that makes Z exact to
    # machine precision). Training/estimation paths are unchanged.
    om = _omega64(prior, log_p0.device)
    log_p0_64 = torch.log_softmax(log_p0.double(), dim=-1)
    ex = model.extra_ell(u)
    out = sentence_information_gain(log_p0_64, om, toks, bu.double(), gamma.double(), valid,
                                    extra_ell=None if ex is None else ex.double())
    return float(out["per_token"].sum())


_OMEGA64: dict = {}


def _omega64(prior, device) -> torch.Tensor:
    key = (id(prior), str(device))
    if key not in _OMEGA64:
        _OMEGA64.clear()
        _OMEGA64[key] = prior.omega.detach().to(device=device, dtype=torch.float64)
    return _OMEGA64[key]


@dataclass
class SelectionResult:
    N: int
    accuracy: float
    hits: int
    n_pools: int
    bits_realized: float          # Fano-inverted information from accuracy at N
    per_pool_hit: list


def fano_bits(accuracy: float, N: int) -> float:
    """Information (bits) implied by top-1 accuracy over N equiprobable candidates.

    I >= log2(N) - H(err) - err*log2(N-1)   (Fano). Clamped at 0.
    """
    if N < 2:
        return 0.0
    # At or below chance the Fano expression is NOT a lower bound on information (it is
    # symmetric around 1/N and rises again toward acc=0: fano(0, 4) = 0.415 bits). Zero
    # information is the only honest reading of chance-or-worse accuracy.
    if accuracy <= 1.0 / N:
        return 0.0
    acc = min(max(accuracy, 1e-9), 1 - 1e-9)
    err = 1 - acc
    h = -acc * math.log2(acc) - err * math.log2(err)
    return max(0.0, math.log2(N) - h - err * math.log2(max(1, N - 1)))


@torch.no_grad()
def run_selection(model, prior, pools: list[Pool], eeg_lookup: dict,
                  window: int = 1, device="cpu",
                  gamma_zero: bool = False, seed: int = 0) -> SelectionResult:
    """Evaluate top-1 selection over pools.

    eeg_lookup: text -> (window, win_obs, win_pad, observed) tensors of the TRUE
    reading (built once by `prepare_eeg_lookup`). gamma_zero forces the structural
    null arm: all scores 0, ties broken uniformly at random -> accuracy must be 1/N.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    hits, per_pool = 0, []
    for p in pools:
        true_text = p.candidates[p.true_idx].text
        w, wo, wp, ob = eeg_lookup[true_text]
        scores = []
        for c in p.candidates:
            if gamma_zero:
                scores.append(0.0)
                continue
            toks = torch.from_numpy(c.token_ids)
            scores.append(score_candidate(model, prior, toks, w, wo, wp, ob, device))
        s = np.asarray(scores, dtype=np.float64)
        best = np.flatnonzero(s == s.max())
        choice = int(rng.choice(best))            # ties resolved at random, not argmax-first
        hit = int(choice == p.true_idx)
        hits += hit
        per_pool.append(hit)
    n = max(1, len(pools))
    acc = hits / n
    return SelectionResult(N=len(pools[0].candidates) if pools else 0,
                           accuracy=acc, hits=hits, n_pools=n,
                           bits_realized=fano_bits(acc, len(pools[0].candidates) if pools else 0),
                           per_pool_hit=per_pool)


def prepare_eeg_lookup(sentences: list, window: int = 1) -> dict:
    """text -> (window, win_obs, win_pad, observed) for the first reading of each text."""
    ds = PRDDataset(sentences, window=window)
    out = {}
    for i in range(len(ds)):
        item = ds[i]
        if item["text"] in out:
            continue
        out[item["text"]] = (item["window"], item["win_observed"],
                             item["win_pad"], item["observed"])
    return out


def predict_n_star(bits_per_sentence: float, target_acc: float = 0.5,
                   n_grid=(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)) -> int:
    """The matched-N prediction, from measured bits ONLY (written to prereg before test).

    Uses the exact ideal-observer accuracy for an I-bit observer over N equiprobable
    candidates via the tilted-posterior approximation acc(N) = 1/(1+(N-1)e^{-I ln2}).
    Returns the largest N in the grid with predicted acc >= target_acc. If NO N in the
    grid reaches the target (bits <= 0 means acc(2) <= 0.5), the smallest grid N is
    returned as a placeholder; use `n_star_meets_target` to record that fact.
    """
    I_nats = bits_per_sentence * LN2
    best = n_grid[0]
    for n in n_grid:
        acc = 1.0 / (1.0 + (n - 1) * math.exp(-I_nats))
        if acc >= target_acc:
            best = n
    return int(best)


def n_star_meets_target(bits_per_sentence: float, n: int, target_acc: float = 0.5) -> bool:
    """Whether the predicted ideal-observer accuracy at `n` actually reaches the target.
    False means predict_n_star's N* is a placeholder, not a prediction of success."""
    acc = 1.0 / (1.0 + (n - 1) * math.exp(-bits_per_sentence * LN2))
    return acc >= target_acc and bits_per_sentence > 0
