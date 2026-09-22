"""Candidate pools for selection experiments, with the validity attack as a hard gate.

A pool is only usable if a NO-BRAIN classifier cannot find the true sentence in it:
text-only features (mean prior log-prob, length, type-token ratio) must score at or
below chance + 2pp. Otherwise the pool — not the EEG — would carry the answer.

Pools are deterministic given (seed, N) and hashed; the hash travels with every result.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Pool:
    true_idx: int                  # position of the true sentence in `candidates`
    candidates: list               # list of Sentence (true + distractors)
    pool_hash: str


def _length(s) -> int:
    return int(len(s.token_ids))


@torch.no_grad()
def mean_prior_logprob(prior, sent, device="cpu") -> float:
    ids = torch.from_numpy(sent.token_ids).unsqueeze(0).to(device)
    lp = prior.log_probs(ids)[0]
    return float(lp[torch.arange(ids.shape[1]), ids[0]].mean())


def build_corpus_pools(sentences: list, N: int, seed: int,
                       length_tol: int = 2, prior=None,
                       logp_tol: float = 0.5, device="cpu") -> list[Pool]:
    """One pool per unique TEXT: the true sentence + (N-1) length-matched distractors
    drawn from OTHER texts. Prior-logprob matching applied when a prior is supplied.
    """
    rng = np.random.default_rng(seed)
    by_text: dict = {}
    for s in sentences:
        by_text.setdefault(s.text, []).append(s)
    texts = sorted(by_text)

    logp = {}
    if prior is not None:
        for t in texts:
            logp[t] = mean_prior_logprob(prior, by_text[t][0], device)

    pools = []
    for t in texts:
        true = by_text[t][0]
        others = [u for u in texts if u != t]
        cand_texts = [u for u in others if abs(_length(by_text[u][0]) - _length(true)) <= length_tol]
        if prior is not None and len(cand_texts) > (N - 1):
            cand_texts = sorted(
                cand_texts, key=lambda u: abs(logp[u] - logp[t]))[: max(N - 1, 4 * N)]
        if len(cand_texts) < N - 1:
            cand_texts = others                       # fall back, never silently shrink N
        pick = rng.choice(len(cand_texts), size=N - 1, replace=len(cand_texts) < N - 1)
        cands = [true] + [by_text[cand_texts[int(i)]][0] for i in pick]
        order = rng.permutation(N)
        true_idx = int(np.flatnonzero(order == 0)[0])
        cands = [cands[int(i)] for i in order]
        h = hashlib.sha256(json.dumps(
            [c.text for c in cands], sort_keys=False).encode()).hexdigest()[:16]
        pools.append(Pool(true_idx=true_idx, candidates=cands, pool_hash=h))
    return pools


def text_only_attack(pools: list[Pool], prior=None, device="cpu",
                     seed: int = 0) -> float:
    """The pool-validity gate: can text features alone find the true sentence?

    Tries a battery of no-brain decision rules and reports the accuracy of the BEST
    one — a pool must survive the strongest cheap attack, not the average one:
    argmax/argmin length, closest-to-median length, argmax type-token ratio, and
    (when a prior is supplied) argmax/argmin mean prior log-prob. Anything materially
    above 1/N invalidates the pool set.
    """
    def lengths(p):
        return [float(_length(c)) for c in p.candidates]

    def med_close(p):
        lens = np.asarray(lengths(p))
        return (-np.abs(lens - np.median(lens))).tolist()

    def ttr(p):
        return [len(set(c.text.split())) / max(1, len(c.text.split()))
                for c in p.candidates]

    rules = [("longest", lengths, +1), ("shortest", lengths, -1),
             ("median_length", med_close, +1), ("ttr", ttr, +1)]
    if prior is not None:
        cache: dict = {}

        def logp(p):
            out = []
            for c in p.candidates:
                if c.text not in cache:
                    cache[c.text] = mean_prior_logprob(prior, c, device)
                out.append(cache[c.text])
            return out
        rules += [("prior_high", logp, +1), ("prior_low", logp, -1)]

    best = 0.0
    for _name, fn, sign in rules:
        hits = 0
        for p in pools:
            v = np.asarray(fn(p), dtype=float) * sign
            if int(np.argmax(v)) == p.true_idx:
                hits += 1
        best = max(best, hits / max(1, len(pools)))
    return best


def pool_validity_gate(pools: list[Pool], prior=None, device="cpu",
                       margin: float = 0.02) -> dict:
    """Returns {passed, attack_acc, chance}. Gate: attack_acc <= chance + margin."""
    if not pools:
        return {"passed": False, "attack_acc": float("nan"), "chance": float("nan")}
    N = len(pools[0].candidates)
    acc = text_only_attack(pools, prior=prior, device=device)
    chance = 1.0 / N
    return {"passed": acc <= chance + margin, "attack_acc": acc, "chance": chance,
            "n_pools": len(pools), "N": N}
