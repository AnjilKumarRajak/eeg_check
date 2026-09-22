"""Prior-residual sequential decoding, with the ablated floor that makes BLEU meaningful.

At each step the prior over the next token is tilted by the EEG evidence and the tilted
distribution is searched:

    log p_phi(v | e_t, w_hat_<t) = log p0(v | w_hat_<t) + gamma_t ell_t(v) - log Z_t

log Z_t is a per-step constant, so it does not change the argmax and can be dropped for
GREEDY decoding -- but it is beam-dependent and must NOT be dropped when ranking beams
against one another, because different beams have different prefixes and therefore
different log Z_t. It is retained here in both paths.

Two labels travel with every result and are never dropped:

    teacher_forced  -- whether the context was gold or the model's own history
    oracle_length   -- whether generation length came from the reference

`oracle_length=True` is a length leak: the reference tells the decoder when to stop,
which also defeats BLEU's brevity penalty. It is allowed for diagnostics only, and no
such number may be compared with a published BLEU.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import torch


@dataclass
class DecodeResult:
    hypotheses: list
    references: list
    arm: str                 # 'real' | 'shuffled' | 'gamma_zero'
    teacher_forced: bool
    oracle_length: bool
    max_len: int
    config_hash: str
    checkpoint: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n"] = len(self.hypotheses)
        return d


@torch.no_grad()
def greedy_prior_residual(model, prior, window, win_obs, win_pad, observed,
                          max_steps: int, eos_id: int, bos_id: int,
                          gamma_zero: bool = False) -> list[int]:
    """Sequential Bayesian update with the model's OWN decoding history.

    The EEG index advances with the generated position. That correspondence is only
    strictly valid under teacher forcing -- once the prefix diverges from the reference,
    EEG step t no longer describes the word actually being generated. This is an
    acknowledged approximation of free-running decoding, not a silent one.
    """
    device = window.device
    gen = [bos_id]
    for t in range(max_steps):
        ids = torch.tensor([gen], device=device)
        lp0 = prior.log_probs(ids)[:, -1, :]                    # (1,V)

        u = model.evidence(window[:, t], win_obs[:, t], win_pad[:, t])  # (1,r) centred
        bu = model.tilt.bu(u)                                   # (1,d)
        p = lp0.exp()
        H = -(p * lp0.clamp_min(-40.0)).sum(dim=-1)             # (1,)
        if gamma_zero:
            gamma = torch.zeros_like(H)
        else:
            gamma = model.tilt.gamma(H, u, observed[:, t])

        ell = bu @ prior.omega.transpose(0, 1)                  # (1,V) exact, full vocab
        logits = lp0 + gamma.unsqueeze(-1) * ell
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)   # proper posterior

        nxt = int(logits.argmax(dim=-1))
        if nxt == eos_id:
            break
        gen.append(nxt)
    return gen[1:]


@torch.no_grad()
def decode_split(model, prior, batches, tokenizer, arm: str = "real",
                 oracle_length: bool = False, max_len: int = 64,
                 config_hash: str = "", checkpoint: str = "",
                 null_seed: int = 0) -> DecodeResult:
    """Decode a whole split under one arm.

    Arms use an IDENTICAL loop, length policy and metric so their BLEU is comparable:
      real        -- real EEG
      shuffled    -- EEG from a length-matched different sentence (cross-sentence null)
      gamma_zero  -- prior only; the floor any EEG claim must beat
    """
    from .nulls import apply_sentence_null

    model.eval()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if eos_id is None:
        eos_id = -1
    if bos_id is None:
        bos_id = eos_id

    hyps, refs = [], []
    for bi, batch in enumerate(batches):
        w, o = batch["window"], batch["win_observed"]
        if arm == "shuffled":
            w, o = apply_sentence_null(w, o, batch["valid"], null_seed + bi)

        B = w.shape[0]
        for i in range(B):
            n_ref = int(batch["valid"][i].sum())
            steps = min(n_ref if oracle_length else max_len, w.shape[1])
            ids = greedy_prior_residual(
                model, prior,
                w[i:i + 1], o[i:i + 1], batch["win_pad"][i:i + 1],
                batch["observed"][i:i + 1],
                max_steps=steps, eos_id=eos_id, bos_id=bos_id,
                gamma_zero=(arm == "gamma_zero"),
            )
            hyps.append(tokenizer.decode(ids, skip_special_tokens=True))
            refs.append(batch["text"][i])

    return DecodeResult(
        hypotheses=hyps, references=refs, arm=arm,
        teacher_forced=False, oracle_length=oracle_length, max_len=max_len,
        config_hash=config_hash, checkpoint=checkpoint,
    )


def corpus_bleu(hyps: list[str], refs: list[str]) -> Optional[float]:
    """Corpus-level BLEU on detokenized word-level text, via sacrebleu.

    Corpus level, not sentence-averaged, and over words rather than BPE pieces --
    scoring BPE where hypothesis and reference both start with a special token grants a
    free unigram match on every sentence. Returns None if sacrebleu is unavailable
    rather than silently substituting a hand-rolled approximation.
    """
    try:
        import sacrebleu
    except ImportError:
        return None
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)
