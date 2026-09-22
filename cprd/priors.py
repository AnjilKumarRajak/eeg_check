
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PriorSanity:
    mean_surprisal: float
    perplexity: float
    uniform_nats: float
    causal: bool
    deterministic: bool
    all_frozen: bool
    min_max_logprob: float          # most uniform-looking row seen
    n_tokens: int

    @property
    def ok(self) -> bool:
        return (
            self.causal
            and self.deterministic
            and self.all_frozen
            and self.mean_surprisal < self.uniform_nats
            and self.mean_surprisal < 8.0
            and self.min_max_logprob > -(self.uniform_nats - 1.0)
        )

    def describe(self) -> str:
        rows = [
            ("mean surprisal (nats/token)", f"{self.mean_surprisal:.3f}", f"< {min(8.0, self.uniform_nats):.2f}"),
            ("perplexity", f"{self.perplexity:.1f}", ""),
            ("uniform baseline (nats)", f"{self.uniform_nats:.3f}", ""),
            ("causal under future perturbation", str(self.causal), "True"),
            ("deterministic (2 forwards equal)", str(self.deterministic), "True"),
            ("all params frozen", str(self.all_frozen), "True"),
            ("worst-row max log p0", f"{self.min_max_logprob:.3f}", f"> {-(self.uniform_nats-1.0):.2f}"),
            ("tokens scored", str(self.n_tokens), ""),
        ]
        w = max(len(r[0]) for r in rows)
        out = [f"  {k:<{w}}  {v:>12}  {req}" for k, v, req in rows]
        return "\n".join(out) + f"\n  => {'PASS' if self.ok else 'FAIL'}"


class ReferenceLM(nn.Module):
    vocab_size: int
    d_model: int

    def log_probs(self, token_ids: torch.LongTensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def omega(self) -> torch.Tensor:
        """Frozen output embedding matrix (V, d). Never receives gradient."""
        raise NotImplementedError

    # -- freezing -------------------------------------------------------
    def freeze(self) -> "ReferenceLM":
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def train(self, mode: bool = True):  # noqa: D102 - deliberate override
        # A frozen reference measure must never enter train mode: dropout would make
        # p0 stochastic, so train-time and eval-time would use different priors.
        return super().train(False)

    def assert_frozen(self) -> None:
        bad = [n for n, p in self.named_parameters() if p.requires_grad]
        if bad:
            raise RuntimeError(f"reference LM has trainable parameters: {bad[:5]}")
        if self.training:
            raise RuntimeError("reference LM is in train mode; p0 would be stochastic")

    # -- acceptance gate ------------------------------------------------
    @torch.no_grad()
    def sanity_report(self, sentences: list[torch.LongTensor], device=None) -> PriorSanity:
        """Acceptance gate. Must pass before any dI is computed."""
        device = device or next(self.parameters()).device
        tot, n, worst = 0.0, 0, float("inf")
        for ids in sentences:
            ids = ids.to(device)
            lp = self.log_probs(ids.unsqueeze(0))[0]                      # (T,V)
            tot += float(-lp[torch.arange(len(ids), device=device), ids].sum())
            n += len(ids)
            worst = min(worst, float(lp.max(dim=-1).values.min()))

        # causality: scrambling w_>=t must not change rows < t
        ids = sentences[0].to(device)
        t = max(1, len(ids) // 2)
        a = self.log_probs(ids.unsqueeze(0))[0]
        ids2 = ids.clone()
        ids2[t:] = (ids2[t:] + 7919) % self.vocab_size
        b = self.log_probs(ids2.unsqueeze(0))[0]
        causal = bool(torch.allclose(a[:t], b[:t], atol=1e-5))

        c = self.log_probs(ids.unsqueeze(0))[0]
        deterministic = bool(torch.equal(a, c))
        all_frozen = not any(p.requires_grad for p in self.parameters())

        surp = tot / max(1, n)
        return PriorSanity(
            mean_surprisal=surp,
            perplexity=float(math.exp(min(surp, 700))),
            uniform_nats=float(math.log(self.vocab_size)),
            causal=causal,
            deterministic=deterministic,
            all_frozen=all_frozen,
            min_max_logprob=worst,
            n_tokens=n,
        )


class BartEmptySourcePrior(ReferenceLM):
    def __init__(self, model_name: str = "facebook/bart-large"):
        super().__init__()
        from transformers import BartForConditionalGeneration, BartTokenizerFast

        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        try:
            self.tokenizer = BartTokenizerFast.from_pretrained(model_name)
        except Exception:
            self.tokenizer = None
        self.vocab_size = int(self.model.config.vocab_size)
        self.d_model = int(self.model.config.d_model)
        self._bos = int(self.model.config.bos_token_id)
        self._eos = int(self.model.config.eos_token_id)
        ds = self.model.config.decoder_start_token_id
        self._dec_start = int(ds if ds is not None else self._eos)
        self.freeze()

    def log_probs(self, token_ids: torch.LongTensor) -> torch.Tensor:
        B, T = token_ids.shape
        dev = token_ids.device
        if next(self.model.parameters()).device != dev:
            self.model.to(dev)
        enc = torch.tensor([[self._bos, self._eos]], device=dev).expand(B, 2)
        start = torch.full((B, 1), self._dec_start, dtype=torch.long, device=dev)
        # position t sees [start, w_0 .. w_{t-1}] and predicts w_t: strictly causal,
        # and no zero-logit row, so position 0 is a real prediction not a uniform.
        dec_in = torch.cat([start, token_ids[:, :-1]], dim=1)
        out = self.model(input_ids=enc, decoder_input_ids=dec_in)
        return F.log_softmax(out.logits, dim=-1)

    @property
    def omega(self) -> torch.Tensor:
        return self.model.get_output_embeddings().weight


class CausalLMPrior(ReferenceLM):

    def __init__(self, model_name: str = "gpt2-large"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = int(self.model.config.vocab_size)
        self.d_model = int(getattr(self.model.config, "n_embd", None)
                           or self.model.config.hidden_size)
        bos = self.model.config.bos_token_id
        self._bos = int(bos if bos is not None else self.tokenizer.eos_token_id)
        self.freeze()

    def log_probs(self, token_ids: torch.LongTensor) -> torch.Tensor:
        B, T = token_ids.shape
        dev = token_ids.device
        if next(self.model.parameters()).device != dev:
            self.model.to(dev)
        start = torch.full((B, 1), self._bos, dtype=torch.long, device=dev)
        inp = torch.cat([start, token_ids[:, :-1]], dim=1)   # position t predicts w_t
        out = self.model(input_ids=inp)
        return F.log_softmax(out.logits, dim=-1)

    @property
    def omega(self) -> torch.Tensor:
        return self.model.get_output_embeddings().weight


def build_prior(name: str, **kw) -> ReferenceLM:
    if name in ("bart_empty_source", "bart"):
        return BartEmptySourcePrior(**kw)
    if name in ("causal_lm", "gpt2"):
        return CausalLMPrior(**kw)
    raise ValueError(f"unknown prior '{name}'")
