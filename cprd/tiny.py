"""TinyCausalLM: a small but genuinely causal reference LM for smoke runs and tests.

Deliberately a working causal model (real masked self-attention) rather than a mock —
the causality and zero-information gates must be able to fail against it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import ReferenceLM


class TinyCausalLM(ReferenceLM):
    def __init__(self, vocab_size: int = 64, d_model: int = 32,
                 n_layers: int = 2, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(512, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=2 * d_model,
            dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self._bos = 0
        self.freeze()

    def log_probs(self, token_ids: torch.LongTensor) -> torch.Tensor:
        B, T = token_ids.shape
        dev = token_ids.device
        if next(self.parameters()).device != dev:
            self.to(dev)
        start = torch.full((B, 1), self._bos, dtype=torch.long, device=dev)
        inp = torch.cat([start, token_ids[:, :-1]], dim=1)
        pos = torch.arange(T, device=dev).unsqueeze(0)
        h = self.embed(inp) + self.pos(pos)
        mask = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), diagonal=1)
        h = self.blocks(h, mask=mask)
        return F.log_softmax(self.head(h), dim=-1)

    @property
    def omega(self) -> torch.Tensor:
        return self.head.weight


class _WhitespaceTok:
    """Deterministic whitespace tokenizer for smoke-mode dataset builds."""

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = [1 + (hash(w) % (self.vocab_size - 2)) for w in text.split()]
        return {"input_ids": ids or [1]}
