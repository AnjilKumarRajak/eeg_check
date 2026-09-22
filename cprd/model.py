"""PriorResidualModel: frozen prior + evidence encoder + rank-budgeted tilt.

phi = {encoder, B, a, b, c, scale} is the complete trainable set. The prior and Omega
never receive gradient, and `train()` is overridden so that putting the model in train
mode cannot put the prior in train mode -- otherwise LM dropout would make p0 stochastic
and train-time and eval-time would silently use different reference measures.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn

from .encoder import CombinedEncoder, EvidenceEncoder, GazeEncoder, RankBudgetedTilt
from .objective import sentence_information_gain, assert_upper_bound
from .priors import ReferenceLM


@dataclass
class ModelConfig:
    r: int = 16
    window: int = 1
    d_s: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    s_max: float = 4.0
    sigma_max: float = 1.0
    gamma_max: float = 5.0
    gamma_mode: str = "learned"
    evidence: str = "eeg"          # "eeg" (840-d) | "gaze" (6-d) | "both" (846-d, EEG-beyond-gaze via report)
    evidence_dim: int = 840
    # Centre u_t on its mean over the data (batch mean in training, running mean at
    # eval). 2026-09-16 E1 finding: without this, raw dI ascent has a collapse basin --
    # a CONSTANT u earns ~1.7 nats on synthetic data (and ~1 nat on ZuCo) by tilting
    # the prior toward the token marginal without reading any evidence, and because
    # uninformative evidence-dependent variation only raises log Z (Jensen), the
    # encoder is driven to a constant function (probe: u std 5e-4 after training vs
    # 0.14 at init; dI real == dI permuted to 4 decimals at 2 injected bits/token).
    # With centring a constant encoder yields zero tilt and zero dI, so the only way
    # to earn dI is evidence-dependent structure. The ESTIMAND is unchanged: a
    # constant tilt component cancels exactly in dI(real) - dI(null) anyway.
    center_u: bool = True
    center_momentum: float = 0.05
    # Nonlinear rank-k tilt (2026-09-17, Findings §3): ell(v) = <Omega_v, B u> + <phi(Omega_v), W u>
    # with phi a small MLP over the FROZEN prior embedding (d_model -> free_tilt_hidden -> k).
    # The linear-in-Omega tilt cannot represent class structure over the vocabulary
    # (oracle: <= 0.14 of 1 injected bit); phi can carve half-spaces of it, and because
    # it is a function of Omega_v it generalises to tokens unseen in training (a free
    # V x k table does not: it memorises). Z stays exact over the full vocabulary.
    # free_tilt_rank = 0 reproduces the old model exactly.
    free_tilt_rank: int = 0
    free_tilt_hidden: int = 64
    # "concat" (default, original) | "gated": u = norm(W_g u_g + sigmoid(gate) * W_e u_e); the
    # EEG branch is only used where the learned gate opens (mixture-of-experts style).
    fusion: str = "concat"

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("fusion") == "concat":
            d.pop("fusion")            # keeps checkpoint keys of pre-existing runs unchanged
        return d


class PriorResidualModel(nn.Module):
    def __init__(self, prior: ReferenceLM, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # NOT registered as a submodule: keeps the LM out of state_dict() entirely, so
        # checkpoints are phi-only and a fine-tuned prior cannot be shipped by accident.
        object.__setattr__(self, "_prior", prior.freeze())
        if cfg.evidence == "gaze":
            self.encoder = GazeEncoder(
                g_dim=cfg.evidence_dim, r=cfg.r, d_model=cfg.d_model, n_heads=cfg.n_heads,
                n_layers=cfg.n_layers, dropout=cfg.dropout, s_max=cfg.s_max,
            )
        elif cfg.evidence == "eeg":
            self.encoder = EvidenceEncoder(
                r=cfg.r, d_s=cfg.d_s, d_model=cfg.d_model, n_heads=cfg.n_heads,
                n_layers=cfg.n_layers, dropout=cfg.dropout, s_max=cfg.s_max,
            )
        elif cfg.evidence == "both":
            self.encoder = CombinedEncoder(
                r=cfg.r, d_s=cfg.d_s, d_model=cfg.d_model, n_heads=cfg.n_heads,
                n_layers=cfg.n_layers, dropout=cfg.dropout, s_max=cfg.s_max,
                g_dim=cfg.evidence_dim - 840, eeg_dim=840, fusion=cfg.fusion,
            )
        else:
            raise ValueError(f"unknown evidence kind {cfg.evidence!r}")
        self.tilt = RankBudgetedTilt(
            d_model=prior.d_model, r=cfg.r, sigma_max=cfg.sigma_max,
            gamma_max=cfg.gamma_max, gamma_mode=cfg.gamma_mode,
        )
        self.free_phi = None
        self.free_W = None
        if cfg.free_tilt_rank > 0:
            self.free_phi = nn.Sequential(nn.Linear(prior.d_model, cfg.free_tilt_hidden), nn.GELU(),
                                          nn.Linear(cfg.free_tilt_hidden, cfg.free_tilt_rank))
            self.free_W = nn.Linear(cfg.r, cfg.free_tilt_rank, bias=False)
            nn.init.zeros_(self.free_W.weight)          # starts as the old model
        self.register_buffer("u_mean", torch.zeros(cfg.r))
        self.register_buffer("u_mean_n", torch.zeros(()))

    def evidence(self, window, win_obs, win_pad, observed=None, valid=None) -> torch.Tensor:
        """Encoder output, centred (see ModelConfig.center_u). In train mode the batch
        mean over observed valid tokens is subtracted and the running mean updated; in
        eval mode the running mean is subtracted. Norm is NOT re-normalised, so a
        near-constant encoder is not amplified back to full scale."""
        u = self.encoder(window, win_obs, win_pad)
        if not self.cfg.center_u:
            return u
        if self.training:
            mask = torch.ones(u.shape[:-1], dtype=torch.bool, device=u.device)
            if valid is not None:
                mask = mask & valid
            if observed is not None:
                mask = mask & observed
            if mask.any():
                mu = u[mask].mean(0)
                with torch.no_grad():
                    m = self.cfg.center_momentum if float(self.u_mean_n) > 0 else 1.0
                    self.u_mean.mul_(1 - m).add_(m * mu.detach())
                    self.u_mean_n += 1
                return u - mu
            return u - self.u_mean
        return u - self.u_mean

    @property
    def prior(self) -> ReferenceLM:
        return self._prior

    def extra_ell(self, u: torch.Tensor):
        """(..., V) free-table tilt logits, or None when free_tilt_rank == 0."""
        if self.free_phi is None:
            return None
        R = self.free_phi(self.prior.omega.detach().to(u.dtype))          # (V, k), recomputed per call
        R = R - R.mean(0, keepdim=True)                                   # no constant-tilt component
        return self.free_W(u) @ R.t()

    def gain_terms(self, batch: dict, log_p0: torch.Tensor, u: torch.Tensor,
                   gamma_override=None) -> dict:
        """dI terms for a given evidence vector u (used by forward and by the NCE
        training objective, which re-scores the same tokens under deranged evidence)."""
        observed, valid, tokens = batch["observed"], batch["valid"], batch["token_ids"]
        bu = self.tilt.bu(u)
        p = log_p0.exp()
        H = -(p * log_p0.clamp_min(-40.0)).sum(dim=-1)
        if gamma_override == "zero":
            gamma = torch.zeros_like(H)
        else:
            gamma = self.tilt.gamma(H, u, observed)
        out = sentence_information_gain(log_p0, self.prior.omega, tokens, bu, gamma, valid,
                                        extra_ell=self.extra_ell(u))
        out["u_norm"] = u.norm(dim=-1)
        out["prior_entropy"] = H
        return out

    def train(self, mode: bool = True):
        super().train(mode)
        self._prior.eval()          # never let the reference measure become stochastic
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def assert_prior_frozen(self) -> None:
        self._prior.assert_frozen()
        for n, p in self._prior.named_parameters():
            if p.grad is not None:
                raise RuntimeError(f"gradient reached frozen prior parameter {n}")

    def forward(self, batch: dict, log_p0: torch.Tensor,
                gamma_override: Optional[str] = None) -> dict:
        """batch: from data.collate. log_p0: (B,T,V) precomputed under the frozen prior.

        gamma_override='zero' forces gamma == 0 for the structural null-invariance arm.
        """
        window = batch["window"]              # (B,T,W,F)
        win_obs = batch["win_observed"]       # (B,T,W)
        win_pad = batch["win_pad"]            # (B,T,W)
        observed = batch["observed"]          # (B,T)
        valid = batch["valid"]                # (B,T)
        tokens = batch["token_ids"]           # (B,T)

        u = self.evidence(window, win_obs, win_pad, observed, valid)   # (B,T,r) centred
        return self.gain_terms(batch, log_p0, u, gamma_override)

    @torch.no_grad()
    def check_upper_bound(self, out: dict, valid: torch.Tensor) -> None:
        assert_upper_bound(out["per_token"], out["log_p0_gold"], valid)
