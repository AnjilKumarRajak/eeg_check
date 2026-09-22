"""EEG evidence encoder f_phi, and the rank-budgeted tilt.

Design choice: the 840-d word vector is not flat. It is 8 frequency bands x 105
scalp channels, and the two axes mean different things -- channels carry spatial
(topographic) structure, bands carry spectral structure with different physiological
interpretations. A flat MLP over 840 dims discards that.

So the encoder is explicitly factorised:

    (840,) -> (8 bands, 105 channels)
       -> per-band spatial projection        Linear(105 -> d_s), band-specific
       -> band attention pooling             learned query over the 8 bands
       -> temporal encoder over the window   small transformer over 2w+1 words
       -> u_t in R^r,  ||u_t|| = s <= s_max

Bounding ||u_t|| matters. ell(v) = <Omega_v, B u_t> scales with ||B|| * ||u||, so
constraining B alone is vacuous -- the encoder simply grows ||u|| to compensate and any
stated capacity budget becomes decorative. Both are constrained here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BANDS = 8
N_CHANNELS = 105
FEAT_DIM = N_BANDS * N_CHANNELS          # 840


class BandSpatialProjection(nn.Module):
    """Per-band spatial projection over the 105 scalp channels.

    Each band gets its own Linear(105 -> d_s): alpha and gamma topographies are not
    the same map, so sharing one projection across bands would force them to be.
    Implemented as one batched einsum rather than 8 separate Linears.
    """

    def __init__(self, d_s: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_s = d_s
        self.weight = nn.Parameter(torch.empty(N_BANDS, N_CHANNELS, d_s))
        self.bias = nn.Parameter(torch.zeros(N_BANDS, d_s))
        nn.init.xavier_uniform_(self.weight)
        self.norm = nn.LayerNorm(d_s)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 840) -> (..., 8, d_s)"""
        lead = x.shape[:-1]
        x = x.reshape(*lead, N_BANDS, N_CHANNELS)
        h = torch.einsum("...bc,bcd->...bd", x, self.weight) + self.bias
        return self.drop(F.gelu(self.norm(h)))


class BandAttentionPool(nn.Module):
    """Pool the 8 band representations with a learned query.

    Attention rather than a fixed sum, so the model can express which bands carry
    evidence; the learned weights are also a reportable diagnostic (which band matters)
    without claiming any physiological localisation.
    """

    def __init__(self, d_s: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_s) * 0.02)
        self.key = nn.Linear(d_s, d_s)
        self.value = nn.Linear(d_s, d_s)
        self.scale = d_s ** -0.5

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (..., 8, d_s) -> ((..., d_s), attention weights (..., 8))"""
        k = self.key(h)
        att = torch.softmax((k @ self.query) * self.scale, dim=-1)     # (...,8)
        v = self.value(h)
        return torch.einsum("...b,...bd->...d", att, v), att


class EvidenceEncoder(nn.Module):
    """f_phi: (window of word-level EEG, missingness mask) -> u_t in R^r."""

    def __init__(self, r: int = 16, d_s: int = 64, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 s_max: float = 4.0):
        super().__init__()
        self.r = r
        self.s_max = s_max
        self.spatial = BandSpatialProjection(d_s, dropout)
        self.band_pool = BandAttentionPool(d_s)
        # REVISED: Linear(d_s, d_model) instead of d_s + 1 (removed gaze leakage)
        self.in_proj = nn.Linear(d_s, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, r)
        # learned scale in (0, s_max): bounds ||u_t|| so the capacity budget binds
        self.scale_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, window: torch.Tensor, observed: torch.Tensor,
                pad: torch.Tensor | None = None) -> torch.Tensor:
        """window:   (B, W, 840) or (B, T, W, 840)
        observed: same shape minus the feature axis, True where EEG is real
        pad:      True where the window slot is off the end of the sentence
        returns u_t with ||u_t|| == s, shape (B, r) or (B, T, r)
        """
        squeeze_t = window.dim() == 3
        if squeeze_t:
            window, observed = window.unsqueeze(1), observed.unsqueeze(1)
            if pad is not None:
                pad = pad.unsqueeze(1)
        B, T, W, _ = window.shape

        # Zero out missing EEG slots
        obs_f = observed.unsqueeze(-1).to(window.dtype)
        x = torch.nan_to_num(window, nan=0.0) * obs_f

        h = self.spatial(x)                                   # (B,T,W,8,d_s)
        h, _ = self.band_pool(h)                              # (B,T,W,d_s)
        # Clean projection: do NOT feed obs_f directly as input feature to prevent gaze leakage
        h = self.in_proj(h)                                   # (B,T,W,d_model)

        h = h.reshape(B * T, W, -1)
        
        # Combined key padding mask: pad OR unobserved EEG slot
        if pad is not None:
            key_pad = pad.reshape(B * T, W) | (~observed.reshape(B * T, W))
        else:
            key_pad = ~observed.reshape(B * T, W)

        # A fully-padded row would make softmax produce NaN; keep one slot alive
        all_pad = key_pad.all(dim=1)
        if bool(all_pad.any()):
            key_pad = key_pad.clone()
            key_pad[all_pad, 0] = False
        h = self.temporal(h, src_key_padding_mask=key_pad)

        # Masked mean over observed slots only: unobserved/padding must not be averaged in
        m = (~key_pad).unsqueeze(-1).to(h.dtype)
        h = (h * m).sum(1) / m.sum(1).clamp(min=1.0)

        u = self.out(h).reshape(B, T, self.r)
        s = self.s_max * torch.sigmoid(self.scale_logit)
        u = s * F.normalize(u, dim=-1, eps=1e-6)
        return u.squeeze(1) if squeeze_t else u


class GazeEncoder(nn.Module):
    """f_phi for the GAZE evidence channel: (window of per-word gaze vectors) -> u_t.

    Same contract as EvidenceEncoder (same temporal encoder, same bounded ||u_t||, same
    masking rule: unobserved slots are padding, never an input flag) but the input is
    the 6-d eye-movement record per word (build.GAZE_COLS) instead of 840 band powers.
    This is the channel the EEG model must NOT see; it is measured here on its own so
    the two channels can be compared under the identical estimator and nulls."""

    def __init__(self, g_dim: int = 6, r: int = 16, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 s_max: float = 4.0):
        super().__init__()
        self.r, self.s_max, self.g_dim = r, s_max, g_dim
        self.in_proj = nn.Sequential(nn.Linear(g_dim, d_model), nn.LayerNorm(d_model),
                                     nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, r)
        self.scale_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, window: torch.Tensor, observed: torch.Tensor,
                pad: torch.Tensor | None = None) -> torch.Tensor:
        squeeze_t = window.dim() == 3
        if squeeze_t:
            window, observed = window.unsqueeze(1), observed.unsqueeze(1)
            if pad is not None:
                pad = pad.unsqueeze(1)
        B, T, W, _ = window.shape
        obs_f = observed.unsqueeze(-1).to(window.dtype)
        x = torch.nan_to_num(window, nan=0.0) * obs_f
        h = self.in_proj(x)                                    # (B,T,W,d_model)
        h = h.reshape(B * T, W, -1)
        if pad is not None:
            key_pad = pad.reshape(B * T, W) | (~observed.reshape(B * T, W))
        else:
            key_pad = ~observed.reshape(B * T, W)
        all_pad = key_pad.all(dim=1)
        if bool(all_pad.any()):
            key_pad = key_pad.clone()
            key_pad[all_pad, 0] = False
        h = self.temporal(h, src_key_padding_mask=key_pad)
        m = (~key_pad).unsqueeze(-1).to(h.dtype)
        h = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        u = F.normalize(self.out(h), dim=-1) * (self.s_max * torch.sigmoid(self.scale_logit))
        u = u.reshape(B, T, self.r)
        return u.squeeze(1) if squeeze_t else u


class CombinedEncoder(nn.Module):
    """f_phi for the COMBINED channel [eeg(840) | gaze(6)] -> u_t.

    Two branches, fused at the evidence-vector level:
      * the EEG branch is the unchanged EvidenceEncoder, and its slots are masked by the
        gaze `fixated` column (col 840) exactly as the EEG-only run masks unfixated
        words -- so the EEG branch here is the EEG-only model, no more, no less;
      * the gaze branch is the unchanged GazeEncoder over the 6 gaze columns.
    u_t = normalize(W [u_eeg ; u_gaze]) * s. The combined channel exists so that
    I(text; EEG | gaze) can be read off as I(both) - I(gaze); it is never the headline."""

    def __init__(self, r: int = 16, d_s: int = 64, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1, s_max: float = 4.0,
                 g_dim: int = 6, eeg_dim: int = FEAT_DIM, fusion: str = "concat"):
        super().__init__()
        assert fusion in ("concat", "gated")
        self.fusion = fusion
        self._aux = None
        self.last_gate = None
        self.r, self.s_max, self.eeg_dim, self.g_dim = r, s_max, eeg_dim, g_dim
        self.eeg = EvidenceEncoder(r=r, d_s=d_s, d_model=d_model, n_heads=n_heads,
                                   n_layers=n_layers, dropout=dropout, s_max=s_max)
        self.gaze = GazeEncoder(g_dim=g_dim, r=r, d_model=d_model, n_heads=n_heads,
                                n_layers=n_layers, dropout=dropout, s_max=s_max)
        self.fuse = nn.Linear(2 * r, r)
        if fusion == "gated":
            self.fuse_g = nn.Linear(r, r)
            self.fuse_e = nn.Linear(r, r)
            self.gate = nn.Linear(2 * r, 1)
            nn.init.constant_(self.gate.bias, -2.0)      # EEG expert starts (mostly) closed
        self.scale_logit = nn.Parameter(torch.tensor(0.0))

    def aux_loss(self):
        """Mean squared cross-correlation between the EEG and gaze evidence vectors of the last
        forward (decorrelation regulariser); 0 if unavailable."""
        return self._aux if self._aux is not None else torch.zeros(())

    def forward(self, window: torch.Tensor, observed: torch.Tensor,
                pad: torch.Tensor | None = None) -> torch.Tensor:
        squeeze_t = window.dim() == 3
        if squeeze_t:
            window, observed = window.unsqueeze(1), observed.unsqueeze(1)
            if pad is not None:
                pad = pad.unsqueeze(1)
        w_eeg = window[..., :self.eeg_dim]
        w_gz = window[..., self.eeg_dim:self.eeg_dim + self.g_dim]
        fixated = w_gz[..., 0] > 0.5                       # (B,T,W): EEG present iff fixated
        obs_eeg = observed & fixated
        u_e = self.eeg(w_eeg, obs_eeg, pad)
        u_g = self.gaze(w_gz, observed, pad)
        if self.training:
            a = (u_e - u_e.mean((0, 1), keepdim=True)).reshape(-1, self.r)
            b = (u_g - u_g.mean((0, 1), keepdim=True)).reshape(-1, self.r)
            a = a / (a.std(0, keepdim=True) + 1e-6); b = b / (b.std(0, keepdim=True) + 1e-6)
            self._aux = ((a.t() @ b) / a.shape[0]).pow(2).mean()
        if self.fusion == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([u_e, u_g], dim=-1)))
            self.last_gate = gate.detach()
            z = self.fuse_g(u_g) + gate * self.fuse_e(u_e)
        else:
            z = self.fuse(torch.cat([u_e, u_g], dim=-1))
        u = F.normalize(z, dim=-1) * (self.s_max * torch.sigmoid(self.scale_logit))
        return u.squeeze(1) if squeeze_t else u


class RankBudgetedTilt(nn.Module):
    """ell_t(v) = <Omega_v, B u_t>, plus the evidence gain gamma_t.

    B is constrained by its SPECTRAL norm (power iteration), not its Frobenius norm.
    They are different quantities: Frobenius mass can concentrate on a single singular
    direction, so a Frobenius bound does not bound the operator's gain.

    gamma has NO lower clamp on the pre-activation. Clamping it (e.g. at -3) floors
    gamma >= softplus(-3) ~ 0.049, which makes gamma = 0 unreachable by learning and so
    makes null-invariance structurally impossible for a trained model -- the exact
    property the objective exists to provide.
    """

    def __init__(self, d_model: int, r: int, sigma_max: float = 1.0,
                 gamma_max: float = 5.0, gamma_mode: str = "learned"):
        super().__init__()
        assert gamma_mode in ("learned", "constant", "entropy_only", "zero")
        self.d_model, self.r = d_model, r
        self.sigma_max, self.gamma_max = sigma_max, gamma_max
        self.gamma_mode = gamma_mode
        self.gamma_const = 1.0          # value used in 'constant' mode (warmup sets it)

        self.B = nn.Parameter(torch.empty(d_model, r))
        nn.init.orthogonal_(self.B)
        with torch.no_grad():
            self.B.mul_(0.1)
        self.register_buffer("_u_iter", F.normalize(torch.randn(d_model), dim=0))

        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(r))
        self.c = nn.Parameter(torch.full((1,), -2.0))

    @torch.no_grad()
    def project_spectral(self, n_iter: int = 2) -> float:
        """Project B onto {||B||_2 <= sigma_max} via power iteration."""
        u = self._u_iter
        for _ in range(n_iter):
            v = F.normalize(self.B.t() @ u, dim=0, eps=1e-8)
            u = F.normalize(self.B @ v, dim=0, eps=1e-8)
        sigma = float(torch.dot(u, self.B @ v))
        self._u_iter.copy_(u)
        if sigma > self.sigma_max:
            self.B.mul_(self.sigma_max / (sigma + 1e-12))
        return sigma

    def bu(self, u: torch.Tensor) -> torch.Tensor:
        """u: (..., r) -> B u : (..., d_model)"""
        return u @ self.B.t()

    def gamma(self, prior_entropy: torch.Tensor, u: torch.Tensor,
              observed: torch.Tensor) -> torch.Tensor:
        """gamma_t >= 0, hard-gated to exactly 0 where the word carried no EEG.

        43.5% of ZuCo tokens are unfixated. Without this gate the encoder can read
        "all zeros" as "skipped word => frequent, predictable word" and earn dI from
        gaze behaviour rather than neural signal.
        """
        if self.gamma_mode == "zero":
            g = torch.zeros_like(prior_entropy)
        elif self.gamma_mode == "constant":
            g = torch.full_like(prior_entropy, float(self.gamma_const))
        elif self.gamma_mode == "entropy_only":
            g = F.softplus(self.a * prior_entropy + self.c)
        else:
            g = F.softplus(self.a * prior_entropy + (u * self.b).sum(-1) + self.c)
        g = g.clamp(max=self.gamma_max)
        return g * observed.to(g.dtype)
