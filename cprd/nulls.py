
from __future__ import annotations

import numpy as np
import torch


def _derangement(n: int, rng: np.random.Generator, tries: int = 100) -> np.ndarray:
    """Permutation with no fixed points. Falls back to a cyclic shift for tiny n."""
    if n < 2:
        return np.arange(n)
    for _ in range(tries):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    return np.roll(np.arange(n), 1)


def length_stratified_derangement(lengths: np.ndarray, rng: np.random.Generator,
                                  bucket: int = 2) -> np.ndarray:
    """Donor index per sentence: deranged within buckets of similar length."""
    n = len(lengths)
    donor = np.arange(n)
    order = np.argsort(lengths, kind="stable")
    i = 0
    while i < n:
        j = i
        while j < n and lengths[order[j]] - lengths[order[i]] <= bucket:
            j += 1
        idx = order[i:j]
        if len(idx) >= 2:
            donor[idx] = idx[_derangement(len(idx), rng)]
        else:
            # singleton bucket: borrow from the neighbouring bucket rather than self
            # (position i in the sorted order; the next-longer sentence, or the
            # next-shorter one at the end of the order)
            nb = i + 1 if i + 1 < n else i - 1
            donor[idx] = order[nb]
        i = j
    assert n < 2 or not np.any(donor == np.arange(n)), "derangement produced a self-donor"
    return donor


def _observed_row_index(win_obs: torch.Tensor, valid: torch.Tensor, i: int) -> torch.Tensor:
    L = int(valid[i].sum())
    W = win_obs.shape[-1]
    centre = win_obs[i, :L, W // 2]
    idx = torch.nonzero(centre, as_tuple=False).flatten()
    if idx.numel() == 0:
        idx = torch.arange(L, device=win_obs.device)
    return idx


def apply_sentence_null(window: torch.Tensor, win_obs: torch.Tensor,
                        valid: torch.Tensor, seed: int, bucket: int = 2):
    B, T, W, F = window.shape
    rng = np.random.default_rng(seed)
    lengths = valid.sum(dim=1).cpu().numpy()
    donor = length_stratified_derangement(lengths, rng, bucket)

    out_w = torch.zeros_like(window)
    out_o = torch.zeros_like(win_obs)
    for i in range(B):
        d = int(donor[i])
        Li = int(lengths[i])
        # wrap the donor's OBSERVED rows to cover the recipient's length; never zero-pad,
        # never hand an observed token an empty window (see _observed_row_index)
        drows = _observed_row_index(win_obs, valid, d)
        idx = drows[torch.arange(Li, device=window.device) % drows.numel()]
        out_w[i, :Li] = window[d, idx]
        out_o[i, :Li] = win_obs[d, idx]
    return out_w, out_o


def apply_temporal_roll_null(window: torch.Tensor, win_obs: torch.Tensor,
                             valid: torch.Tensor, seed: int):
    B, T, W, F = window.shape
    rng = np.random.default_rng(seed)
    out_w = window.clone()
    out_o = win_obs.clone()
    for i in range(B):
        L = int(valid[i].sum())
        if L < 2:
            continue
        # roll among OBSERVED rows only (missing rows stay where they are), so every
        # observed token still carries a real EEG row -- see _observed_row_index
        rows = _observed_row_index(win_obs, valid, i)
        m = int(rows.numel())
        if m < 2:
            continue
        k = int(rng.integers(1, m))            # non-zero offset
        src = rows[(torch.arange(m, device=window.device) + k) % m]
        out_w[i, rows] = window[i, src]
        out_o[i, rows] = win_obs[i, src]
    return out_w, out_o


def apply_gaussian_null(window: torch.Tensor, win_obs: torch.Tensor,
                        valid: torch.Tensor, seed: int,
                        channel_mean: torch.Tensor | None = None,
                        channel_std: torch.Tensor | None = None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(window.shape, generator=g).to(window.device)
    if channel_mean is None or channel_std is None:
        F = window.shape[-1]
        flat = window.reshape(-1, F)
        obs = win_obs.reshape(-1)
        if bool(obs.any()):
            vals = flat[obs]                                  # (n_obs, F)
            channel_mean = vals.mean(dim=0)                   # (F,)
            channel_std = vals.std(dim=0).clamp(min=1e-8)     # (F,)
        else:
            channel_mean = torch.zeros(F, device=window.device)
            channel_std = torch.ones(F, device=window.device)
    return channel_mean + channel_std * noise, win_obs.clone()


def apply_zeroed_null(window: torch.Tensor, win_obs: torch.Tensor,
                      valid: torch.Tensor, seed: int):

    return torch.zeros_like(window), win_obs.clone()


def apply_amplitude_only_null(window: torch.Tensor, win_obs: torch.Tensor,
                              valid: torch.Tensor, seed: int):

    B, T, W, F = window.shape
    out = torch.zeros_like(window)
    for i in range(B):
        m = win_obs[i]                                        # (T, W)
        if bool(m.any()):
            mean_vec = window[i][m].mean(dim=0)               # (F,)
        else:
            mean_vec = torch.zeros(F, device=window.device)
        out[i] = mean_vec
    return out, win_obs.clone()


def apply_position_only_null(window: torch.Tensor, win_obs: torch.Tensor,
                             valid: torch.Tensor, seed: int):

    B, T, W, F = window.shape
    pos = torch.arange(T, dtype=window.dtype, device=window.device)
    enc = torch.zeros(T, F, dtype=window.dtype, device=window.device)
    half = min(F, 64) // 2
    for k in range(half):
        enc[:, 2 * k] = torch.sin(pos / (10000 ** (k / max(1, half))))
        enc[:, 2 * k + 1] = torch.cos(pos / (10000 ** (k / max(1, half))))
    out = enc.unsqueeze(1).expand(T, W, F).unsqueeze(0).expand(B, T, W, F).contiguous()
    return out, win_obs.clone()


def text_cluster_derangement(texts: list, lengths: np.ndarray,
                             rng: np.random.Generator, bucket: int = 2) -> np.ndarray:
    n = len(texts)
    donor = length_stratified_derangement(lengths, rng, bucket)
    if n < 2:
        return donor
    codes = np.unique(np.asarray(texts, dtype=object).astype(str), return_inverse=True)[1]
    by_len: dict = {}
    for i, L in enumerate(np.asarray(lengths).tolist()):
        by_len.setdefault(L, []).append(i)

    def ok(r, d):
        return codes[d] != codes[r] and d != r

    for i in np.flatnonzero(codes[donor] == codes).tolist():
        if ok(i, donor[i]):
            continue                              # already repaired by an earlier swap
        same = by_len.get(int(lengths[i]), [])
        fixed = False
        for pool, tries in ((same, 64), (range(n), 512)):
            m = len(pool)
            if m < 2:
                continue
            for _ in range(tries):
                j = pool[int(rng.integers(m))]
                if j != i and ok(i, donor[j]) and ok(j, donor[i]):
                    donor[i], donor[j] = donor[j], donor[i]
                    fixed = True
                    break
            if fixed:
                break
    return donor


def apply_text_cluster_null(window: torch.Tensor, win_obs: torch.Tensor,
                            valid: torch.Tensor, seed: int,
                            texts: list | None = None, bucket: int = 2):
    """Cross-sentence null with donor text != recipient text, wrapped never zero-padded."""
    B, T, W, F = window.shape
    rng = np.random.default_rng(seed)
    lengths = valid.sum(dim=1).cpu().numpy()
    if texts is None:
        donor = length_stratified_derangement(lengths, rng, bucket)
    else:
        donor = text_cluster_derangement(list(texts), lengths, rng, bucket)
    out_w = torch.zeros_like(window)
    out_o = torch.zeros_like(win_obs)
    for i in range(B):
        d = int(donor[i])
        Li = int(lengths[i])
        drows = _observed_row_index(win_obs, valid, d)
        idx = drows[torch.arange(Li, device=window.device) % drows.numel()]
        out_w[i, :Li] = window[d, idx]
        out_o[i, :Li] = win_obs[d, idx]
    return out_w, out_o


NULLS = {
    "sentence_derangement": apply_sentence_null,
    "temporal_roll": apply_temporal_roll_null,
    "gaussian_matched": apply_gaussian_null,
    "zeroed": apply_zeroed_null,
    "amplitude_only": apply_amplitude_only_null,
    "position_only": apply_position_only_null,
    "text_cluster_derangement": apply_text_cluster_null,
}
