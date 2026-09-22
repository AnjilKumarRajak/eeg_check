
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .nulls import NULLS, length_stratified_derangement, text_cluster_derangement, _observed_row_index
from .objective import assert_upper_bound


@dataclass
class DIResult:
    dI_real: float                       # nats/token, all tokens
    dI_real_observed: float              # nats/token, EEG-present tokens only
    dI_real_missing: float               # nats/token, EEG-absent tokens (should be ~0)
    dI_null: dict                        # null name -> mean dI
    dI_hat: dict                         # null name -> real - null
    p_value: dict                        # null name -> permutation p
    ci95: tuple                          # clustered bootstrap CI on dI_real_observed (raw)
    n_tokens: int
    n_observed: int
    n_sentences: int
    n_clusters: int
    frac_positive: float
    dI_hat_nce: dict = None              # null -> per-token InfoNCE bound with the null draws as
                                         # negatives (positive included; <= log(n_perm+1)); a
                                         # valid lower bound on I and the estimand of the
                                         # 2026-09-17 instrument (docs/FINDINGS_20260917.md)
    dI_hat_dv: dict = None               # null -> Donsker-Varadhan lower bound: real - log E_null[e^dI]
                                         # (a valid lower bound on I; the mean-null dI_hat is not, see
                                         # docs/FINDINGS_20260917.md)
    ci95_dI_hat: dict = None             # null -> CI on the CORRECTED dI_hat (raw CI shifted
                                         # by that null's mean; null-mean SE is negligible at
                                         # n_perm >= 1e3, so the shift is exact to O(1/sqrt(n)))
    meta: dict = field(default_factory=dict)

    def describe(self) -> str:
        L = [
            f"  sentences={self.n_sentences}  unique-text clusters={self.n_clusters}",
            f"  tokens={self.n_tokens}  with EEG={self.n_observed} "
            f"({100*self.n_observed/max(1,self.n_tokens):.1f}%)",
            f"  dI (all tokens)        = {self.dI_real:+.4f} nats/token",
            f"  dI (EEG present)       = {self.dI_real_observed:+.4f} nats/token  "
            f"[95% CI {self.ci95[0]:+.4f}, {self.ci95[1]:+.4f}]  <- headline",
            f"  dI (EEG absent)        = {self.dI_real_missing:+.4f} nats/token  "
            f"(structurally 0; non-zero indicates a bug)",
            f"  fraction of tokens with dI > 0: {100*self.frac_positive:.1f}%",
        ]
        for k in self.dI_null:
            L.append(f"  null[{k:<22}] = {self.dI_null[k]:+.4f}   "
                     f"dI_hat = {self.dI_hat[k]:+.4f}   p = {self.p_value[k]:.4f}"
                     + (f"   DV = {self.dI_hat_dv[k]:+.4f}" if self.dI_hat_dv else "")
                     + (f"   InfoNCE = {self.dI_hat_nce[k]:+.4f}" if self.dI_hat_nce else ""))
        return "\n".join(L)
GLOBAL_NULLS = ("sentence_derangement", "text_cluster_derangement")


def _global_null_windows(batches, null: str, seed: int):
    rng = np.random.default_rng(seed)
    loc, lengths, texts = [], [], []
    for bi, b in enumerate(batches):
        v = b["valid"]
        for i in range(v.shape[0]):
            loc.append((bi, i)); lengths.append(int(v[i].sum()))
            texts.append(b["text"][i] if "text" in b else "")
    lengths = np.asarray(lengths)
    if any(texts):
        donor = text_cluster_derangement(texts, lengths, rng, bucket=0)
    else:
        donor = length_stratified_derangement(lengths, rng, bucket=0)
    out = []
    g = 0
    for bi, b in enumerate(batches):
        W, O = b["window"], b["win_observed"]
        out_w, out_o = torch.zeros_like(W), torch.zeros_like(O)
        for i in range(W.shape[0]):
            bd, idd = loc[int(donor[g])]
            Wd, Od, Vd = batches[bd]["window"], batches[bd]["win_observed"], batches[bd]["valid"]
            rows = _observed_row_index(Od, Vd, idd)
            Li = int(lengths[g])
            pos = torch.arange(Li, device=rows.device)
            j = torch.searchsorted(rows, pos).clamp(max=rows.numel() - 1)
            jm = (j - 1).clamp(min=0)
            pick = torch.where((rows[j] - pos).abs() <= (pos - rows[jm]).abs(), j, jm)
            idx = rows[pick]
            out_w[i, :Li] = Wd[idd, idx].to(W.device)
            out_o[i, :Li] = Od[idd, idx].to(O.device)
            g += 1
        out.append((out_w, out_o))
    return out


@torch.no_grad()
def _forward_tokens(model, batches, log_p0s, gamma_override=None,
                    null: Optional[str] = None, seed: int = 0,
                    return_mean_only: bool = False):
    """Run the model over prepared batches, returning flat per-token arrays or observed mean."""
    dev = next(model.parameters()).device
    gnull = _global_null_windows(batches, null, seed) if null in GLOBAL_NULLS else None
    if return_mean_only:
        total_sum, total_n = 0.0, 0
        lse = None                                     # running logsumexp of per-token dI (DV term)
        per_tok = []                                   # per-token dI on observed tokens, in order
        for bi, batch in enumerate(batches):
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if log_p0s is not None and bi < len(log_p0s) and log_p0s[bi] is not None:
                lp0_dev = log_p0s[bi].to(dev)
            else:
                lp0_dev = model.prior.log_probs(b["token_ids"])
            if gnull is not None:
                b["window"], b["win_observed"] = gnull[bi][0].to(dev), gnull[bi][1].to(dev)
            elif null is not None:
                w, o = NULLS[null](b["window"], b["win_observed"], b["valid"], seed + bi)
                b["window"], b["win_observed"] = w, o
            out = model(b, lp0_dev, gamma_override=gamma_override)
            assert_upper_bound(out["per_token"], out["log_p0_gold"], b["valid"])
            obs_mask = b["observed"] & b["valid"]
            if obs_mask.any():
                pt = out["per_token"][obs_mask].double()
                per_tok.append(pt.cpu().numpy())
                total_sum += float(pt.sum().item())
                total_n += int(obs_mask.sum().item())
                l = float(torch.logsumexp(pt, 0).item())
                lse = l if lse is None else float(np.logaddexp(lse, l))
        if total_n == 0:
            return float("nan"), float("nan"), np.zeros(0)
        # (mean dI, log-mean-exp dI, per-token dI)
        return total_sum / total_n, (lse - np.log(total_n)), np.concatenate(per_tok)

    dI, obs, texts, sent_id = [], [], [], []
    for bi, batch in enumerate(batches):
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if log_p0s is not None and bi < len(log_p0s) and log_p0s[bi] is not None:
            lp0_dev = log_p0s[bi].to(dev)
        else:
            lp0_dev = model.prior.log_probs(b["token_ids"])
        if gnull is not None:
            b["window"], b["win_observed"] = gnull[bi][0].to(dev), gnull[bi][1].to(dev)
        elif null is not None:
            w, o = NULLS[null](b["window"], b["win_observed"], b["valid"], seed + bi)
            b["window"], b["win_observed"] = w, o
        out = model(b, lp0_dev, gamma_override=gamma_override)
        assert_upper_bound(out["per_token"], out["log_p0_gold"], b["valid"])
        v = b["valid"]
        pt = out["per_token"]
        for i in range(v.shape[0]):
            n = int(v[i].sum())
            dI.append(pt[i, :n].detach().cpu().numpy())
            obs.append(b["observed"][i, :n].detach().cpu().numpy())
            texts.extend([b["text"][i]] * n)
            sent_id.extend([f"{bi}:{i}"] * n)
    return (np.concatenate(dI) if dI else np.zeros(0),
            np.concatenate(obs) if obs else np.zeros(0, bool),
            np.array(texts), np.array(sent_id))


def _load_checkpoint(path: str) -> np.ndarray | None:
    if path and os.path.exists(path):
        try:
            return np.load(path)
        except Exception:
            return None    # corrupt/truncated checkpoint (e.g. killed mid-write): redo it
    return None


def _save_checkpoint(path: str, arr: np.ndarray) -> None:
    """Atomic write: never leaves a truncated/corrupt file for a crash to load."""
    if not path:
        return
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _clustered_bootstrap(values: np.ndarray, clusters: np.ndarray,
                         n_boot: int = 2000, seed: int = 0,
                         checkpoint_path: str | None = None,
                         checkpoint_every: int = 500) -> tuple:

    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uniq}
    means = np.full(n_boot, np.nan)
    start = 0
    cached = _load_checkpoint(checkpoint_path) if checkpoint_path else None
    if cached is not None and len(cached) == n_boot:
        done = np.flatnonzero(~np.isnan(cached))
        start = int(done[-1]) + 1 if len(done) else 0
        means[:start] = cached[:start]
        for _ in range(start):
            rng.choice(uniq, size=len(uniq), replace=True)
    for b in range(start, n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by[c] for c in pick])
        means[b] = values[sel].mean()
        if checkpoint_path and ((b + 1) % checkpoint_every == 0 or b + 1 == n_boot):
            _save_checkpoint(checkpoint_path, means)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


@torch.no_grad()
def evaluate(model, batches, log_p0s=None, nulls=("sentence_derangement", "temporal_roll"),
             n_perm: int = 200, n_boot: int = 2000, seed: int = 0,
             meta: Optional[dict] = None, progress: bool = False,
             checkpoint_dir: Optional[str] = None, checkpoint_tag: str = "eval",
             checkpoint_every: int = 500) -> DIResult:
    model.eval()
    if log_p0s is None or any(lp is None for lp in log_p0s):
        dev = next(model.parameters()).device
        log_p0s = [model.prior.log_probs(b["token_ids"].to(dev)).cpu() for b in batches]
    dI, obs, texts, sent_id = _forward_tokens(model, batches, log_p0s)

    real_all = float(dI.mean()) if len(dI) else float("nan")
    real_obs = float(dI[obs].mean()) if obs.any() else float("nan")
    real_mis = float(dI[~obs].mean()) if (~obs).any() else 0.0

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    null_mean, dhat, pvals, dv, nce = {}, {}, {}, {}, {}
    for nm in nulls:
        if progress:
            print(f"  evaluating null[{nm}] with {n_perm} permutations ...", flush=True)
        ckpt_path = (os.path.join(checkpoint_dir, f"{checkpoint_tag}_{nm}.npy")
                    if checkpoint_dir else None)
        draws = np.full(n_perm, np.nan)
        lme = np.full(n_perm, np.nan)                  # per-draw log-mean-exp (DV)
        lse_tok = None                                 # per-token running logsumexp over draws (NCE)
        tok_path = (ckpt_path[:-4] + "_lsetok.npy") if ckpt_path else None
        start = 0
        cached = _load_checkpoint(ckpt_path) if ckpt_path else None
        cached_tok = _load_checkpoint(tok_path) if tok_path else None
        if cached is not None and cached.shape == (2, n_perm) and cached_tok is not None \
                and len(cached_tok) == int(obs.sum()):
            done = np.flatnonzero(~np.isnan(cached[0]))
            start = int(done[-1]) + 1 if len(done) else 0
            draws[:start] = cached[0, :start]
            lme[:start] = cached[1, :start]
            lse_tok = cached_tok if start else None
            if progress and start:
                print(f"    resuming null[{nm}] from checkpoint at {start}/{n_perm}",
                      flush=True)
        tick = max(1, n_perm // 10)
        for k in range(start, n_perm):
            draws[k], lme[k], pt_k = _forward_tokens(model, batches, log_p0s, null=nm, seed=seed + 1000 * k, return_mean_only=True)
            lse_tok = pt_k.copy() if lse_tok is None else np.logaddexp(lse_tok, pt_k)
            if ckpt_path and ((k + 1) % checkpoint_every == 0 or k + 1 == n_perm):
                _save_checkpoint(ckpt_path, np.stack([draws, lme]))
                _save_checkpoint(tok_path, lse_tok)
            if progress and ((k + 1) % tick == 0 or k + 1 == n_perm):
                print(f"    null[{nm}] {k + 1}/{n_perm}", flush=True)
        null_mean[nm] = float(np.nanmean(draws))
        dhat[nm] = real_obs - null_mean[nm]
        # DV: log E_{product}[e^T] over all (token, draw) pairs = logmeanexp of per-draw logmeanexps
        ok = ~np.isnan(lme)
        dv_null = float(np.logaddexp.reduce(lme[ok]) - np.log(ok.sum())) if ok.any() else float("nan")
        dv[nm] = real_obs - dv_null
        # per-token InfoNCE with the n_perm null draws as negatives (positive included)
        if lse_tok is not None and len(lse_tok) == int(obs.sum()):
            T_pos = dI[obs].astype(np.float64)
            denom = np.logaddexp(T_pos, lse_tok) - np.log(n_perm + 1)
            nce[nm] = float(np.mean(T_pos - denom))
        else:
            nce[nm] = float("nan")
        # (1 + #{null >= real}) / (1 + n): can never be exactly 0
        pvals[nm] = float((1 + np.sum(draws >= real_obs)) / (1 + n_perm))

    if progress:
        print(f"  clustered bootstrap with {n_boot} resamples ...", flush=True)
    boot_ckpt = (os.path.join(checkpoint_dir, f"{checkpoint_tag}_bootstrap.npy")
                if checkpoint_dir else None)
    ci = _clustered_bootstrap(dI[obs], texts[obs], n_boot=n_boot, seed=seed,
                              checkpoint_path=boot_ckpt,
                              checkpoint_every=checkpoint_every) if obs.any() else (np.nan, np.nan)

    return DIResult(
        dI_real=real_all, dI_real_observed=real_obs, dI_real_missing=real_mis,
        dI_null=null_mean, dI_hat=dhat, p_value=pvals, ci95=ci, dI_hat_dv=dv, dI_hat_nce=nce,
        ci95_dI_hat={nm: (ci[0] - null_mean[nm], ci[1] - null_mean[nm])
                     for nm in null_mean} if ci is not None else None,
        n_tokens=int(len(dI)), n_observed=int(obs.sum()),
        n_sentences=int(len(np.unique(sent_id))), n_clusters=int(len(np.unique(texts))),
        frac_positive=float((dI > 0).mean()) if len(dI) else float("nan"),
        meta=meta or {},
    )


@torch.no_grad()
def zero_gamma_check(model, batches, log_p0s=None) -> float:
    dev = next(model.parameters()).device
    if log_p0s is None or any(lp is None for lp in log_p0s):
        log_p0s = [model.prior.log_probs(b["token_ids"].to(dev)).cpu() for b in batches]
    worst = 0.0
    for batch, lp0 in zip(batches, log_p0s):
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        lp0_dev = lp0.to(dev) if torch.is_tensor(lp0) else lp0
        out = model(b, lp0_dev, gamma_override="zero")
        v = b["valid"]
        worst = max(worst, float(out["per_token"][v].abs().max()) if v.any() else 0.0)
    return worst
