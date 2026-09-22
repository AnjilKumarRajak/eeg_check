"""Training phi against the information-gain objective.

The prior is frozen and the token sequences are fixed, so `log p0` is computed ONCE per
split and reused for every epoch and every permutation. That is a correctness property
as much as a speed one: it guarantees train-time and eval-time use a bit-identical
reference measure.

Model selection is on held-out validation dI_hat (bias-corrected), never on raw dI and
never on train dI. Raw dI rewards exploiting the prior's weaknesses; train dI rewards
memorisation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import PRDDataset, collate
from .evaluate import evaluate
from .model import ModelConfig, PriorResidualModel
from .nulls import NULLS


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    patience: int = 5
    seed: int = 0
    n_perm_val: int = 40
    device: str = "cpu"
    train_null_contrast: bool = False
    contrast_null: str = "sentence_derangement"
    contrast_weight: float = 1.0
    raw_weight: float = 1.0        # weight on raw-dI ascent (0 => contrast-only training)
    objective: str = "raw"         # "raw" (pre-registered dI ascent) | "nce" (InfoNCE over
                                   # per-token dI with in-batch deranged evidence as negatives;
                                   # a constant tilt scores 0, so there is no collapse basin)
    n_null_train: int = 4          # negatives per token for objective="nce"
    gamma_warmup_value: float = 1.0  # the constant gamma during warmup. Small (~0.1) makes
                                   # the objective ~linear in the evidence signal (the
                                   # Jensen penalty is O(gamma^2)), so the encoder learns
                                   # the correlation before the gate opens (E1 collapse fix)
    gamma_warmup_epochs: int = 0   # hold the gain gate at gamma_mode='constant' for the
                                   # first K epochs (encoder/B learn ell before the gate
                                   # can collapse to 0); then restore the configured mode

    decorr_weight: float = 0.0     # weight on the EEG/gaze decorrelation regulariser (combined channel)

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d.get("decorr_weight"):
            d.pop("decorr_weight", None)   # default 0: keeps existing checkpoint keys unchanged
        return d


def config_hash(*objs) -> str:
    blob = json.dumps([o if isinstance(o, dict) else o.to_dict() for o in objs], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@torch.no_grad()
def precompute_log_p0(prior, loader, device) -> tuple[list, Optional[list]]:
    """Prepare batches. Prior log-probs are computed on-demand to keep memory footprint minimal."""
    prior.assert_frozen()
    batches = []
    for b in loader:
        b_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        batches.append(b_gpu)
    return batches, None


def make_scheduler(opt, total_steps: int, warmup_frac: float):
    warm = max(1, int(total_steps * warmup_frac))

    def fn(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        prog = min(1.0, prog)          # clamp: an overshoot would make cosine rise again
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train(model: PriorResidualModel, train_sents, val_sents,
          mcfg: ModelConfig, tcfg: TrainConfig, out_dir: str,
          state_tag: str | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(tcfg.seed)
    device = torch.device(tcfg.device)
    model.to(device)

    tr_loader = DataLoader(PRDDataset(train_sents, window=mcfg.window),
                           batch_size=tcfg.batch_size, shuffle=False, collate_fn=collate)
    va_loader = DataLoader(PRDDataset(val_sents, window=mcfg.window),
                           batch_size=tcfg.batch_size, shuffle=False, collate_fn=collate)

    tr_batches, tr_lp0 = precompute_log_p0(model.prior, tr_loader, device)
    va_batches, va_lp0 = precompute_log_p0(model.prior, va_loader, device)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay)
    # ceil, not floor: a floor'd total makes the cosine overshoot and turn back upward
    steps_per_epoch = math.ceil(len(tr_batches))
    sched = make_scheduler(opt, tcfg.epochs * steps_per_epoch, tcfg.warmup_frac)

    chash = config_hash(mcfg, tcfg)
    state_tag = state_tag or chash
    state_path = os.path.join(out_dir, f"train_state_{state_tag}.pt")
    start_epoch = 0
    history, best, best_state, patience_ctr = [], -float("inf"), None, 0
    if os.path.exists(state_path):
        try:
            st = torch.load(state_path, map_location=device)
            model.load_state_dict(st["phi"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            start_epoch = int(st["epoch"]) + 1
            history = st.get("history", [])
            best = st.get("best", best)
            best_state = st.get("best_state")
            patience_ctr = int(st.get("patience_ctr", 0))
            print(f"resuming training from epoch {start_epoch} ({state_path})", flush=True)
        except Exception as e:
            print(f"train-state resume failed ({e}); starting fresh", flush=True)

    order = list(range(len(tr_batches)))
    gamma_mode_cfg = model.tilt.gamma_mode

    # Model selection is only meaningful in the gate mode the model will be USED in:
    # during warmup gamma is held constant and the a/b/c gate parameters receive no
    # gradient, so a warmup-epoch state evaluated at gamma=const is a different model
    # from the same weights under the learned gate. Warmup epochs are therefore not
    # eligible as "best" and do not count toward patience (unless the whole run is
    # warmup, e.g. a tiny smoke run).
    warm = tcfg.gamma_warmup_epochs if tcfg.gamma_warmup_epochs < tcfg.epochs else 0

    _out: dict = {}
    try:
        _train_epochs(model, tcfg, start_epoch, order, tr_batches, tr_lp0, va_batches, va_lp0,
                      opt, sched, history, state_path, gamma_mode_cfg, warm,
                      best, best_state, patience_ctr, _out)
    finally:
        model.tilt.gamma_mode = gamma_mode_cfg      # never leave the warmup mode behind
        model.tilt.gamma_const = 1.0
    best, best_state = _out.get("best", best), _out.get("best_state", best_state)

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = os.path.join(out_dir, f"phi_{chash}.pt")
    torch.save({"phi": model.state_dict(), "model_config": mcfg.to_dict(),
                "train_config": tcfg.to_dict(), "config_hash": chash}, ckpt)
    with open(os.path.join(out_dir, f"config_{chash}.json"), "w") as fh:
        json.dump({"model": mcfg.to_dict(), "train": tcfg.to_dict(),
                   "config_hash": chash, "history": history,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=2)

    return {"history": history, "best_val_dI_hat": best,
            "checkpoint": ckpt, "config_hash": chash}


def _train_epochs(model, tcfg, start_epoch, order, tr_batches, tr_lp0, va_batches, va_lp0,
                  opt, sched, history, state_path, gamma_mode_cfg, warm,
                  best, best_state, patience_ctr, out):
    out["best"], out["best_state"] = best, best_state
    # resume continues from the saved epoch (the scheduler state was restored with it)
    for epoch in range(start_epoch, tcfg.epochs):
        model.train()
        # gain-gate warmup: constant gate while ell is learned, then the configured mode
        if tcfg.gamma_warmup_epochs > 0:
            want = "constant" if epoch < tcfg.gamma_warmup_epochs else gamma_mode_cfg
            if model.tilt.gamma_mode != want:
                model.tilt.gamma_mode = want
                model.tilt.gamma_const = tcfg.gamma_warmup_value if want == "constant" else 1.0
                print(f"  [gamma warmup] epoch {epoch}: gamma_mode -> {want}", flush=True)
        torch.manual_seed(tcfg.seed + epoch)
        g = torch.Generator().manual_seed(tcfg.seed + epoch)
        perm = torch.randperm(len(order), generator=g).tolist()

        losses, gammas, unorms, neg_fracs = [], [], [], []
        for i in perm:
            batch = tr_batches[i]
            lp0 = tr_lp0[i].to(device) if tr_lp0 is not None else model.prior.log_probs(batch["token_ids"])
            out = model(batch, lp0)
            model.check_upper_bound(out, batch["valid"])
            aux_real = model.encoder.aux_loss() if hasattr(model.encoder, "aux_loss") else None
            objective = out["per_sentence_mean"].mean()
            if tcfg.objective == "nce":
                # InfoNCE: per token, the real evidence must out-score M deranged
                # evidence vectors. T = dI (log-likelihood ratio vs the prior); the
                # positive is included in the denominator, so the bound is <= log(M+1).
                v = batch["valid"] & batch["observed"]
                T_pos = out["per_token"]
                negs = []
                for m in range(tcfg.n_null_train):
                    w, o = NULLS["sentence_derangement"](batch["window"], batch["win_observed"],
                                                          batch["valid"],
                                                          seed=tcfg.seed + 7919 * (epoch * max(1, len(order)) + i) + m)
                    u_n = model.evidence(w, o, batch["win_pad"], batch["observed"], batch["valid"])
                    negs.append(model.gain_terms(batch, lp0, u_n)["per_token"])
                allT = torch.stack([T_pos] + negs, dim=0)                      # (M+1,B,T)
                nce = T_pos - torch.logsumexp(allT, dim=0) + math.log(allT.shape[0])
                loss = -(nce[v].mean() if v.any() else nce.mean())
            else:
                loss = -tcfg.raw_weight * objective       # ascent on raw dI (weighted)
            if getattr(tcfg, "decorr_weight", 0.0) > 0 and aux_real is not None:
                loss = loss + tcfg.decorr_weight * aux_real.to(loss.device)
            if tcfg.train_null_contrast:
                # A LINEAR contrast (real - null) is degenerate no matter which null it's
                # taken against: any permutation-based null disrupts local temporal
                # continuity in a way that's much easier for phi to key on than the
                # genuine, often-subtle per-token EEG-token correspondence, and a linear
                # objective has no saturation -- there is always more gradient reward for
                # widening the gap further, so it runs away (confirmed empirically with
                # sentence_derangement AND temporal_roll alike, and a small linear weight
                # 0.05 just reproduces the zero-contrast local optimum instead). Fix:
                # a logistic/margin (softplus) contrast, the standard NCE-style choice.
                # Its gradient saturates once real already beats null by a comfortable
                # margin, removing the incentive to keep crashing the null indefinitely,
                # while still supplying a non-vanishing gradient at margin==0 -- unlike
                # pure real-ascent, which the model satisfies for free via an
                # evidence-independent marginal shortcut that cancels exactly under the
                # null correction (also confirmed empirically: recovered dI_hat pinned at
                # 0.000 regardless of the true injected bits).
                w, o = NULLS[tcfg.contrast_null](
                    batch["window"], batch["win_observed"], batch["valid"],
                    seed=tcfg.seed + epoch * max(1, len(order)) + i,
                )
                null_batch = dict(batch)
                null_batch["window"] = w
                null_batch["win_observed"] = o
                null_out = model(null_batch, lp0)
                model.check_upper_bound(null_out, batch["valid"])
                margin = out["per_sentence_mean"] - null_out["per_sentence_mean"]
                loss = loss + tcfg.contrast_weight * nn.functional.softplus(-margin).mean()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch}. Not silently skipped: a diverging "
                    "dI is the degenerate-optimum signature and must fail loudly."
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            model.assert_prior_frozen()
            nn.utils.clip_grad_norm_(model.trainable_parameters(), tcfg.grad_clip)
            opt.step()
            sched.step()
            model.tilt.project_spectral()

            losses.append(float(objective))     # raw dI, not the (possibly contrastive) loss
            v = batch["valid"]
            gammas.append(float(out["gamma"][v].mean()))
            unorms.append(float(out["u_norm"][v].mean()))
            neg_fracs.append(float((out["per_token"][v] <= 0).float().mean()))

        # Both nulls, and checkpoint selection requires the MIN of the two: picking the
        # epoch that peaks against a single null (30 noisy looks at held-out data) is
        # itself a selection-bias route to an inflated number, and a gap that survives
        # only one null is exactly the length/amplitude-artifact case nulls.py warns
        # about. Requiring both to agree makes that shortcut much harder to reach.
        res = evaluate(model, va_batches, va_lp0,
                       nulls=("sentence_derangement", "temporal_roll"), n_perm=tcfg.n_perm_val,
                       seed=tcfg.seed)
        if tcfg.objective == "nce":
            crit = min(res.dI_hat_nce["sentence_derangement"], res.dI_hat_nce["temporal_roll"])
        else:
            crit = min(res.dI_hat["sentence_derangement"], res.dI_hat["temporal_roll"])

        # Degeneracy signature: real dI drifting negative while the null-corrected gap
        # keeps climbing means phi is learning to crash the null arm, not to predict the
        # gold token better -- the gap is real but not evidence of recovered information.
        if epoch > 5 and res.dI_real_observed < -0.5 and crit > 0 and \
                history and res.dI_real_observed < history[-1]["val_dI_observed"] - 1e-6:
            print(f"  [degeneracy warning] val dI(obs)={res.dI_real_observed:+.4f} still "
                  f"falling while val dI_hat={crit:+.4f} rises -- likely null-contrast "
                  "exploiting the null's construction rather than real evidence", flush=True)

        history.append({
            "epoch": epoch,
            "train_dI": sum(losses) / len(losses),
            "train_neg_frac": sum(neg_fracs) / len(neg_fracs),
            "val_dI_observed": res.dI_real_observed,
            "val_dI_hat": crit,
            "gamma_mean": sum(gammas) / len(gammas),
            "u_norm_mean": sum(unorms) / len(unorms),
        })
        print(f"epoch {epoch:3d}  train dI={history[-1]['train_dI']:+.4f}  "
              f"val dI(obs)={res.dI_real_observed:+.4f}  val dI_hat={crit:+.4f}  "
              f"gamma={history[-1]['gamma_mean']:.4f}  "
              f"neg_frac={history[-1]['train_neg_frac']:.3f}", flush=True)

        if epoch < warm:
            pass                                 # warmup: not eligible, no patience
        elif crit > best:
            best, patience_ctr = crit, 0
            # phi only: the prior is not a submodule, so it cannot land in the checkpoint
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        out["best"], out["best_state"] = best, best_state

        # resumable training state: an interruption loses at most one epoch
        torch.save({"epoch": epoch, "phi": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "best": best, "best_state": best_state,
                    "patience_ctr": patience_ctr, "history": history}, state_path)

        if patience_ctr >= tcfg.patience:
            print(f"early stopping at epoch {epoch} (best val dI_hat={best:+.4f})")
            break
