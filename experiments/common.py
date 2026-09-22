
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cprd.audit import Ledger, Record, code_git_rev, sha256_of          # noqa: E402
from cprd.data import PRDDataset, collate, dataset_fingerprint, read_split  # noqa: E402
from cprd.model import ModelConfig, PriorResidualModel                   # noqa: E402
from cprd.prereg import (check_gate_artifact, collect_green_gates,        # noqa: E402
                         write_gate_artifact)
from cprd.train import TrainConfig, precompute_log_p0, train             # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--data-dir", default="runs/data", help="dir with zuco2_{split}.h5")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prior", default="causal_lm", choices=["causal_lm", "tiny"])
    ap.add_argument("--prior-model", default="gpt2-large")
    ap.add_argument("--evidence", default="eeg", choices=["eeg", "gaze", "both"])
    ap.add_argument("--gamma-warmup-value", type=float, default=1.0,
                    help="constant gamma during the warmup epochs (see TrainConfig)")
    ap.add_argument("--gamma-warmup-epochs", type=int, default=3,
                    help="epochs with the gain gate held constant before it is learned (0 = off)")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--no-center-u", dest="center_u", action="store_false",
                    help="disable evidence centring (reproduces the raw-objective collapse; ablation only)")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--eval-gamma", type=float, default=None,
                    help="evaluate at this constant gain (default: the checkpoint's own gate = softplus(c) ~ 0.127)")
    ap.add_argument("--extra-eval-gamma", type=float, default=None,
                    help="E1 only: ALSO evaluate each trained synthetic model at this constant gain (side file e1_extra_eval.jsonl)")
    ap.add_argument("--spike-gaze", type=float, default=0.0,
                    help="EEG power analysis: add alpha x (random 6->840 projection of standardised gaze) x per-feature std to the EEG features (observed slots only)")
    ap.add_argument("--s-max", type=float, default=4.0, help="upper bound on ||u_t|| (tilt-norm budget)")
    ap.add_argument("--fusion", default="concat", choices=["concat", "gated"], help="combined-channel fusion")
    ap.add_argument("--decorr-weight", type=float, default=0.0, help="EEG/gaze decorrelation regulariser weight (combined channel)")
    ap.add_argument("--gaze-control", default="none", choices=["none", "structure_only"],
                    help="structure_only: replace each word's gaze vector by a random other word's, keeping sub-token replication")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap sentences (smoke)")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-contrast", action="store_true",
                    help="EXPLORATORY: add softplus(-(dI_real - dI_null)) to the loss")
    ap.add_argument("--contrast-weight", type=float, default=1.0)
    ap.add_argument("--raw-weight", type=float, default=1.0,
                    help="weight on raw-dI ascent (only meaningful with --train-contrast)")
    ap.add_argument("--contrast-null", default="sentence_derangement")
    ap.add_argument("--objective", default="raw", choices=["raw", "nce"],
                    help="raw = pre-registered dI ascent; nce = InfoNCE over per-token dI (2026-09-17 instrument)")
    ap.add_argument("--n-null-train", type=int, default=4, help="negatives per token for --objective nce")
    ap.add_argument("--free-tilt-rank", type=int, default=0,
                    help="rank of the learned free token table added to the prior-embedding tilt (0 = off)")
    ap.add_argument("--free-tilt-hidden", type=int, default=64, help="hidden width of the nonlinear tilt MLP")
    ap.add_argument("--estimand", default="mean_null", choices=["mean_null", "nce"],
                    help="which statistic is the reported dI_hat (gates/ledger); the other is recorded alongside")
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                    help="allow non-deterministic CUDA kernels (faster, not reproducible)")
    ap.set_defaults(deterministic=True)
    return ap


def apply_determinism(args) -> None:
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass
    if not getattr(args, "deterministic", True):
        print("[determinism] DISABLED by --no-deterministic", flush=True)
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cuda = getattr(torch.backends, "cuda", None)
    if cuda is not None:
        for fn, val in (("enable_flash_sdp", False), ("enable_mem_efficient_sdp", False),
                        ("enable_math_sdp", True)):
            if hasattr(cuda, fn):
                getattr(cuda, fn)(val)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        print("[determinism] strict (math SDPA only; CUBLAS_WORKSPACE_CONFIG=:4096:8)", flush=True)
    except Exception as e:                       # an op with no deterministic impl
        torch.use_deterministic_algorithms(True, warn_only=True)
        print(f"[determinism] warn_only fallback: {e}", flush=True)


def train_config_from_args(args, seed=None, **overrides) -> TrainConfig:
    kw = dict(epochs=args.epochs, batch_size=args.batch_size, device=args.device,
              seed=args.seed if seed is None else seed, lr=args.lr,
              train_null_contrast=bool(getattr(args, "train_contrast", False)),
              contrast_weight=float(getattr(args, "contrast_weight", 1.0)),
              raw_weight=float(getattr(args, "raw_weight", 1.0)),
              contrast_null=str(getattr(args, "contrast_null", "sentence_derangement")),
              gamma_warmup_epochs=int(getattr(args, "gamma_warmup_epochs", 0)),
              gamma_warmup_value=float(getattr(args, "gamma_warmup_value", 1.0)),
              objective=str(getattr(args, "objective", "raw")),
              n_null_train=int(getattr(args, "n_null_train", 4)),
              decorr_weight=float(getattr(args, "decorr_weight", 0.0)))
    kw.update(overrides)
    return TrainConfig(**kw)


def objective_label(args) -> str:
    if getattr(args, "objective", "raw") == "nce":
        return "infonce"
    return "hybrid_contrast" if getattr(args, "train_contrast", False) else "raw_dI_ascent"


def build_prior(args):
    if args.prior == "tiny":
        from cprd.tiny import TinyCausalLM
        return TinyCausalLM()
    from cprd.priors import CausalLMPrior
    return CausalLMPrior(args.prior_model)


def prior_gate_or_die(prior, sentences, device) -> dict:
    probe = [torch.from_numpy(s.token_ids) for s in sentences[:40]]
    rep = prior.sanity_report(probe, device=torch.device(device))
    print(rep.describe(), flush=True)
    if not rep.ok:
        raise SystemExit("prior failed the acceptance gate; refusing to continue "
                         "(dI <= -log p0 makes every number a prior artifact)")
    return {"mean_surprisal": rep.mean_surprisal, "causal": rep.causal}


def load_split_sentences(args, split: str):
    path = os.path.abspath(os.path.join(args.data_dir, f"zuco2_{split}.h5"))
    sents = read_split(path, limit=args.limit or None,
                       evidence=getattr(args, "evidence", "eeg"))
    if getattr(args, "spike_gaze", 0.0) > 0 and getattr(args, "evidence", "eeg") == "eeg":
        sents = apply_spike(sents, args, path)
    if getattr(args, "gaze_control", "none") == "structure_only":
        sents = apply_structure_only(sents, seed=args.seed + {"train": 1, "val": 2, "test": 3}.get(split, 4))
    return sents, dataset_fingerprint(path)


_SPIKE = {}


def apply_spike(sents, args, path):
    if "stats" not in _SPIKE:
        trp = os.path.join(os.path.dirname(path), "zuco2_train.h5")
        tr_e = read_split(trp, limit=args.limit or None, evidence="eeg")
        tr_b = read_split(trp, limit=args.limit or None, evidence="both")
        E = np.concatenate([s.eeg[s.observed] for s in tr_e], 0)
        Gz = np.concatenate([b.eeg[:, 840:] for b in tr_b], 0)
        rng = np.random.default_rng(123)
        _SPIKE["stats"] = (np.nanstd(E, 0) + 1e-8, Gz.mean(0), Gz.std(0) + 1e-6,
                           rng.standard_normal((Gz.shape[1], E.shape[1])).astype(np.float32) / np.sqrt(Gz.shape[1]))
    sd, gm, gs, G = _SPIKE["stats"]
    both = read_split(path, limit=args.limit or None, evidence="both")
    import copy
    out = []
    for s, b in zip(sents, both):
        z = (b.eeg[:, 840:] - gm) / gs
        s2 = copy.copy(s)
        s2.eeg = (s.eeg + args.spike_gaze * sd * (z @ G)).astype(np.float32)
        out.append(s2)
    return out


def apply_structure_only(sents, seed: int):
    import copy
    rng = np.random.default_rng(seed)
    pool = []
    for s in sents:
        wi = s.word_index if s.word_index is not None else np.arange(len(s.token_ids))
        first = np.r_[True, wi[1:] != wi[:-1]]
        pool.append(s.eeg[first])
    pool = np.concatenate(pool, 0)
    out = []
    for s in sents:
        wi = s.word_index if s.word_index is not None else np.arange(len(s.token_ids))
        first = np.r_[True, wi[1:] != wi[:-1]]
        gid = np.cumsum(first) - 1
        rep = pool[rng.integers(0, len(pool), size=int(gid.max()) + 1)]
        s2 = copy.copy(s)
        s2.eeg = rep[gid].astype(np.float32)
        out.append(s2)
    return out


def model_config_from_args(args) -> ModelConfig:
    ev = getattr(args, "evidence", "eeg")
    from cprd.build import GAZE_DIM
    dim = {"eeg": 840, "gaze": GAZE_DIM, "both": 840 + GAZE_DIM}[ev]
    return ModelConfig(r=args.r, window=args.window, evidence=ev, evidence_dim=dim,
                       center_u=getattr(args, "center_u", True),
                       free_tilt_rank=int(getattr(args, "free_tilt_rank", 0)),
                       free_tilt_hidden=int(getattr(args, "free_tilt_hidden", 64)),
                       s_max=float(getattr(args, "s_max", 4.0)),
                       fusion=str(getattr(args, "fusion", "concat")))


def prepare_batches(sentences, prior, device, window=1, batch_size=8):
    from torch.utils.data import DataLoader
    loader = DataLoader(PRDDataset(sentences, window=window),
                        batch_size=batch_size, shuffle=False, collate_fn=collate)
    return precompute_log_p0(prior, loader, torch.device(device))


def train_model(prior, train_sents, val_sents, args) -> PriorResidualModel:
    from cprd.audit import sha256_of
    from cprd.data import selection_measurement_split
    mcfg = model_config_from_args(args)
    tcfg = train_config_from_args(args)
    fp_tr = dataset_fingerprint(os.path.abspath(os.path.join(args.data_dir, "zuco2_train.h5")))
    fp_va = dataset_fingerprint(os.path.abspath(os.path.join(args.data_dir, "zuco2_val.h5")))
    val_sel, _ = selection_measurement_split(val_sents)
    key = train_model_key(args)
    ckpt_dir = os.path.join(args.runs_dir, "ckpt")
    ckpt = os.path.join(ckpt_dir, f"phi_{key}.pt")
    model = PriorResidualModel(prior, mcfg)
    model.ckpt_key = key
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=torch.device(args.device))
        model.load_state_dict(state["phi"])
        model.to(torch.device(args.device))
        print(f"reusing trained checkpoint (same config+data+prior): {ckpt}", flush=True)
        return model
    train(model, train_sents, val_sel, mcfg, tcfg, ckpt_dir, state_tag=key)
    # persist under the full key so only an identical (config, data, prior) reuses it
    tmp = ckpt + ".tmp"
    torch.save({"phi": model.state_dict(), "key": key,
                "model_config": mcfg.to_dict(), "train_config": tcfg.to_dict(),
                "fingerprints": {"train": fp_tr, "val": fp_va},
                "prior": f"{args.prior}:{args.prior_model}"}, tmp)
    os.replace(tmp, ckpt)
    return model


def train_model_key(args) -> str:
    """The checkpoint key train_model() would use for these args (for --resume-eval)."""
    from cprd.audit import sha256_of
    fp_tr = dataset_fingerprint(os.path.abspath(os.path.join(args.data_dir, "zuco2_train.h5")))
    fp_va = dataset_fingerprint(os.path.abspath(os.path.join(args.data_dir, "zuco2_val.h5")))
    return sha256_of([model_config_from_args(args).to_dict(), train_config_from_args(args).to_dict(),
                      fp_tr, fp_va, f"{args.prior}:{args.prior_model}", "val_selection_half:v1",
                      f"limit={getattr(args, 'limit', 0)}"])


def read_gate_optional(runs_dir: str, name: str) -> dict:
    """Read a gate artifact WITHOUT requiring it to have passed ({} if missing)."""
    import json
    p = os.path.join(runs_dir, f"gate_{name}.json")
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        return json.load(fh)


def ledger_for(args) -> Ledger:
    return Ledger(os.path.join(args.runs_dir, "ledger.jsonl"))


def base_record(args, split_fp: str, experiment: str, gates: list) -> dict:
    actual = collect_green_gates(args.runs_dir)
    recorded = [a for a in actual
                if any(a == g or a.startswith(f"{g}_") for g in gates)]
    # non-gate provenance markers the driver asserts itself (e.g. prior_sanity)
    recorded += [g for g in gates if g == "prior_sanity"]
    return dict(split_fingerprint=split_fp,
                config_sha=sha256_of(vars(args)),
                code_rev=code_git_rev(),
                prior_id=f"{args.prior}:{args.prior_model}",
                seed=args.seed, n_perm=args.n_perm, n_boot=args.n_boot,
                gates_passed=recorded, experiment=experiment,
                objective=objective_label(args),
                evidence=getattr(args, "evidence", "eeg"))
