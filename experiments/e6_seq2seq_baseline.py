
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import base_parser, base_record, ledger_for, load_split_sentences  # noqa: E402
from cprd.audit import Record                                                  # noqa: E402
from cprd.decode import corpus_bleu                                            # noqa: E402
from cprd.prereg import check_gate_artifact, write_gate_artifact               # noqa: E402
from cprd.wangji_official import BrainTranslator, apply_step1_freeze           # noqa: E402

FEAT = 840


class _SmokeTok:
    pad_token_id, eos_token_id = 1, 2

    def __call__(self, t, **kw):
        return {"input_ids": [3 + (hash(w) % 120) for w in t.split()][:16] + [2]}

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(f"w{int(i)}" for i in row if int(i) > 2) for row in ids]


class WangJiOfficial(nn.Module):

    def __init__(self, backbone, d_dec: int):
        super().__init__()
        self.inner = BrainTranslator(backbone, in_feature=FEAT,
                                     decoder_embedding_size=d_dec,
                                     additional_encoder_nhead=8,
                                     additional_encoder_dim_feedforward=2048)

    def forward(self, eeg, obs, labels):
        x = torch.nan_to_num(eeg, nan=0.0)
        return self.inner(x, obs.long(), ~obs, labels)

    @torch.no_grad()
    def generate(self, eeg, obs, **kw):
        enc = self.inner.addin_forward(torch.nan_to_num(eeg, nan=0.0), ~obs)
        return self.inner.pretrained.generate(inputs_embeds=enc,
                                              attention_mask=obs.long(), **kw)


class Seq2SeqBaseline(nn.Module):
    def __init__(self, backbone, d_dec: int, faithful: bool = False):
        super().__init__()
        n_layers = 6 if faithful else 2
        self.frontend = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=FEAT, nhead=8, dim_feedforward=2048,
                                       batch_first=True, dropout=0.1),
            num_layers=n_layers)
        self.proj = nn.Linear(FEAT, d_dec)
        self.backbone = backbone

    def _embed(self, eeg, obs):
        x = torch.nan_to_num(eeg, nan=0.0) * obs.unsqueeze(-1)
        h = self.frontend(x, src_key_padding_mask=~obs)
        return torch.relu(self.proj(h))

    def forward(self, eeg, obs, labels):
        return self.backbone(inputs_embeds=self._embed(eeg, obs),
                             attention_mask=obs.long(), labels=labels)

    @torch.no_grad()
    def generate(self, eeg, obs, **kw):
        return self.backbone.generate(inputs_embeds=self._embed(eeg, obs),
                                      attention_mask=obs.long(), **kw)


def word_level(sentences, tok, max_words=56, max_tgt=56):
    """Fixated-word EEG sequences (W&J convention) + tokenized target text."""
    rows = []
    for s in sentences:
        feats = s.eeg[s.observed]
        if feats.shape[0] == 0:
            continue
        ids = tok(s.text, truncation=True, max_length=max_tgt)["input_ids"]
        rows.append((feats[:max_words].astype(np.float32), np.asarray(ids), s.text))
    return rows


def batches(rows, bs, device, pad_id):
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        S = max(r[0].shape[0] for r in chunk)
        L = max(len(r[1]) for r in chunk)
        eeg = torch.zeros(len(chunk), S, FEAT)
        obs = torch.zeros(len(chunk), S, dtype=torch.bool)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        for j, (f, ids, _) in enumerate(chunk):
            eeg[j, :f.shape[0]] = torch.from_numpy(f)
            obs[j, :f.shape[0]] = True
            lab[j, :len(ids)] = torch.from_numpy(ids)
        yield (eeg.to(device), obs.to(device), lab.to(device),
               [r[2] for r in chunk])


def dev_ce(model, rows, args, device, tok) -> float:
    was_training = model.training
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for eeg, obs, lab, _ in batches(rows, args.batch_size, device, tok.pad_token_id):
            tot += float(model(eeg, obs, lab).loss) * eeg.shape[0]
            n += eeg.shape[0]
    if was_training:
        model.train()
    return tot / max(1, n)


STAGE_ORDER = ["step1", "step2", "single"]


def train_stage(model, tr_rows, va_rows, opt, sched, n_epochs, stage, state_path,
                state, args, device, tok, clip: bool):
    start = 0
    best_loss = float("inf")
    best_state = None
    if state is not None and state.get("stage") == stage:
        start = int(state["epoch"]) + 1
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        if sched is not None and state.get("sched") is not None:
            sched.load_state_dict(state["sched"])
        best_loss = float(state.get("best_loss", float("inf")))
        best_state = state.get("best_state")
        print(f"  [resume] {stage}: continuing at epoch {start} "
              f"(best dev CE so far {best_loss:.4f})", flush=True)
    model.train()
    for ep in range(start, n_epochs):
        tot, n = 0.0, 0
        for eeg, obs, lab, _ in batches(tr_rows, args.batch_size, device, tok.pad_token_id):
            out = model(eeg, obs, lab)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(out.loss)
            n += 1
        if sched is not None:
            sched.step()
        d = dev_ce(model, va_rows, args, device, tok)
        if d < best_loss:
            best_loss = d
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{stage}] epoch {ep}: train CE {tot / max(1, n):.4f}  dev CE {d:.4f}"
              + ("  *best*" if d == best_loss else ""), flush=True)
        torch.save({"stage": stage, "epoch": ep, "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched is not None else None,
                    "best_loss": best_loss, "best_state": best_state}, state_path)
    return best_loss, best_state


def main():
    ap = base_parser("seq2seq comparability baselines (three-arm, variant-tagged)")
    ap.add_argument("--backbone", default="facebook/bart-large")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--their-code", action="store_true",
                    help="use the base paper's VERBATIM BrainTranslator class "
                         "(cprd/wangji_official.py) — the 1:1 architecture row")
    ap.add_argument("--faithful", action="store_true",
                    help="our reimplementation of their 6-layer front end "
                         "(cross-check of the port)")
    ap.add_argument("--their-recipe", action="store_true",
                    help="their verbatim two-step recipe: step1 SGD 5e-5 (BART mostly "
                         "frozen), step2 SGD 5e-7 (all unfrozen), best-on-dev")
    ap.add_argument("--epochs-step1", type=int, default=20,
                    help="their default: 20 (smoke: 1)")
    ap.add_argument("--epochs-step2", type=int, default=30,
                    help="their default: 30 (smoke: 1)")
    ap.add_argument("--lr-step1", type=float, default=5e-5)
    ap.add_argument("--lr-step2", type=float, default=5e-7)
    args = ap.parse_args()
    if args.their_code and args.faithful:
        raise SystemExit("--their-code and --faithful are mutually exclusive")
    check_gate_artifact(args.runs_dir, "build")
    device = torch.device(args.device)

    from transformers import AutoTokenizer, BartConfig, BartForConditionalGeneration
    if args.smoke:
        cfg = BartConfig(vocab_size=128, d_model=32, encoder_layers=1, decoder_layers=1,
                         encoder_attention_heads=2, decoder_attention_heads=2,
                         encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=128)
        backbone = BartForConditionalGeneration(cfg)
        tok = _SmokeTok()
        d_dec = 32
    else:
        # transformers>=5 ships bart-large fp16; the EEG encoder is fp32 and the
        # mix crashes in layernorm_embedding. Force fp32.
        backbone = BartForConditionalGeneration.from_pretrained(args.backbone).float()
        tok = AutoTokenizer.from_pretrained(args.backbone)
        d_dec = backbone.config.d_model

    if args.their_code:
        model = WangJiOfficial(backbone, d_dec).to(device)
        arch_tag = "wangji_official"
    else:
        model = Seq2SeqBaseline(backbone, d_dec, faithful=args.faithful).to(device)
        arch_tag = "wangji_faithful" if args.faithful else "modernized"
    variant = arch_tag + ("_their_recipe" if args.their_recipe else "_adamw")
    print(f"baseline variant: {variant}", flush=True)

    tr, _ = load_split_sentences(args, "train")
    va, fp = load_split_sentences(args, "val")
    tr_rows, va_rows = word_level(tr, tok), word_level(va, tok)

    # ---- training (per-epoch resumable state, MANDATORY for every case) ----
    ck_dir = os.path.join(args.runs_dir, "ckpt_e6")
    os.makedirs(ck_dir, exist_ok=True)
    state_path = os.path.join(ck_dir, f"state_{variant}.pt")
    state = None
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device, weights_only=False)
        print(f"  found training state ({state['stage']} epoch {state['epoch']}) — resuming",
              flush=True)

    if args.their_recipe:
        # their verbatim two-step schedule (train_decoding.py); no grad clipping
        done_step1 = state is not None and \
            STAGE_ORDER.index(state["stage"]) > STAGE_ORDER.index("step1")
        if not done_step1:
            trainable = apply_step1_freeze(model)
            print(f"  step1: {len(trainable)} trainable param tensors "
                  f"(BART mostly frozen)", flush=True)
            opt1 = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                                   lr=args.lr_step1, momentum=0.9)
            sched1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=20, gamma=0.1)
            train_stage(model, tr_rows, va_rows, opt1, sched1, args.epochs_step1,
                        "step1", state_path, state, args, device, tok, clip=False)
            state = None  # step2 starts fresh from the step1-trained weights
        for p in model.parameters():
            p.requires_grad = True
        opt2 = torch.optim.SGD(model.parameters(), lr=args.lr_step2, momentum=0.9)
        sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=30, gamma=0.1)
        best_loss, best_state = train_stage(model, tr_rows, va_rows, opt2, sched2,
                                            args.epochs_step2, "step2", state_path,
                                            state, args, device, tok, clip=False)
        if best_state is not None:
            model.load_state_dict(best_state)   # their code evaluates best-on-dev
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        train_stage(model, tr_rows, va_rows, opt, None, args.epochs, "single",
                    state_path, state, args, device, tok, clip=True)

    # ---- three arms through the IDENTICAL generate path ----
    model.eval()
    results, arm_texts = {}, {}
    for arm in ("real", "shuffled", "noise"):
        hyps, refs = [], []
        rng = np.random.default_rng(args.seed)
        rows = va_rows
        if arm == "shuffled":
            perm = rng.permutation(len(rows))
            rows = [(va_rows[int(perm[i])][0], va_rows[i][1], va_rows[i][2])
                    for i in range(len(va_rows))]
        for eeg, obs, lab, texts in batches(rows, args.batch_size, device, tok.pad_token_id):
            if arm == "noise":
                mu, sd = eeg[obs].mean(0), eeg[obs].std(0).clamp(min=1e-6)
                eeg = mu + sd * torch.randn_like(eeg)
            ids = model.generate(eeg, obs, max_length=32, num_beams=4, do_sample=False)
            hyps.extend(tok.batch_decode(ids, skip_special_tokens=True))
            refs.extend(texts)
        b = corpus_bleu(hyps, refs)
        results[arm] = b
        arm_texts[arm] = (hyps, refs)
        print(f"  arm={arm:<9} corpus BLEU = {b}", flush=True)

    # final checkpoint for e9 retrofit (and reuse)
    torch.save({"model": model.state_dict(), "smoke": bool(args.smoke),
                "arch": arch_tag, "faithful": bool(args.faithful), "d_dec": d_dec,
                "backbone_name": None if args.smoke else args.backbone,
                "backbone_config": (cfg.to_dict() if args.smoke else None)},
               os.path.join(ck_dir, f"seq2seq_{variant}.pt"))

    # paper metrics per arm (same hyps/refs the BLEU used)
    from cprd.text_metrics import compute_all
    paper = {arm: compute_all(h, r, light=(args.smoke or args.device == "cpu"),
                              device=args.device)
             for arm, (h, r) in arm_texts.items()}
    with open(os.path.join(args.runs_dir, f"e6_paper_metrics_{variant}.json"), "w") as fh:
        json.dump(paper, fh, indent=2)

    led = ledger_for(args)
    base = base_record(args, split_fp=fp, experiment="e6_seq2seq_baseline",
                       gates=["build"])
    for arm, b in results.items():
        if b is not None:
            led.append(Record(metric="bleu_corpus_free_running", value=b,
                              arm=f"{variant}/{arm}", **base))
        for k in ("bleu1", "chrf", "meteor", "bertscore_f1", "rouge1", "wer"):
            v = paper.get(arm, {}).get(k)
            if isinstance(v, (int, float)):
                led.append(Record(metric=f"generation_{k}", value=float(v),
                                  arm=f"{variant}/{arm}", **base))
    # merge variant results into a single gate artifact
    gate_path = os.path.join(args.runs_dir, "gate_seq2seq_baseline.json")
    merged = {}
    if os.path.exists(gate_path):
        with open(gate_path) as fh:
            merged = (json.load(fh).get("detail") or {})
    merged[variant] = {"bleu": results,
                       "checkpoint": os.path.join(ck_dir, f"seq2seq_{variant}.pt")}
    merged["bleu"] = merged.get(variant, {}).get("bleu", results)   # legacy key for fig2
    merged["checkpoint"] = merged[variant]["checkpoint"]
    write_gate_artifact(args.runs_dir, "seq2seq_baseline", True, merged)
    return 0


if __name__ == "__main__":
    sys.exit(main())
