"""Error analysis for the selection system: WHAT fails, not just how often.

Consumes e4_selected_texts.csv + the pools' scores (written by e4) and produces:

  e4_error_analysis.json    per-bucket accuracies and the failure taxonomy
  fig9_error_analysis       four-panel figure: correctness vs sentence length,
                            vs prior log-prob, margin distributions (correct vs wrong),
                            and near-miss semantic similarity of the errors

Failure taxonomy (per wrong selection):
  near_miss     chosen candidate shares >= 40% content words with the reference
  same_topic    chosen candidate has the same task/corpus label as the reference
  off_target    neither — the informative failure class
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

STOP = set("the a an and or of to in is was were are be been on at for with by "
           "that this it as his her he she they i you we".split())


def content_words(text: str) -> set:
    return {w.lower().strip(".,;:!?'\"()") for w in text.split()} - STOP - {""}


def overlap(a: str, b: str) -> float:
    ca, cb = content_words(a), content_words(b)
    if not ca:
        return 0.0
    return len(ca & cb) / len(ca)


def analyze(runs_dir: str, task_of: dict | None = None) -> dict:
    """Build the error-analysis JSON from the artifacts e4 writes."""
    csv_path = os.path.join(runs_dir, "e4_selected_texts.csv")
    if not os.path.exists(csv_path):
        return {"_error": f"{csv_path} missing (run e4 first)"}
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return {"_error": "no rows"}

    margins = None
    rc_path = os.path.join(runs_dir, "e4_margins.json")
    if os.path.exists(rc_path):
        margins = json.load(open(rc_path))

    recs = []
    for i, r in enumerate(rows):
        ref, hyp = r["reference"], r["hypothesis_real"]
        correct = bool(int(r["correct"]))
        rec = {"len_words": len(ref.split()), "correct": correct,
               "overlap": overlap(ref, hyp) if not correct else 1.0}
        if not correct:
            if rec["overlap"] >= 0.4:
                rec["failure_class"] = "near_miss"
            elif task_of and task_of.get(ref) == task_of.get(hyp):
                rec["failure_class"] = "same_topic"
            else:
                rec["failure_class"] = "off_target"
        if margins and i < len(margins.get("margin", [])):
            rec["margin"] = margins["margin"][i]
            rec["plogp"] = margins.get("plogp", [None] * len(rows))[i]
        recs.append(rec)

    # length-quartile accuracy
    lens = np.array([r["len_words"] for r in recs])
    corr = np.array([r["correct"] for r in recs], dtype=float)
    qs = np.quantile(lens, [0.25, 0.5, 0.75])
    by_len = {}
    labels = ["Q1_short", "Q2", "Q3", "Q4_long"]
    bins = [lens <= qs[0], (lens > qs[0]) & (lens <= qs[1]),
            (lens > qs[1]) & (lens <= qs[2]), lens > qs[2]]
    for lab, m in zip(labels, bins):
        if m.any():
            by_len[lab] = {"n": int(m.sum()), "acc": float(corr[m].mean()),
                           "mean_len": float(lens[m].mean())}

    wrong = [r for r in recs if not r["correct"]]
    taxonomy = {}
    for r in wrong:
        taxonomy[r["failure_class"]] = taxonomy.get(r["failure_class"], 0) + 1
    near_ov = [r["overlap"] for r in wrong]

    out = {
        "n": len(recs), "accuracy": float(corr.mean()),
        "accuracy_by_length_quartile": by_len,
        "failure_taxonomy": taxonomy,
        "wrong_overlap_mean": float(np.mean(near_ov)) if near_ov else None,
        "wrong_overlap_median": float(np.median(near_ov)) if near_ov else None,
        "interpretation_note": (
            "near_miss errors share content words with the truth — partial semantic "
            "evidence reached the verifier; off_target errors are the true failures. "
            "A high near_miss share supports the coarse-semantic-evidence account."),
    }
    if margins:
        m = np.asarray(margins["margin"], dtype=float)[: len(corr)]
        out["margin_mean_correct"] = float(m[corr.astype(bool)].mean()) if corr.any() else None
        out["margin_mean_wrong"] = float(m[~corr.astype(bool)].mean()) if (~corr.astype(bool)).any() else None
    with open(os.path.join(runs_dir, "e4_error_analysis.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def fig9_error_analysis(runs: str) -> str | None:
    """Four-panel error-analysis figure. Registered in cprd.figures.ALL_FIGURES."""
    path = os.path.join(runs, "e4_error_analysis.json")
    if not os.path.exists(path):
        analyze(runs)
    if not os.path.exists(path):
        return None
    ea = json.load(open(path))
    if "_error" in ea:
        return None
    csv_path = os.path.join(runs, "e4_selected_texts.csv")
    rows = list(csv.DictReader(open(csv_path)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.6))

    # (a) accuracy by length quartile
    bl = ea.get("accuracy_by_length_quartile", {})
    if bl:
        axes[0].bar(list(bl.keys()), [v["acc"] for v in bl.values()], color="tab:blue")
        axes[0].set_title("accuracy vs sentence length")
        axes[0].tick_params(axis="x", rotation=30)

    # (b) failure taxonomy
    tx = ea.get("failure_taxonomy", {})
    if tx:
        colors = {"near_miss": "tab:orange", "same_topic": "tab:purple",
                  "off_target": "tab:red"}
        axes[1].bar(list(tx.keys()), list(tx.values()),
                    color=[colors.get(k, "tab:gray") for k in tx])
        axes[1].set_title("failure taxonomy (wrong picks)")
        axes[1].tick_params(axis="x", rotation=20)

    # (c) content-word overlap of wrong picks
    ovs = [float(json.dumps(0)) for _ in ()]  # placeholder removal below
    ovs = []
    for r in rows:
        if not int(r["correct"]):
            ovs.append(overlap(r["reference"], r["hypothesis_real"]))
    if ovs:
        axes[2].hist(ovs, bins=10, color="tab:orange")
        axes[2].axvline(0.4, ls=":", c="k", lw=1)
        axes[2].set_title("wrong picks: content overlap w/ truth")
        axes[2].set_xlabel("overlap")

    # (d) margin split, when margins were exported
    mpath = os.path.join(runs, "e4_margins.json")
    if os.path.exists(mpath):
        mm = json.load(open(mpath))
        m = np.asarray(mm["margin"], dtype=float)
        c = np.asarray([int(r["correct"]) for r in rows], dtype=bool)[: len(m)]
        if c.any() and (~c).any():
            axes[3].hist(m[c], bins=12, alpha=0.6, label="correct", color="tab:blue")
            axes[3].hist(m[~c], bins=12, alpha=0.6, label="wrong", color="tab:red")
            axes[3].set_title("verifier margin by outcome")
            axes[3].legend(frameon=False, fontsize=7)
    else:
        axes[3].text(0.5, 0.5, "margins not exported", ha="center", va="center")
        axes[3].set_axis_off()

    d = os.path.join(runs, "figures")
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"fig9_error_analysis.{ext}"), bbox_inches="tight")
    plt.close(fig)
    return os.path.join(d, "fig9_error_analysis.pdf")
