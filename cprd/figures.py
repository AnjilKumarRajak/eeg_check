"""ICLR-oriented figures, rendered straight from gate artifacts + the ledger.

Every figure is regenerable from provenance-carrying files only — no hand-fed numbers.
Saved as both PDF (camera-ready) and PNG (draft) into <runs>/figures/.

  fig1_selection_vs_N      accuracy vs N (log2 x): real/gamma0/zeroed/derangement arms,
                           chance curve, and the theoretical ideal-observer curve from
                           the measured bits — the paper's money figure
  fig2_dissociation        same-system bars: generation BLEU real-vs-noise beside
                           selection accuracy real-vs-zeroed (argument #1)
  fig3_risk_coverage       selective risk vs coverage, real arm vs cheap-confidence
  fig4_matched_n           predicted vs realized bits (non-additivity finding)
  fig5_itr_axis            our ITR point on the classical BCI axis (MI/P300/SSVEP bands)
  fig6_estimator           recovered vs injected bits with y=x and the validated range
  fig7_scaling             realized bits / accuracy vs training fraction (if e8 ran)
  fig8_metric_collapse     paper metrics real-vs-zeroed bars (BLEU/BERTScore/METEOR/...)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

LN2 = math.log(2.0)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _gate(runs, name):
    p = os.path.join(runs, f"gate_{name}.json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


def _save(plt, fig, runs, name):
    d = os.path.join(runs, "figures")
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    return os.path.join(d, f"{name}.pdf")


def ideal_observer_acc(N, bits):
    return 1.0 / (1.0 + (N - 1) * math.exp(-bits * LN2))


def fig1_selection_vs_N(runs) -> str | None:
    g = _gate(runs, "selection")
    if not g:
        return None
    plt = _mpl()
    sweep = g["detail"]["sweep"]
    Ns = [row["N"] for row in sweep]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    styles = {"real": ("o-", "tab:blue"), "gamma_zero": ("s--", "tab:gray"),
              "zeroed": ("^--", "tab:red"), "derangement": ("v--", "tab:orange")}
    for arm, (st, c) in styles.items():
        ys = [row[arm]["acc"] for row in sweep if arm in row]
        if ys:
            ax.plot(Ns[:len(ys)], ys, st, color=c, label=arm.replace("_", "-"), ms=4)
    ax.plot(Ns, [1.0 / n for n in Ns], ":", color="k", lw=1, label="chance")
    # theoretical curve from realized bits (median across N)
    bits = np.median([row["real"]["bits"] for row in sweep if "real" in row])
    grid = np.logspace(np.log2(min(Ns)), np.log2(max(Ns)), 50, base=2)
    ax.plot(grid, [ideal_observer_acc(n, bits) for n in grid], "-", color="tab:green",
            lw=1, alpha=0.7, label=f"ideal obs. @ {bits:.2f} bits")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("candidate-set size N")
    ax.set_ylabel("top-1 selection accuracy")
    ax.legend(frameon=False, fontsize=7)
    return _save(plt, fig, runs, "fig1_selection_vs_N")


def fig2_dissociation(runs) -> str | None:
    gs, gq = _gate(runs, "system"), _gate(runs, "seq2seq_baseline")
    if not gs:
        return None
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.8))
    d = gs["detail"]
    arms = d.get("arms") or {}
    def _acc(name, legacy):
        if name in arms:
            return arms[name]["acc"]
        return d.get(legacy, float("nan"))
    axes[0].bar(["real", "zeroed", "γ=0"],
                [_acc("real", "acc_real"), _acc("zeroed", "acc_zeroed"),
                 _acc("gamma_zero", "acc_gamma_zero")],
                color=["tab:blue", "tab:red", "tab:gray"])
    axes[0].axhline(1.0 / d.get("N_star", 133), ls=":", c="k", lw=1)
    axes[0].set_title(f"selection @ N={d.get('N_star')}", fontsize=9)
    axes[0].set_ylabel("accuracy")
    if gq and isinstance(gq["detail"].get("bleu"), dict):
        b = gq["detail"]["bleu"]
        names = [k for k in ("real", "shuffled", "noise") if b.get(k) is not None]
        axes[1].bar(names, [b[k] for k in names],
                    color=["tab:blue", "tab:orange", "tab:red"][:len(names)])
        axes[1].set_title("generation BLEU (same EEG)", fontsize=9)
        axes[1].set_ylabel("corpus BLEU")
    else:
        axes[1].text(0.5, 0.5, "e6 not run", ha="center", va="center")
        axes[1].set_axis_off()
    fig.suptitle("Same system: selection depends on the brain; generation does not",
                 fontsize=9, y=1.02)
    return _save(plt, fig, runs, "fig2_dissociation")


def fig3_risk_coverage(runs) -> str | None:
    p = os.path.join(runs, "e4_risk_coverage.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        rc = json.load(fh)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot(rc["coverages"], rc["risks"], "o-", color="tab:blue", ms=3,
            label=f"ΔI margin (AURC {rc['aurc']:.3f})")
    if "risks_cheap" in rc:
        ax.plot(rc["coverages"], rc["risks_cheap"], "s--", color="tab:gray", ms=3,
                label=f"length+logp baseline (AURC {rc['aurc_cheap']:.3f})")
    ax.axhline(rc["full_risk"], ls=":", c="k", lw=1, label="no abstention")
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk")
    ax.legend(frameon=False, fontsize=7)
    return _save(plt, fig, runs, "fig3_risk_coverage")


def fig4_matched_n(runs) -> str | None:
    g = _gate(runs, "system")
    if not g:
        return None
    d = g["detail"]
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.bar(["predicted\n(ΔI × tokens)", "realized\n(selection)"],
           [d.get("predicted_bits", np.nan), d.get("realized_bits", np.nan)],
           color=["tab:gray", "tab:blue"])
    ax.set_ylabel("bits / sentence")
    ax.set_title("per-token evidence does not add", fontsize=9)
    for i, v in enumerate([d.get("predicted_bits"), d.get("realized_bits")]):
        if v is not None:
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    return _save(plt, fig, runs, "fig4_matched_n")


def fig5_itr_axis(runs) -> str | None:
    g = _gate(runs, "system")
    if not g:
        return None
    itr = g["detail"].get("itr", {})
    ours = [v for k, v in itr.items() if k.startswith("nykopp")]
    if not ours:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.0, 1.8))
    bands = [("MI", 17, 27, "tab:gray"), ("P300", 30, 52, "tab:orange"),
             ("SSVEP", 100, 325, "tab:green")]
    for name, lo, hi, c in bands:
        ax.barh([0], [hi - lo], left=[lo], height=0.5, color=c, alpha=0.35)
        ax.text((lo + hi) / 2, 0.38, name, ha="center", fontsize=7)
    for v in ours:
        ax.plot([v], [0], "D", color="tab:blue", ms=7)
    ax.annotate("this work\n(passive reading)", (min(ours), 0), (min(ours), -0.45),
                ha="center", fontsize=7, color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlabel("information transfer rate (bits/min, Nykopp)")
    ax.set_yticks([])
    return _save(plt, fig, runs, "fig5_itr_axis")


def fig6_estimator(runs) -> str | None:
    g = _gate(runs, "estimator")
    if not g:
        return None
    sweep = (g.get("detail") or {}).get("sweep") or []
    if not sweep:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    bt = [r["b_true"] for r in sweep]
    rec = [r.get("recovered_median", r.get("recovered")) for r in sweep]
    ok = [r["recovery_ok"] and r["no_overshoot"] for r in sweep]
    vmax = (g.get("detail") or {}).get("validated_max_bits_per_token")
    lim = max(bt + rec + [0.1]) * 1.15
    ax.plot([0, lim], [0, lim], "k:", lw=1, label="y = x (truth)")
    ax.fill_between([0, lim], [0, 0.8 * lim], color="tab:gray", alpha=0.12,
                    label="pass band (≥80%)")
    for x, y, o in zip(bt, rec, ok):
        ax.plot([x], [y], "o", color="tab:green" if o else "tab:red", ms=6)
    if vmax:
        ax.axvspan(0, vmax, color="tab:green", alpha=0.08)
        ax.text(vmax, 0.02, f" validated ≤ {vmax}", fontsize=7, color="tab:green")
    ax.set_xlabel("injected capacity (bits/token, MC-exact)")
    ax.set_ylabel("recovered Δ̂I (bits/token)")
    ax.legend(frameon=False, fontsize=7)
    return _save(plt, fig, runs, "fig6_estimator")


def fig7_scaling(runs) -> str | None:
    g = _gate(runs, "scaling")
    if not g:
        return None
    rows = g["detail"].get("rows", [])
    if not rows:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    fr = [r["fraction"] for r in rows]
    ax.plot(fr, [r["bits_realized"] for r in rows], "o-", color="tab:blue")
    ax.set_xlabel("training-data fraction")
    ax.set_ylabel("realized bits / sentence")
    ax2 = ax.twinx()
    ax2.plot(fr, [r["accuracy"] for r in rows], "s--", color="tab:green")
    ax2.set_ylabel("selection accuracy", color="tab:green")
    return _save(plt, fig, runs, "fig7_scaling")


def fig8_metric_collapse(runs) -> str | None:
    p = os.path.join(runs, "e4_paper_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        pm = json.load(fh)
    if "real" not in pm or "zeroed" not in pm:
        return None
    plt = _mpl()
    keys = [k for k in ("bleu1", "bleu4", "chrf", "meteor", "bertscore_f1",
                        "rouge1", "rougeL")
            if isinstance(pm["real"].get(k), (int, float))
            and isinstance(pm["zeroed"].get(k), (int, float))]
    scale = {k: (100.0 if pm["real"][k] <= 1.0 else 1.0) for k in keys}
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [pm["real"][k] * scale[k] for k in keys], 0.38,
           color="tab:blue", label="real EEG")
    ax.bar(x + 0.2, [pm["zeroed"][k] * scale[k] for k in keys], 0.38,
           color="tab:red", label="zeroed")
    ax.set_xticks(x, [k.replace("bertscore_f1", "BERTScore") for k in keys],
                  rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("score")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("every metric carries its own collapse floor", fontsize=9)
    return _save(plt, fig, runs, "fig8_metric_collapse")


def fig9_error_analysis(runs):
    from .error_analysis import fig9_error_analysis as _f
    return _f(runs)


ALL_FIGURES = [fig1_selection_vs_N, fig2_dissociation, fig3_risk_coverage,
               fig4_matched_n, fig5_itr_axis, fig6_estimator, fig7_scaling,
               fig8_metric_collapse, fig9_error_analysis]


def make_all(runs: str) -> list:
    out = []
    for fn in ALL_FIGURES:
        try:
            p = fn(runs)
            if p:
                out.append(p)
                print(f"  wrote {p}", flush=True)
        except Exception as e:
            print(f"  !! {fn.__name__}: {type(e).__name__}: {e}", flush=True)
    return out
