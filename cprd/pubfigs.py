"""Publication figures (ICLR): 600 dpi, large bold type on both axes, PDF + PNG.

Rendered ONLY from gate artifacts / ledgers of the runs dirs (v1: runs, runs_gaze,
runs_both; v2: runs_v2, runs_v2_gaze, runs_v2_both) and the E1 probe JSON. No number is
typed in by hand. Output: <out>/figN_<name>.{pdf,png}.

    python -m cprd.pubfigs --out paper_figs
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

LN2 = math.log(2.0)
CH = [("gaze", "Gaze", "#1f77b4"), ("eeg", "EEG", "#d62728"), ("both", "EEG + gaze", "#2ca02c")]
DIRS = {"v1": {"eeg": "runs", "gaze": "runs_gaze", "both": "runs_both"},
        "v2": {"eeg": "runs_v2", "gaze": "runs_v2_gaze", "both": "runs_v2_both"}}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600,
        "font.size": 15, "font.weight": "bold", "font.family": "DejaVu Sans",
        "axes.labelsize": 17, "axes.labelweight": "bold", "axes.titlesize": 17, "axes.titleweight": "bold",
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12.5,
        "axes.linewidth": 1.8, "xtick.major.width": 1.6, "ytick.major.width": 1.6,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "lines.linewidth": 2.8, "lines.markersize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def _gate(d, name):
    p = os.path.join(d, f"gate_{name}.json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


def _save(plt, fig, out, name):
    os.makedirs(out, exist_ok=True)
    name = name.replace("_v2", "")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=600)
    plt.close(fig)
    return name


def _bold_ticks(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


# ---------------------------------------------------------------- Fig 1: bits per channel
def fig1_channels(out, version="v2"):
    plt = _mpl()
    rows = []
    for key, label, col in CH:
        g = _gate(DIRS[version][key], "channel")
        if not g:
            continue
        d = g["detail"]
        val = d.get("bits_per_token_nce_derangement") if version == "v2" else d["bits_per_token_derangement"]
        rows.append((label, col, float(val), d.get("p_derangement"), d.get("ci95_bits_per_token_derangement"),
                     d.get("bits_per_token_meannull_derangement")))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    x = np.arange(len(rows))
    for i, (label, col, val, p, ci, mn) in enumerate(rows):
        ax.bar(i, val, color=col, width=0.62, edgecolor="black", linewidth=1.6)
        ax.text(i, max(val, 0) + 0.006, f"{val:+.3f}\np={p:.3f}" if p is not None else f"{val:+.3f}",
                ha="center", va="bottom", fontsize=12.5, fontweight="bold")
    ax.axhline(0, color="black", lw=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("Information (bits / token)\n" + ("InfoNCE lower bound" if version == "v2" else "Δ̂I (real − null)"))
    ax.set_title(f"Text information per evidence channel")
    ax.set_ylim(min(0, min(r[2] for r in rows)) - 0.02, max(r[2] for r in rows) * 1.45 + 0.02)
    _bold_ticks(ax)
    return _save(plt, fig, out, f"fig1_channels_{version}")


# ---------------------------------------------------------------- Fig 2: selection vs N
def fig2_selection_vs_N(out, version="v2"):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    any_ = False
    for key, label, col in CH:
        g = _gate(DIRS[version][key], "selection")
        if not g:
            continue
        sweep = g["detail"]["sweep"]
        Ns = [r["N"] for r in sweep]
        ax.plot(Ns, [r["real"]["acc"] for r in sweep], "o-", color=col, label=f"{label}: real")
        ax.plot(Ns, [r["zeroed"]["acc"] for r in sweep], "s--", color=col, alpha=0.55, label=f"{label}: zeroed")
        any_ = True
    if not any_:
        return None
    Ns = np.array([2, 4, 8, 16, 32, 64])
    ax.plot(Ns, 1.0 / Ns, ":", color="black", lw=2.2, label="chance 1/N")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Candidate-set size N")
    ax.set_ylabel("Top-1 selection accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Selection from N candidates (validation)")
    ax.legend(ncol=2, loc="upper right")
    _bold_ticks(ax)
    return _save(plt, fig, out, f"fig2_selection_vs_N_{version}")


# ---------------------------------------------------------------- Fig 3: E1 calibration
def fig3_calibration(out):
    plt = _mpl()
    g2 = _gate("runs_v2", "estimator")
    if not g2:
        return None
    sw = g2["detail"]["sweep"]
    b = [r["b_true"] for r in sw]
    med = [r["recovered_median"] for r in sw]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax in (a1, a2):
        for r in sw:
            for (bt, rec, p) in r["attempts"]:
                ax.plot(bt, rec, "o", color="#1f77b4", alpha=0.35, ms=8)
        ax.plot(b, med, "o-", color="#1f77b4", label="Estimate (median of 3 seeds)")
        ax.set_xlabel("Injected capacity (bits / token)")
        ax.set_ylabel("Recovered (bits / token)")
    lim = 2.15
    a1.plot([0, lim], [0, lim], "-", color="black", lw=1.8, alpha=0.7, label="Truth (y = x)")
    a1.set_xlim(-0.05, lim); a1.set_ylim(-0.05, lim)
    a1.set_title("Full range: a valid lower bound")
    a1.legend(loc="upper left")
    a2.axhline(0, color="gray", lw=1.2)
    a2.set_xlim(-0.05, lim); a2.set_ylim(-0.03, 0.15)
    a2.set_title("Zoom: zero point and monotone rise")
    a2.legend(loc="upper left")
    for ax in (a1, a2):
        _bold_ticks(ax)
    fig.suptitle("Estimator calibration on a channel of known capacity", fontweight="bold", fontsize=17, y=1.03)
    return _save(plt, fig, out, "fig3_calibration")


# ---------------------------------------------------------------- Fig 4: one-shot test arms
def fig4_test_arms(out, version="v2"):
    plt = _mpl()
    arms = ["real", "zeroed", "gamma_zero", "derangement", "gaussian_matched", "amplitude_only", "position_only"]
    names = ["real", "zeroed", "γ = 0", "deranged", "Gaussian", "amplitude", "position"]
    panels = [(k, l, c) for k, l, c in CH if _gate(DIRS[version][k], "system")]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 4.8), sharey=False)
    fig.subplots_adjust(wspace=0.38)
    axes = np.atleast_1d(axes)
    for ax, (key, label, col) in zip(axes, panels):
        d = _gate(DIRS[version][key], "system")["detail"]
        accs = [d["arms"][a]["acc"] for a in arms]
        cols = [col] + ["#bbbbbb"] * (len(arms) - 1)
        ax.bar(range(len(arms)), accs, color=cols, edgecolor="black", linewidth=1.5)
        ax.axhline(1.0 / d["N_star"], color="black", ls=":", lw=2.2, label=f"chance (N = {d['N_star']})")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(names, rotation=40, ha="right")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"{label}: N* = {d['N_star']}")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper right")
        _bold_ticks(ax)
    fig.suptitle(f"One-shot test with the control battery", fontweight="bold", y=1.02)
    return _save(plt, fig, out, f"fig4_test_arms_{version}")


# ---------------------------------------------------------------- Fig 5: ITR axis
def fig5_itr(out, version="v2"):
    plt = _mpl()
    pts = []
    for key, label, col in CH:
        g = _gate(DIRS[version][key], "system")
        if g and "nykopp_T2_trial" in g["detail"]["itr"]:
            pts.append((label, col, g["detail"]["itr"]["nykopp_T2_trial"]))
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bands = [("MI-BCI", 5, 25, "#dddddd"), ("P300", 10, 40, "#c9c9c9"), ("SSVEP", 30, 100, "#b5b5b5")]
    for i, (name, lo, hi, c) in enumerate(bands):
        ax.axvspan(lo, hi, color=c, alpha=0.5)
        ax.text(math.sqrt(lo * hi), 0.93 - 0.085 * i, name, ha="center", fontsize=12.5, fontweight="bold")
    for k, (label, col, v) in enumerate(pts):
        y = (0.50, 0.50, 0.20)[k]
        ax.plot([max(v, 0.05)], [y], "D", color=col, ms=13, markeredgecolor="black")
        ax.annotate(f"{label}: {v:.1f} b/min", (max(v, 0.05), y), textcoords="offset points",
                    xytext=(0, 20 if k < 2 else -34), ha="center", fontsize=12.5, fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set_xscale("log")
    ax.set_xlim(0.04, 150)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nykopp ITR, trial-time denominator (bits / min)")
    ax.set_title(f"Where the channels sit on the classical BCI axis")
    _bold_ticks(ax)
    return _save(plt, fig, out, f"fig5_itr_axis_{version}")


# ---------------------------------------------------------------- Fig 6: risk-coverage
def fig6_risk_coverage(out, version="v2"):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    any_ = False
    for key, label, col in CH:
        p = os.path.join(DIRS[version][key], "e4_risk_coverage.json")
        if not os.path.exists(p):
            continue
        rc = json.load(open(p))
        ax.plot(rc["coverages"], rc["risks"], "o-", color=col, label=f"{label}: model confidence (AURC {rc['aurc']:.2f})")
        ax.plot(rc["coverages"], rc["risks_cheap"], "--", color=col, alpha=0.5, lw=2.0,
                label=f"{label}: cheap baseline (AURC {rc['aurc_cheap']:.2f})")
        any_ = True
    if not any_:
        return None
    ax.set_xlabel("Coverage (fraction of test pools answered)")
    ax.set_ylabel("Selective risk (error rate)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Abstention: risk–coverage on the test split")
    ax.legend(fontsize=11)
    _bold_ticks(ax)
    return _save(plt, fig, out, f"fig6_risk_coverage_{version}")


# ---------------------------------------------------------------- Fig 7: scaling
def fig7_scaling(out, version="v2"):
    g = _gate(DIRS[version]["gaze"], "scaling")
    if not g:
        return None
    plt = _mpl()
    d = g["detail"]
    rows = d.get("rows") or d.get("sweep") or []
    if not rows:
        return None
    fr = [r.get("fraction", r.get("frac")) for r in rows]
    key = "acc" if "acc" in rows[0] else ("accuracy" if "accuracy" in rows[0] else None)
    if key is None:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot(fr, [r[key] for r in rows], "o-", color="#1f77b4")
    ax.set_xlabel("Fraction of training sentences")
    ax.set_ylabel("Selection accuracy")
    ax.set_title("Scaling with training data (gaze)")
    _bold_ticks(ax)
    return _save(plt, fig, out, f"fig7_scaling_{version}")


# ---------------------------------------------------------------- Fig 8: estimator inconsistency (oracle probe)
def fig8_oracle(out, probe_json="probes/probe_e1_oracle_tilt_bound.json"):
    if not os.path.exists(probe_json):
        return None
    plt = _mpl()
    d = json.load(open(probe_json))
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    rows = [r for r in d["rows"] if r["b"] == 1.0]
    for gamma, mk in ((0.25, "o"), (1.0, "s"), (5.0, "^")):
        rr = [r for r in rows if r["gamma"] == gamma]
        ax.plot([r["bu"] for r in rr], [r["mean_null"] for r in rr], mk + "-", color="#d62728", label=f"real − mean null, γ = {gamma}")
        ax.plot([r["bu"] for r in rr], [r["dv"] for r in rr], mk + "--", color="#1f77b4", alpha=0.8, label=f"DV bound, γ = {gamma}")
    ax.axhline(1.0, color="black", lw=1.8, ls=":", label="truth (1 bit)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Tilt scale ‖B u‖")
    ax.set_ylabel("Estimate (bits / token)")
    ax.set_title("Oracle encoder, 1 injected bit: v1 has no plateau")
    ax.legend(fontsize=10, ncol=2)
    _bold_ticks(ax)
    return _save(plt, fig, out, "fig8_oracle_inconsistency")


# ---------------------------------------------------------------- Fig 9: v1 vs v2 side by side (gaze/eeg/both bits)
def fig9_v1_vs_v2(out):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    x = np.arange(len(CH)); w = 0.36
    ok = False
    for j, (version, hatch) in enumerate((("v1", ""), ("v2", "//"))):
        vals = []
        for key, label, col in CH:
            g = _gate(DIRS[version][key], "channel")
            if not g:
                vals.append(np.nan); continue
            d = g["detail"]
            vals.append(d.get("bits_per_token_nce_derangement") if version == "v2" else d["bits_per_token_derangement"])
        if not all(np.isnan(v) for v in vals):
            ok = True
        ax.bar(x + (j - 0.5) * w, vals, width=w, color=[c for _, _, c in CH], alpha=0.55 + 0.45 * j,
               hatch=hatch, edgecolor="black", linewidth=1.5,
               label=("v1: Δ̂I = real − mean null (pre-registered)" if version == "v1" else "v2: InfoNCE lower bound (validated)"))
    if not ok:
        return None
    ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in CH])
    ax.axhline(0, color="black", lw=1.2)
    ax.set_ylabel("bits / token")
    ax.set_title("Same channels, two instruments")
    ax.legend(loc="upper right", fontsize=11)
    _bold_ticks(ax)
    return _save(plt, fig, out, "fig9_v1_vs_v2")


# ---------------------------------------------------------------- Fig 0: methodology pipeline
def fig0_pipeline(out):
    """Six-lane pipeline schematic (same content as docs/methodology_pipeline.drawio).
    Text is wrapped to the box width and box heights follow the wrapped text."""
    import textwrap
    plt = _mpl()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    W, H = 12.0, 5.8                      # inches
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    FS, LEAD = 7.6, 0.135                 # box font (pt), line leading (in)
    CPI = 0.0072 * FS                     # approx inches per character at FS

    def wrap(text, w):
        n = max(12, int((w - 0.18) / CPI))
        out_lines = []
        for para in text.split("\n"):
            out_lines += textwrap.wrap(para, n) or [""]
        return out_lines

    def lane(x, y, w, h, title, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.12",
                                    fc=fc, ec="#555555", lw=1.4))
        ax.text(x + 0.1, y + h - 0.17, title, fontsize=9.6, fontweight="bold", va="center")

    def stack(x, top, w, items, gap=0.12):
        """Place boxes top-down; returns list of (cx, ytop, ybottom)."""
        y = top; pos = []
        for text, fc in items:
            lines = wrap(text, w); h = 0.16 + LEAD * len(lines)
            ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.01,rounding_size=0.07",
                                        fc=fc, ec="#333333", lw=1.1))
            ax.text(x + 0.09, y - h / 2, "\n".join(lines), fontsize=FS, va="center", ha="left",
                    fontweight="normal", linespacing=1.15)
            pos.append((x + w / 2, y, y - h)); y -= h + gap
        for (cx, yt, yb), (cx2, yt2, yb2) in zip(pos, pos[1:]):
            ax.add_patch(FancyArrowPatch((cx, yb), (cx2, yt2), arrowstyle="-|>", mutation_scale=9, lw=1.1, color="#333333"))
        return pos

    def arrow(p, q, color="#333333", ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.3, color=color, linestyle=ls))

    # ---- lanes (x, y, w, h)
    LA = (0.15, 2.55, 2.75, 3.1); LB = (3.0, 2.55, 2.95, 3.1); LC = (6.05, 2.55, 2.95, 3.1); LD = (9.1, 2.55, 2.75, 3.1)
    LE = (0.15, 0.1, 8.85, 2.35); LF = (9.1, 0.1, 2.75, 2.35)
    lane(*LA, "A · Data (E0)", "#E8F1FA"); lane(*LB, "B · Prior-residual model", "#EAF6EA")
    lane(*LC, "C · Channel measurement (E2)", "#F3ECFA"); lane(*LD, "D · Calibration (E1)", "#FDECEC")
    lane(*LE, "E · Selection as a communication system (E3 → E4)", "#FFF0E0"); lane(*LF, "F · Provenance", "#F2F2F2")
    top = LA[1] + LA[3] - 0.36
    pa = stack(LA[0] + 0.1, top, LA[2] - 0.2, [
        ("ZuCo 1.0 + 2.0: 18,890 readings of 1,319 texts by 30 subjects. 840-d band power per word; fixation durations from the .mat files.", "white"),
        ("Text-clustered split 15,131 / 1,834 / 1,925 readings, zero text overlap. Validation → selection half / measurement half (by text hash).", "white"),
        ("Evidence per word: EEG 840-d (NaN if unfixated) · gaze 6-d · both 846-d. Missingness is a mask, never a feature.", "white"),
        ("Frozen prior p₀ = gpt2-large; exact log p₀ over the full vocabulary.", "#FFF4D6")])
    pb = stack(LB[0] + 0.1, top, LB[2] - 0.2, [
        ("Encoder f_φ: window (t−1, t, t+1) → u_t ∈ ℝ¹⁶. EEG: band-spatial projection → band attention → 2-layer transformer. Gaze: Linear(6→128). ‖u‖ ≤ s_max, u centred.", "white"),
        ("Tilt over the full vocabulary (exact Z): ℓ_t(v) = ⟨Ω_v, B u_t⟩ + ⟨φ(Ω_v), W u_t⟩; φ = MLP 1280→64→16 over frozen Ω; γ_t ≥ 0, hard-gated to 0 on unfixated words.", "white"),
        ("T_t = γ_t ℓ_t(w_t) − log Z_t ≤ −log p₀(w_t) (tripwire); γ = 0 ⇒ T ≡ 0 (zero anchor).", "white"),
        ("Training: InfoNCE with 4 in-batch deranged negatives per token.", "#F5FBF5")])
    pc = stack(LC[0] + 0.1, top, LC[2] - 0.2, [
        ("Nulls at dataset level, M = 1,000 draws each: ① cross-sentence derangement (length-matched, different text, position-preserving); ② temporal roll within the sentence.", "white"),
        ("Estimand — InfoNCE lower bound: Î_t = T_t(e_t) − log[(e^{T_t(e_t)} + Σ_m e^{T_t(e^{(m)})}) / (M+1)]; E[Î] ≤ I; a constant ℓ gives Î = 0. DV bound and v1 statistic stored alongside.", "white"),
        ("Exact permutation p under both nulls; text-clustered bootstrap CI (10k); H1: p < 0.05 under both.", "white"),
        ("Output: bits/token and bits/sentence per channel; EEG beyond gaze = both − gaze.", "#E6D6F5")])
    pd = stack(LD[0] + 0.1, top, LD[2] - 0.2, [
        ("Prior-matched synthetic channel: sentences sampled from p₀; clusters = equal-prior-mass slabs of Ω; e_t = μ_c(w_t) + σε in 16 dims; I(E; c) = b, Monte-Carlo exact.", "white"),
        ("Same train–select–measure procedure as E2; b ∈ {0, 0.25, 1, 2}; 3 seeds; 20 null datasets.", "white"),
        ("Lower-bound gate (median over seeds): zero point · detection (p < 0.05) · soundness · monotone · KS-uniform p. Tightness Î/b reported downstream.", "white"),
        ("v1 fails: encoder collapse · no plateau · linear tilt cannot represent the code.", "#F9D6D2")])
    topE = LE[1] + LE[3] - 0.36
    pe1 = stack(LE[0] + 0.1, topE, 2.75, [("Candidate pools: true sentence + N−1 length- and prior-log-p-matched distractors; no-brain attack must stay ≤ chance + 2 pp.", "white")])
    pe2 = stack(LE[0] + 3.0, topE, 2.85, [("Likelihood-ratio verifier S(y) = Σ_t T_t(y_t, e); argmax, random ties. E3: accuracy vs N ∈ {2…64} on validation, 4 arms.", "white")])
    pe3 = stack(LE[0] + 6.0, topE, 2.75, [("Matched-N freeze: N* = largest N with ideal-observer accuracy ≥ 0.5 from the E2 bits — written to the pre-registration before any test access.", "white")])
    ybot = min(pe1[0][2], pe2[0][2], pe3[0][2]) - 0.15
    pe4 = stack(LE[0] + 0.1, ybot, LE[2] - 0.2, [("E4 — the ONE guarded test session: real · zeroed · γ = 0 (tie asserted) · deranged · Gaussian-matched · amplitude-only · position-only → accuracy, Fano bits, Wolpaw/Nykopp ITR (T1/T2/T3), risk–coverage & AURC, partial AUC, TOST, selected-text BLEU/chrF/BERTScore (comparability only).", "#FFE3C2")])
    pf = stack(LF[0] + 0.1, topE, LF[2] - 0.2, [
        ("Gates build → estimator → channel → selection → system; never hand-edited; overrides logged, downstream records stamped.", "white"),
        ("Ledger: every number with split fingerprint, objective, estimand, evidence, seed, gates passed.", "white"),
        ("Pre-registration: H1–H5, N*, config hash; guarded test loader; deviations log.", "white")])
    # cross-lane arrows
    arrow((LA[0] + LA[2] - 0.1, pa[2][1] - 0.3), (LB[0] + 0.1, pb[0][1] - 0.3), color="#5B8DB8")
    arrow((LA[0] + LA[2] - 0.1, pa[3][1] - 0.2), (LB[0] + 0.1, pb[2][1] - 0.2), color="#C48A00")
    arrow((LB[0] + LB[2] - 0.1, pb[2][1] - 0.2), (LC[0] + 0.1, pc[1][1] - 0.4), color="#3C8C3C")
    arrow((LD[0] + 0.1, pd[2][1] - 0.4), (LC[0] + LC[2] - 0.1, pc[3][1] - 0.2), color="#C0392B", ls="--")
    arrow((pe1[0][0] + 1.4, pe1[0][1] - 0.35), (pe2[0][0] - 1.45, pe2[0][1] - 0.35))
    arrow((pe2[0][0] + 1.45, pe2[0][1] - 0.35), (pe3[0][0] - 1.4, pe3[0][1] - 0.35))
    arrow((pe3[0][0], pe3[0][2]), (pe3[0][0], pe4[0][1]))
    arrow((pb[3][0], pb[3][2]), (pe2[0][0], pe2[0][1]), color="#3C8C3C")
    arrow((pc[3][0], pc[3][2]), (pe3[0][0], pe3[0][1]), color="#7B4FA8")
    arrow((LE[0] + LE[2], pe4[0][1] - 0.2), (LF[0] + 0.1, pf[1][1] - 0.2), color="#666666", ls="--")
    return _save(plt, fig, out, "fig0_pipeline")



# ---------------------------------------------------------------- Fig 10: language quality vs information
def fig10_quality_vs_information(out, version="v2"):
    """Headline separation: text-similarity of the selected candidate under real vs zeroed
    evidence (left) against the information lower bound (right), per channel."""
    plt = _mpl()
    rows = []
    for key, label, col in CH:
        d = DIRS[version][key]
        pm = os.path.join(d, "e4_paper_metrics.json"); g = _gate(d, "channel")
        if not (os.path.exists(pm) and g):
            continue
        m = json.load(open(pm)); det = g["detail"]
        bits = det.get("bits_per_token_nce_derangement") if version == "v2" else det["bits_per_token_derangement"]
        rows.append((label, col, m["real"]["bleu4"], m["zeroed"]["bleu4"], m["real"]["bertscore_f1"], m["zeroed"]["bertscore_f1"], float(bits)))
    if not rows:
        return None
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 4.6))
    fig.subplots_adjust(wspace=0.38)
    x = np.arange(len(rows)); w = 0.36
    for ax, ir, iz, title, ylim in ((a1, 2, 3, "Selected-text BLEU-4", (0, 118)), (a2, 4, 5, "BERTScore-F1", (0.8, 1.035))):
        ax.bar(x - w / 2, [r[ir] for r in rows], w, color=[r[1] for r in rows], edgecolor="black", linewidth=1.4, label="real evidence")
        ax.bar(x + w / 2, [r[iz] for r in rows], w, color="#bbbbbb", edgecolor="black", linewidth=1.4, label="zeroed evidence")
        ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylim(*ylim); ax.set_title(title); ax.legend(loc="upper center", ncol=2, fontsize=11, columnspacing=1.0, handlelength=1.2)
        _bold_ticks(ax)
    a3.bar(x, [r[6] for r in rows], 0.55, color=[r[1] for r in rows], edgecolor="black", linewidth=1.4)
    for i, r in enumerate(rows):
        a3.text(i, max(r[6], 0) + 0.004, f"{r[6]:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    a3.axhline(0, color="black", lw=1.2); a3.set_xticks(x); a3.set_xticklabels([r[0] for r in rows])
    a3.set_ylim(-0.01, max(r[6] for r in rows) * 1.35 + 0.01); a3.set_title("Information (bits / token)")
    _bold_ticks(a3)
    fig.suptitle("Language quality is not evidence dependence: EEG keeps its text scores with zeroed input and carries 0 bits",
                 fontweight="bold", fontsize=13.5, y=1.03)
    return _save(plt, fig, out, f"fig10_quality_vs_information_{version}")


# ---------------------------------------------------------------- Fig 11: breakdown (task/subject/surprisal)
def fig11_breakdown(out, version="v2"):
    """Per-task, per-subject and per-surprisal-decile breakdown of the E2 InfoNCE bound
    (reviewer items 14, 29, 30), gaze and EEG side by side; re-analysis of the stored
    E2 permutation draws, no new training or test access (experiments/e10_breakdown.py)."""
    plt = _mpl()
    chans = [("gaze", "Gaze", "#1f77b4"), ("eeg", "EEG", "#d62728")]
    data = {}
    for key, label, col in chans:
        p_ = os.path.join(DIRS[version][key], "gate_breakdown.json")
        if not os.path.exists(p_):
            continue
        data[key] = (label, col, json.load(open(p_))["detail"])
    if not data:
        return None
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18.5, 5.2))
    fig.subplots_adjust(wspace=0.3, bottom=0.16, top=0.86)

    # --- panel 1: by task, grouped bars with 95% CI whiskers
    task_order = ["task1-SR", "task2-NR", "task3-TSR", "task2-NR-2.0"]
    task_names = ["SR", "NR", "TSR", "NR-2.0"]
    x = np.arange(len(task_order))
    w = 0.36
    for i, (key, (label, col, det)) in enumerate(data.items()):
        by_task = det["by_task"]
        ys = [by_task[t]["sentence_derangement"]["bits_per_token"] if t in by_task else np.nan for t in task_order]
        cis = [by_task[t]["sentence_derangement"]["ci95"] if t in by_task else (np.nan, np.nan) for t in task_order]
        errs = np.array([[y - c[0], c[1] - y] for y, c in zip(ys, cis)]).T
        a1.bar(x + (i - 0.5) * w, ys, w, yerr=np.abs(errs), capsize=3, color=col,
              edgecolor="black", linewidth=1.3, label=label)
    a1.axhline(0, color="black", lw=1.1)
    a1.set_xticks(x); a1.set_xticklabels(task_names)
    a1.set_ylabel("bits / token"); a1.set_title("By ZuCo task"); a1.legend(fontsize=10.5)
    _bold_ticks(a1)

    # --- panel 2: by subject, sorted strip/dot plot
    for i, (key, (label, col, det)) in enumerate(data.items()):
        by_subj = det["by_subject"]
        subs = sorted(by_subj.keys())
        ys = np.array([by_subj[s]["sentence_derangement"]["bits_per_token"] for s in subs])
        order = np.argsort(ys)
        a2.plot(np.arange(len(subs)) + (i - 0.5) * 0.15, ys[order], "o", color=col, ms=5,
               alpha=0.85, label=f"{label} (n={len(subs)})")
    a2.axhline(0, color="black", lw=1.1)
    a2.set_xlabel("subject (sorted per channel)")
    a2.set_ylabel("bits / token"); a2.set_title("By subject"); a2.legend(fontsize=10.5)
    _bold_ticks(a2)

    # --- panel 3: by prior-surprisal decile
    for key, (label, col, det) in data.items():
        by_dec = det["by_surprisal_decile"]
        ks = sorted(by_dec.keys(), key=int)
        xs = [by_dec[k]["mean_surprisal_nats"] for k in ks]
        ys = [by_dec[k]["sentence_derangement"]["bits_per_token"] for k in ks]
        rho = det["surprisal_relation"]["spearman_rho_deciles"]
        a3.plot(xs, ys, "o-", color=col, label=f"{label} (Spearman ρ={rho:+.2f})")
    a3.axhline(0, color="black", lw=1.1)
    a3.set_xlabel("prior surprisal (nats/tok)")
    a3.set_ylabel("bits / token"); a3.set_title("By prior-surprisal decile"); a3.legend(fontsize=10.5)
    _bold_ticks(a3)

    fig.suptitle("The gaze channel is present in every task and every subject; EEG is flat at zero throughout",
                 fontweight="bold", fontsize=13, y=1.04)
    return _save(plt, fig, out, f"fig11_breakdown_{version}")


# ---------------------------------------------------------------- Fig 12: rank sweep and capacity controls
def fig12_capacity(out):
    plt = _mpl()
    spec = [("Linear\ntilt", "runs_v2_gaze_lineartilt", "#999999"), ("Rank 4", "runs_v2_gaze_r4", "#1f77b4"),
            ("Rank 8", "runs_v2_gaze_r8", "#1f77b4"), ("Rank 16\n(main)", "runs_v2_gaze", "#0b3d91"),
            ("Rank 32", "runs_v2_gaze_r32", "#1f77b4"), ("Rank 64,\nwide MLP", "runs_v2_gaze_bigtilt", "#ff7f0e")]
    rows = []
    for lab, d, c in spec:
        g = _gate(d, "channel")
        if g:
            rows.append((lab, c, g["detail"]["bits_per_token_nce_derangement"], g["detail"]["bits_per_token_DV_derangement"]))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w / 2, [r[2] for r in rows], w, color=[r[1] for r in rows], edgecolor="black", linewidth=1.4, label="InfoNCE bound")
    ax.bar(x + w / 2, [r[3] for r in rows], w, color="#dddddd", edgecolor="black", linewidth=1.4, label="Donsker-Varadhan bound")
    for i, r in enumerate(rows):
        ax.text(i - w / 2, r[2] + 0.004, f"{r[2]:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("Gaze information (bits / token)")
    ax.set_title("Tilt capacity: nonlinearity matters more than rank")
    ax.set_ylim(0, max(r[2] for r in rows) * 1.22)
    ax.legend(loc="upper right")
    _bold_ticks(ax)
    return _save(plt, fig, out, "fig12_capacity")


# ---------------------------------------------------------------- Fig 13: split-leakage ablation
def fig13_leakage(out):
    plt = _mpl()
    pairs = {"Gaze": ("runs_v2_gaze", "runs_v2_instance_gaze", "#1f77b4"), "EEG": ("runs_v2", "runs_v2_instance_eeg", "#d62728")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 5.0), gridspec_kw={"width_ratios": [1, 1.4]})
    x = np.arange(len(pairs)); w = 0.36
    vals = {}
    for k, (lab, (dc, di, c)) in enumerate(pairs.items()):
        gc, gi = _gate(dc, "channel"), _gate(di, "channel")
        if not (gc and gi):
            return None
        vals[lab] = (gc["detail"]["bits_per_token_nce_derangement"], gi["detail"]["bits_per_token_nce_derangement"])
    a1.bar(x - w / 2, [vals[k][0] for k in pairs], w, color=[v[2] for v in pairs.values()], edgecolor="black", linewidth=1.4, label="Text-clustered split")
    a1.bar(x + w / 2, [vals[k][1] for k in pairs], w, color=[v[2] for v in pairs.values()], edgecolor="black", linewidth=1.4, hatch="//", alpha=0.6, label="Instance split (texts shared)")
    for i, k in enumerate(pairs):
        a1.text(i - w / 2, max(vals[k][0], 0) + 0.003, f"{vals[k][0]:.3f}", ha="center", fontsize=12, fontweight="bold")
        a1.text(i + w / 2, max(vals[k][1], 0) + 0.003, f"{vals[k][1]:.3f}", ha="center", fontsize=12, fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels(list(pairs)); a1.set_ylabel("Information (bits / token)")
    a1.set_title("Channel bound"); a1.set_ylim(0, 0.14); a1.legend(loc="upper right", fontsize=11)
    gs = {"cl": _gate("runs_v2_gaze", "selection"), "in": _gate("runs_v2_instance_gaze", "selection"), "ine": _gate("runs_v2_instance_eeg", "selection")}
    if not all(gs.values()):
        return None
    Ns = [r["N"] for r in gs["cl"]["detail"]["sweep"]]
    a2.plot(Ns, [r["real"]["acc"] for r in gs["cl"]["detail"]["sweep"]], "o-", color="#1f77b4", label="Gaze, text-clustered")
    a2.plot(Ns, [r["real"]["acc"] for r in gs["in"]["detail"]["sweep"]], "s--", color="#1f77b4", alpha=0.7, label="Gaze, instance split")
    a2.plot(Ns, [r["real"]["acc"] for r in gs["ine"]["detail"]["sweep"]], "^--", color="#d62728", label="EEG, instance split")
    a2.plot(Ns, [1.0 / n for n in Ns], ":", color="black", label="Chance")
    a2.set_xscale("log", base=2); a2.set_xticks(Ns); a2.set_xticklabels([str(n) for n in Ns])
    a2.set_xlabel("Candidate-set size N"); a2.set_ylabel("Top-1 selection accuracy"); a2.set_ylim(0, 1.0)
    a2.set_title("Selection (validation)"); a2.legend(loc="upper right", fontsize=11)
    for a in (a1, a2):
        _bold_ticks(a)
    fig.suptitle("Sharing texts across splits does not create evidence-dependent information", fontweight="bold", fontsize=16, y=1.03)
    return _save(plt, fig, out, "fig13_leakage")


# ---------------------------------------------------------------- Fig 14: leave-one-subject-out (gaze)
def fig14_loso(out):
    plt = _mpl()
    p = os.path.join("runs_v2_gaze", "ledger.jsonl")
    if not os.path.exists(p):
        return None
    acc = {4: {}, 32: {}}; zer = {4: {}, 32: {}}
    for l in open(p):
        r = json.loads(l); m = r["metric"]
        if not m.startswith("loso_selection_acc_N"):
            continue
        N = int(m.rsplit("N", 1)[1]); arm = r["arm"].replace("heldout=", "")
        if arm.endswith("/zeroed"):
            zer[N][arm[:-7]] = r["value"]
        else:
            acc[N][arm] = r["value"]
    if not acc[4]:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, N in zip(axes, (4, 32)):
        subs = sorted(acc[N], key=lambda k: acc[N][k])
        xs = np.arange(len(subs))
        ax.bar(xs, [acc[N][k] for k in subs], 0.7, color="#1f77b4", edgecolor="black", linewidth=1.2, label="Held-out subject")
        ax.plot(xs, [zer[N].get(k, np.nan) for k in subs], "o", color="#555555", ms=8, label="Zeroed evidence")
        ax.axhline(1.0 / N, color="black", ls=":", lw=2, label=f"Chance (1/{N})")
        ax.set_xticks(xs); ax.set_xticklabels(subs, rotation=45, ha="right")
        ax.set_ylabel("Top-1 selection accuracy"); ax.set_title(f"N = {N}  (mean {np.mean(list(acc[N].values())):.3f}, {len(subs)} subjects)")
        ax.set_ylim(0, 1.0); ax.legend(loc="upper left", fontsize=11); _bold_ticks(ax)
    fig.suptitle("Leave-one-subject-out: gaze generalises to unseen readers", fontweight="bold", fontsize=16, y=1.03)
    return _save(plt, fig, out, "fig14_loso")


def make_all(out):
    made = []
    for fn in (lambda: fig1_channels(out, "v2"), lambda: fig2_selection_vs_N(out, "v2"),
               lambda: fig3_calibration(out), lambda: fig4_test_arms(out, "v2"),
               lambda: fig5_itr(out, "v2"), lambda: fig6_risk_coverage(out, "v2"),
               lambda: fig7_scaling(out, "v2"), lambda: fig10_quality_vs_information(out, "v2"),
               lambda: fig11_breakdown(out, "v2"), lambda: fig12_capacity(out),
               lambda: fig13_leakage(out), lambda: fig14_loso(out)):
        try:
            r = fn()
            if r:
                made.append(r)
        except Exception as e:  # a missing input must not kill the rest
            print(f"  (skipped: {type(e).__name__}: {e})")
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_figs")
    a = ap.parse_args()
    for m in make_all(a.out):
        print("wrote", m)


