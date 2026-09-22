
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LN2 = math.log(2.0)

GATES = ["build", "estimator", "channel", "selection", "system", "attributes",
         "seq2seq_baseline", "retrofit", "loso", "scaling"]


def _load_gate(runs, name):
    p = os.path.join(runs, f"gate_{name}.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception as e:
        return {"passed": False, "detail": {"_corrupt": str(e)}}


def _ledger(runs):
    p = os.path.join(runs, "ledger.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _status(runs):
    p = os.path.join(runs, "campaign_status.tsv")
    if not os.path.exists(p):
        return []
    rows = [l.rstrip("\n").split("\t") for l in open(p) if l.strip()]
    return rows[1:] if rows and rows[0][0] == "stage" else rows


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.2e}"
        return f"{v:.{nd}f}"
    return str(v)


def _gate_label(g):
    if g is None:
        return "MISSING"
    d = g.get("detail", {}) or {}
    if g.get("override") or d.get("override"):
        return "OVERRIDDEN (logged)"
    if not g.get("passed"):
        return "FAILED"
    if d.get("partial_pass"):
        return (f"PARTIAL (resolution floor {d.get('resolution_floor_bits_per_token', '?')} bits/tok; "
                f"validated {d.get('validated_bits_per_token', '?')}; mode={d.get('gate_mode', 'point')}; "
                f"tightness={d.get('tightness_by_b')})")
    return "green"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = args.runs_dir
    out = args.out or os.path.join(runs, "REPORT.md")

    gates = {n: _load_gate(runs, n) for n in GATES}
    led = _ledger(runs)
    status = _status(runs)
    L = []
    w = L.append

    w(f"# Campaign report — `{os.path.abspath(runs)}`")
    w(f"_rendered {time.strftime('%Y-%m-%d %H:%M:%S')} from gate_*.json + ledger.jsonl; "
      f"{len(led)} ledger records_\n")

    # ------------------------------------------------------------ stage status
    w("## 1. What ran\n")
    if status:
        w("| stage | started | seconds | exit | status | gate | note |")
        w("|---|---|---:|---:|---|---|---|")
        for r in status:
            r = (r + [""] * 8)[:8]
            w(f"| {r[0]} | {r[1]} | {r[3]} | {r[4]} | **{r[5]}** | {r[6]} | {r[7][:90]} |")
    else:
        w("_(no campaign_status.tsv — stages were run by hand)_")
    w("")

    # ------------------------------------------------------------ gates
    w("## 2. Gates\n")
    w("| gate | status | key detail |")
    w("|---|---|---|")
    for n in GATES:
        g = gates[n]
        d = (g or {}).get("detail", {}) or {}
        key = ""
        if n == "build" and g:
            key = f"split={d.get('split_kind')} n_texts={d.get('n_unique_texts')} counts={d.get('counts')} hash={d.get('split_hash')}"
        elif n == "estimator" and g:
            sw = d.get("sweep", [])
            key = "; ".join(f"b={p['b_nominal']}→{_fmt(p['recovered_median'],3)}{'✓' if p['recovery_ok'] else '✗'}" for p in sw)
            key += (f" | **resolution floor = {d.get('resolution_floor_bits_per_token', 'none')} bits/tok** "
                    f"| KS p={_fmt(d.get('ks_p'),3)} | mode={d.get('mode')}")
        elif n == "channel" and g:
            key = (f"Δ̂I={_fmt(d.get('bits_per_token_derangement'))} bits/tok (derangement), "
                   f"{_fmt(d.get('bits_per_token_roll'))} (roll); {_fmt(d.get('bits_per_sentence'),3)} bits/sent; "
                   f"p_der={_fmt(d.get('p_derangement'),3)} p_roll={_fmt(d.get('p_roll'),3)}; "
                   f"**H1 primary endpoint: {'PASS' if d.get('H1_primary_endpoint_pass') else 'FAIL'}**; "
                   f"γ=0 anchor={_fmt(d.get('zero_gamma'))}; "
                   f"**{d.get('reading_vs_floor', 'floor: n/a')}**")
        elif n == "selection" and g:
            key = f"N*={d.get('matched_N_prediction')} from {_fmt(d.get('bits_per_sentence_input'),3)} bits/sent (E2)"
        elif n == "system" and g:
            key = (f"N*={d.get('N_star')} realized={_fmt(d.get('realized_bits'),4)} bits vs predicted "
                   f"{_fmt(d.get('predicted_bits'),4)} → matched-1.5×: {_fmt(d.get('matched_within_1p5x'))}; "
                   f"zero-anchor spread={_fmt(d.get('zero_anchor_max_spread'))}")
        w(f"| {n} | **{_gate_label(g)}** | {key} |")
    w("")

    # ------------------------------------------------------------ three channels
    # Sibling runs dirs by convention: <runs>, <runs>_gaze, <runs>_both (or runs_smoke_*).
    base_dir = runs.rstrip("/")
    sib = {"eeg": base_dir, "gaze": base_dir + "_gaze", "both": base_dir + "_both"}
    chan = {}
    for name, dd in sib.items():
        gg = _load_gate(dd, "channel")
        if not gg:
            continue
        det = gg.get("detail", {}) or {}
        ci = None
        if det.get("ci95_bits_per_token_derangement") and None not in det["ci95_bits_per_token_derangement"]:
            ci = tuple(float(x) for x in det["ci95_bits_per_token_derangement"])
        else:
            for r in _ledger(dd):
                if r.get("experiment") == "e2_channel" and r.get("arm") == "real_vs_derangement" \
                        and r.get("metric") == "dI_hat_bits_per_token" and r.get("ci_lo") is not None:
                    ci = (float(r["ci_lo"]), float(r["ci_hi"]))
        chan[name] = {"bits_tok": det.get("bits_per_token_derangement"),
                      "bits_sent": det.get("bits_per_sentence"),
                      "tokens": det.get("mean_observed_tokens"),
                      "p": det.get("p_derangement"), "H1": det.get("H1_primary_endpoint_pass"),
                      "ci": ci, "floor": det.get("reading_vs_floor")}
    if len(chan) >= 2:
        w("## 3a. The three evidence channels, one estimator\n")
        w("_Same prior, same nulls, same pre-registration; only the evidence array differs. "
          "`EEG | gaze` is the increment I(both) − I(gaze): what EEG adds beyond the eyes. "
          "Its CI is the two bootstrap CIs combined in quadrature (independent-run approximation)._\n")
        w("| channel | Δ̂I bits/token | 95% CI | bits/sentence | p (derangement) | H1 | vs resolution floor |")
        w("|---|---:|---|---:|---:|---|---|")
        for name in ("gaze", "eeg", "both"):
            c = chan.get(name)
            if not c:
                w(f"| {name} | _not run_ | | | | | |"); continue
            ci = f"[{_fmt(c['ci'][0])}, {_fmt(c['ci'][1])}]" if c["ci"] else "—"
            w(f"| **{name}** | {_fmt(c['bits_tok'])} | {ci} | {_fmt(c['bits_sent'],3)} | "
              f"{_fmt(c['p'],3)} | {'PASS' if c['H1'] else 'FAIL'} | {(c['floor'] or '')[:60]} |")
        if chan.get("both") and chan.get("gaze") and chan["both"]["bits_tok"] is not None:
            inc = float(chan["both"]["bits_tok"]) - float(chan["gaze"]["bits_tok"])
            toks = float(chan["both"].get("tokens") or 0)
            ci_txt = "—"
            if chan["both"]["ci"] and chan["gaze"]["ci"]:
                hb = (chan["both"]["ci"][1] - chan["both"]["ci"][0]) / 2
                hg = (chan["gaze"]["ci"][1] - chan["gaze"]["ci"][0]) / 2
                h = math.sqrt(hb * hb + hg * hg)
                ci_txt = f"[{_fmt(inc - h)}, {_fmt(inc + h)}]"
            w(f"| **EEG \\| gaze** (both − gaze) | {_fmt(inc)} | {ci_txt} | {_fmt(inc * toks, 3)} | — | — | "
              f"{'CI includes 0 → no EEG increment beyond gaze' if ci_txt != '—' and (inc - h) <= 0 <= (inc + h) else ''} |")
        w("")

    # ------------------------------------------------------------ hypotheses
    w("## 3. Pre-registered hypotheses\n")
    gch, gsys = gates["channel"], gates["system"]
    dch = (gch or {}).get("detail", {}) or {}
    dsy = (gsys or {}).get("detail", {}) or {}
    w("| id | statement | outcome |")
    w("|---|---|---|")
    h1 = "PASS" if dch.get("H1_primary_endpoint_pass") else ("FAIL" if gch else "not run")
    w(f"| **H1 (primary)** | Δ̂I > 0 on EEG-present tokens under BOTH nulls | **{h1}** "
      f"(p_der={_fmt(dch.get('p_derangement'),3)}, p_roll={_fmt(dch.get('p_roll'),3)}) |")
    h2 = "not run" if not gsys else ("PASS" if dsy.get("matched_within_1p5x") else "FAIL")
    w(f"| H2 | realized selection bits within 1.5× of E2-predicted at N* | **{h2}** "
      f"(N*={dsy.get('N_star', '—')}, predicted={_fmt(dsy.get('predicted_bits'))}, "
      f"realized={_fmt(dsy.get('realized_bits'))}"
      + ("; no N reaches 50% predicted accuracy — N* is a placeholder" if dsy.get("matched_N_meets_target") is False else "")
      + ") |")
    itr = (dsy.get("itr") or {})
    # pre-registered primary denominator is T2 (trial time); fall back to the most
    # conservative available one and SAY so
    prim = next((k for k in ("nykopp_T2_trial", "nykopp_T1_reading", "nykopp_T3_nominal") if k in itr), None)
    real_acc = ((dsy.get("arms") or {}).get("real") or {}).get("acc")
    above_chance = real_acc is not None and dsy.get("N_star") and real_acc > 1.0 / float(dsy["N_star"])
    h3 = ("not run" if not gsys else
          ("PASS" if prim and above_chance and 1 <= itr[prim] <= 60 else "FAIL"))
    w(f"| H3 | Nykopp ITR in 1–60 bits/min (primary denominator T2) | **{h3}** "
      f"({prim or '—'}={_fmt(itr.get(prim), 3) if prim else '—'}; all: "
      f"{', '.join(k + '=' + _fmt(v, 3) for k, v in sorted(itr.items()) if k.startswith('nykopp')) or '—'}) |")
    h4 = "not run" if not gsys else ("PASS" if (dsy.get("partial_auc") or 0) > 0.5 and dsy.get("dominates_cheap_confidence") else "FAIL")
    w(f"| H4 | partial confidence AUC > 0.5 and beats cheap baseline | **{h4}** "
      f"(pAUC={_fmt(dsy.get('partial_auc'),3)}; dominates cheap baseline: {_fmt(dsy.get('dominates_cheap_confidence'))}) |")
    tc = dsy.get("tost_collapse") or {}
    td = dsy.get("tost_detail") or {}
    powered = all(v.get("powered", True) for v in td.values()) if td else True
    h5 = "not run" if not gsys else ("PASS" if tc and all(tc.values()) else f"FAIL ({sum(1 for v in tc.values() if v)}/{len(tc)} arms collapsed)")
    if gsys and not powered:
        mm = max(v.get("min_resolvable_margin", 0) for v in td.values())
        h5 += (f" — UNDERPOWERED: {(dsy.get('pool_gate') or {}).get('n_pools', '?')} pools cannot "
               f"establish ±2pp equivalence (smallest resolvable margin {mm:.3f})")
    w(f"| H5 | every noise arm collapses to chance (TOST ±2pp) | **{h5}** |")
    # additional-seed replications (never overwrite the primary run)
    reps = sorted(fn for fn in os.listdir(runs) if fn.startswith("gate_system_seed") and fn.endswith(".json")) if os.path.isdir(runs) else []
    if reps:
        w("")
        w("_E4 seed replications (same prereg hypotheses, own prereg sub-dir):_\n")
        w("| run | N | real acc | zeroed acc | realized bits | partial AUC |")
        w("|---|---:|---:|---:|---:|---:|")
        for fn in reps:
            d = (_load_gate(runs, fn[len("gate_"):-len(".json")]) or {}).get("detail", {}) or {}
            arms = d.get("arms") or {}
            w(f"| {fn[len('gate_'):-len('.json')]} | {d.get('N_star', '—')} | {_fmt((arms.get('real') or {}).get('acc'))} | "
              f"{_fmt((arms.get('zeroed') or {}).get('acc'))} | {_fmt(d.get('realized_bits'))} | {_fmt(d.get('partial_auc'), 3)} |")
    w("")

    # ------------------------------------------------------------ headline numbers
    w("## 4. Numbers that may be quoted\n")
    w("_Every row below is a ledger record. `gates` is what was actually green when the "
      "number was produced (`estimator_partial` = calibration incomplete; `*_OVERRIDDEN` = "
      "a logged override; an absent gate = it was not green). `objective` names the "
      "training loss; only `raw_dI_ascent` is the pre-registered one._\n")
    want = collections.OrderedDict([
        ("e2_channel", ["dI_hat_bits_per_token", "bits_per_sentence"]),
        ("e3_selection", None),
        ("e4_system", ["selection_acc_Nstar", "bits_realized_per_sentence", "itr_nykopp", "aurc", "partial_confidence_auc", "selectedtext_bleu4"]),
        ("e6_seq2seq_baseline", ["bleu1", "bleu4", "chrf", "bertscore_f1"]),
        ("e9_retrofit", None),
        ("e7_loso", None),
        ("e8_scaling", None),
    ])
    by_exp = collections.defaultdict(list)
    for r in led:
        by_exp[r.get("experiment", "?")].append(r)
    for exp, keys in want.items():
        rows = by_exp.get(exp, [])
        if not rows:
            continue
        if keys:
            rows = [r for r in rows if any(str(r.get("metric", "")).startswith(k) for k in keys)]
        if not rows:
            continue
        w(f"### {exp}\n")
        w("| metric | arm | value | 95% CI | p | seed | evidence | objective | gates |")
        w("|---|---|---:|---|---:|---:|---|---|---|")
        for r in sorted(rows, key=lambda r: (str(r.get("metric")), str(r.get("arm")), r.get("seed") or 0)):
            ci = "—"
            if r.get("ci_lo") is not None and r.get("ci_hi") is not None:
                ci = f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}]"
            w(f"| {r.get('metric')} | {r.get('arm','')} | {_fmt(r.get('value'))} | {ci} | "
              f"{_fmt(r.get('p'),3)} | {r.get('seed','')} | {r.get('evidence','eeg')} | "
              f"{r.get('objective','')} | "
              f"{', '.join(r.get('gates_passed') or []) or '(none)'} |")
        w("")

    # ------------------------------------------------------------ reading guide
    w("## 5. How to read this honestly\n")
    w("- **Δ̂I (bits/token)** is a null-corrected lower bound on information the estimator can "
      "*recover*; it is not the brain's capacity. If the estimator gate is PARTIAL or FAILED, "
      "the E2 number is uncalibrated and must be described as such.")
    w("- **If H1 fails**, the honest headline is *\"indistinguishable from zero\"*, and the CI "
      "on the `dI_hat_bits_per_token` row is the quantity to report — not the point estimate.")
    w("- **Selected-text BLEU/BERTScore** (`selectedtext_*`) are computed on the chosen candidate "
      "and are a near-linear function of selection accuracy. They are comparability rows. They "
      "are not generation metrics and nothing in this pipeline generates text except E6.")
    w("- **E6 baseline rows** are free-running generation; identical metrics across real / "
      "shuffled / noise arms mean the model's output is input-independent.")
    w("- **`evidence=gaze` rows** come from the eye-movement channel (fixation count and "
      "durations per word), measured under the identical estimator and nulls in its own "
      "runs dir. They are the comparison for the EEG rows, never a substitute: if gaze "
      "shows bits and EEG does not, the honest sentence is *eye movements carry X bits; "
      "EEG adds nothing beyond them on these features*.")
    w("- **Any number missing from §4 is not reportable.** If you need it, it has to land in "
      "the ledger with a gate, not in a table by hand.")
    w("")
    dev = os.path.join(runs, "prereg", "deviations.log")
    if os.path.exists(dev):
        w("## 6. Deviations log (verbatim)\n")
        w("```")
        w(open(dev).read().rstrip())
        w("```")
    ta = os.path.join(runs, "prereg", "test_access.log")
    if os.path.exists(ta):
        w("## 7. Test-split access log (verbatim)\n")
        w("```")
        w(open(ta).read().rstrip())
        w("```")

    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w").write(txt)
    print(txt)
    print(f"[report written: {out}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
