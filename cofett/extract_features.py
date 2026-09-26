#!/usr/bin/env python3
"""COFETT (OpenNeuro ds006317) run -> per-character 8-band EEG power, ZuCo-style.

Per trial the events carry three triggers (parallel-port code + 65279):
    65329 (50)  reading onset  : characters highlighted one at a time, 0.4 s each,
                                 punctuation skipped without consuming time
    65379 (100) recall onset   : silent verbatim recall, nominal 0.4*(n+1) s
    65381 (102) rest
Sentences are presented in xlsx list order (verified: character counts match the
inter-trigger intervals on every run). Features mirror cprd/build.py: 8 bands
(theta1..gamma2, ZuCo definitions) x 105 channels = 840-d, mean Hilbert power
over each character's window, log-transformed. The 105 channels are the 124 EEG
channels minus the 19 most peripheral (ring '9'/'10' sites), fixed list written
to the output so every run uses the same set.

Output <out>/<run>.npz : feats_read (n_trials, max_chars, 840) NaN-padded,
                         feats_recall (same, nominal uniform alignment),
                         n_chars, sentences (unicode), keep_chars (mask), channels
"""
from __future__ import annotations
import argparse, json, os, string, sys
import numpy as np
import openpyxl

BANDS = [("t1", 4, 6), ("t2", 6.5, 8), ("a1", 8.5, 10), ("a2", 10.5, 13),
         ("b1", 13.5, 18), ("b2", 18.5, 30), ("g1", 30.5, 40), ("g2", 40, 49.5)]
PUNCT = set(string.punctuation + "，。？！：；、“”‘’")
CHAR_DUR = 0.4
FS_OUT = 250.0


def load_list(xlsx):
    ws = openpyxl.load_workbook(xlsx, read_only=True).worksheets[0]
    return [str(r[0]) for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]


def list_for(run_name, stim_dir):
    task = run_name.split("task-")[1].split("_")[0]
    run = int(run_name.split("run-")[1][:2])
    f = f"text1-{run}.xlsx" if task == "para1" else "text2.xlsx"
    return f, load_list(os.path.join(stim_dir, f))


def pick_channels(names, n_keep=105):
    ring = [n for n in names if n.endswith("9") or n.endswith("10") or n.endswith("9h") or n.endswith("10h")]
    keep = [n for n in names if n not in ring]
    if len(keep) > n_keep:                       # drop remaining from the ring-adjacent end, deterministic
        keep = keep[:n_keep]
    assert len(keep) == n_keep, (len(keep), len(ring))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--stim-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import mne
    mne.set_log_level("ERROR")
    run_name = os.path.basename(args.edf).replace("_eeg.edf", "")
    list_file, sents = list_for(run_name, args.stim_dir)

    raw = mne.io.read_raw_edf(args.edf, preload=True)
    eeg_names = [n for n, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    # channels.tsv marks 124 EEG + 2 EOG (+1 trigger); the EDF may type them all 'eeg'
    chan_tsv = args.edf.replace("_eeg.edf", "_channels.tsv")
    if os.path.exists(chan_tsv):
        rows = [l.split("\t") for l in open(chan_tsv, encoding="utf-8-sig").read().strip().split("\n")[1:]]
        eeg_names = [r[0] for r in rows if r[1].strip().upper() == "EEG" and r[0] in raw.ch_names]
    keep = pick_channels(eeg_names)
    raw.pick(keep)
    raw.notch_filter(50.0)
    raw.resample(FS_OUT)
    data = raw.get_data() * 1e6                                  # (C, N) microvolts (ZuCo feature scale)
    C, N = data.shape
    fs = FS_OUT
    # band power envelopes (C, N) per band -> stacked (8, C, N)
    from scipy.signal import butter, sosfiltfilt, hilbert
    env = np.empty((len(BANDS), C, N), dtype=np.float32)
    for bi, (_, lo, hi) in enumerate(BANDS):
        sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
        x = sosfiltfilt(sos, data, axis=1)
        env[bi] = np.abs(hilbert(x, axis=1)).astype(np.float32) ** 2

    ev = [l.split("\t") for l in open(args.events, encoding="utf-8-sig").read().strip().split("\n")[1:]]
    ev = [(float(r[0]), int(r[3])) for r in ev]
    on = [t for t, v in ev if v == 65329]
    rec = [t for t, v in ev if v == 65379]
    rest = [t for t, v in ev if v == 65381]
    n_tr = min(len(on), len(rec), len(rest), len(sents))
    maxc = max(len(s) for s in sents[:n_tr])
    F = len(BANDS) * C
    feats_read = np.full((n_tr, maxc, F), np.nan, dtype=np.float32)
    feats_rec = np.full((n_tr, maxc, F), np.nan, dtype=np.float32)
    keep_chars = np.zeros((n_tr, maxc), dtype=bool)
    n_chars = np.zeros(n_tr, dtype=np.int32)
    mismatch = 0

    def window_power(t0, t1):
        a, b = int(round(t0 * fs)), int(round(t1 * fs))
        if b <= a or b > N:
            return None
        return np.log(env[:, :, a:b].mean(axis=2) + 1e-12).reshape(-1)   # (8*C,)

    for i in range(n_tr):
        s = sents[i]
        chars = list(s)
        nonp = [k for k, ch in enumerate(chars) if ch not in PUNCT]
        n_chars[i] = len(chars)
        exp = len(nonp) * CHAR_DUR
        if abs((rec[i] - on[i]) - exp) > 0.6:
            mismatch += 1
            continue
        # reading: exact schedule
        for j, k in enumerate(nonp):
            w = window_power(on[i] + j * CHAR_DUR, on[i] + (j + 1) * CHAR_DUR)
            if w is not None:
                feats_read[i, k] = w; keep_chars[i, k] = True
        # recall: nominal uniform pacing over the recall interval (rest - recall onset)
        span = rest[i] - rec[i]
        step = span / max(1, len(nonp) + 1)
        for j, k in enumerate(nonp):
            w = window_power(rec[i] + j * step, rec[i] + (j + 1) * step)
            if w is not None:
                feats_rec[i, k] = w
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, feats_read=feats_read, feats_recall=feats_rec,
                        keep_chars=keep_chars, n_chars=n_chars,
                        sentences=np.array(sents[:n_tr], dtype=object),
                        channels=np.array(keep), bands=np.array([b[0] for b in BANDS]),
                        list_file=list_file, run=run_name, n_mismatch=mismatch)
    print(f"{run_name}: {n_tr} trials, {mismatch} timing mismatches skipped, F={F}, "
          f"chars kept {keep_chars.sum()} -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
