#!/usr/bin/env bash
# Quick end-to-end check of the COFETT path without downloading recordings: writes synthetic
# per-character feature files in the exact format of extract_features.py (real sentence lists,
# random 840-d features), then runs the adapter and the unchanged framework (E2, E3) on a small
# subset with the float32 Qwen2.5-0.5B prior. Checks that the code runs; the numbers are meaningless.
#
#   bash cofett/smoke_test.sh [WORK_DIR]        (env: PYTHON, DEVICE=cuda|cpu, PRIOR_FP32)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(dirname "$HERE")"; cd "$REPO"
W=${1:-$REPO/cofett_smoke}; PY=${PYTHON:-python}; DEV=${DEVICE:-cuda}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONIOENCODING=utf-8 PYTHONUTF8=1
mkdir -p "$W/feat" "$W/runs"
$PY - "$HERE/stim" "$W/feat" <<'EOF'
import os, sys, numpy as np
sys.path.insert(0, sys.argv[1] + "/..")
from extract_features import BANDS, load_list
stim, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
for sub in ("sub-01", "sub-02"):
    for r in (1, 2):
        sents = load_list(os.path.join(stim, f"text1-{r}.xlsx"))[:40]
        maxc = max(len(s) for s in sents)
        f = rng.normal(size=(len(sents), maxc, 8 * 105)).astype(np.float32)
        keep = np.zeros((len(sents), maxc), bool)
        for i, s in enumerate(sents): keep[i, :len(s)] = True
        f[~keep] = np.nan
        run = f"{sub}_ses-01_task-para1_run-0{r}"
        np.savez(os.path.join(out, run + ".npz"), feats_read=f, feats_recall=f.copy(), keep_chars=keep,
                 n_chars=np.array([len(s) for s in sents]), sentences=np.array(sents, dtype=object), channels=np.array([f"c{i}" for i in range(105)]),
                 bands=np.array([b[0] for b in BANDS]), list_file=f"text1-{r}.xlsx", run=run, n_mismatch=0)
print("synthetic features written")
EOF
PRIOR_FP32=${PRIOR_FP32:-$W/prior_fp32}
[ -f "$PRIOR_FP32/config.json" ] || $PY "$HERE/make_fp32_prior.py" "$PRIOR_FP32"
$PY "$HERE/e0_build_cofett.py" --prior-model "$PRIOR_FP32" --feat-dir "$W/feat" --out "$W/data" --runs-dir "$W/runs" --phase reading
A="--prior causal_lm --prior-model $PRIOR_FP32 --evidence eeg --device $DEV --objective nce --free-tilt-rank 16 --estimand nce
   --gamma-warmup-value 1.0 --gamma-warmup-epochs 20 --runs-dir $W/runs --data-dir $W/data --epochs 1 --lr 1e-3"
# a stage exits 1 when its pre-registered gate does not pass (expected on random features);
# the smoke test only requires that every stage runs and writes its gate file
$PY experiments/e2_channel.py $A --n-perm 20 --n-boot 50 || true
$PY experiments/e3_selection.py $A --n-grid 2,4 || true
for g in build channel selection; do
  [ -f "$W/runs/gate_$g.json" ] || { echo "COFETT SMOKE TEST FAILED: no gate_$g.json"; exit 1; }
done
echo "COFETT SMOKE TEST PASSED"
