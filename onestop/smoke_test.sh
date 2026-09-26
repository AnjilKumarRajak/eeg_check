
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(dirname "$HERE")"; cd "$REPO"
W=${1:-$REPO/onestop_smoke}; PY=${PYTHON:-python}; DEV=${DEVICE:-cuda}; PRIOR=${PRIOR:-distilgpt2}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1
mkdir -p "$W"
$PY - "$W/ia_report.csv" <<'EOF'
import sys, numpy as np, pandas as pd
rng = np.random.default_rng(0)
paras = ["The city council approved a new plan to build more parks near the river this year .",
         "Scientists found that the small birds travel thousands of miles every winter to find food .",
         "The company said its new phone would go on sale next month in several countries .",
         "Heavy rain caused flooding in many towns and forced families to leave their homes .",
         "The museum opened an exhibition about the history of flight and early aircraft .",
         "Farmers are worried that the dry summer will reduce the size of the harvest .",
         "A new study shows that regular exercise can improve sleep and reduce stress .",
         "The football team won the championship after a long and difficult season .",
         "Engineers are testing a train that could travel faster than any before it .",
         "The library will stay open late during exams so that students can study ."]
rows = []
for p in range(12):
    for t, text in enumerate(paras):
        for lvl in ("Adv", "Ele"):
            words = text.split() if lvl == "Adv" else ["Simply"] + text.split()[:-2] + ["."]
            for i, w in enumerate(words):
                n = int(rng.integers(0, 3))
                rows.append(dict(participant_id=f"p{p}", TRIAL_INDEX=t * 2 + (lvl == "Ele"), IA_ID=i + 1, IA_LABEL=w,
                                 IA_FIXATION_COUNT=n, IA_FIRST_FIXATION_DURATION=200 * (n > 0) + rng.integers(0, 50),
                                 IA_FIRST_RUN_DWELL_TIME=250 * n, IA_DWELL_TIME=300 * n, IA_REGRESSION_PATH_DURATION=320 * n,
                                 IA_SKIP=int(n == 0), article_id=t, paragraph_id=1, difficulty_level=lvl,
                                 practice_trial=False, repeated_reading_trial=False))
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
print("synthetic interest-area report written")
EOF
R="$W/runs"; mkdir -p "$R"
$PY "$HERE/e0_build_onestop.py" --csv "$W/ia_report.csv" --out "$W/data" --runs-dir "$R" --prior-model "$PRIOR"
A="--prior causal_lm --prior-model $PRIOR --evidence gaze --device $DEV --data-dir $W/data --objective nce
   --free-tilt-rank 16 --estimand nce --gamma-warmup-value 1.0 --gamma-warmup-epochs 20 --epochs 1 --lr 1e-3"
# a stage exits 1 when its pre-registered gate does not pass (expected on synthetic data);
# the smoke test only requires that every stage runs and writes its gate file
$PY experiments/e2_channel.py $A --runs-dir "$R" --n-perm 20 --n-boot 50 || true
$PY experiments/e13_word_level.py $A --runs-dir "$R" --n-perm 20 --n-boot 50 || true
$PY experiments/e3_selection.py $A --runs-dir "$R" --n-grid 2,4 || true
S="$W/runs_struct"; mkdir -p "$S"; cp "$R/gate_build.json" "$S/"
$PY experiments/e2_channel.py $A --runs-dir "$S" --n-perm 20 --n-boot 50 --gaze-control structure_only || true
for g in "$R/gate_build.json" "$R/gate_channel.json" "$R/gate_wordlevel.json" "$R/gate_selection.json" "$S/gate_channel.json"; do
  [ -f "$g" ] || { echo "ONESTOP SMOKE TEST FAILED: missing $g"; exit 1; }
done
echo "ONESTOP SMOKE TEST PASSED"
