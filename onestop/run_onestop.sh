
set -u
CSV=${1:?usage: run_onestop.sh <IA_REPORT_CSV> <WORK_DIR>}; W=${2:?usage: run_onestop.sh <IA_REPORT_CSV> <WORK_DIR>}
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(dirname "$HERE")"; cd "$REPO"
PY=${PYTHON:-python}; DEV=${DEVICE:-cuda}
E1_GATE=${E1_GATE:-$REPO/runs_v2/gate_estimator.json}
[ -f "$E1_GATE" ] || { echo "missing E1 gate $E1_GATE (run the ZuCo E1 stage first, or set E1_GATE)"; exit 1; }
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1
mkdir -p "$W/logs"; LOG="$W/logs"

# 1. build: text-disjoint split (7,618 / 960 / 1,140 readings, 126 texts)
R="$W/runs_gaze"; mkdir -p "$R"
[ -f "$R/gate_build.json" ] || $PY "$HERE/e0_build_onestop.py" --csv "$CSV" --out "$W/data" --runs-dir "$R" > "$LOG/build.log" 2>&1
cp "$E1_GATE" "$R/gate_estimator.json"

# 2. measurement with the ZuCo gaze instrument
P="--prior causal_lm --prior-model gpt2-large --evidence gaze --device $DEV --data-dir $W/data"
INST="--objective nce --free-tilt-rank 16 --estimand nce --gamma-warmup-value 1.0 --gamma-warmup-epochs 20"
TR="--epochs 20 --lr 1e-3"
step () { local id=$1; shift; "$@" > "$LOG/$id.log" 2>&1; echo "== $id rc=$?"; }
[ -f "$R/gate_channel.json" ]   || step e2_gaze    $PY experiments/e2_channel.py    $P --runs-dir "$R" $INST $TR --n-perm 1000 --n-boot 10000
[ -f "$R/gate_wordlevel.json" ] || step e13_words  $PY experiments/e13_word_level.py $P --runs-dir "$R" $INST $TR --n-perm 1000 --n-boot 10000
[ -f "$R/gate_selection.json" ] || step e3_select  $PY experiments/e3_selection.py  $P --runs-dir "$R" $INST $TR --n-grid 2,4,8,16

# 3. structure-only control: word grouping and sentence length kept, gaze content replaced
S="$W/runs_gaze_struct"; mkdir -p "$S"; cp "$R/gate_build.json" "$S/"; cp "$E1_GATE" "$S/gate_estimator.json"
[ -f "$S/gate_channel.json" ]   || step e2_struct  $PY experiments/e2_channel.py    $P --runs-dir "$S" $INST $TR --n-perm 300 --n-boot 10000 --gaze-control structure_only
echo "ONESTOP ALL DONE"
