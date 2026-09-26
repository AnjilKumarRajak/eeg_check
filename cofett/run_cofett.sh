
set -u
RAW=${1:?usage: run_cofett.sh <RAW_DIR> <WORK_DIR>}; W=${2:?usage: run_cofett.sh <RAW_DIR> <WORK_DIR>}
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(dirname "$HERE")"
PY=${PYTHON:-python}; DEV=${DEVICE:-cuda}; PM=${PRIOR_MODEL:-Qwen/Qwen2.5-0.5B}
E1_GATE=${E1_GATE:-$REPO/runs_v2/gate_estimator.json}
[ -f "$E1_GATE" ] || { echo "missing E1 gate $E1_GATE (run the ZuCo E1 stage first, or set E1_GATE)"; exit 1; }
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1
mkdir -p "$W/feat" "$W/feat_ses234" "$W/logs"; LOG="$W/logs"

for edf in $(find "$RAW" -name "*_eeg.edf" | sort); do
  n=$(basename "$edf" _eeg.edf); case $n in *_ses-01_*) O="$W/feat";; *) O="$W/feat_ses234";; esac
  [ -f "$O/$n.npz" ] || $PY "$HERE/extract_features.py" --edf "$edf" --events "${edf%_eeg.edf}_events.tsv" \
      --stim-dir "$HERE/stim" --out "$O/$n.npz" >> "$LOG/extract.log" 2>&1
done
F1="$W/feat"; F16="$W/feat,$W/feat_ses234"   # session 1 only / all four sessions

  local name=$1; shift; local R="$W/runs_$name"; mkdir -p "$R"
  [ -f "$R/gate_build.json" ] || $PY "$HERE/e0_build_cofett.py" --prior-model "$PM" --out "$W/data_built_$name" \
      --runs-dir "$R" "$@" > "$LOG/build_$name.log" 2>&1
  cp "$E1_GATE" "$R/gate_estimator.json"; }
for PH in reading recall; do
  build $PH                  --feat-dir "$F1"  --phase $PH                                     # unseen sentences, pooled
  build ${PH}_sub01          --feat-dir "$F1"  --phase $PH --only-subject sub-01              # unseen, per participant
  build ${PH}_sub02          --feat-dir "$F1"  --phase $PH --only-subject sub-02
  build repeat_$PH           --feat-dir "$F1"  --phase $PH --split-kind repeat                # session-1 repeated split
  build repeat16_$PH         --feat-dir "$F16" --phase $PH --split-kind repeat16              # COFETT scheme, pooled
  build repeat16_${PH}_sub01 --feat-dir "$F16" --phase $PH --split-kind repeat16 --only-subject sub-01
  build repeat16_${PH}_sub02 --feat-dir "$F16" --phase $PH --split-kind repeat16 --only-subject sub-02
done
for A in 1.0 0.3 0.1; do build spike_a$A --feat-dir "$F1" --phase reading --spike-alpha $A; done  # injected controls

PRIOR_FP32=${PRIOR_FP32:-$W/prior_qwen25_05b_fp32}
[ -f "$PRIOR_FP32/config.json" ] || $PY "$HERE/make_fp32_prior.py" "$PRIOR_FP32" --model "$PM"
cd "$REPO"
INST="--objective nce --free-tilt-rank 16 --estimand nce --gamma-warmup-value 1.0 --gamma-warmup-epochs 20"
INST_NR="--objective nce --estimand nce --gamma-warmup-value 1.0 --gamma-warmup-epochs 20"
TR="--epochs 20 --lr 1e-3"; EV="--n-perm 1000 --n-boot 10000"; EVC="--n-perm 300 --n-boot 10000"
P="--prior causal_lm --prior-model $PRIOR_FP32 --evidence eeg --device $DEV"
step () { local id=$1; shift; "$@" > "$LOG/$id.log" 2>&1; echo "== $id rc=$?"; }
e2 () {  # $1 = run name, $2 = data name, rest = extra args
  local R="$W/runs_$1" D="$W/data_built_$2"; shift 2
  [ -f "$R/gate_channel.json" ] || step "e2_$(basename $R)" $PY experiments/e2_channel.py $P --runs-dir "$R" --data-dir "$D" $TR "$@"; }
newrun () { mkdir -p "$W/runs_$1"; cp "$E1_GATE" "$W/runs_$1/gate_estimator.json"; cp "$W/runs_reading/gate_build.json" "$W/runs_$1/"; }

# 14 real-EEG conditions (Table: COFETT bits/token), 1,000 permutation draws
for PH in reading recall; do
  for T in $PH ${PH}_sub01 ${PH}_sub02 repeat_$PH repeat16_$PH repeat16_${PH}_sub01 repeat16_${PH}_sub02; do e2 $T $T $INST $EV; done
done
# injected text-dependent controls, 300 draws
for A in 1.0 0.3 0.1; do e2 spike_a$A spike_a$A $INST $EVC; done
# robustness (pooled reading, unseen sentences), 300 draws: evidence window, tilt capacity
for WIN in 0 3; do newrun reading_w$WIN; e2 reading_w$WIN reading $INST $EVC --window $WIN; done
for RK in 4 8 32; do newrun reading_r$RK; e2 reading_r$RK reading $INST_NR --free-tilt-rank $RK $EVC; done
newrun reading_bigtilt;    e2 reading_bigtilt reading $INST_NR --free-tilt-rank 64 --free-tilt-hidden 256 $EVC
newrun reading_lineartilt; e2 reading_lineartilt reading $INST_NR --free-tilt-rank 0 $EVC

A () { echo "$P --runs-dir $W/runs_$1 --data-dir $W/data_built_$1 $INST"; }
OVR="text-only pool attack fails at some pool sizes (small candidate sets); E4 exploratory, same treatment as the ZuCo runs"
for PH in reading recall; do R="$W/runs_$PH"
  [ -f "$R/gate_selection.json" ] || step e3_$PH $PY experiments/e3_selection.py $(A $PH) $TR --n-grid 2,4,8,16,32
  $PY -c "import json,sys;sys.exit(0 if json.load(open('$R/gate_selection.json')).get('passed') else 1)" 2>/dev/null \
    || $PY experiments/override_gate.py "$R" selection --reason "$OVR"
  [ -f "$R/gate_system.json" ]    || step e4_$PH  $PY experiments/e4_system.py    $(A $PH) $TR $EV     # one-shot test session
  [ -f "$R/gate_breakdown.json" ] || step e10_$PH $PY experiments/e10_breakdown.py $(A $PH) $TR $EV
done
R="$W/runs_reading"
[ -f "$R/gate_loso.json" ]        || step e7_loso    $PY experiments/e7_loso.py    $(A reading) --epochs 15 --lr 1e-3 --subject-prefix sub- --n-list 4,32
[ -f "$R/gate_scaling.json" ]     || step e8_scaling $PY experiments/e8_scaling.py $(A reading) --epochs 15 --lr 1e-3 --n-pool 32
[ -f "$R/gate_sensitivity.json" ] || step e12_gain   $PY experiments/e12_sensitivity.py $(A reading) $TR --m-draws 120 --gamma-grid 1,2 --uscale-grid ""
echo "COFETT ALL DONE"
