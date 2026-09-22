#!/usr/bin/env bash
# v2 instrument (2026-09-17, docs/FINDINGS_20260917.md §5): InfoNCE estimand + objective,
# nonlinear rank-16 tilt over the prior embedding, prior-matched E1 with the lower-bound
# gate. Same data build (runs/data), own runs dirs (runs_v2*), no baselines.
set -uo pipefail; cd "$(dirname "$0")"
export HF_HUB_DISABLE_PROGRESS_BARS=1 NLTK_DATA=$HOME/nltk_data PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
V=$HOME/venvs/cprd/bin/python; PRIOR="--prior causal_lm --prior-model gpt2-large"
INST="--objective nce --free-tilt-rank 16 --estimand nce --gamma-warmup-value 1.0 --gamma-warmup-epochs 20"
TR="--epochs 20 --lr 1e-3"; EV="--n-perm 1000 --n-boot 10000"
mkdir -p runs_v2/logs; cp runs/gate_build.json runs_v2/
run () { local id=$1 dir=$2; shift 2; local t0=$(date +%s); mkdir -p $dir/logs
  timeout --signal=INT --kill-after=120 28800 "$@" > $dir/logs/$id.log 2>&1; local rc=$?
  local st=ok; [ $rc -ne 0 ] && st=failed
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$id" "$(date -d @$t0 +%FT%T)" "$(date +%FT%T)" "$(( $(date +%s)-t0 ))" "$rc" "$st" "-" "run_v2.sh" >> $dir/campaign_status.tsv
  echo "== $id rc=$rc ($(( $(date +%s)-t0 ))s) $(date +%FT%T)"; }
run e1_estimator runs_v2 $V experiments/e1_estimator.py $PRIOR --runs-dir runs_v2 --data-dir runs/data --device cuda $INST --sweep 0,0.25,1,2 --seeds 0 --n-sent 2000 --n-null 20 --epochs 20 --lr 1e-3 --n-perm 300 --synth-tokens prior --gate-mode lower_bound
for EV_ in gaze eeg both; do
  D=runs_v2_$EV_; [ "$EV_" = eeg ] && D=runs_v2      # report.py pairs <runs>, <runs>_gaze, <runs>_both
  mkdir -p $D/logs; cp runs/gate_build.json $D/; cp runs_v2/gate_estimator.json $D/ 2>/dev/null
  C="$PRIOR --evidence $EV_ --runs-dir $D --data-dir runs/data --device cuda $INST"
  run e2_channel $D $V experiments/e2_channel.py $C $TR $EV
  run e3_selection $D $V experiments/e3_selection.py $C $TR --n-grid 2,4,8,16,32,64
  [ "$($V - $D <<'PY'
import json,sys,os; p=os.path.join(sys.argv[1],'gate_selection.json'); print('green' if os.path.exists(p) and json.load(open(p)).get('passed') else 'failed')
PY
)" = green ] || $V experiments/override_gate.py $D selection --reason "text-only pool attack at N<=16 (pool construction limitation, identical across channels); E4 exploratory, records carry selection_OVERRIDDEN"
  run e4_system $D $V experiments/e4_system.py $C $TR $EV
  $V experiments/report.py --runs-dir $D --out $D/REPORT.md > /dev/null 2>&1 || true
done
$V experiments/report.py --runs-dir runs_v2 --out runs_v2/REPORT.md > /dev/null 2>&1 || true
echo "== V2 ALL DONE $(date +%FT%T)"
