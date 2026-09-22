
set -uo pipefail; cd "$(dirname "$0")"
export HF_HUB_DISABLE_PROGRESS_BARS=1 NLTK_DATA=$HOME/nltk_data PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
V=$HOME/venvs/cprd/bin/python; PRIOR="--prior causal_lm --prior-model gpt2-large"
INST="--objective nce --free-tilt-rank 16 --estimand nce --gamma-warmup-value 1.0 --gamma-warmup-epochs 20"
TR="--epochs 20 --lr 1e-3"; EV="--n-perm 1000 --n-boot 10000"
until grep -q "V2 ALL DONE" v2_console.log; do sleep 300; done
run () { local id=$1 dir=$2; shift 2; local t0=$(date +%s)
  timeout --signal=INT --kill-after=120 28800 "$@" > $dir/logs/$id.log 2>&1; local rc=$?
  local st=ok; [ $rc -ne 0 ] && st=failed
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$id" "$(date -d @$t0 +%FT%T)" "$(date +%FT%T)" "$(( $(date +%s)-t0 ))" "$rc" "$st" "-" "run_v2_followup.sh" >> $dir/campaign_status.tsv
  echo "== $id rc=$rc ($(( $(date +%s)-t0 ))s) $(date +%FT%T)"; }
G="$PRIOR --evidence gaze --runs-dir runs_v2_gaze --data-dir runs/data --device cuda $INST"
for SD in 1 2; do run g4_system_seed$SD runs_v2_gaze $V experiments/e4_system.py $G $TR $EV --seed $SD; done
run e8_scaling runs_v2_gaze $V experiments/e8_scaling.py $G --epochs 15 --lr 1e-3
# calibration-label refresh: g2/g3/g4 ran under the 1-seed E1 verdict; the 3-seed verdict is FULL
$V - <<'PY'
import json, os, time
est = json.load(open("runs_v2/gate_estimator.json"))
for d in ("runs_v2_gaze", "runs_v2", "runs_v2_both"):
    p = os.path.join(d, "gate_channel.json")
    if not os.path.exists(p): continue
    g = json.load(open(p)); det = g["detail"]
    det["estimator_mode"] = est["detail"]["mode"]; det["estimator_gate_passed"] = bool(est["passed"])
    det["estimator_tightness_by_b"] = est["detail"].get("tightness_by_b")
    det["reading_vs_floor"] = ("validated lower bound (E1 FULL, 3 seeds; tightness %s)" % est["detail"].get("tightness_by_b")) if est["passed"] else det["reading_vs_floor"]
    json.dump(g, open(p, "w"), indent=2)
    with open(os.path.join(d, "prereg", "deviations.log"), "a") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%S") + "  LABEL REFRESH: gate_channel.detail.reading_vs_floor/estimator_* rewritten from the final 3-seed gate_estimator (numbers unchanged)\n")
print("labels refreshed")
PY
for d in runs_v2_gaze runs_v2_both runs_v2; do $V experiments/report.py --runs-dir $d --out $d/REPORT.md > /dev/null 2>&1 || echo "report $d failed"; done
$V experiments/make_figures.py --runs-dir runs_v2 > runs_v2/logs/figures.log 2>&1 || true
echo "== FOLLOWUP ALL DONE $(date +%FT%T)"
