
set -uo pipefail; cd "$(dirname "$0")"
export HF_HUB_DISABLE_PROGRESS_BARS=1 NLTK_DATA=$HOME/nltk_data PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
V=$HOME/venvs/cprd/bin/python; PRIOR="--prior causal_lm --prior-model gpt2-large"
INST="--objective nce --free-tilt-rank 16 --estimand nce --gamma-warmup-value 1.0 --gamma-warmup-epochs 20"
TR="--epochs 20 --lr 1e-3"; EV="--n-perm 1000 --n-boot 10000"; EV3="--n-perm 300 --n-boot 10000"
run () { local id=$1 dir=$2; shift 2; local t0=$(date +%s); mkdir -p $dir/logs
  timeout --signal=INT --kill-after=120 43200 "$@" > $dir/logs/$id.log 2>&1; local rc=$?
  echo "== $id [$dir] rc=$rc ($(( $(date +%s)-t0 ))s) $(date +%FT%T)"; }
STEP="${1:-all}"

# per-task/per-subject breakdown of the channel measurement (all 30 readers, all 4 ZuCo tasks)
if [ "$STEP" = "all" ] || [ "$STEP" = "breakdown" ]; then
  for EV_ in gaze eeg both; do D=runs_v2_$EV_; [ "$EV_" = eeg ] && D=runs_v2
    run e10_breakdown $D $V experiments/e10_breakdown.py $PRIOR --evidence $EV_ --runs-dir $D --data-dir runs/data --device cuda $INST $TR $EV
  done
fi

# instance-level (non-text-clustered) split, gaze and EEG, E2 + E3: leakage-robustness check
if [ "$STEP" = "all" ] || [ "$STEP" = "leakage" ]; then
  DI=runs_v2_instance/data; mkdir -p runs_v2_instance/logs
  run i0_build runs_v2_instance $V experiments/e0_build.py --pickle-root "$ZUCO_PICKLES" --mat-root "$ZUCO_MATS" --out $DI --runs-dir runs_v2_instance --prior-model gpt2-large --split-kind instance
  for EV_ in gaze eeg; do D=runs_v2_instance_$EV_; mkdir -p $D/logs; cp runs_v2_instance/gate_build.json $D/; cp runs_v2/gate_estimator.json $D/
    C="$PRIOR --evidence $EV_ --runs-dir $D --data-dir $DI --device cuda $INST"
    run i2_channel $D $V experiments/e2_channel.py $C $TR $EV
    run i3_selection $D $V experiments/e3_selection.py $C $TR --n-grid 2,4,8,16,32,64
  done
fi

# leave-one-subject-out on the gaze channel, the 12 ZuCo-1.0 subjects, N in {4, 32}
if [ "$STEP" = "all" ] || [ "$STEP" = "loso" ]; then
  run e7_loso runs_v2_gaze $V experiments/e7_loso.py $PRIOR --evidence gaze --runs-dir runs_v2_gaze --data-dir runs/data --device cuda $INST --epochs 15 --lr 1e-3 --subject-prefix Z --n-list 4,32
fi

# structure-only control on gaze: sub-token replication/segmentation kept, gaze content replaced
if [ "$STEP" = "all" ] || [ "$STEP" = "struct" ]; then
  D=runs_v2_gaze_struct; mkdir -p $D/logs; cp runs/gate_build.json $D/ 2>/dev/null; cp runs_v2/gate_estimator.json $D/
  run e2_struct $D $V experiments/e2_channel.py $PRIOR --evidence gaze --runs-dir $D --data-dir runs/data --device cuda $INST $TR $EV3 --gaze-control structure_only
fi

# four-way task/session attribute probe (gaze-only / EEG-only / EEG-residualized-on-gaze)
if [ "$STEP" = "all" ] || [ "$STEP" = "attributes" ]; then
  run e5_attributes runs_v2 $V experiments/e5_attributes.py $PRIOR --runs-dir runs_v2 --data-dir runs/data --device cuda
fi

# generation-baseline reproduction: the verbatim BrainTranslator architecture, two optimizer recipes
if [ "$STEP" = "all" ] || [ "$STEP" = "baselines" ]; then
  run e6_their_recipe runs_v2 $V experiments/e6_seq2seq_baseline.py $PRIOR --runs-dir runs_v2 --data-dir runs/data --device cuda --their-code --their-recipe
  run e6_adamw runs_v2 $V experiments/e6_seq2seq_baseline.py $PRIOR --runs-dir runs_v2 --data-dir runs/data --device cuda --their-code
fi

# low-bit calibration sweep (Table e1_calib): 11 capacities x 3 seeds at the fixed evaluation gain
if [ "$STEP" = "all" ] || [ "$STEP" = "lowbit" ]; then
  D=runs_v2_lowbit; mkdir -p $D/logs; cp runs/gate_build.json $D/ 2>/dev/null; cp runs_v2/gate_estimator.json $D/
  run e14_lowbit $D $V experiments/e14_lowbit.py $PRIOR --runs-dir $D --data-dir runs/data --device cuda $INST $TR --sweep 0,0.005,0.01,0.025,0.05,0.10,0.15,0.25,0.5,1,2 --seeds 0,1,2
fi

echo "== STEP $STEP DONE $(date +%FT%T)"
