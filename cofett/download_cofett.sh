#!/usr/bin/env bash
# Download the COFETT recordings (OpenNeuro ds006317, CC0) used in the paper.
#   session 1, both participants, tasks para1 + para2 (16 runs)      -> unseen-sentence and session-1 repeated splits
#   sessions 2-4, both participants, task para2 only (24 runs)        -> 16-repetition scheme (COFETT's own design)
# Resumable (curl -C -). Usage: bash cofett/download_cofett.sh <RAW_DIR>
set -u
RAW=${1:?usage: download_cofett.sh <RAW_DIR>}
B=https://s3.amazonaws.com/openneuro.org/ds006317
get () {  # $1 = sub, $2 = ses, $3 = task, $4 = run
  local n="sub-$1_ses-$2_task-$3_run-$4" d="$RAW/sub-$1/ses-$2/eeg"; mkdir -p "$d"
  for e in events.tsv channels.tsv eeg.json eeg.edf; do
    for try in 1 2 3; do curl -s -f -C - "$B/sub-$1/ses-$2/eeg/${n}_$e" -o "$d/${n}_$e" && break; sleep 5; done
  done
  echo "done $n"
}
for s in 01 02; do for t in para1 para2; do for r in 01 02 03 04; do get $s 01 $t $r; done; done; done
for ses in 02 03 04; do for s in 01 02; do for r in 01 02 03 04; do get $s $ses para2 $r; done; done; done
echo "COFETT DOWNLOAD DONE"
