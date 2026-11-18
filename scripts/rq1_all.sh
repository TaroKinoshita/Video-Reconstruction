#!/usr/bin/env bash
# Everything left of RQ1 compute, in order, unattended (dev completion, held-out S0/S1/L0/L1, H5 edits, rows,
# comparisons). Each step skips work already done, so the script can simply be started again. Usage limits are
# waited out by scripts/run_rq1_l0.sh. Progress: runs/rq1/progress.log.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
PY=.venv/Scripts/python.exe; [[ -x $PY ]] || PY=.venv/bin/python
LOG=runs/rq1/progress.log
say() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$LOG"; }
DEV="syn2_02 syn2_04 syn2_05 syn2_07 syn2_09 syn2_12"
HELD="syn2_01 syn2_03 syn2_06 syn2_08 syn2_10 syn2_11"

say "waiting for running L0 jobs"
while ps -ef | grep -v grep | grep -q -e "run_baseline.sh" -e "run_rq1_l1.py"; do sleep 60; done

say "dev L0 runs 1-3"; bash scripts/run_rq1_l0.sh "$DEV" "1 2 3" 2 >> "$LOG" 2>&1
$PY scripts/run_rq1_l1.py --clips $DEV >> "$LOG" 2>&1
EXTRA=$($PY scripts/rq1_extra_runs.py dev); say "dev clips with SD > .03: ${EXTRA:-none}"
[[ -n "$EXTRA" ]] && bash scripts/run_rq1_l0.sh "$EXTRA" "4 5" 2 >> "$LOG" 2>&1
$PY scripts/run_rq1_l1.py --clips $DEV >> "$LOG" 2>&1

say "held-out S0"; [[ -f runs/rq1/start/syn2_11/run/config.json ]] || $PY scripts/build_rq1_starts.py --clips $HELD >> "$LOG" 2>&1
say "held-out S1"; $PY - <<PYX >> "$LOG" 2>&1
import json, subprocess, sys
from pathlib import Path
done = json.loads(Path("runs/rq1/S1/scores.json").read_text(encoding="utf-8"))
todo = [c for c in "$HELD".split() if c not in done]
if todo: subprocess.run([sys.executable, "scripts/run_rq1_s1.py", "--clips", *todo], check=True)
PYX
N=$($PY scripts/rq1_extra_runs.py held_out); say "held-out L0 runs per clip: $N"
bash scripts/run_rq1_l0.sh "$HELD" "$(seq -s ' ' 1 $N)" 2 >> "$LOG" 2>&1
$PY scripts/run_rq1_l1.py --clips $HELD >> "$LOG" 2>&1

say "H5 edits"; $PY scripts/run_rq1_edits.py >> "$LOG" 2>&1
say "rows and comparisons"; $PY scripts/rq1_rows.py >> "$LOG" 2>&1
mkdir -p runs/rq1/compare
for pair in "S0 S1" "L0 L1" "L1 S1" "L0 S1"; do
  set -- $pair
  for split in dev held_out; do
    $PY scripts/compare_conditions.py --rows runs/rq1/all.json --ref $1 --cand $2 --split $split \
      --out runs/rq1/compare/${1}_vs_${2}_${split}.md >> "$LOG" 2>&1
  done
done
say "done"
