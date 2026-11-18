#!/usr/bin/env bash
# Start scripts/rq1_all.sh detached unless it is already running (pid in runs/rq1/rq1_all.lock) or finished
# (progress.log ends with "done"). Safe to call from a scheduler at any time.
cd "$(dirname "$0")/.."
LOCK=runs/rq1/rq1_all.lock
if [[ -f $LOCK ]] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "running (pid $(cat $LOCK))"; exit 0; fi
if tail -1 runs/rq1/progress.log 2>/dev/null | grep -q " done$"; then echo "finished"; exit 0; fi
nohup bash scripts/rq1_all.sh >> runs/rq1/rq1_all.out 2>&1 &
echo $! > $LOCK; echo "started (pid $!)"
