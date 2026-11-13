#!/usr/bin/env bash
# Condition C with an LLM (PLAN.md 11.1): the `claude` CLI repairs an existing run through vrecon-mcp only.
#
#   bash scripts/run_llm_repair.sh RUN_DIR [--model MODEL] [--turns 60] [--minutes 30] [--clip CLIP_DIR]
#
#   RUN_DIR  a run made with `vrecon run init` (its config fixes the start scene, masks and budget)
#   --clip   synthetic clip folder: score the adopted scene against gt.json afterwards (scoring only;
#            the agent never sees ground truth, PLAN.md 11.8)
#
# The agent gets the vrecon tools and nothing else (no shell, no file access), so every change goes
# through the run manager's proposal -> constraints -> accept rule. Logs go to RUN_DIR/llm/.
set -euo pipefail
abspath() { (cd "$1" && { pwd -W 2>/dev/null || pwd; }); }
RUN="$(abspath "$1")"; shift
MODEL="claude-opus-5-5"; TURNS=60; MINUTES=30; CLIP=""; USE_BUDGET=0; LOOKER=0; VIEWER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift ;;
    --turns) TURNS="$2"; shift ;;
    --minutes) MINUTES="$2"; shift ;;
    --clip) CLIP="$(abspath "$2")"; shift ;;
    --use-budget) USE_BUDGET=1 ;;
    --looker) LOOKER=1 ;;  # option 2 prototype: run_look (picture) + run_fix (issue words)
    --viewer) LOOKER=1; VIEWER=1 ;;  # looker + close looking (run/view.py): overview, events, clip, compare, track
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
REPO="$(abspath "$(dirname "${BASH_SOURCE[0]}")/..")"
PY="$REPO/.venv/bin/python"; [[ -x "$REPO/.venv/Scripts/python.exe" ]] && PY="$REPO/.venv/Scripts/python.exe"
LOG="$RUN/llm"; mkdir -p "$LOG"
if [[ $USE_BUDGET == 1 ]]; then
  # --use-budget: the first trials stopped after 6-12 of 30 attempts; the fixed order tries every
  # hypothesis twice. A rejected attempt costs only budget, so keep going.
  STOP_RULE="Keep working until the budget is used up. A rejected attempt is not a failure: it only costs one
attempt. When the obvious hypotheses are done, revisit every object's position, size, timing and orientation
again (earlier fits were made with other parts still wrong), and try the camera focal length again."
else
  STOP_RULE="Stop when it is used up or when two diagnoses in a row suggest nothing that is likely to help."
fi
LOOK_RULE=""; EXTRA=()
if [[ $LOOKER == 1 ]]; then
  LOOK_RULE="You can also LOOK: run_look shows the reference video next to the reconstruction's boxes. Look before
repairing and after each adopted change, say in one line what looks wrong, and fix it with run_fix (one issue
word and a target). run_fix and run_optimize share the budget and the accept rule."
  EXTRA=("mcp__vrecon__run_look" "mcp__vrecon__run_fix")
fi
if [[ $VIEWER == 1 ]]; then
  LOOK_RULE="${LOOK_RULE}
Look closely before you judge timing or motion: start with run_view_overview (the whole clip, numbered, with each
object's speed), then run_view_events (machine-found starts, stops, turns, occlusions with frame ranges). Step
through the frames that matter with run_view_clip (its own frame rate; give an object label as region to zoom in)
and run_view_compare (video and reconstruction on the same frames). Before fixing timing or a path, check the
numbers with run_view_track (observed and reconstructed start and stop frames, offsets). Say what you saw with
frame numbers."
  EXTRA+=("mcp__vrecon__run_view_overview" "mcp__vrecon__run_view_events" "mcp__vrecon__run_view_clip"
          "mcp__vrecon__run_view_compare" "mcp__vrecon__run_view_track")
fi

cat > "$LOG/mcp.json" <<EOF
{"mcpServers": {"vrecon": {"command": "$PY", "args": ["-m", "vrecon.mcp_server"],
  "env": {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "VRECON_LOOKER": "$LOOKER"}}}}
EOF
cat > "$LOG/prompt.txt" <<EOF
You are repairing a 3D reconstruction of a short video. The run directory is: $RUN

Use only the vrecon tools. Each repair attempt is a hypothesis about ONE thing that is wrong (the camera
focal length, or one object's floor position, size, timing or orientation) tested with run_optimize; the
run manager adopts it only if the loss improves. Read run_state and run_diagnose first, and diagnose again
after each adopted change. The attempt budget is fixed in the run (run_state shows attempts and
max_attempts).
${STOP_RULE}
${LOOK_RULE}
Finish with a short summary of what you changed and what still looks wrong.
EOF

START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
(cd "$LOG" && timeout "${MINUTES}m" claude -p "$(cat "$LOG/prompt.txt")" \
  --model "$MODEL" --max-turns "$TURNS" --mcp-config "$LOG/mcp.json" --strict-mcp-config --tools "" \
  --allowedTools "mcp__vrecon__run_state" "mcp__vrecon__run_diagnose" "mcp__vrecon__run_optimize" \
    "mcp__vrecon__run_history" "mcp__vrecon__run_rollback" "${EXTRA[@]}" \
  --output-format stream-json --verbose) > "$LOG/agent.jsonl" 2> "$LOG/agent.stderr"
EXIT=$?
set -e
END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
"$PY" - "$RUN" "$CLIP" "$MODEL" "$START" "$END" "$EXIT" <<'EOF'
import json, sys
from pathlib import Path
from vrecon.run import service
run, clip, model, start, end, code = sys.argv[1:]
mgr, _, _ = service._manager(Path(run))
rec = {"model": model, "start": start, "end": end, "exit_code": int(code), "attempts": mgr.attempts,
       "revision": mgr.adopted.revision, "metrics": mgr.metrics, "history": service.history(run, 200)}
if clip:
    from vrecon.evaluate import score
    rec["score"] = score(Path(clip), mgr.adopted)["summary"]
(Path(run) / "llm" / "result.json").write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
print(json.dumps({k: rec[k] for k in ("exit_code", "attempts", "revision", "metrics")} | {"score": rec.get("score")}, default=str))
EOF
