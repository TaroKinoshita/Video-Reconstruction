#!/usr/bin/env bash
# Baseline B (PLAN.md 11.1): a general-purpose LLM + blender-mcp reconstructs one clip, then we score it.
#
#   bash baselines/b_llm_mcp/run_baseline.sh CLIP_DIR RUN_DIR [--model MODEL] [--minutes 60] [--turns 200]
#
#   CLIP_DIR  a reference clip folder (needs video.mp4; for scoring also gt_passes/, gt.json, meta.json)
#   RUN_DIR   new, empty folder for this run (one model x clip x attempt)
#
# Needs on this machine: full Blender 5.2.x (GUI build; blender-mcp does not work in `blender -b`),
# uv/uvx, the `claude` CLI, ffmpeg, and this repo's .venv for scoring. On a headless Linux box,
# Blender runs under Xvfb, as in CineRecon's setup.
set -euo pipefail

CLIP="$(cd "$1" && pwd)"; RUN="$2"; shift 2
MODEL="claude-opus-5-5"; MINUTES=60; TURNS=200
PORT="${BLENDER_PORT:-9876}"
MCP_VERSION="2.2.6"          # mcp-for-blender (blender-mcp) version used for every run
BLENDER="${BLENDER:-blender}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift ;;
    --minutes) MINUTES="$2"; shift ;;
    --turns) TURNS="$2"; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

if [[ -e "$RUN" && -n "$(ls -A "$RUN" 2>/dev/null)" ]]; then echo "$RUN is not empty"; exit 2; fi
mkdir -p "$RUN/workspace/input" "$RUN/workspace/output" "$RUN/logs"
RUN="$(cd "$RUN" && pwd)"
WS="$RUN/workspace"

# Only the video goes into the agent's workspace. Ground truth never does (PLAN.md 11.8).
cp "$CLIP/video.mp4" "$WS/input/reference.mp4"

# Prompt: CineRecon L1 (MIT, see LICENSE.CineRecon) with our one change (GPU optional).
sed -e "s#<VIDEO_PATH>#$WS/input/reference.mp4#" -e "s#<OUTPUT_DIR>#$WS/output#g" \
  "$HERE/prompt_L1.txt" > "$RUN/prompt.txt"

cat > "$RUN/mcp.json" <<EOF
{"mcpServers": {"blender": {"command": "uvx", "args": ["mcp-for-blender==$MCP_VERSION"],
  "env": {"BLENDER_HOST": "127.0.0.1", "BLENDER_PORT": "$PORT", "BLENDER_MCP_DISABLE_TELEMETRY": "1"}}}}
EOF

# --- Blender with the add-on's socket server ------------------------------------------------------
"$BLENDER" --version | head -1 > "$RUN/logs/blender_version.txt"
uvx "mcp-for-blender==$MCP_VERSION" install-addon >"$RUN/logs/install_addon.log" 2>&1 || {
  echo "could not install the blender-mcp add-on automatically; install it in Blender and enable auto-start"; }
BLENDER_CMD=("$BLENDER" --factory-startup --python-expr
  "import bpy, addon_utils; [addon_utils.enable(m.__name__, default_set=True) for m in addon_utils.modules() if 'blender' in m.__name__.lower() and 'mcp' in m.__name__.lower()]; bpy.context.scene.blendermcp_port=$PORT")
if [[ -z "${DISPLAY:-}" && "$(uname -s)" == "Linux" ]]; then
  xvfb-run -a -s "-screen 0 1920x1080x24" "${BLENDER_CMD[@]}" >"$RUN/logs/blender.log" 2>&1 &
else
  "${BLENDER_CMD[@]}" >"$RUN/logs/blender.log" 2>&1 &
fi
BLENDER_PID=$!
trap 'kill $BLENDER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break
  sleep 1
done
(exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null || { echo "blender-mcp server did not start on port $PORT (see logs/blender.log)"; exit 1; }

# --- The agent ------------------------------------------------------------------------------------
START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
(cd "$WS" && timeout "${MINUTES}m" claude -p "$(cat "$RUN/prompt.txt")" \
  --model "$MODEL" --max-turns "$TURNS" \
  --mcp-config "$RUN/mcp.json" \
  --allowedTools "mcp__blender__*" "Bash" "Read" "Write" "Edit" "Glob" "Grep" \
  --output-format stream-json --verbose) > "$RUN/logs/agent.jsonl" 2> "$RUN/logs/agent.stderr"
EXIT=$?
set -e
END=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# --- Record, check, score -------------------------------------------------------------------------
"$REPO/.venv/bin/python" "$HERE/finish_run.py" --run "$RUN" --clip "$CLIP" --model "$MODEL" \
  --mcp-version "$MCP_VERSION" --start "$START" --end "$END" --exit-code "$EXIT" --minutes "$MINUTES" --turns "$TURNS"
