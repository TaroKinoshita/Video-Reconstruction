#!/usr/bin/env bash
# Baseline B (PLAN.md 11.1): a general-purpose LLM + blender-mcp reconstructs one clip, then we score it.
#
#   bash baselines/b_llm_mcp/run_baseline.sh CLIP_DIR RUN_DIR [--model MODEL] [--minutes 60] [--turns 200]
#
#   CLIP_DIR  a reference clip folder (needs video.mp4; for scoring also gt_passes/, gt.json, meta.json)
#   RUN_DIR   new, empty folder for this run (one model x clip x attempt)
#   --mcp blend-ai   use the blend-ai server (BLEND_AI_DIR, default ~/Claude/blend-ai) instead of the pinned
#                    mcp-for-blender; it attaches to a Blender you started yourself with its add-on server
#                    running on BLENDER_PORT (no Blender is launched or killed by this script)
#
# Needs on this machine: full Blender 5.2.x (GUI build; blender-mcp does not work in `blender -b`),
# uv/uvx, the `claude` CLI, ffmpeg, and this repo's .venv for scoring. On a headless Linux box,
# Blender runs under Xvfb, as in CineRecon's setup.
set -euo pipefail

# Absolute path that native programs understand too (Git Bash: C:/..., not /c/...).
abspath() { (cd "$1" && { pwd -W 2>/dev/null || pwd; }); }

CLIP="$(abspath "$1")"; RUN="$2"; shift 2
MODEL="claude-opus-5-5"; MINUTES=60; TURNS=200; MCP="mcp-for-blender"; REFINE=0
BLEND_AI_DIR="${BLEND_AI_DIR:-$HOME/Claude/blend-ai}"
PORT="${BLENDER_PORT:-9876}"
MCP_VERSION="2.2.6"          # mcp-for-blender (blender-mcp) version used for every run
WIN_BLENDER="/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"
if [[ -z "${BLENDER:-}" ]]; then
  if ! command -v blender >/dev/null 2>&1 && [[ -x "$WIN_BLENDER" ]]; then BLENDER="$WIN_BLENDER"; else BLENDER=blender; fi
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift ;;
    --minutes) MINUTES="$2"; shift ;;
    --turns) TURNS="$2"; shift ;;
    --mcp) MCP="$2"; shift ;;
    --refine) REFINE=1 ;;  # option 4: also give the agent vrecon's refine_scene tool
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
HERE="$(abspath "$(dirname "${BASH_SOURCE[0]}")")"
REPO="$(abspath "$HERE/../..")"
VENV_PY="$REPO/.venv/bin/python"; [[ -x "$REPO/.venv/Scripts/python.exe" ]] && VENV_PY="$REPO/.venv/Scripts/python.exe"

if [[ -e "$RUN" && -n "$(ls -A "$RUN" 2>/dev/null)" ]]; then echo "$RUN is not empty"; exit 2; fi
mkdir -p "$RUN/workspace/input" "$RUN/workspace/output" "$RUN/logs"
RUN="$(abspath "$RUN")"
WS="$RUN/workspace"

# Only the video goes into the agent's workspace. Ground truth never does (PLAN.md 11.8).
cp "$CLIP/video.mp4" "$WS/input/reference.mp4"

# Prompt: CineRecon L1 (MIT, see LICENSE.CineRecon) with our one change (GPU optional).
sed -e "s#<VIDEO_PATH>#$WS/input/reference.mp4#" -e "s#<OUTPUT_DIR>#$WS/output#g" \
  "$HERE/prompt_L1.txt" > "$RUN/prompt.txt"

if [[ "$MCP" == "blend-ai" ]]; then
  BLEND_AI_WIN="$(abspath "$BLEND_AI_DIR")"
  UV="$(command -v uv)"; command -v cygpath >/dev/null && UV="$(cygpath -m "$UV")"  # a path native programs understand
  cat > "$RUN/mcp.json" <<EOF
{"mcpServers": {"blender": {"command": "$UV", "args": ["run", "--directory", "$BLEND_AI_WIN", "blend-ai"]}}}
EOF
  MCP_VERSION="blend-ai-$(git -C "$BLEND_AI_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "attached (user-started Blender)" > "$RUN/logs/blender_version.txt"
  (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null || { echo "no Blender add-on server on port $PORT: start it in Blender first"; exit 1; }
else
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

fi

EXTRA_TOOLS=()
if [[ $REFINE == 1 ]]; then
  # option 4 (docs/DIRECTION_OPTIONS.md): the repair loop as one more tool. Its masks are the clip's
  # ground-truth stand-in masks, the same observation condition C gets; the agent sees only losses.
  "$VENV_PY" - "$RUN/mcp.json" "$VENV_PY" "$CLIP" "$RUN/refine" "$REPO" <<'PY'
import json, sys
path, py, clip, out, repo = sys.argv[1:]
cfg = json.load(open(path, encoding="utf-8"))
cfg["mcpServers"]["vrecon"] = {"command": py, "args": ["-m", "vrecon.mcp_server"], "cwd": repo,
    "env": {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "VRECON_REFINE_MASKS": clip, "VRECON_REFINE_DIR": out}}
json.dump(cfg, open(path, "w", encoding="utf-8"), indent=1)
PY
  cat >> "$RUN/prompt.txt" <<'EOF'

Extra tool: refine_scene (vrecon). After you have saved a first full scene (camera, floor, wall and one mesh
object per moving thing, animated), call refine_scene with the path of the saved .blend. It fine-tunes the
camera focal length and each moving object's position, size, facing and timing against the video's
silhouettes, and writes a refined copy. Open the refined copy, check it against the video, keep improving
it if needed, and save your final scene to the required output path.
EOF
  EXTRA_TOOLS=("mcp__vrecon__refine_scene")
fi

# --- The agent ------------------------------------------------------------------------------------
START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
(cd "$WS" && timeout "${MINUTES}m" claude -p "$(cat "$RUN/prompt.txt")" \
  --model "$MODEL" --max-turns "$TURNS" \
  --mcp-config "$RUN/mcp.json" --strict-mcp-config \
  --allowedTools "mcp__blender__*" "${EXTRA_TOOLS[@]}" "Bash" "Read" "Write" "Edit" "Glob" "Grep" \
  --output-format stream-json --verbose) > "$RUN/logs/agent.jsonl" 2> "$RUN/logs/agent.stderr"
EXIT=$?
set -e
END=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# --- Record, check, score -------------------------------------------------------------------------
"$VENV_PY" "$HERE/finish_run.py" --run "$RUN" --clip "$CLIP" --model "$MODEL" \
  --mcp-version "$MCP_VERSION" --start "$START" --end "$END" --exit-code "$EXIT" --minutes "$MINUTES" --turns "$TURNS"
