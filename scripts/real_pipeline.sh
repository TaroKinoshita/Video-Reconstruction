#!/usr/bin/env bash
# A real clip from video to a repaired Blender scene, with no LLM:
#   project (vrecon new) -> masks (SAM 3.1) -> first scene (4RC) -> fixed-order repair -> .blend + renders
#
#   bash scripts/real_pipeline.sh VIDEO OUT_DIR [--movers "toy car,box"] [--floor table] [--wall wall]
#                                 [--prompt-frame auto] [--quick] [--skip-sam] [--skip-4rc]
#   --quick  SAM stops at the first clean mover prompt instead of trying every word x 3 frames
#   --floor/--wall default "none": no SAM pass for them (each pass takes as long as the movers, up to 2 hours
#            for a floor filling the frame on 8 GB); the floor plane then comes from the 4RC background points
#   SAM_ARGS="--det-thresh 0.4" passes extra options to gpu/observe/run_sam3.py (e.g. when it runs out of memory)
#
# Tool envs (see docs/HANDOFF.md 2.1): VRECON_ENVS (default %LOCALAPPDATA%/vrecon/envs) with observe/ (SAM)
# and 4rc/, VRECON_SRC (default %LOCALAPPDATA%/vrecon/src) with 4RC/. Outputs in OUT_DIR:
#   project/, masks.npz, frames/, 4rc.npz, run/ (repair run: config, history, init.json = adopted scene),
#   scene.blend, render/ and summary.txt. Existing masks.npz / 4rc.npz are kept with --skip-sam / --skip-4rc.
set -euo pipefail
abspath() { (cd "$1" && { pwd -W 2>/dev/null || pwd; }); }
VIDEO="$1"; OUT="$2"; shift 2
MOVERS="object"; FLOOR="none"; WALL="none"; PROMPT_FRAME="auto"; SKIP_SAM=0; SKIP_4RC=0; QUICK=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --movers) MOVERS="$2"; shift ;;
    --floor) FLOOR="$2"; shift ;;
    --wall) WALL="$2"; shift ;;
    --prompt-frame) PROMPT_FRAME="$2"; shift ;;
    --quick) QUICK="--quick" ;;
    --skip-sam) SKIP_SAM=1 ;;
    --skip-4rc) SKIP_4RC=1 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
REPO="$(abspath "$(dirname "${BASH_SOURCE[0]}")/..")"
PY="$REPO/.venv/Scripts/python.exe"; [[ -x "$PY" ]] || PY="$REPO/.venv/bin/python"
LOCAL="${LOCALAPPDATA:-$HOME/.local/share}/vrecon"
ENVS="${VRECON_ENVS:-$LOCAL/envs}"; SRC="${VRECON_SRC:-$LOCAL/src}"
OBS_PY="$ENVS/observe/Scripts/python.exe"; [[ -x "$OBS_PY" ]] || OBS_PY="$ENVS/observe/bin/python"
RC_PY="$ENVS/4rc/Scripts/python.exe"; [[ -x "$RC_PY" ]] || RC_PY="$ENVS/4rc/bin/python"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
mkdir -p "$OUT"; OUT="$(abspath "$OUT")"; VIDEO="$(abspath "$(dirname "$VIDEO")")/$(basename "$VIDEO")"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/summary.txt"; }
: > "$OUT/summary.txt"
log "video $VIDEO"

# 1. project (one shot: real test clips are single cuts)
rm -rf "$OUT/project"; "$PY" -m vrecon.cli new "$VIDEO" --out "$OUT/project" --no-shots > "$OUT/new.log" 2>&1
log "project: $OUT/project/project.json"

# 2. masks: movers by text (best prompt chosen without ground truth), floor and wall as set pieces
if [[ $SKIP_SAM == 0 || ! -f "$OUT/masks.npz" ]]; then
  SETS=(); [[ $FLOOR != none ]] && SETS+=(--set "$FLOOR=obj_floor"); [[ $WALL != none ]] && SETS+=(--set "$WALL=obj_wall")
  log "SAM 3.1: movers='$MOVERS' floor='$FLOOR' wall='$WALL' (can take 10-90 minutes)"
  (cd "$REPO" && "$OBS_PY" gpu/observe/run_sam3.py "$VIDEO" "$OUT/masks.npz" --movers "$MOVERS" \
     --prompt-frame "$PROMPT_FRAME" "${SETS[@]}" $QUICK ${SAM_ARGS:-}) > "$OUT/sam.log" 2>&1
fi
log "masks: $OUT/masks.npz"

# 3. first scene from 4RC (it samples 30 frames from the folder)
if [[ $SKIP_4RC == 0 || ! -f "$OUT/4rc.npz" ]]; then
  rm -rf "$OUT/frames"; mkdir -p "$OUT/frames"
  ffmpeg -v error -i "$VIDEO" "$OUT/frames/%05d.png"
  log "4RC on $(ls "$OUT/frames" | wc -l) frames (about 6 minutes)"
  (cd "$SRC/4RC" && "$RC_PY" inference.py --input "$OUT/frames" --save "$OUT/4rc.npz") > "$OUT/4rc.log" 2>&1
fi
log "4RC: $OUT/4rc.npz"

# 4. repair run (focal fixed by default with a 4D start) and fixed-order repair
rm -rf "$OUT/run"
"$PY" -m vrecon.cli run init "$OUT/project/project.json" "$OUT/run" --masks "$OUT/masks.npz" --video "$VIDEO" \
  --from-4d "$OUT/4rc.npz" > "$OUT/init.log" 2>&1
log "repair (fixed order, 2 passes)"
"$PY" -m vrecon.cli run auto "$OUT/run" > "$OUT/auto.log" 2>&1
"$PY" - "$OUT" <<'PYEOF' | tee -a "$OUT/summary.txt"
import sys
from vrecon.run import service
out = sys.argv[1]
s = service.state(out + "/run")
h = service.history(out + "/run", 200)
acc = [x for x in h if x["status"] == "accepted"]
print(f"attempts {len(h)}, accepted {len(acc)}, loss -> {s['metrics']}")
for o in s["objects"]:
    if o["kind"] != "set":
        print(f"  {o['id']}: size {[round(v, 4) for v in o['size']]} ({'moving' if o['animated'] else 'still'})")
print(f"  camera fx {[round(c['fx'], 1) for c in s['cameras']]}")
PYEOF

# 5. Blender scene and a few renders to look at
"$PY" -m vrecon.cli build "$OUT/run/init.json" --blend "$OUT/scene.blend" --render-dir "$OUT/render" \
  --frames 0,36,72 > "$OUT/build.log" 2>&1 || log "build failed (see build.log)"
log "done: $OUT/scene.blend, renders in $OUT/render"
