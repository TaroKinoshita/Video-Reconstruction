#!/bin/bash
# Cameras and depth with MegaSaM (mega-sam/mega-sam, Apache-2.0). Linux only (DROID-SLAM CUDA extensions);
# on Windows it runs inside WSL. Usage:
#
#     bash gpu/observe/run_megasam.sh VIDEO OUT.npz [N_FRAMES=30]
#
# Needs: MEGASAM_ROOT (checkout with base/ built and checkpoints downloaded), an activated env with its
# dependencies, ffmpeg. Runs the repo's three steps (Depth-Anything, UniDepth, camera tracking) on every
# frame, then megasam_to_4d.py writes the same keys as a 4RC output for `vrecon run init --from-4d`.
set -euo pipefail
VIDEO=$(realpath "$1"); OUT=$(realpath -m "$2"); N=${3:-30}
ROOT=${MEGASAM_ROOT:?set MEGASAM_ROOT}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d); SCENE=clip
mkdir -p "$WORK/$SCENE"
ffmpeg -v error -i "$VIDEO" "$WORK/$SCENE/%05d.png"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT/UniDepth:$ROOT/base/droid_slam"
start=$(date +%s)
python Depth-Anything/run_videos.py --encoder vitl --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  --img-path "$WORK/$SCENE" --outdir "$WORK/da/$SCENE"
python UniDepth/scripts/demo_mega-sam.py --scene-name $SCENE --img-path "$WORK/$SCENE" --outdir "$WORK/unidepth"
(cd "$WORK" && python "$ROOT/camera_tracking_scripts/test_demo.py" --datapath="$WORK/$SCENE" \
  --weights="$ROOT/checkpoints/megasam_final.pth" --scene_name $SCENE \
  --mono_depth_path "$WORK/da" --metric_depth_path "$WORK/unidepth" --disable_vis)
secs=$(( $(date +%s) - start ))
python "$HERE/megasam_to_4d.py" "$WORK/outputs/${SCENE}_droid.npz" "$VIDEO" "$OUT" --frames "$N" --seconds "$secs"
rm -rf "$WORK"
