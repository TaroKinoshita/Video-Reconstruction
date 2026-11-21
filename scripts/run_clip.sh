#!/usr/bin/env bash
# PLAN v2: one clip from its video to the A/B jobs and, after generation, the motion scores. Every step skips when
# its output exists, so the script can be run again after any stop. Steps that spend money stop unless allowed.
#   bash scripts/run_clip.sh CLIP VIDEO START FRAMES SIZE [--draft] [--fal] [--score]
#     START seconds, FRAMES at 24 fps (17n+5), SIZE WxH (multiples of 32)
#   --draft  allow the bare Claude draft (baselines/b_llm_mcp, Claude usage; needs GUI Blender)
#   --fal    allow the first-frame pictures (fal, paid) and so the s2 jobs
#   --score  after the pod run: motion scores of every generated pair (SAM 2 + pose, local GPU)
# Needs: the observe env (SAM 3.1, local), GVHMR in WSL (~/vrecon), the SAM 2 env (C:/Users/<you>/sam2env).
# Hand-made per clip: runs/p5/CLIP/look.txt (one sentence on how the clip looks), checked by a human.
set -euo pipefail
C=$1; VIDEO=$2; START=$3; N=$4; SIZE=$5; shift 5
DRAFT=0; FAL=0; SCORE=0
for x in "$@"; do case $x in --draft) DRAFT=1;; --fal) FAL=1;; --score) SCORE=1;; esac; done
P=.venv/Scripts/python.exe
OBS="$LOCALAPPDATA/vrecon/envs/observe/Scripts/python.exe"
SAM2="$USERPROFILE/sam2env"
H=runs/human/$C; D=runs/p5/$C; I=runs/p6/$C; CL=runs/p1/clips/$C
mkdir -p "$H" "$D" "$I" "$CL"
[[ -f $CL/video.mp4 ]] || cp "$VIDEO" "$CL/video.mp4"
END=$($P -c "print($START + $N / 24)")

# 1 observations (local, free)
[[ -f $H/masks.npz ]] || "$OBS" gpu/observe/run_sam3.py "$CL/video.mp4" "$H/masks.npz" --movers person
if [[ ! -f $H/gvhmr_skeleton.npz ]]; then
  W=$(wsl wslpath -a "$(pwd -W)")
  wsl -e bash -lc "cd ~/vrecon/GVHMR && ~/vrecon/bin/micromamba run -r ~/vrecon/mamba -n gvhmr python '$W/gpu/observe/run_gvhmr.py' '$W/$CL/video.mp4' '$W/$H'"
fi

# 2 bare draft (Claude; spends usage)
BARE=runs/p1/drafts/$C/r1/workspace/output/scene.blend
if [[ ! -f $BARE ]]; then
  if (( DRAFT )); then bash scripts/run_p1_drafts.sh "$C" "1"; else echo "STOP: no bare draft ($BARE); rerun with --draft"; exit 3; fi
fi

# 3 improvements (local)
[[ -f $I/final.blend ]] || bash scripts/improve_all.sh "$BARE" "$I" --skeleton "$H/gvhmr_skeleton.npz" --people "$H/masks.npz"

# 4 references for the generator (cut from the source)
if [[ ! -d $D/source_refs ]]; then
  $P scripts/make_ref_crops.py "$CL/video.mp4" "$H/masks.npz" "$I/outside/project.json" "$D/refs_raw" --start "$START" --end "$END"
  mkdir -p "$D/source_refs"; k=0
  for f in "$D/refs_raw/frame_full.png" "$D"/refs_raw/crop_*.png; do
    [[ -f $f ]] || continue; cp "$f" "$D/source_refs/$(printf %02d $k)_$(basename "$f")"; k=$((k + 1))
  done
  cp "$D/refs_raw/frame_full.png" "$D/source_first.png"
fi
[[ -f $D/look.txt ]] || { echo "STOP: write $D/look.txt (how the clip looks, one sentence), then rerun"; exit 3; }

# 5 controls, QC, jobs (first-frame pictures only with --fal)
if [[ ! -f $D/jobs.json ]]; then
  S1_ONLY=$(( 1 - FAL )) bash scripts/prep_ab.sh "$C" "$BARE" "$I/final.blend" "$SIZE" "$N" "$START" \
    "$D/source_first.png" "$D/look.txt" --check "cat:person=$H/masks.npz"
fi
echo "JOBS $D/jobs.json  (pod: gpu/h3_control/comfy_jobs.py --jobs $D/jobs.json --out $D/out --group s1 ...)"

# 6 motion scores after generation (local GPU)
if (( SCORE )); then
  pairs=()
  for mv in orbit45 dolly-60; do for k in bare improved; do
    g=$(ls -d "$D"/out/s1_${k}_${mv}__s* 2>/dev/null | head -1)
    [[ -n $g && -f $g/video.mp4 ]] && pairs+=("$D/${k}_$mv/control_sem.mp4:$g/video.mp4")
  done; done
  mkdir -p runs/p7/$C
  "$SAM2/Scripts/python.exe" scripts/score_motion_sam2.py --ckpt "$SAM2/ckpt/sam2.1_hiera_small.pt" \
    --pose "$SAM2/ckpt/yolo11m-pose.pt" --source-video "$CL/video.mp4" --source-masks "$H/masks.npz" \
    --source-label auto --start "$START" --out "runs/p7/$C/motion.json" "${pairs[@]}"
fi
