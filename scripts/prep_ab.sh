#!/usr/bin/env bash
# PLAN v2 7.7 (after the second review): staged V2V A/B for one clip, everything up to a dry run.
#   bash scripts/prep_ab.sh CLIP BARE.blend IMPROVED.blend SIZE FRAMES START SOURCE.png LOOK.txt [QC args...]
# QC args are passed to scripts/proxy_qc.py (e.g. --check "cat:person=MASKS.npz" --needs "...").
# Stages (groups in runs/p5/<clip>/jobs.json):
#   s1  bare vs improved: ONLY the control differs (same camera path from the bare draft, common references = the
#       source, common prompt = look + colour legend, seed 43)
#   s2  improved + the generated first frame as picture 1 (fal; reused when its inputs are unchanged)
#   s3  improved + the view-independent action timeline
#   s4  improved with coarse figures for unobserved people
# A control with a wrong format stops the run (proxy_qc exit 2).
# S1_ONLY=1: only what s1 needs (no coarse controls, no first-frame pictures: nothing paid).
set -euo pipefail
C=$1; BARE=$2; IMP=$3; SIZE=$4; N=$5; START=$6; SRC=$7; LOOK=$8; shift 8
P=.venv/Scripts/python.exe; D=runs/p5/$C
mkdir -p $D
[[ -f $D/categories.json ]] || $P scripts/categorize_blend.py "$BARE" $D/categories.json
$P scripts/scene_timeline.py "$IMP" $D/categories.json $D/timeline_improved.txt --start $START --frames $N
for mv in orbit45 dolly-60; do
  CP=$D/campath_$mv.json; rm -f $CP
  KINDS="bare improved improved_coarse"; [[ ${S1_ONLY:-0} == 1 ]] && KINDS="bare improved"
  for k in $KINDS; do
    B=$BARE; extra=""; [[ $k != bare ]] && B=$IMP; [[ $k == improved_coarse ]] && extra="--coarse"
    out=$D/${k}_$mv; mkdir -p $out
    $P scripts/export_clay_blend.py "$B" $out/control_sem.mp4 --move $mv --size $SIZE --frames $N --start $START \
      --semantic $D/categories.json --camera-path $CP $extra
    $P scripts/proxy_qc.py "$B" $D/categories.json $out/control_sem.mp4 $out/qc.json --start $START --frames $N "$@"
  done
  [[ ${S1_ONLY:-0} == 1 ]] || $P scripts/first_frame_ref.py $D/improved_$mv/control_sem.mp4 "$SRC" $D/improved_$mv/first_frame.png --look "$LOOK"
done
$P scripts/make_ab_jobs.py $C
echo PREP_DONE
