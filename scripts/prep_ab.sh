#!/usr/bin/env bash
# PLAN v2 7.6: everything a V2V A/B needs for one clip, both conditions (bare draft / improved), each camera move:
# categories (once, from the bare draft), shaded semantic-colour controls (improved also: coarse figures for
# unobserved people), time-coded action text per condition, first-frame references on fal per condition x move,
# proxy QC, jobs.json. Then a dry run of the jobs.
#   bash scripts/prep_ab.sh CLIP BARE.blend IMPROVED.blend SIZE FRAMES START SOURCE_FRAME.png LOOK.txt [MASKS.npz] [ROLES.json]
set -euo pipefail
C=$1; BARE=$2; IMP=$3; SIZE=$4; N=$5; START=$6; SRC=$7; LOOK=$8; MASKS=${9:-}; ROLES=${10:-}
P=.venv/Scripts/python.exe; D=runs/p5/$C
mkdir -p $D
[[ -f $D/categories.json ]] || $P scripts/categorize_blend.py "$BARE" $D/categories.json
for k in bare improved; do
  B=$BARE; extra=""; [[ $k == improved ]] && { B=$IMP; extra="--coarse"; }
  $P scripts/scene_timeline.py "$B" $D/categories.json $D/timeline_$k.txt --start $START --frames $N
  for mv in orbit45 dolly-60; do
    out=$D/${k}_$mv; mkdir -p $out
    $P scripts/export_clay_blend.py "$B" $out/control_sem.mp4 --move $mv --size $SIZE --frames $N --start $START \
      --semantic $D/categories.json $extra ${ROLES:+--roles $ROLES}
    $P scripts/proxy_qc.py "$B" $D/categories.json $out/control_sem.mp4 $out/qc.json ${MASKS:+--masks $MASKS} || echo "QC FAILED: $out"
    [[ -f $out/first_frame.png ]] || $P scripts/first_frame_ref.py $out/control_sem.mp4 "$SRC" $out/first_frame.png --look "$LOOK"
  done
done
echo PREP_DONE
