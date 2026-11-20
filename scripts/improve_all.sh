#!/usr/bin/env bash
# PLAN v2 8: every improvement step on a draft .blend, in order; each step's output feeds the next and is kept.
#   bash scripts/improve_all.sh DRAFT.blend OUT_DIR [--skeleton GVHMR.npz] [--held MASKS.npz:LABEL] [--ball MASKS.npz]
#        [--people MASKS.npz]
# Steps: outside (improve_blend) -> person (swap_person_blend, needs --skeleton) -> held (fit_held_blend, needs
# --held) -> ball (fit_ball_blend --one, needs --ball) -> feet (pin_feet_blend, after person) -> people
# (follow_masks_blend, needs --people). Missing observations skip their step. OUT_DIR/final.blend is the last one.
set -euo pipefail
DRAFT=$1; OUT=$2; shift 2
SK=""; HELD=""; BALL=""; PEOPLE=""
while [[ $# -gt 0 ]]; do case $1 in --skeleton) SK=$2;; --held) HELD=$2;; --ball) BALL=$2;; --people) PEOPLE=$2;; esac; shift 2; done
P=.venv/Scripts/python.exe; mkdir -p "$OUT"
$P scripts/improve_blend.py "$DRAFT" "$OUT/outside"
cur="$OUT/outside/improved.blend"
if [[ -n $SK ]]; then $P scripts/swap_person_blend.py "$cur" "$SK" "$OUT/person.blend" --report "$OUT/person.json"; cur="$OUT/person.blend"; fi
if [[ -n $HELD && -n $SK ]]; then $P scripts/fit_held_blend.py "$cur" "${HELD%:*}" "${HELD##*:}" "$OUT/held.blend" --report "$OUT/held.json"; cur="$OUT/held.blend"; fi
if [[ -n $BALL && -n $SK ]]; then $P scripts/fit_ball_blend.py "$cur" "$BALL" "$OUT/ball.blend" --one --report "$OUT/ball.json"; cur="$OUT/ball.blend"; fi
if [[ -n $SK ]]; then $P scripts/pin_feet_blend.py "$cur" "$OUT/feet.blend" --report "$OUT/feet.json"; cur="$OUT/feet.blend"; fi
if [[ -n $PEOPLE ]]; then
  [[ -f "$OUT/categories.json" ]] || $P scripts/categorize_blend.py "$cur" "$OUT/categories.json"
  $P scripts/follow_masks_blend.py "$cur" "$OUT/categories.json" "$PEOPLE" "$OUT/people.blend" --report "$OUT/people.json"; cur="$OUT/people.blend"
fi
cp "$cur" "$OUT/final.blend"
echo "IMPROVED $OUT/final.blend"
