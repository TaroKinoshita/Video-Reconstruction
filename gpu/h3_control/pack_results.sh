#!/usr/bin/env bash
# Collect phase-1a results after run_jobs.py.
#   bash gpu/h3_control/pack_results.sh [runs/phase1a/generated] [evaluation/phase1a/results]
#
# - Small files (meta.json, sheet.jpg, results_manifest.json) are copied into the repo folder so they can
#   be committed and pushed (pattern A): the cloud session can then read them.
# - Videos are packed into one tar file to download to your PC (videos are not committed).
set -euo pipefail
SRC="${1:-runs/phase1a/generated}"
DST="${2:-evaluation/phase1a/results}"
mkdir -p "$DST"
cp "$SRC/results_manifest.json" "$DST/" 2>/dev/null || echo "no results_manifest.json yet"
for d in "$SRC"/*/; do
  [[ -f "$d/meta.json" ]] || continue
  name="$(basename "$d")"
  mkdir -p "$DST/$name"
  cp "$d/meta.json" "$DST/$name/"
  [[ -f "$d/sheet.jpg" ]] && cp "$d/sheet.jpg" "$DST/$name/"
done
TAR="runs/phase1a_videos_$(date +%Y%m%d_%H%M).tar"
tar -cf "$TAR" -C "$SRC" $(cd "$SRC" && ls -d */video.mp4 2>/dev/null)
echo "small results -> $DST (commit and push these)"
echo "videos        -> $TAR (download to your PC)"
du -sh "$DST" "$TAR"
