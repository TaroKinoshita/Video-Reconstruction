#!/usr/bin/env bash
# PLAN v2 P1: bare Claude drafts (baselines/b_llm_mcp) for clips under runs/p1/clips/<name>/video.mp4, two lanes,
# usage-limit retry as in run_rq1_l0.sh. Skips drafts that already have workspace/output/scene.blend.
#   bash scripts/run_p1_drafts.sh "tennis duck mograph" "1 2"
set -uo pipefail
CLIPS=($1); RUNS=($2); LANES=2; WAIT_MIN="${WAIT_MIN:-20}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
jobs_list=(); for r in "${RUNS[@]}"; do for c in "${CLIPS[@]}"; do jobs_list+=("$c:$r"); done; done
lane() {
  local i=$1 port=$((9876 + $1)) k=0
  for j in "${jobs_list[@]}"; do
    if (( k++ % LANES != i )); then continue; fi
    local c=${j%%:*} r=${j##*:} run="$REPO/runs/p1/drafts/${j%%:*}/r${j##*:}"
    [[ -f "$run/workspace/output/scene.blend" ]] && continue
    mkdir -p "$(dirname "$run")"; local tries=0
    while :; do
      rm -rf "$run"
      BLENDER_PORT=$port bash "$REPO/baselines/b_llm_mcp/run_baseline.sh" "$REPO/runs/p1/clips/$c" "$run" > "$run.log" 2>&1
      if grep -q '"api_error_status":429' "$run/logs/agent.jsonl" 2>/dev/null && (( tries++ < 48 )); then
        echo "lane$i $c r$r usage limit, waiting ${WAIT_MIN}m ($(date +%H:%M))"; sleep $(( WAIT_MIN * 60 )); continue
      fi
      break
    done
    echo "lane$i $c r$r done ($(date +%H:%M))"
  done
}
for ((i = 0; i < LANES; i++)); do lane $i & sleep 20; done
wait
