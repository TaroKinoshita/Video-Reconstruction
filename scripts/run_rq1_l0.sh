#!/usr/bin/env bash
# RQ1 cell L0 (Claude builds the scene, baselines/b_llm_mcp) on several clips x runs, in LANES parallel lanes
# (one Blender per lane on its own port). Skips runs that already have score.json. A run stopped by a usage
# limit (HTTP 429) is retried every WAIT_MIN minutes (default 20, at most 48 times).
#
#   bash scripts/run_rq1_l0.sh "syn2_02 syn2_04" "1 2 3" [LANES=2]
set -uo pipefail
CLIPS=($1); RUNS=($2); LANES="${3:-2}"; WAIT_MIN="${WAIT_MIN:-20}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
jobs_list=()
for r in "${RUNS[@]}"; do for c in "${CLIPS[@]}"; do jobs_list+=("$c:$r"); done; done
lane() {
  local i=$1 port=$((9876 + $1)) k=0
  for j in "${jobs_list[@]}"; do
    if (( k++ % LANES != i )); then continue; fi
    local c=${j%%:*} r=${j##*:} run="$REPO/runs/rq1/L0/${j%%:*}/r${j##*:}"
    [[ -f "$run/score.json" ]] && continue
    rm -rf "$run"; mkdir -p "$(dirname "$run")"
    local s=$(date +%s) tries=0
    while :; do
      rm -rf "$run"
      BLENDER_PORT=$port bash "$REPO/baselines/b_llm_mcp/run_baseline.sh" "$REPO/evaluation/synthetic_v2/$c" "$run" > "$run.log" 2>&1
      # a usage limit (HTTP 429) ends the agent at once with nothing built: wait and run it again, unscored
      if grep -q '"api_error_status":429' "$run/logs/agent.jsonl" 2>/dev/null && (( tries++ < 48 )); then
        rm -f "$run/score.json"; echo "lane$i $c r$r usage limit, waiting ${WAIT_MIN}m ($(date +%H:%M))"
        sleep $(( WAIT_MIN * 60 )); continue
      fi
      break
    done
    echo "lane$i $c r$r exit=$? $(( $(date +%s) - s ))s"
  done
}
for ((i = 0; i < LANES; i++)); do lane $i & sleep 20; done
wait
