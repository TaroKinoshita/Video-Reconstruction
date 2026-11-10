"""Score a hand-driven B or C run (see prepare.py, README.md).

    python baselines/gpt_codex/finish.py b RUN_DIR --model gpt-x [--minutes 60]
    python baselines/gpt_codex/finish.py c RUN_DIR --model gpt-x

Optional: save the agent's session log as RUN_DIR/logs/agent.jsonl (B) or RUN_DIR/llm/agent.jsonl (C) first;
B's leak check reads it. Writes the same result files as the Claude scripts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["b", "c"])
    ap.add_argument("run")
    ap.add_argument("--model", required=True)
    ap.add_argument("--minutes", type=int, default=60)
    args = ap.parse_args()
    run = Path(args.run).resolve()
    clip = Path((run / "clip.txt").read_text(encoding="utf-8").strip())
    start = _iso((run / "prompt.txt").stat().st_mtime)
    end = _iso(datetime.now().timestamp())
    if args.kind == "b":
        return subprocess.call([sys.executable, str(REPO / "baselines" / "b_llm_mcp" / "finish_run.py"),
                                "--run", str(run), "--clip", str(clip), "--model", args.model,
                                "--mcp-version", "blend-ai (codex app)", "--start", start, "--end", end,
                                "--exit-code", "0", "--minutes", str(args.minutes), "--turns", "0"])
    from vrecon.evaluate import score
    from vrecon.run import service

    mgr, _, _ = service._manager(run)
    rec = {"model": args.model, "client": "codex app", "start": start, "end": end, "attempts": mgr.attempts,
           "revision": mgr.adopted.revision, "metrics": mgr.metrics, "history": service.history(str(run), 200),
           "score": score(clip, mgr.adopted)["summary"]}
    (run / "llm").mkdir(exist_ok=True)
    (run / "llm" / "result.json").write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("attempts", "revision", "metrics", "score")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
