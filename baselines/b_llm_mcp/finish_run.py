"""After a baseline-B run: record what happened, check for ground-truth leaks, and score the result.

Writes RUN/run.json and, when the agent produced scene.blend, RUN/score.json.
A run counts as INVALID if the agent's log mentions ground-truth files (PLAN.md 11.8).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LEAK_MARKERS = ("gt.json", "gt_passes", "meta.json", "/evaluation/")


def agent_summary(log: Path) -> dict:
    """Totals from the stream-json log (the final 'result' event), and tool-call counts."""
    out: dict = {"tool_calls": {}, "result": None}
    if not log.is_file():
        return out
    for line in log.read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    out["tool_calls"][name] = out["tool_calls"].get(name, 0) + 1
        elif ev.get("type") == "result":
            out["result"] = {k: ev.get(k) for k in ("subtype", "is_error", "num_turns", "duration_ms",
                                                    "total_cost_usd", "usage")}
    return out


def leaks(log: Path) -> list[str]:
    if not log.is_file():
        return []
    text = log.read_text(errors="replace")
    return [m for m in LEAK_MARKERS if m in text]


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("run", "clip", "model", "mcp-version", "start", "end"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--exit-code", type=int, required=True)
    ap.add_argument("--minutes", type=int, required=True)
    ap.add_argument("--turns", type=int, required=True)
    a = ap.parse_args()
    run, clip = Path(a.run), Path(a.clip)
    out_dir = run / "workspace" / "output"
    deliverables = ["scene.blend", "reconstruction.mp4", "comparison/original_vs_reconstruction.mp4",
                    "comparison/original_vs_reconstruction_contact_sheet.jpg", "report.md"]
    found = {d: (out_dir / d).is_file() and (out_dir / d).stat().st_size > 0 for d in deliverables}
    leaked = leaks(run / "logs" / "agent.jsonl")

    def version(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return None

    record = {
        "condition": "B (LLM + blender-mcp)",
        "clip": clip.name,
        "model": a.model,
        "mcp_for_blender": a.mcp_version,
        "claude_cli": version(["claude", "--version"]),
        "blender": (run / "logs" / "blender_version.txt").read_text().strip()
        if (run / "logs" / "blender_version.txt").is_file() else None,
        "prompt": "baselines/b_llm_mcp/prompt_L1.txt (CineRecon L1, GPU made optional)",
        "budget": {"minutes": a.minutes, "max_turns": a.turns},
        "start": a.start,
        "end": a.end,
        "exit_code": a.exit_code,
        "timed_out": a.exit_code == 124,
        "deliverables": found,
        "agent": agent_summary(run / "logs" / "agent.jsonl"),
        "ground_truth_leak": leaked,
        "valid": not leaked,
    }
    (run / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("clip", "model", "exit_code", "timed_out", "valid")}, indent=2))
    if leaked:
        print(f"INVALID RUN: the agent log mentions {leaked}", file=sys.stderr)

    blend = out_dir / "scene.blend"
    if blend.is_file() and (clip / "gt_passes").is_dir():
        from vrecon.evaluate import score_blend

        try:
            report = score_blend(clip, blend)
        except Exception as exc:  # an unscoreable scene is itself a result
            report = {"error": f"{type(exc).__name__}: {exc}"}
        (run / "score.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report.get("summary", report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
