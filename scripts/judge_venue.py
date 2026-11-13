"""Plausibility of a completed venue (docs/VENUE_PLAN.md, phase A questions), judged by an LLM from pictures.

    python scripts/judge_venue.py VIEWS.png SOURCE.png OUT.json [--model claude-opus-5-5]

VIEWS.png: one row of renders from the fixed cameras (scripts/clay_views.py, source first). The judge sees only
the two pictures (no method name) and answers the seven questions with yes / no / unsure and one line each.
Several answers can be right for the unseen side: this scores plausibility, not agreement with any ground truth.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

QUESTIONS = [
    "Does anything float in the air without support?",
    "Do objects pass through each other?",
    "Is anything cut off, with a hole or a missing piece where there should be structure?",
    "Are the stands stepped and facing the court?",
    "Are there aisles or entrances, so that people could get in and move around?",
    "Does this place look like a tennis venue?",
    "In the first render (the original camera), does the scene keep what source.png shows (court, net, people, the stand and wall behind)?",
]
GOOD = ["no", "no", "no", "yes", "yes", "yes", "yes"]

PROMPT = """Two pictures are in this folder. source.png is a frame from a video. views.png is one row of plain renders
(simple shapes, flat colours) of a 3D scene rebuilt from that video, seen from: the original camera, pulled back and
up, from the side, from high above, and turned around (looking where the original camera never looked).
Judge the 3D scene. Answer each question with "yes", "no" or "unsure" and one short reason, as a JSON list of
objects {"q": number, "answer": ..., "reason": ...}. Simple shapes and flat colours are expected; do not count them
against the scene.
""" + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(QUESTIONS)) + "\nAnswer with ONLY the JSON list."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("views")
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--model", default="claude-opus-5-5")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copyfile(a.views, Path(tmp) / "views.png")
        shutil.copyfile(a.source, Path(tmp) / "source.png")
        (Path(tmp) / "mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
        proc = subprocess.run(["claude", "-p", PROMPT, "--model", a.model, "--max-turns", "6", "--strict-mcp-config",
                               "--mcp-config", str(Path(tmp) / "mcp.json"), "--tools", "Read", "--allowedTools", "Read",
                               "--output-format", "json"], cwd=tmp, capture_output=True, text=True, encoding="utf-8",
                              timeout=600)
    res = json.loads(proc.stdout)
    text = res.get("result", "")
    ans = json.loads(text[text.find("["):text.rfind("]") + 1])
    good = sum(1 for x in ans if 1 <= int(x["q"]) <= 7 and str(x["answer"]).lower() == GOOD[int(x["q"]) - 1])
    out = {"answers": ans, "good": good, "of": 7, "cost_usd": res.get("total_cost_usd")}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"good": good, "answers": [(x["q"], x["answer"]) for x in ans]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
