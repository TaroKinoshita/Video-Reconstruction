"""Phase D: an LLM writes the venue layout (venue/layout.py) from the source picture and the observation summary;
the same fit and build as the hand-made layout follow (scripts/build_venue.py).

    python scripts/llm_venue.py SOURCE.png OBSERVATIONS.json OUT_LAYOUT.json [--model claude-opus-5-5]
        [--extra "text about the clip"] [--image-prompt none|fal]

The model runs as `claude -p` in a temporary folder holding only source.png, observations.json and
layout_format.md, with the Read tool only (no ground truth, no other files). --image-prompt fal is phase E and not
wired yet: only `none` (no generated outside picture) is accepted. Writes the layout, the raw answer, cost and
turns (OUT_LAYOUT.log.json).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROMPT = """You are laying out a whole venue as an editable 3D scene, from one camera's picture. The camera saw only
part of it; the scene must also hold up when the camera pulls back, turns sideways or looks behind itself, so lay
out the whole venue: what the picture shows, and plausible guesses for what it does not.

Files in this folder:
  source.png          the camera's picture
  observations.json   what the picture measured, in court metres (origin at the court centre, X across, Y along,
                      away from the camera is +Y, Z up): the camera position, and per observed thing (segmenter
                      labels, hints only) its centroid, extent, height range, and for large things a front line.
                      Far away the depths carry errors of metres; the court and net positions come from the
                      court lines and are exact (standard).
  layout_format.md    the layout format: part types, their numbers, statuses

Rules:
- Court and net: type court / net, status "standard" (they are placed by the rules, do not move them).
- Things the picture shows: status "observed"; add "fit": LABEL (a key of observations.json "things") when the
  part is that thing, so its position is fitted to the measurement.
- Things the picture does not show: status "hypothesis", and say why in "assumptions".
- Nothing may stand between the camera and what it sees, and the camera itself must stay outside every part.
- Prefer few, large parts (whole stands, walls) over many small ones. Use the same units as observations.json.
{extra}
Answer with ONLY the layout JSON object (site, assumptions, parts), no other text."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("observations")
    ap.add_argument("out")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--extra", default="")
    ap.add_argument("--image-prompt", default="none", choices=["none", "fal", "fake"])
    a = ap.parse_args()
    if a.image_prompt != "none":
        raise SystemExit(f"--image-prompt {a.image_prompt}: not wired yet (phase E); use none")
    from vrecon.venue import layout as lay
    from vrecon.venue.layout import check

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copyfile(a.source, Path(tmp) / "source.png")
        shutil.copyfile(a.observations, Path(tmp) / "observations.json")
        (Path(tmp) / "layout_format.md").write_text(lay.__doc__, encoding="utf-8")
        (Path(tmp) / "mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
        prompt = PROMPT.format(extra=("Note: " + a.extra) if a.extra else "")
        proc = subprocess.run(["claude", "-p", prompt, "--model", a.model, "--max-turns", "8", "--strict-mcp-config",
                               "--mcp-config", str(Path(tmp) / "mcp.json"), "--tools", "Read", "--allowedTools", "Read",
                               "--output-format", "json"], cwd=tmp, capture_output=True, text=True, encoding="utf-8",
                              timeout=900)
    res = json.loads(proc.stdout)
    text = res.get("result", "")
    i, j = text.find("{"), text.rfind("}")
    layout = json.loads(text[i:j + 1])
    probs = check(layout)
    log = {"model": a.model, "image_prompt": a.image_prompt, "cost_usd": res.get("total_cost_usd"), "turns": res.get("num_turns"),
           "problems": probs, "answer": text}
    Path(str(out) + ".log.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    out.write_text(json.dumps(layout, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"parts": len(layout.get("parts", [])), "problems": probs, "cost_usd": log["cost_usd"]}))
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
