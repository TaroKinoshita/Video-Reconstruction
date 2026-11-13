"""Phase D: an LLM writes the venue layout (venue/layout.py) from the source picture and the observation summary;
the same fit and build as the hand-made layout follow (scripts/build_venue.py).

    python scripts/llm_venue.py SOURCE.png OBSERVATIONS.json OUT_LAYOUT.json [--model claude-opus-5-5]
        [--extra "text about the clip"] [--image-prompt none|fal --outside OUTSIDE_DIR]

The model runs as `claude -p` in a temporary folder holding only source.png, observations.json and
layout_format.md, with the Read tool only (no ground truth, no other files). --image-prompt fal (phase E) adds the
outside pictures made by scripts/fal_outside.py (OUTSIDE_DIR: the pasted-back candidates of the given quality, the
mask of the source area and the placement); the model picks one, says why, and keeps unseen parts hypotheses. Writes the layout, the raw answer, cost and
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
{outside}Answer with ONLY the layout JSON object (site, assumptions, parts), no other text."""


OUTSIDE = """Also in this folder, pictures of what may lie OUTSIDE the camera's view, drawn by an image model:
{files}
  outside_mask.png        black = where the camera's picture (source.png) sits in them, white = drawn
  outside_placement.json  where source.png sits in them (box x, y, w, h in pixels) and its scale
They have the same camera, lens and perspective as source.png, extended outward; the black area is the real picture.
The drawn parts are guesses by the image model, not observations: use them only as suggestions for what is
outside the view (how far stands continue, corners, walls, aisles), never to move what observations.json
measured. Pick the ONE picture you find most plausible and consistent with source.png, do not mix them. Add to
the layout object "outside_choice": {{"file": ..., "why": ...}}. Parts suggested only by the drawn area stay
status "hypothesis"; say in "assumptions" when a hypothesis came from it.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("observations")
    ap.add_argument("out")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--extra", default="")
    ap.add_argument("--image-prompt", default="none", choices=["none", "fal"])
    ap.add_argument("--outside", default=None, help="fal_outside.py output folder (with --image-prompt fal)")
    ap.add_argument("--quality", default="medium", help="which outside candidates to show")
    a = ap.parse_args()
    if a.image_prompt == "fal" and not a.outside:
        raise SystemExit("--image-prompt fal needs --outside DIR (made by scripts/fal_outside.py)")
    from vrecon.venue import layout as lay
    from vrecon.venue.layout import check

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copyfile(a.source, Path(tmp) / "source.png")
        shutil.copyfile(a.observations, Path(tmp) / "observations.json")
        (Path(tmp) / "layout_format.md").write_text(lay.__doc__, encoding="utf-8")
        (Path(tmp) / "mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
        outside = ""
        if a.image_prompt == "fal":
            od = Path(a.outside)
            olog = json.loads((od / "outside.json").read_text(encoding="utf-8"))
            names = [c["name"] for c in olog["candidates"] if "error" not in c and c["quality"] == a.quality]
            for n in names:
                shutil.copyfile(od / f"{n}.png", Path(tmp) / f"outside_{n}.png")
            shutil.copyfile(od / "mask.png", Path(tmp) / "outside_mask.png")
            (Path(tmp) / "outside_placement.json").write_text(json.dumps(olog["placement"], indent=1), encoding="utf-8")
            outside = OUTSIDE.format(files="\n".join(f"  outside_{n}.png" for n in names))
        prompt = PROMPT.format(extra=("Note: " + a.extra) if a.extra else "", outside=outside)
        proc = subprocess.run(["claude", "-p", prompt, "--model", a.model, "--max-turns", "8", "--strict-mcp-config",
                               "--mcp-config", str(Path(tmp) / "mcp.json"), "--tools", "Read", "--allowedTools", "Read",
                               "--output-format", "json"], cwd=tmp, capture_output=True, text=True, encoding="utf-8",
                              timeout=900)
    res = json.loads(proc.stdout)
    text = res.get("result", "")
    i, j = text.find("{"), text.rfind("}")
    layout = json.loads(text[i:j + 1])
    probs = check(layout)
    log = {"model": a.model, "image_prompt": a.image_prompt, "outside": a.outside, "outside_choice": layout.get("outside_choice"), "cost_usd": res.get("total_cost_usd"), "turns": res.get("num_turns"),
           "problems": probs, "answer": text}
    Path(str(out) + ".log.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    out.write_text(json.dumps(layout, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"parts": len(layout.get("parts", [])), "problems": probs, "cost_usd": log["cost_usd"]}))
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
