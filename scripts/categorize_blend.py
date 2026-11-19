"""Object categories for a .blend's things (semantic proxy colours, coarse/fine people; PLAN v2 7.6, after CWM 2.0).

    python scripts/categorize_blend.py IN.blend OUT.json [--model claude-sonnet-5-5]

The top-level objects' names (with their child mesh names as hints) go to `claude -p` (no tools, names only, no
image, nothing else from the project) with a fixed list of categories; the answer is checked against the list.
Writes {"categories": {top_name: category}, "model", "cost_usd"}. A bare draft and its improved copy share the names,
so one file serves both (copies "Name.001" take "Name").
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CATEGORIES = {
    "person": "a human (any part of one: body, limbs, clothing, hair)",
    "held": "something a person holds or pulls (racket, bat, string, leash, bag in hand, toy being pulled)",
    "ground": "floor, ground, court surface, road, field, court lines on it",
    "structure": "walls, stands, seating rows, buildings, fences, railings, nets, pillars, doors, windows",
    "furniture": "beds, tables, chairs, benches, shelves, baskets, large containers",
    "prop": "small loose objects (toys, balls, bottles, boxes, blocks)",
    "vehicle": "cars, bikes, carts",
    "nature": "trees, plants, rocks, water",
    "sky": "sky, distant background, backdrop planes",
}


def top_names(blend: Path) -> dict[str, list[str]]:
    import bpy

    bpy.ops.wm.open_mainfile(filepath=str(blend.resolve()))

    def top(o):
        while o.parent is not None:
            o = o.parent
        return o

    out: dict[str, list[str]] = {}
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        out.setdefault(top(o).name.split(".")[0], []).append(o.name.split(".")[0])
    return {k: sorted(set(v))[:12] for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("blend", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    a = ap.parse_args()
    names = top_names(a.blend)
    prompt = (
        "Classify each object of a 3D scene by its name. Categories:\n"
        + "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
        + "\n\nObjects (top-level name: names of its parts):\n"
        + "\n".join(f"- {k}: {', '.join(v)}" for k, v in sorted(names.items()))
        + "\n\nAnswer with JSON only: {\"<top-level name>\": \"<category>\", ...} for every object listed."
    )
    r = subprocess.run(["claude", "-p", prompt, "--model", a.model, "--output-format", "json", "--max-turns", "1",
                        "--allowedTools", ""], capture_output=True, encoding="utf-8", errors="replace")
    res = json.loads(r.stdout)
    text = res.get("result", "")
    text = text[text.index("{"): text.rindex("}") + 1]
    cats = json.loads(text)
    cats = {k: (v if v in CATEGORIES else "prop") for k, v in cats.items() if k in names}
    for k in names:
        cats.setdefault(k, "prop")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"categories": cats, "model": a.model, "cost_usd": res.get("total_cost_usd")},
                                indent=1) + "\n", encoding="utf-8")
    print(json.dumps(cats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
