"""Render category label images of a finished multi-cut .blend (bpy side of scripts/multishot_check.py).

    python scripts/multishot_labels.py SCENE.blend OUT.npz WIDTH HEIGHT FRAMES_JSON [--hide PREFIX ...]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("blend", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("w", type=int)
ap.add_argument("h", type=int)
ap.add_argument("frames", help="JSON list of timeline frames")
ap.add_argument("--hide", nargs="*", default=[])
a = ap.parse_args()
import bpy  # noqa: E402

from vrecon.scene.multishot import render_labels  # noqa: E402

bpy.ops.wm.open_mainfile(filepath=str(a.blend.resolve()))
render_labels(a.out.resolve(), json.loads(a.frames), (a.w, a.h), tuple(a.hide))
print(json.dumps({"out": str(a.out)}))
