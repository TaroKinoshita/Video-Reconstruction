"""Build the finished multi-cut .blend and render the semantic control (and optionally check frames).

    python scripts/multishot_render.py IN.npz OUT.blend OUT_control.mp4 [--frames N] [--check DIR F F ...]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("npz", type=Path)
ap.add_argument("blend", type=Path)
ap.add_argument("control", type=Path)
ap.add_argument("--frames", type=int, default=0, help="17n+5; default: the most that fit")
ap.add_argument("--check", nargs="+", help="DIR then timeline frames: semantic PNGs only, no video")
a = ap.parse_args()
import bpy  # noqa: E402
import numpy as np  # noqa: E402

from vrecon.scene.multishot import build, render_semantic  # noqa: E402

r = build(a.npz.resolve(), a.blend.resolve(), finished=True)
z = np.load(a.npz)
size = (int(z["W"]) // 32 * 32, int(z["H"]) // 32 * 32)
if a.check:
    d = Path(a.check[0]).resolve()
    d.mkdir(parents=True, exist_ok=True)
    sc = bpy.context.scene
    render_semantic(None, 0, size)  # sets up the look only
    for f in map(int, a.check[1:]):
        sc.frame_set(f)
        sc.render.filepath = str(d / f"{f:04d}.png")
        bpy.ops.render.render(write_still=True)
else:
    n = a.frames or ((r["frames"] - 5) // 17 * 17 + 5)
    render_semantic(a.control.resolve(), n, size)
print(r)
