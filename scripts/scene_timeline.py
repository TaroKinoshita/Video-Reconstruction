"""Time-coded action lines from a .blend's own motion (two-level text, after CWM 2.0's whole-window caption + local
actions): which thing moves, when, and which way as seen from the source camera, for an output window.

    python scripts/scene_timeline.py IN.blend CATEGORIES.json OUT.txt [--start 0] [--frames 124]

Per top-level thing of category person / held / prop / vehicle (and an observed body): its middle (bounding-box
centre of its meshes) at every output frame (24 fps, scene time start + i/24); speed above STILL (a share of the
camera-to-scene distance per second) is "moving"; runs of moving frames shorter than 0.25 s are dropped. Each run is
described without view-dependent words (a camera move changes left/right): goes up / comes down (world vertical
dominant) or moves along the floor.
Nothing per clip; names come from the scene (CamelCase and underscores split).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

STILL = 0.04  # share of the camera-to-scene distance per second


def nice(name: str) -> str:
    s = re.sub(r"[_\-.]+", " ", name)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).strip().lower()
    return re.sub(r"\s+\d+$", "", s) or name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("blend", type=Path)
    ap.add_argument("categories", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=124)
    a = ap.parse_args()
    import bpy
    from mathutils import Vector

    from vrecon.metrics.poses import camera_pose
    from vrecon.scene.novel_blend import open_scene

    cats = json.loads(a.categories.read_text(encoding="utf-8"))["categories"]
    rec = open_scene(a.blend)
    cam = rec.reconstruction.cameras[0]
    fps, n_src = rec.source.fps.value, rec.source.frame_count
    sc = bpy.context.scene

    def top(o):
        while o.parent is not None:
            o = o.parent
        return o

    groups: dict = {}
    for o in bpy.data.objects:
        if o.type != "MESH" or o.hide_render:
            continue
        t = top(o)
        observed = bool(t.get("vrecon_rig_of")) or o.get("vrecon_id") == "obj_person_observed"
        cat = "person" if observed else cats.get(t.name.split(".")[0])
        if cat in ("person", "held", "prop", "vehicle"):
            key = "person" if observed else nice(t.name.split(".")[0])
            groups.setdefault((key, cat), []).append(o)
    times = [a.start + i / 24.0 for i in range(a.frames)]
    r0, c0, _ = camera_pose(cam, min(a.start * fps, n_src - 1))
    tracks = {k: [] for k in groups}
    for t in times:
        f = min(t * fps, n_src - 1)
        sc.frame_set(sc.frame_start + int(f), subframe=float(f - int(f)))
        for k, parts in groups.items():
            pts = np.array([list(p.matrix_world @ Vector(c)) for p in parts for c in p.bound_box])
            tracks[k].append((pts.min(0) + pts.max(0)) / 2)
    mids = np.array([tr[0] for tr in tracks.values()]) if tracks else np.zeros((1, 3))
    dist = float(np.median(np.linalg.norm(mids - c0, axis=1))) or 1.0
    lines = []
    for (name, cat), tr in tracks.items():
        p = np.array(tr)
        v = np.gradient(p, axis=0) * 24.0
        moving = np.linalg.norm(v, axis=1) > STILL * dist
        i = 0
        while i < len(moving):
            if not moving[i]:
                i += 1
                continue
            j = i
            while j < len(moving) and moving[j]:
                j += 1
            if (j - i) / 24.0 >= 0.25:
                # view-independent words only (a camera move turns left/right around): world up / down, or along
                # the floor
                d = p[j - 1] - p[i]
                vert = abs(d[2]) > np.linalg.norm(d[:2])
                word = ("goes up" if d[2] > 0 else "comes down") if vert else "moves along the floor"
                # "moves", not "walks": a person's box centre also shifts when an arm goes up
                lines.append((times[i] - a.start, times[j - 1] - a.start, f"the {name} {word}"))
            i = j
    lines.sort()
    text = "\n".join(f"{s:.1f}-{e:.1f} s: {what}." for s, e, what in lines) or "Nothing moves."
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
