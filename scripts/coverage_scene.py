"""Plan step 3 on a reconstructed scene: which faces the source video saw, drawn from the source camera and from
cameras orbited around the scene.

    python scripts/coverage_scene.py PROJECT.json OUT_DIR [--orbits 30,-30] [--frames 0,120,240] [--step 13]

The orbit pivot is the first character's position (the subject a camera move goes around), else where the first
camera's centre ray meets the floor (z = 0); every camera key is turned about the vertical axis through it. Writes OUT_DIR/<view>/ (renders of the coverage colours: green observed, yellow
oblique, red unseen) and OUT_DIR/coverage.json (per object shares).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def orbit(cam, deg: float, pivot: np.ndarray):
    from vrecon.metrics.poses import _rot
    from vrecon.scene.convert import matrix_to_quat
    from vrecon.schema import Pose

    a = np.radians(deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    keys = []
    for k in cam.keys:
        t = rz @ (np.array(k.pose.translation) - pivot) + pivot
        r = rz @ np.asarray(_rot(k.pose.rotation))
        keys.append(k.model_copy(update={"pose": Pose(translation=tuple(map(float, t)), rotation=matrix_to_quat(r.tolist()))}))
    return cam.model_copy(update={"keys": keys})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("out")
    ap.add_argument("--orbits", default="30,-30")
    ap.add_argument("--frames", default="0,120,240")
    ap.add_argument("--step", type=int, default=13)
    a = ap.parse_args()
    from vrecon.metrics.poses import camera_pose
    from vrecon.scene.coverage import render_coverage
    from vrecon.schema import load_project

    proj_path = Path(a.project).resolve()
    p = load_project(proj_path)
    out = Path(a.out).resolve()
    cam = p.reconstruction.cameras[0]
    r, c, _ = camera_pose(cam, 0)
    d = np.asarray(r)[:, 2]
    pivot = c + d * (-c[2] / d[2]) if d[2] < -1e-6 else c + d * 3 * abs(c[2])
    ch = next((o for o in p.reconstruction.objects if o.kind == "character"), None)
    if ch is not None:
        pivot = np.array((ch.keys[0] if ch.keys else ch.rest).pose.translation, float)
    pivot[2] = 0.0
    n = p.source.frame_count
    src = list(range(0, n, a.step))
    frames = [int(f) for f in a.frames.split(",")]
    res = {"pivot": pivot.round(4).tolist(), "views": {}}
    views = {"source": None} | {f"orbit{int(x):+d}": orbit(cam, float(x), pivot) for x in a.orbits.split(",")}
    for name, view in views.items():
        paths, summary = render_coverage(p, out / name, src, view=view, view_frames=frames, root=proj_path.parent)
        res["views"][name] = [str(x) for x in paths]
        res["coverage"] = summary
    (out / "coverage.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items() if kk != "faces"} for k, v in res["coverage"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
