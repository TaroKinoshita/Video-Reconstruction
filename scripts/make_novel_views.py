"""Ground truth for views the reference video never shows (evaluation of completion, docs/HANDOFF.md).

    python scripts/make_novel_views.py [--clips evaluation/synthetic/syn_01 ...] [--frames 0,36,72]

For each synthetic clip, renders the ground-truth scene from new cameras made from the clip's own camera path:
    orbit_l10 / orbit_l30 / orbit_r30   the path turned 10 / 30 deg about the vertical axis through the objects' centre
    back_1p5                every camera pulled back 1.5x from that centre
Writes <clip>/gt_passes/novel/<name>/{camera.json, index.npz, depth.npz, id_map.json} (git-ignored with gt_passes).
Evaluation data only: reconstruction code must never read it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

NOVEL = {"orbit_l10": ("orbit", 10.0), "orbit_l30": ("orbit", 30.0), "orbit_r30": ("orbit", -30.0), "back_1p5": ("back", 1.5)}


def novel_camera(gt, kind: str, value: float):
    from vrecon.scene.convert import matrix_to_quat, quat_to_matrix
    from vrecon.schema import CameraKey, Pose

    cam = gt.reconstruction.cameras[0]
    movers = [o for o in gt.reconstruction.objects if o.kind != "set"]
    pts = [np.array(k.pose.translation) for o in movers for k in (o.keys or [o.rest])]
    p = np.mean(pts, 0) if pts else np.zeros(3)
    a = math.radians(value) if kind == "orbit" else 0.0
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    keys = []
    for k in cam.keys:
        c = np.array(k.pose.translation)
        r = np.array(quat_to_matrix(k.pose.rotation))
        if kind == "orbit":
            c, r = rz @ (c - p) + p, rz @ r
        else:
            c = p + value * (c - p)
        keys.append(CameraKey(shot_frame=k.shot_frame, intrinsics=k.intrinsics,
                              pose=Pose(translation=tuple(map(float, c)), rotation=matrix_to_quat(r.tolist()))))
    return cam.model_copy(update={"keys": keys}), p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--frames", default="0,36,72")
    a = ap.parse_args()
    from vrecon.scene.builder import build_scene
    from vrecon.scene.passes import render_passes
    from vrecon.schema import load_project

    clips = a.clips or sorted(str(p) for p in Path("evaluation/synthetic").glob("syn_*"))
    frames = [int(f) for f in a.frames.split(",")]
    for clip in clips:
        gt = load_project(Path(clip) / "gt.json")
        n = gt.shots[0].end - gt.shots[0].start
        fr = [f for f in frames if f < n]
        for name, (kind, value) in NOVEL.items():
            cam, centre = novel_camera(gt, kind, value)
            proj = gt.model_copy(deep=True)
            proj.reconstruction.cameras = [cam]
            out = (Path(clip) / "gt_passes" / "novel" / name).resolve()
            build_scene(proj)
            render_passes(out, shot_frames=fr, shaded=False)
            (out / "camera.json").write_text(json.dumps({"camera": cam.model_dump(mode="json"), "frames": fr,
                                                         "kind": kind, "value": value, "centre": centre.tolist()},
                                                        indent=1) + "\n", encoding="utf-8")
            print(clip, name, "frames", fr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
