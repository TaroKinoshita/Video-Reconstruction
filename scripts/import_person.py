"""Make a project that shows a person from a GVHMR skeleton file, then build it (a .blend and a few renders).

    python scripts/import_person.py SKELETON.npz OUT_DIR [--render 0,60,120]

OUT_DIR gets project.json (with the skeleton copied next to it), scene.blend and renders/. The camera is GVHMR's
estimate (static camera: the first frame's intrinsics, at the origin of the camera frame), placed in the
skeleton's world with the person's in-camera position, so the render can be laid over the input video.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("skeleton")
    ap.add_argument("out")
    ap.add_argument("--render", default="")
    ap.add_argument("--placement", choices=["camera", "world"], default="camera")
    a = ap.parse_args()

    from vrecon.analysis.from_gvhmr import character_from_gvhmr, load_skeleton
    from vrecon.schema import (Camera, CameraKey, FileRef, Intrinsics, Pose, Project, Rational, Shot, ShotMapping,
                               SourceVideo, save_project)

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.skeleton, out / "skeleton.npz")
    d = load_skeleton(out / "skeleton.npz")
    n, fps = int(d["frames"]), float(d["fps"])

    # camera: world pose from one frame's joints (camera frame vs world frame, Kabsch on the 22 joints)
    pc, pw = d["joints_cam"][0], d["joints_world"][0]
    mc, mw = pc.mean(0), pw.mean(0)
    u, _, vt = np.linalg.svd((pw - mw).T @ (pc - mc))
    r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt  # world <- camera
    c = mw - r @ mc
    from vrecon.scene.convert import matrix_to_quat

    qw, qx, qy, qz = matrix_to_quat(r.tolist())
    k = d["K"][0]
    intr = Intrinsics(fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]),
                      width=int(d["width"]), height=int(d["height"]))
    cam = Camera(id="cam_001", shot_id="shot_001", intrinsics=intr,
                 keys=[CameraKey(shot_frame=0, pose=Pose(translation=tuple(map(float, c)), rotation=(qw, qx, qy, qz)))])
    person = character_from_gvhmr(out / "skeleton.npz", "skeleton.npz",
                                  camera=None if a.placement == "world" else cam)
    fr = Rational(num=round(fps * 1000), den=1000)
    project = Project(
        id="proj_person", source=SourceVideo(file=FileRef(id="file_video", path=str(d["meta"]["video"])), fps=fr,
                                             frame_count=n, width=int(d["width"]), height=int(d["height"])),
        shots=[Shot(id="shot_001", start=0, end=n, transition_from_previous="first")],
        reconstruction={"scale": {"source": "estimated", "note": "GVHMR body model"},
                        "cameras": [cam],
                        "objects": [person], "shot_mapping": [ShotMapping(shot_id="shot_001", scene_time_start=0.0)]},
    )
    save_project(project, out / "project.json")

    from vrecon.scene.builder import build_scene, render_frames, save_blend

    report = build_scene(project, root=out)
    print("\n".join(report.warnings))
    save_blend(out / "scene.blend")
    if a.render:
        frames = [int(f) for f in a.render.split(",")]
        for p in render_frames((out / "renders").resolve(), frames, engine="CYCLES",
                               resolution_percentage=50):
            print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
