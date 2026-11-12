"""A person (GVHMR skeleton) in the 4RC scene of the same video: cameras and floor from 4RC, person on the floor.

    python scripts/place_person.py VIDEO SKELETON.npz 4RC.npz OUT_DIR [--render 0,100,200]

The 4RC run must be on the same frames GVHMR read (frames extracted with -fps_mode passthrough). The floor is
the largest plane of the 4RC points (no masks; the person covers a small part of the picture). Writes
OUT_DIR/project.json, skeleton.npz, placement.json, scene.blend, renders/, overlay_XXXXXX.png.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("skeleton")
    ap.add_argument("rc")
    ap.add_argument("out")
    ap.add_argument("--render", default="")
    a = ap.parse_args()

    from vrecon.analysis.from_4d import initialize, load_4rc
    from vrecon.analysis.place_person import place_in_scene
    from vrecon.control.core import read_video_rgb
    from vrecon.schema import FileRef, Project, Rational, Shot, ShotMapping, SourceVideo, save_project

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.skeleton, out / "skeleton.npz")
    video = read_video_rgb(a.video)
    t, h, w = video.shape[:3]
    sk = np.load(out / "skeleton.npz")
    fps = float(sk["fps"])
    if int(sk["frames"]) != t:
        raise SystemExit(f"skeleton has {int(sk['frames'])} frames, video {t}: not the same frames")
    with np.load(a.rc) as z:
        n = int(z["n_frames"])
    frames = [int(f) for f in np.linspace(0, t - 1, n, dtype=int)]
    m = load_4rc(a.rc, w, h, frames)
    cams, objs, scale, rep4 = initialize(m, np.zeros((t, h, w), np.uint8), {}, w, h, "shot_001", fps=fps)
    person, rep = place_in_scene(out / "skeleton.npz", "skeleton.npz", cams[0])
    project = Project(
        id="proj_person_scene",
        source=SourceVideo(file=FileRef(id="file_video", path=str(Path(a.video).resolve())),
                           fps=Rational(num=round(fps * 1000), den=1000), frame_count=t, width=w, height=h),
        shots=[Shot(id="shot_001", start=0, end=t, transition_from_previous="first")],
        reconstruction={"scale": scale, "cameras": cams, "objects": objs + [person],
                        "shot_mapping": [ShotMapping(shot_id="shot_001", scene_time_start=0.0)]},
    )
    save_project(project, out / "project.json")
    rep["4rc"] = {k: v for k, v in rep4.items() if k != "objects"}
    (out / "placement.json").write_text(json.dumps(rep, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps(rep, default=float))

    from vrecon.scene.builder import build_scene, render_frames, save_blend

    report = build_scene(project, root=out)
    print("\n".join(report.warnings))
    save_blend(out / "scene.blend")
    if a.render:
        import cv2

        from vrecon.control.core import load_frames

        fr = [int(f) for f in a.render.split(",")]
        for f, p in zip(fr, render_frames(out / "renders", fr, engine="CYCLES", resolution_percentage=50)):
            r = load_frames([p])[0]
            src = cv2.resize(video[f], (r.shape[1], r.shape[0]))
            blend = (0.5 * src + 0.5 * r).astype(np.uint8)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(np.concatenate([src, r, blend], 1), cv2.COLOR_RGB2BGR))
            buf.tofile(out / f"overlay_{f:06d}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
