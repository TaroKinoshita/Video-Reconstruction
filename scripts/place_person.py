"""A person (GVHMR skeleton) in the 4RC scene of the same video: cameras and floor from 4RC, person on the floor.

    python scripts/place_person.py VIDEO SKELETON.npz 4RC.npz OUT_DIR [--render 0,100,200] [--controls depth,canny,clay,id,look]

The 4RC run must be on the same frames GVHMR read (frames extracted with -fps_mode passthrough). The floor is
the largest plane of the 4RC points (no masks; the person covers a small part of the picture). Writes
OUT_DIR/project.json, skeleton.npz, background.npz, placement.json, scene.blend, renders/, overlay_XXXXXX.png,
and with --controls the control videos in OUT_DIR/controls (vrecon.scene.controls).

The scene is filled with: the 4RC floor, a background mesh from the 4RC points (the person's box left out in each
sampled frame), the person; rough appearance colours from the video (person per body part, floor).
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
    ap.add_argument("--controls", default="")
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

    # where the person is in the picture: GVHMR's joint pixels; boxes around them are left out of the background
    from vrecon.analysis.appearance import character_look, region_look
    from vrecon.analysis.background import background_mesh

    pc, kg = sk["joints_cam"], sk["K"]
    uv = np.stack([(pc[i, :, :2] / pc[i, :, 2:]) * [kg[i][0, 0], kg[i][1, 1]] + [kg[i][0, 2], kg[i][1, 2]] for i in range(t)])

    def box(i, grow=0.3):
        lo, hi = uv[i].min(0), uv[i].max(0)
        size = hi - lo
        return (lo[0] - grow * size[0] - 20, lo[1] - grow * size[1] - 40, hi[0] + grow * size[0] + 20, hi[1] + grow * size[1] + 20)

    A = np.array(rep4["alignment"])
    floor_tol = 0.03 * abs(rep4["camera_height"])
    exclude = {k: [box(f)] for k, f in enumerate(frames)}
    verts, faces, colors, rep_bg = background_mesh(m, A, video, exclude, floor_tol=floor_tol)
    np.savez_compressed(out / "background.npz", verts=verts, faces=faces, colors=colors)
    from vrecon.schema import Evidence, FileRef as FR, SceneObject

    background = SceneObject(id="obj_background", name="background", kind="set",
                             asset=FR(id="file_background", path="background.npz", media_type="application/x-vrecon-mesh+npz"),
                             size=tuple(float(x) for x in np.ptp(verts, 0)),
                             evidence=[Evidence(observation_id=None, note="4RC point maps, median over sampled frames, person boxes left out")])
    names = [str(x) for x in sk["joint_names"]]
    person.look = character_look(video, uv, names, [int(p) for p in sk["parents"]], frames=range(0, t, 3))
    floor = objs[0]
    va, _, ca, _ = background_mesh(m, A, video, exclude)
    on_floor = np.abs(va[:, 2]) < floor_tol
    if on_floor.sum() > 100:
        floor.look = {"all": tuple(float(c) for c in np.median(ca[on_floor], 0))}
    objs = [floor] + objs[1:] + [background]
    rep["background"] = rep_bg
    rep["looks"] = {"person": person.look, "floor": floor.look}
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

    if a.controls:
        from vrecon.scene.controls import export_controls

        man = export_controls(project, out / "controls", kinds=tuple(a.controls.split(",")), preview=False, root=out)
        print(json.dumps({k: v["path"] for k, v in man["controls"].items()}))
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
