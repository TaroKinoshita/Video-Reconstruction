"""Synthetic scenes for the completion comparison (plan step 6: none / rules / LLM).

    python scripts/make_completion_clips.py [--out evaluation/completion]

Each scene is still; the source camera sees only part of it, a new camera ("novel/wide") shows the rest:
    cmp_01 symmetric   a table with a chair on each side; the source sees the table and the left chair only
    cmp_02 repeat      six equal crates in a row; the source sees the first three
    cmp_03 nothing     two boxes on the left; the hidden right side is empty (adding anything there is wrong)

Writes <out>/<id>/: gt.json, meta.json (what is hidden: evaluation only), source.png (textured render of the
source view: what a model may look at), start.json (the starting reconstruction: the objects the source camera
sees, with small errors, and a floor and wall cut to the source view), gt_passes/ (source and novel passes).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import bpy  # noqa: F401

from vrecon.control.core import load_frames
from vrecon.scene.builder import build_scene, render_frames
from vrecon.scene.convert import look_at_opencv
from vrecon.scene.look import apply_look
from vrecon.scene.passes import render_passes
from vrecon.schema import (Camera, CameraKey, FileRef, Intrinsics, Pose, Project, Rational, SceneObject, Shot,
                           ShotMapping, SourceVideo, TransformKey, save_project)

W, H, FRAMES = 640, 352, 9


def box(oid, size, x, y, yaw=0.0):
    import math

    return SceneObject(id=oid, kind="rigid", size=size, status="human_confirmed",
                       rest=TransformKey(time=0.0, pose=Pose(translation=(x, y, size[2] / 2),
                                                             rotation=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))))


def scene(cid: str):
    objs = [SceneObject(id="obj_floor", kind="set", size=(30, 20, 0.02), rest=TransformKey(time=0, pose=Pose(translation=(0, 0, -0.01)))),
            SceneObject(id="obj_wall", kind="set", size=(30, 0.2, 6), rest=TransformKey(time=0, pose=Pose(translation=(0, 6.1, 3.0))))]
    if cid == "cmp_01":
        objs += [box("obj_table", (1.6, 0.8, 0.75), 0.0, 2.0), box("obj_chair_l", (0.5, 0.5, 0.9), -1.3, 2.0),
                 box("obj_chair_r", (0.5, 0.5, 0.9), 1.3, 2.0)]
        src, look, novel, nlook = (-2.6, -3.0, 1.5), (-1.7, 2.0, 0.45), (0.0, -4.5, 1.8), (0.0, 2.0, 0.4)
        hidden, structure = ["obj_chair_r"], "mirror of obj_chair_l across x = 0 (the table's centre)"
    elif cid == "cmp_02":
        objs += [box(f"obj_crate_{i + 1}", (0.7, 0.7, 0.7), -3.0 + 1.2 * i, 2.5) for i in range(6)]
        src, look, novel, nlook = (-3.4, -2.6, 1.4), (-2.3, 2.5, 0.35), (0.0, -5.5, 2.0), (0.0, 2.5, 0.35)
        hidden, structure = ["obj_crate_4", "obj_crate_5", "obj_crate_6"], "row continues: x = -3.0 + 1.2 i, i = 0..5"
    else:
        objs += [box("obj_box_a", (0.8, 0.6, 0.6), -2.4, 2.0, 0.3), box("obj_box_b", (0.5, 0.5, 1.1), -1.2, 2.8)]
        src, look, novel, nlook = (-2.8, -2.8, 1.5), (-1.4, 2.4, 0.5), (0.0, -5.0, 1.8), (0.0, 2.4, 0.4)
        hidden, structure = [], "nothing is hidden: the right side is empty"
    f = 1.4 * W  # source: narrow, so part of the scene stays out of the picture
    intr = Intrinsics(width=W, height=H, fx=f, fy=f, cx=(W - 1) / 2, cy=(H - 1) / 2)
    nintr = Intrinsics(width=W, height=H, fx=0.8 * W, fy=0.8 * W, cx=(W - 1) / 2, cy=(H - 1) / 2)
    cam = Camera(id="cam_001", shot_id="shot_001", intrinsics=intr, status="human_confirmed",
                 keys=[CameraKey(shot_frame=0, pose=Pose(translation=src, rotation=look_at_opencv(src, look)))])
    ncam = Camera(id="cam_001", shot_id="shot_001", intrinsics=nintr,
                  keys=[CameraKey(shot_frame=0, pose=Pose(translation=novel, rotation=look_at_opencv(novel, nlook)))])
    p = Project(id=f"proj_{cid}", source=SourceVideo(file=FileRef(id="file_video", path="source.png"), fps=Rational(num=24, den=1),
                                                     frame_count=FRAMES, width=W, height=H),
                shots=[Shot(id="shot_001", start=0, end=FRAMES, transition_from_previous="first")],
                reconstruction={"scale": {"source": "human", "note": "synthetic ground truth, meters"}, "cameras": [cam],
                                "objects": objs, "shot_mapping": [ShotMapping(shot_id="shot_001", scene_time_start=0.0)]})
    return p, ncam, {"id": cid, "hidden": hidden, "structure": structure,
                     "note": "evaluation only: reconstruction and the LLM must not read meta.json or gt.json"}


def start_from(p: Project, visible: set[str], seed: int) -> Project:
    """The source view's objects with small errors (position 5 cm, size 5%); floor and wall cut to a few metres
    around what the source sees (they are completed by the rule condition)."""
    rng = random.Random(seed)
    objs = []
    for o in p.reconstruction.objects:
        if o.kind == "set":
            size = (8.0, 8.0, 0.02) if o.id == "obj_floor" else (8.0, 0.2, 3.0)
            t = o.rest.pose.translation
            centre = (-1.5, 1.5, -0.01) if o.id == "obj_floor" else (-1.5, t[1], 1.5)
            objs.append(o.model_copy(update={"size": size, "status": "estimated",
                                             "rest": TransformKey(time=0, pose=Pose(translation=centre))}))
        elif o.id in visible:
            t = o.rest.pose.translation
            s = [v * (1 + rng.uniform(-0.05, 0.05)) for v in o.size]
            nt = (t[0] + rng.uniform(-0.05, 0.05), t[1] + rng.uniform(-0.05, 0.05), s[2] / 2)
            objs.append(o.model_copy(update={"size": tuple(s), "status": "estimated",
                                             "rest": o.rest.model_copy(update={"pose": o.rest.pose.model_copy(update={"translation": nt})})}))
    q = p.model_copy(deep=True)
    q.id = p.id.replace("proj_", "proj_start_")
    q.reconstruction.objects = objs
    return q


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/completion")
    a = ap.parse_args()
    for i, cid in enumerate(("cmp_01", "cmp_02", "cmp_03")):
        out = (Path(a.out) / cid).resolve()
        out.mkdir(parents=True, exist_ok=True)
        p, ncam, meta = scene(cid)
        save_project(p, out / "gt.json")
        build_scene(p)
        render_passes(out / "gt_passes", shot_frames=[0], shaded=False)
        idx = np.load(out / "gt_passes" / "index.npz")["index"][0]
        id_map = json.loads((out / "gt_passes" / "id_map.json").read_text(encoding="utf-8"))
        visible = {v for k, v in id_map.items() if (idx == int(k)).sum() > 50}
        meta["visible_in_source"] = sorted(visible)
        apply_look(10 + i)
        bpy.context.scene.cycles.use_denoising = True
        with tempfile.TemporaryDirectory() as tmp:
            png = render_frames(tmp, [0], engine="CYCLES", samples=16)[0]
            (out / "source.png").write_bytes(png.read_bytes())
        q = p.model_copy(deep=True)
        q.reconstruction.cameras = [ncam]
        build_scene(q)
        nd = out / "gt_passes" / "novel" / "wide"
        render_passes(nd, shot_frames=[0], shaded=False)
        (nd / "camera.json").write_text(json.dumps({"camera": ncam.model_dump(mode="json"), "frames": [0], "kind": "set",
                                                    "value": 0}, indent=1) + "\n", encoding="utf-8")
        save_project(start_from(p, visible, seed=100 + i), out / "start.json")
        (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        print(cid, "visible", sorted(visible), "hidden", meta["hidden"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
