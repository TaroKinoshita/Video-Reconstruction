"""Synthetic clips with a fast rigid prop and a ball, rendered with motion blur, for the prop tools
(racket-like swing, toss and hit). Same layout as scripts/make_synthetic_clips.py (video.mp4, gt.json, meta.json,
gt_passes/); evaluation only: reconstruction must not read gt.json or gt_passes/.

    python scripts/make_prop_clips.py [--out evaluation/props] [--only prop_01] [--samples 16] [--passes-only]

prop_01: static camera. A paddle (0.68 x 0.27 x 0.03 m, a racket's size) held still, swung through about 200
degrees in 6 frames, held again; a ball (0.067 m) tossed up, hit at the swing, flying off; motion blur on.
prop_02: the same with a slow camera pan and a second toss that is not hit (bounces on the floor).
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import bpy
import numpy as np

from vrecon.control.core import encode_mp4, load_frames
from vrecon.schema import (Camera, CameraKey, FileRef, Intrinsics, Pose, Project, Rational, SceneObject, Shot,
                           ShotMapping, SourceVideo, TransformKey, save_project)
from vrecon.scene.builder import build_scene, render_frames
from vrecon.scene.convert import look_at_opencv
from vrecon.scene.look import apply_look
from vrecon.scene.passes import render_passes

W, H, FPS, FRAMES = 640, 352, 24, 73
G = 9.81
PADDLE = (0.68, 0.27, 0.03)
BALL = 0.067
SPECS = [{"id": "prop_01", "seed": 11, "camera": "static", "second_toss": False},
         {"id": "prop_02", "seed": 12, "camera": "pan", "second_toss": True}]


def _quat_axis(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = math.sin(a / 2)
    return (math.cos(a / 2), float(axis[0] * s), float(axis[1] * s), float(axis[2] * s))


def paddle_keys(swing_start: int, swing_len: int = 6):
    """Held at the side (handle down), then swung about a horizontal axis through the grip; one key a frame."""
    grip = np.array([0.45, 1.0, 1.0])
    keys = []
    for f in range(FRAMES):
        s = min(max((f - swing_start) / swing_len, 0.0), 1.0)
        s = 0.5 - 0.5 * math.cos(math.pi * s)  # ease in/out
        a = math.radians(-100 + 200 * s)        # swing angle about the x axis
        q = _quat_axis((1, 0, 0), a)
        # the paddle's long axis is its local x; turn it to point up first, then swing about world x
        up = _quat_axis((0, 1, 0), -math.pi / 2)
        r = _qmul(q, up)
        centre = grip + _rot(r, np.array([PADDLE[0] / 2, 0, 0]))
        keys.append(TransformKey(time=f / FPS, pose=Pose(translation=tuple(float(x) for x in centre), rotation=r)))
    return keys


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def _rot(q, v):
    w, x, y, z = q
    m = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return m @ v


def ball_keys(toss: int, hit: int, second_toss: bool):
    """Held, tossed straight up (apex 0.9 s later... cut short by the hit), hit towards +y/-x at 22 m/s;
    optionally a second toss that falls and bounces (restitution 0.7)."""
    p0 = np.array([0.45, 0.8, 1.2])
    keys, events = [], []
    v_up = G * (hit - toss) / FPS / 2 + 1.0  # still rising a little before the hit
    out_v = np.array([-6.0, 21.0, 2.0])
    p_hit = None
    for f in range(FRAMES):
        t = f / FPS
        if f <= toss:
            p = p0
        elif f <= hit:
            dt = t - toss / FPS
            p = p0 + np.array([0, 0, v_up * dt - G * dt * dt / 2])
            p_hit = p
        else:
            dt = t - hit / FPS
            p = p_hit + out_v * dt + np.array([0, 0, -G * dt * dt / 2])
        keys.append(TransformKey(time=t, pose=Pose(translation=tuple(float(x) for x in p))))
    events += [{"object": "obj_ball", "kind": "toss", "frame": toss}, {"object": "obj_ball", "kind": "hit", "frame": hit}]
    if second_toss:
        b0 = np.array([-0.6, 0.9, 1.3])
        t2 = 48
        bk = []
        for f in range(FRAMES):
            if f <= t2:
                p = b0
            else:
                p = _bounce(b0, 3.0, (f - t2) / FPS)
            bk.append(TransformKey(time=f / FPS, pose=Pose(translation=tuple(float(x) for x in p))))
        events.append({"object": "obj_ball_2", "kind": "toss", "frame": t2})
        return keys, bk, events
    return keys, None, events


def _bounce(p0, vz, t, e=0.7, r=BALL / 2):
    z, v, x = p0[2], vz, 0.0
    tt = 0.0
    while True:
        # time to hit the floor from z with velocity v
        disc = v * v + 2 * G * (z - r)
        th = (v + math.sqrt(max(disc, 0))) / G
        if tt + th >= t:
            dt = t - tt
            return np.array([p0[0], p0[1], z + v * dt - G * dt * dt / 2])
        tt += th
        v = (v - G * th) * -e
        z = r
        if abs(v) < 0.3:
            return np.array([p0[0], p0[1], r])


def make_project(spec: dict):
    f = 0.95 * W
    intr = Intrinsics(width=W, height=H, fx=f, fy=f, cx=(W - 1) / 2, cy=(H - 1) / 2)
    target = (0.2, 1.0, 1.6)
    cams = []
    for fr in sorted(set(range(0, FRAMES, 4)) | {FRAMES - 1}):
        s = fr / (FRAMES - 1)
        eye = (0.1, -3.4, 1.6)
        look = target if spec["camera"] == "static" else (target[0] - 0.6 + 1.2 * s, target[1], target[2])
        cams.append(CameraKey(shot_frame=fr, pose=Pose(translation=eye, rotation=look_at_opencv(eye, look))))
    swing, toss = 30, 14
    hit = swing + 3
    bk, bk2, events = ball_keys(toss, hit, spec["second_toss"])
    objects = [
        SceneObject(id="obj_floor", kind="set", size=(40, 20, 0.02), rest=TransformKey(time=0, pose=Pose(translation=(0, 0, -0.01)))),
        SceneObject(id="obj_wall", kind="set", size=(40, 0.2, 8), rest=TransformKey(time=0, pose=Pose(translation=(0, 6.0, 4.0)))),
        SceneObject(id="obj_paddle", kind="rigid", size=PADDLE, keys=paddle_keys(swing), status="human_confirmed",
                    look={"all": (0.85, 0.15, 0.12)}),
        SceneObject(id="obj_ball", kind="rigid", size=(BALL, BALL, BALL), keys=bk, status="human_confirmed",
                    look={"all": (0.85, 0.95, 0.2)}),
    ]
    if bk2:
        objects.append(SceneObject(id="obj_ball_2", kind="rigid", size=(BALL, BALL, BALL), keys=bk2,
                                   status="human_confirmed", look={"all": (0.85, 0.95, 0.2)}))
    events.append({"object": "obj_paddle", "kind": "swing", "frame": swing, "end_frame": swing + 6})
    project = Project(
        id=f"proj_{spec['id']}",
        source=SourceVideo(file=FileRef(id="file_video", path="video.mp4"), fps=Rational(num=FPS, den=1),
                           frame_count=FRAMES, width=W, height=H),
        shots=[Shot(id="shot_001", start=0, end=FRAMES, transition_from_previous="first")],
        reconstruction={"scale": {"source": "human", "note": "synthetic ground truth, meters"},
                        "cameras": [Camera(id="cam_001", shot_id="shot_001", intrinsics=intr, keys=cams, status="human_confirmed")],
                        "objects": objects,
                        "shot_mapping": [ShotMapping(shot_id="shot_001", scene_time_start=0.0)]})
    meta = {"id": spec["id"], "camera_motion": spec["camera"], "frames": FRAMES, "fps": FPS, "size": [W, H],
            "motion_blur": {"shutter_frames": 0.5}, "prop_events": events,
            "note": "ground truth for evaluation only; reconstruction must not read gt.json or gt_passes/"}
    return project, meta


def _prop_colours(project):
    """apply_look gives every mesh a random material; the props keep their plain colours (the masks key on them)."""
    for o in project.reconstruction.objects:
        if not o.look:
            continue
        for ob in bpy.data.objects:
            if ob.type == "MESH" and ob.get("vrecon_id") == o.id:
                m = bpy.data.materials.new(f"prop_{o.id}")
                m.use_nodes = True
                m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*o.look["all"], 1.0)
                ob.data.materials.clear()
                ob.data.materials.append(m)


def render_clip(spec, out: Path, samples: int):
    project, meta = make_project(spec)
    out.mkdir(parents=True, exist_ok=True)
    build_scene(project)
    apply_look(spec["seed"])
    _prop_colours(project)
    sc = bpy.context.scene
    sc.cycles.use_denoising = True
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5
    with tempfile.TemporaryDirectory() as tmp:
        paths = render_frames(tmp, list(range(FRAMES)), engine="CYCLES", samples=samples)
        encode_mp4(load_frames(paths), out / "video.mp4", fps=FPS, crf=16)
    build_scene(project)
    bpy.context.scene.render.use_motion_blur = False
    render_passes(out / "gt_passes", shaded=False)
    save_project(project, out / "gt.json")
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{spec['id']} -> {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/props")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--samples", type=int, default=16)
    a = ap.parse_args()
    for spec in SPECS:
        if a.only and spec["id"] not in a.only:
            continue
        render_clip(spec, Path(a.out) / spec["id"], a.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
