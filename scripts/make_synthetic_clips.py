"""Render synthetic reference clips with known ground truth (PLAN.md 11.8: checking the measuring tools).

    python scripts/make_synthetic_clips.py [--out evaluation/synthetic] [--only syn_01 ...] [--samples 16]

For each clip, writes to <out>/<clip>/:
    video.mp4     the "reference video" (textured, lit, Cycles on CPU)
    gt.json       ground-truth vrecon project (cameras, objects, timing); source file = video.mp4
    meta.json     what happens in the clip (camera motion type, motion onset per object)
    gt_passes/    exact depth / object-index passes (git-ignored, regenerate with this script)

These are evaluation data only: reconstruction code must never read gt.json or gt_passes/.
Deterministic: the same spec and seed give the same scene.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from pathlib import Path

import bpy  # noqa: F401  (rendering happens through vrecon.scene)

from vrecon.control.core import encode_mp4, load_frames
from vrecon.schema import (
    Camera,
    CameraKey,
    FileRef,
    Intrinsics,
    Pose,
    Project,
    Rational,
    SceneObject,
    Shot,
    ShotMapping,
    SourceVideo,
    TransformKey,
    save_project,
)
from vrecon.scene.builder import build_scene, render_frames
from vrecon.scene.convert import look_at_opencv
from vrecon.scene.look import apply_look
from vrecon.scene.passes import render_passes

W, H, FPS, FRAMES = 640, 352, 24, 73  # 73 = 17*4+5, about 3 s

# camera motion types: static, pan, dolly, truck, orbit, handheld
SPECS = [
    {"id": "syn_01", "seed": 1, "camera": "static", "objects": 1},
    {"id": "syn_02", "seed": 2, "camera": "pan", "objects": 2},
    {"id": "syn_03", "seed": 3, "camera": "dolly", "objects": 1},
    {"id": "syn_04", "seed": 4, "camera": "truck", "objects": 2},
    {"id": "syn_05", "seed": 5, "camera": "orbit", "objects": 3},
    {"id": "syn_06", "seed": 6, "camera": "handheld", "objects": 2},
]


def camera_path(kind: str, rng: random.Random, target):
    keys = []
    shake = [(rng.uniform(0, 6.3), rng.uniform(0.6, 1.4)) for _ in range(3)]
    for f in list(range(0, FRAMES, 4)) + [FRAMES - 1]:
        if keys and keys[-1].shot_frame == f:
            continue
        s = f / (FRAMES - 1)
        look = list(target)
        if kind == "static":
            eye = (0.5, -6.0, 1.7)
        elif kind == "pan":
            eye = (0.0, -6.0, 1.6)
            look = (target[0] - 1.5 + 3.0 * s, target[1], target[2])
        elif kind == "dolly":
            eye = (0.0, -7.5 + 2.5 * s, 1.5)
        elif kind == "truck":
            eye = (-2.0 + 4.0 * s, -6.5, 1.6)
        elif kind == "orbit":
            a = math.radians(-35 + 70 * s)
            eye = (target[0] + 6.5 * math.sin(a), target[1] - 6.5 * math.cos(a), 2.0)
        elif kind == "handheld":
            t = f / FPS
            eye = tuple(b + 0.04 * math.sin(2 * math.pi * fr * t + ph)
                        for b, (ph, fr) in zip((0.0, -6.0, 1.6), shake))
            look = (target[0] + 0.05 * math.sin(3.1 * t), target[1], target[2] + 0.04 * math.sin(2.3 * t + 1))
        else:
            raise ValueError(kind)
        keys.append(CameraKey(shot_frame=f, pose=Pose(translation=eye, rotation=look_at_opencv(eye, look))))
    return keys


def make_project(spec: dict) -> tuple[Project, dict]:
    rng = random.Random(spec["seed"])
    target = (0.0, 1.0, 0.5)
    f = rng.uniform(0.75, 1.1) * W  # focal length in pixels
    intr = Intrinsics(width=W, height=H, fx=f, fy=f, cx=(W - 1) / 2, cy=(H - 1) / 2)
    objects = [
        SceneObject(id="obj_floor", kind="set", size=(40, 20, 0.02), rest=TransformKey(time=0, pose=Pose(translation=(0, 0, -0.01)))),
        SceneObject(id="obj_wall", kind="set", size=(40, 0.2, 8), rest=TransformKey(time=0, pose=Pose(translation=(0, 5.1, 4.0)))),
    ]
    events = []
    for i in range(spec["objects"]):
        size = (rng.uniform(0.4, 1.0), rng.uniform(0.4, 1.0), rng.uniform(0.4, 1.4))
        y = rng.uniform(0.0, 2.5)
        x0, x1 = rng.uniform(-2.2, -0.8), rng.uniform(0.8, 2.2)
        if rng.random() < 0.5:
            x0, x1 = x1, x0
        onset = rng.randint(6, 30)
        dur = rng.randint(24, FRAMES - 1 - onset)
        yaw0, yaw1 = 0.0, rng.uniform(-1.2, 1.2)
        q = lambda a: (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))  # noqa: E731  yaw about z
        oid = f"obj_{i + 1:02d}"
        objects.append(SceneObject(
            id=oid, kind="rigid", size=size,
            keys=[
                TransformKey(time=onset / FPS, pose=Pose(translation=(x0, y, size[2] / 2), rotation=q(yaw0))),
                TransformKey(time=(onset + dur) / FPS, pose=Pose(translation=(x1, y, size[2] / 2), rotation=q(yaw1))),
            ],
            status="human_confirmed",
        ))
        events.append({"object": oid, "onset_frame": onset, "end_frame": onset + dur})
    project = Project(
        id=f"proj_{spec['id']}",
        source=SourceVideo(file=FileRef(id="file_video", path="video.mp4"), fps=Rational(num=FPS, den=1),
                           frame_count=FRAMES, width=W, height=H),
        shots=[Shot(id="shot_001", start=0, end=FRAMES, transition_from_previous="first")],
        reconstruction={
            "scale": {"source": "human", "note": "synthetic ground truth, meters"},
            "cameras": [Camera(id="cam_001", shot_id="shot_001", intrinsics=intr,
                               keys=camera_path(spec["camera"], rng, target), status="human_confirmed")],
            "objects": objects,
            "shot_mapping": [ShotMapping(shot_id="shot_001", scene_time_start=0.0)],
        },
    )
    meta = {"id": spec["id"], "seed": spec["seed"], "camera_motion": spec["camera"], "frames": FRAMES, "fps": FPS,
            "size": [W, H], "events": events,
            "note": "ground truth for evaluation only; reconstruction must not read gt.json or gt_passes/"}
    return project, meta


def render_clip(spec: dict, out: Path, samples: int) -> None:
    project, meta = make_project(spec)
    out.mkdir(parents=True, exist_ok=True)
    build_scene(project)
    apply_look(spec["seed"])
    scene = bpy.context.scene
    scene.cycles.use_denoising = True
    with tempfile.TemporaryDirectory() as tmp:
        paths = render_frames(tmp, list(range(FRAMES)), engine="CYCLES", samples=samples)
        encode_mp4(load_frames(paths), out / "video.mp4", fps=FPS, crf=16)
    build_scene(project)
    render_passes(out / "gt_passes", shaded=False)
    save_project(project, out / "gt.json")
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{spec['id']}: {spec['camera']}, {spec['objects']} object(s) -> {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/synthetic")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--passes-only", action="store_true",
                    help="only (re)render missing gt_passes from the committed gt.json (fast; for new sessions)")
    args = ap.parse_args()
    for spec in SPECS:
        if args.only and spec["id"] not in args.only:
            continue
        out = Path(args.out) / spec["id"]
        if args.passes_only:
            if not (out / "gt_passes" / "index.npz").is_file():
                from vrecon.schema import load_project

                build_scene(load_project(out / "gt.json"))
                render_passes(out / "gt_passes", shaded=False)
                print(f"{spec['id']}: passes rendered", flush=True)
            continue
        render_clip(spec, out, args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
