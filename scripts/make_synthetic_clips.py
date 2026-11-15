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


# Set v2 (docs/REVIEW_INSTRUCTIONS.md F-17, F-18): wider variety than v1, decided per clip so every feature appears
# at least twice. Clips come in pairs with the same camera motion; one clip of each pair goes to dev, the other to
# held-out (seeded coin per pair, SPLIT_SEED), so both halves see every camera motion.
#   shapes: box / cylinder / sphere / two_box (a box with a smaller box on top); surfaces: plain / wood / tiles /
#   checker for the floor (walls follow); extras per mover: tilt (leans 15-25 deg), stop (moves, holds, moves
#   again), from0 (already moving at frame 0). Camera "zoom": still camera, focal length changes by 1.4x.
SPLIT_SEED = 20261115
SPECS_V2 = [  # the two clips of a pair share camera motion, floor surface and kind of extra
    {"id": "syn2_01", "seed": 101, "camera": "static", "shapes": ["cylinder", "box"], "floor": "plain", "wall": "plain", "extras": {1: "stop"}},
    {"id": "syn2_02", "seed": 102, "camera": "static", "shapes": ["sphere"], "floor": "plain", "wall": "brick", "extras": {0: "stop"}},
    {"id": "syn2_03", "seed": 103, "camera": "pan", "shapes": ["two_box", "box"], "floor": "tiles", "wall": "brick", "extras": {1: "from0"}},
    {"id": "syn2_04", "seed": 104, "camera": "pan", "shapes": ["cylinder", "sphere", "box"], "floor": "tiles", "wall": "plain", "extras": {2: "from0"}},
    {"id": "syn2_05", "seed": 105, "camera": "truck", "shapes": ["box", "cylinder"], "floor": "checker", "wall": "brick", "extras": {1: "tilt"}},
    {"id": "syn2_06", "seed": 106, "camera": "truck", "shapes": ["two_box"], "floor": "checker", "wall": "plain", "extras": {0: "tilt"}},
    {"id": "syn2_07", "seed": 107, "camera": "dolly", "shapes": ["sphere", "box"], "floor": "wood", "wall": "plain", "extras": {1: "stop"}},
    {"id": "syn2_08", "seed": 108, "camera": "dolly", "shapes": ["cylinder"], "floor": "wood", "wall": "brick", "extras": {0: "stop"}},
    {"id": "syn2_09", "seed": 109, "camera": "zoom", "shapes": ["box", "two_box"], "floor": "plain", "wall": "brick", "extras": {0: "from0"}},
    {"id": "syn2_10", "seed": 110, "camera": "zoom", "shapes": ["sphere", "cylinder"], "floor": "plain", "wall": "plain", "extras": {1: "from0"}},
    {"id": "syn2_11", "seed": 111, "camera": "orbit", "shapes": ["box", "sphere", "cylinder"], "floor": "tiles", "wall": "plain", "extras": {0: "tilt"}},
    {"id": "syn2_12", "seed": 112, "camera": "handheld", "shapes": ["two_box", "box"], "floor": "tiles", "wall": "brick", "extras": {0: "tilt"}},
]


def split_v2() -> dict[str, list[str]]:
    """dev / held_out ids: one clip of each consecutive pair to each side, by a seeded coin."""
    rng = random.Random(SPLIT_SEED)
    dev, held = [], []
    for i in range(0, len(SPECS_V2), 2):
        a, b = SPECS_V2[i]["id"], SPECS_V2[i + 1]["id"]
        if rng.random() < 0.5:
            a, b = b, a
        dev.append(a)
        held.append(b)
    return {"dev": dev, "held_out": held}


def shape_mesh(shape: str, size) -> tuple:
    """verts [V,3], faces [F,3] in metres, centred on the bounding box (as the builder's proxy boxes are)."""
    import numpy as np

    sx, sy, sz = size

    def box(cx, cy, cz, dx, dy, dz, base=0):
        v = np.array([[cx + x * dx / 2, cy + y * dy / 2, cz + z * dz / 2]
                      for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)])
        f = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                      [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]]) + base
        return v, f

    if shape == "box":
        return box(0, 0, 0, sx, sy, sz)
    if shape == "two_box":  # lower box 60% of the height, upper box 40%, narrower and set back
        v1, f1 = box(0, 0, -sz / 2 + 0.3 * sz, sx, sy, 0.6 * sz)
        v2, f2 = box(0.15 * sx, 0.1 * sy, sz / 2 - 0.2 * sz, 0.5 * sx, 0.6 * sy, 0.4 * sz, base=8)
        return np.vstack([v1, v2]), np.vstack([f1, f2])
    n = 32
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if shape == "cylinder":
        ring = np.c_[sx / 2 * np.cos(a), sy / 2 * np.sin(a)]
        v = np.vstack([np.c_[ring, np.full(n, -sz / 2)], np.c_[ring, np.full(n, sz / 2)],
                       [[0, 0, -sz / 2], [0, 0, sz / 2]]])
        f = []
        for i in range(n):
            j = (i + 1) % n
            f += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
        return v, np.array(f)
    if shape == "sphere":
        rings = 16
        v = [[0, 0, -sz / 2]]
        for r in range(1, rings):
            t = np.pi * r / rings - np.pi / 2
            v += [[sx / 2 * np.cos(t) * np.cos(b), sy / 2 * np.cos(t) * np.sin(b), sz / 2 * np.sin(t)] for b in a]
        v.append([0, 0, sz / 2])
        f = []
        for i in range(n):
            j = (i + 1) % n
            f.append([0, 1 + j, 1 + i])
            top = len(v) - 1
            f.append([top, 1 + (rings - 2) * n + i, 1 + (rings - 2) * n + j])
        for r in range(rings - 2):
            for i in range(n):
                j = (i + 1) % n
                p, q = 1 + r * n, 1 + (r + 1) * n
                f += [[p + i, p + j, q + j], [p + i, q + j, q + i]]
        return np.array(v), np.array(f)
    raise ValueError(shape)


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
        elif kind == "zoom":  # still camera; the focal length is keyed in make_project_v2
            eye = (0.3, -6.2, 1.6)
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


def _yaw(a):
    return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))


def _quat_mul(q, r):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def make_project_v2(spec: dict, out: Path) -> tuple[Project, dict]:
    """Like make_project, with the shapes, surfaces and extras of SPECS_V2. Writes the mover meshes to
    <out>/gt_assets/<id>.npz (ground truth like gt.json: evaluation only)."""
    import numpy as np

    rng = random.Random(spec["seed"])
    target = (0.0, 1.0, 0.5)
    f = rng.uniform(0.75, 1.1) * W
    intr = Intrinsics(width=W, height=H, fx=f, fy=f, cx=(W - 1) / 2, cy=(H - 1) / 2)
    objects = [
        SceneObject(id="obj_floor", kind="set", size=(40, 20, 0.02), rest=TransformKey(time=0, pose=Pose(translation=(0, 0, -0.01)))),
        SceneObject(id="obj_wall", kind="set", size=(40, 0.2, 8), rest=TransformKey(time=0, pose=Pose(translation=(0, 5.1, 4.0)))),
    ]
    events = []
    (out / "gt_assets").mkdir(parents=True, exist_ok=True)
    lanes = rng.sample([0.0, 0.8, 1.6, 2.4], len(spec["shapes"]))  # depth lanes so movers do not run into each other
    for i, shape in enumerate(spec["shapes"]):
        extra = spec["extras"].get(i)
        if shape == "sphere":
            d = rng.uniform(0.45, 0.9)
            size = (d, d, d)
        elif shape == "cylinder":
            d = rng.uniform(0.35, 0.8)
            size = (d, d, rng.uniform(0.5, 1.4))
        else:
            size = (rng.uniform(0.4, 1.0), rng.uniform(0.4, 1.0), rng.uniform(0.5, 1.4))
        y = lanes[i]
        x0, x1 = rng.uniform(-2.2, -0.8), rng.uniform(0.8, 2.2)
        if rng.random() < 0.5:
            x0, x1 = x1, x0
        yaw1 = rng.uniform(-1.2, 1.2) if shape != "sphere" else 0.0
        tilt = math.radians(rng.uniform(15, 25)) if extra == "tilt" else 0.0
        lean = (math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0)  # about the object's x axis
        z = (size[1] / 2) * math.sin(tilt) + (size[2] / 2) * math.cos(tilt)  # lowest corner on the floor
        onset = 0 if extra == "from0" else rng.randint(6, 20 if extra == "stop" else 30)
        if extra == "stop":  # move, hold 10-16 frames, move again
            dur1, hold = rng.randint(12, 18), rng.randint(10, 16)
            dur2 = rng.randint(12, FRAMES - 1 - onset - dur1 - hold)
            xm = x0 + (x1 - x0) * 0.45
            times = [onset, onset + dur1, onset + dur1 + hold, onset + dur1 + hold + dur2]
            xs, yaws = [x0, xm, xm, x1], [0.0, yaw1 / 2, yaw1 / 2, yaw1]
        else:
            dur = rng.randint(24, FRAMES - 1 - onset)
            times, xs, yaws = [onset, onset + dur], [x0, x1], [0.0, yaw1]
        oid = f"obj_{i + 1:02d}"
        verts, faces = shape_mesh(shape, size)
        asset = out / "gt_assets" / f"{oid}.npz"
        np.savez_compressed(asset, verts=verts.astype(np.float32), faces=faces.astype(np.int32))
        objects.append(SceneObject(
            id=oid, kind="rigid", size=size,
            asset=FileRef(id=f"file_{oid}", path=f"gt_assets/{oid}.npz"),
            keys=[TransformKey(time=t / FPS, pose=Pose(translation=(x, y, z), rotation=_quat_mul(_yaw(a), lean)))
                  for t, x, a in zip(times, xs, yaws)],
            status="human_confirmed",
        ))
        ev = {"object": oid, "shape": shape, "onset_frame": onset, "end_frame": times[-1]}
        if extra:
            ev["extra"] = extra
        if extra == "stop":
            ev["hold"] = [times[1], times[2]]
        events.append(ev)
    keys = camera_path(spec["camera"], rng, target)
    if spec["camera"] == "zoom":
        keys = [k.model_copy(update={"intrinsics": intr.model_copy(update={
            "fx": f * (1 + 0.4 * k.shot_frame / (FRAMES - 1)), "fy": f * (1 + 0.4 * k.shot_frame / (FRAMES - 1))})})
            for k in keys]
    project = Project(
        id=f"proj_{spec['id']}",
        source=SourceVideo(file=FileRef(id="file_video", path="video.mp4"), fps=Rational(num=FPS, den=1),
                           frame_count=FRAMES, width=W, height=H),
        shots=[Shot(id="shot_001", start=0, end=FRAMES, transition_from_previous="first")],
        reconstruction={
            "scale": {"source": "human", "note": "synthetic ground truth, meters", "units_per_metre": 1.0,
                      "basis": "measured"},
            "cameras": [Camera(id="cam_001", shot_id="shot_001", intrinsics=intr, keys=keys, status="human_confirmed")],
            "objects": objects,
            "shot_mapping": [ShotMapping(shot_id="shot_001", scene_time_start=0.0)],
        },
    )
    meta = {"id": spec["id"], "set": "v2", "seed": spec["seed"], "camera_motion": spec["camera"], "frames": FRAMES,
            "fps": FPS, "size": [W, H], "surfaces": {"floor": spec["floor"], "wall": spec["wall"]},
            "split": next(k for k, v in split_v2().items() if spec["id"] in v), "events": events,
            "note": "ground truth for evaluation only; reconstruction must not read gt.json, gt_assets/ or gt_passes/"}
    return project, meta


def render_clip(spec: dict, out: Path, samples: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    v2 = "shapes" in spec
    project, meta = make_project_v2(spec, out) if v2 else make_project(spec)
    root = out if v2 else None
    build_scene(project, root=root)
    apply_look(spec["seed"], {"floor": spec["floor"], "wall": spec["wall"]} if v2 else None)
    scene = bpy.context.scene
    scene.cycles.use_denoising = True
    with tempfile.TemporaryDirectory() as tmp:
        paths = render_frames(tmp, list(range(FRAMES)), engine="CYCLES", samples=samples)
        encode_mp4(load_frames(paths), out / "video.mp4", fps=FPS, crf=16)
    build_scene(project, root=root)
    render_passes(out / "gt_passes", shaded=False)
    save_project(project, out / "gt.json")
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{spec['id']}: {spec['camera']}, {spec.get('objects') or len(spec['shapes'])} object(s) -> {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="v1", choices=["v1", "v2"],
                    help="v1: syn_01-06 (regression set); v2: syn2_01-12 (dev-v2 + held-out, SPECS_V2)")
    ap.add_argument("--out", default=None, help="default evaluation/synthetic (v1) or evaluation/synthetic_v2 (v2)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--passes-only", action="store_true",
                    help="only (re)render missing gt_passes from the committed gt.json (fast; for new sessions)")
    args = ap.parse_args()
    specs = SPECS_V2 if args.set == "v2" else SPECS
    args.out = args.out or ("evaluation/synthetic_v2" if args.set == "v2" else "evaluation/synthetic")
    if args.set == "v2":
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "split.json").write_text(json.dumps(
            {**split_v2(), "seed": SPLIT_SEED, "rule": "one clip of each same-camera pair per side, seeded coin",
             "note": "do not run any method on held_out before evaluation/thresholds.yaml is frozen"},
            indent=2) + "\n", encoding="utf-8")
    for spec in specs:
        if args.only and spec["id"] not in args.only:
            continue
        out = Path(args.out) / spec["id"]
        if args.passes_only:
            if not (out / "gt_passes" / "index.npz").is_file():
                from vrecon.schema import load_project

                build_scene(load_project(out / "gt.json"), root=out)
                render_passes(out / "gt_passes", shaded=False)
                print(f"{spec['id']}: passes rendered", flush=True)
            continue
        render_clip(spec, out, args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
