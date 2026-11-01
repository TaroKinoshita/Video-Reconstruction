"""Generate the phase 1a test scenes (PLAN.md 12, stage 1a): one base scene and three edits.

    python scripts/make_phase1a_scenes.py [--out evaluation/phase1a] [--size 1280x704]

Each edit changes exactly one thing relative to the base:
    edit_camera   camera path: slow dolly-in  ->  lateral truck from left to right
    edit_object   main object: 0.6 m cube     ->  tall narrow block (0.35 x 0.35 x 1.1 m)
    edit_timing   main object starts moving 1.0 s later (24 frames)

Writes scenes/<name>.json (vrecon projects), edits.json (what changed, what must stay the same)
and jobs.json (generation jobs for the GPU runner in gpu/h3_control/).
All scenes are 24 fps and 124 frames (17n+5, H3's frame grid), width and height multiples of 32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vrecon.control.core import check_h3_constraints
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
from vrecon.scene.convert import look_at_opencv

FPS = 24
FRAMES = 124  # 17 * 7 + 5, about 5.17 s
TARGET = (0.0, 1.5, 0.6)  # where the camera looks

BASE_PROMPT = (
    "A small red cube slides slowly across a light wooden floor in a quiet, minimalist room. "
    "Behind it is a plain white wall. A gray concrete pillar stands on the left and a wooden crate "
    "sits on the right. Soft, even daylight. Realistic, sharp, natural colors."
)
OBJECT_PROMPT = BASE_PROMPT.replace("A small red cube slides", "A tall, narrow red block slides")
NEGATIVE = "blurry, distorted geometry, flicker, extra objects, text, watermark, low quality"


def camera_keys(path: str) -> list[CameraKey]:
    keys = []
    for f in range(0, FRAMES, 6):
        s = f / (FRAMES - 1)
        if path == "dolly_in":
            eye = (0.0, -7.0 + 1.5 * s, 1.6)
        elif path == "truck_right":
            eye = (-3.0 + 6.0 * s, -6.5, 1.6)
        else:
            raise ValueError(path)
        keys.append(CameraKey(shot_frame=f, pose=Pose(translation=eye, rotation=look_at_opencv(eye, TARGET))))
    last = FRAMES - 1
    if keys[-1].shot_frame != last:
        s = 1.0
        eye = (0.0, -5.5, 1.6) if path == "dolly_in" else (3.0, -6.5, 1.6)
        keys.append(CameraKey(shot_frame=last, pose=Pose(translation=eye, rotation=look_at_opencv(eye, TARGET))))
    return keys


def hero(size, start: float) -> SceneObject:
    z = size[2] / 2
    return SceneObject(
        id="obj_hero",
        name="hero",
        kind="rigid",
        size=size,
        keys=[
            TransformKey(time=start, pose=Pose(translation=(-2.0, 1.0, z))),
            TransformKey(time=start + 3.0, pose=Pose(translation=(2.0, 1.0, z))),
        ],
        status="assumed",
    )


def scene(name: str, width: int, height: int, cam_path="dolly_in", hero_size=(0.6, 0.6, 0.6), hero_start=1.0) -> Project:
    f = 1000.0 * width / 1280  # about 65 degrees horizontal field of view
    intr = Intrinsics(width=width, height=height, fx=f, fy=f, cx=(width - 1) / 2, cy=(height - 1) / 2)
    return Project(
        id=f"proj_p1a_{name}",
        source=SourceVideo(
            file=FileRef(id="file_none", path="synthetic/none.mp4"),
            fps=Rational(num=FPS, den=1),
            frame_count=FRAMES,
            width=width,
            height=height,
        ),
        shots=[Shot(id="shot_001", start=0, end=FRAMES, transition_from_previous="first")],
        reconstruction={
            "scale": {"source": "assumed", "note": "synthetic test scene, meters"},
            "cameras": [
                Camera(id="cam_001", shot_id="shot_001", intrinsics=intr, keys=camera_keys(cam_path), status="assumed")
            ],
            "objects": [
                SceneObject(id="obj_floor", kind="set", size=(14, 14, 0.02), rest=TransformKey(time=0, pose=Pose(translation=(0, 0, -0.01)))),
                SceneObject(id="obj_wall", kind="set", size=(14, 0.1, 5), rest=TransformKey(time=0, pose=Pose(translation=(0, 4.05, 2.5)))),
                SceneObject(id="obj_pillar", kind="rigid", size=(0.4, 0.4, 2.2), rest=TransformKey(time=0, pose=Pose(translation=(-2.6, 2.6, 1.1)))),
                SceneObject(id="obj_crate", kind="rigid", size=(0.8, 0.8, 0.8), rest=TransformKey(time=0, pose=Pose(translation=(2.6, 2.4, 0.4)))),
                hero(hero_size, hero_start),
            ],
            "shot_mapping": [ShotMapping(shot_id="shot_001", scene_time_start=0.0)],
        },
    )


EDITS = [
    {
        "id": "edit_camera",
        "type": "camera_path",
        "description": "camera path: slow dolly-in -> lateral truck from left to right, same look-at target",
        "changed": ["cam_001"],
        "unchanged": ["obj_hero", "obj_pillar", "obj_crate", "obj_wall", "obj_floor"],
        "expected": "camera trajectory follows the truck path; object layout and hero motion timing unchanged",
    },
    {
        "id": "edit_object",
        "type": "object_replace",
        "description": "main object: 0.6 m cube -> tall narrow block 0.35 x 0.35 x 1.1 m (same path and timing)",
        "changed": ["obj_hero"],
        "unchanged": ["cam_001", "obj_pillar", "obj_crate", "obj_wall", "obj_floor"],
        "expected": "hero silhouette matches the new block; its path and timing unchanged",
    },
    {
        "id": "edit_timing",
        "type": "action_timing",
        "description": "main object starts moving 1.0 s (24 frames) later; same path and duration",
        "changed": ["obj_hero"],
        "unchanged": ["cam_001", "obj_pillar", "obj_crate", "obj_wall", "obj_floor"],
        "expected": "hero motion onset at frame 48 instead of 24; everything else unchanged",
        "event": {"target": "obj_hero", "onset_frame_base": 24, "onset_frame_edit": 48},
    },
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/phase1a")
    ap.add_argument("--size", default="1280x704", help="WIDTHxHEIGHT, multiples of 32")
    args = ap.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))
    problems = check_h3_constraints(width, height, FRAMES, FPS)
    if problems:
        raise SystemExit("bad size: " + "; ".join(problems))

    out = Path(args.out)
    scenes = {
        "base": scene("base", width, height),
        "edit_camera": scene("edit_camera", width, height, cam_path="truck_right"),
        "edit_object": scene("edit_object", width, height, hero_size=(0.35, 0.35, 1.1)),
        "edit_timing": scene("edit_timing", width, height, hero_start=2.0),
    }
    for name, project in scenes.items():
        save_project(project, out / "scenes" / f"{name}.json")

    (out / "edits.json").write_text(json.dumps({"base": "base", "edits": EDITS}, indent=2) + "\n")

    prompts = {"base": BASE_PROMPT, "edit_camera": BASE_PROMPT, "edit_object": OBJECT_PROMPT, "edit_timing": BASE_PROMPT}
    jobs = {
        "about": "Generation jobs for gpu/h3_control/run_jobs.py. Control videos come from "
        "`vrecon control scenes/<scene>.json --out controls/<scene>`.",
        "defaults": {
            "sample_size": [height, width],
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "control_context_scale": 1.0,
            "seeds": [43, 44, 45],
            "negative_prompt": NEGATIVE,
        },
        "jobs": [
            {"id": f"{name}__{kind}", "scene": name, "control": f"controls/{name}/control_{kind}.mp4",
             "control_kind": kind, "prompt": prompts[name], "group": "main" if kind == "depth" else "extra"}
            for name in scenes
            for kind in ("depth", "canny")
        ],
    }
    (out / "jobs.json").write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(scenes)} scenes, edits.json and jobs.json ({len(jobs['jobs'])} jobs) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
