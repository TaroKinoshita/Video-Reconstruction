"""Smoke test: can Blender (bpy) build, render, save and reload a scene here?

Usage:
    python scripts/smoke_render.py --out runs/smoke [--engines CYCLES,BLENDER_WORKBENCH]

Prints one JSON summary line and exits non-zero if any check fails.
This is an environment check, not part of the vrecon pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import bpy
from bpy_extras import anim_utils

OBJ_ID = "obj_cube"


def build_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 24
    scene.render.fps = 24

    bpy.ops.mesh.primitive_plane_add(size=10)
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 1.0))
    cube = bpy.context.object
    cube.name = "Cube"
    cube["vrecon_id"] = OBJ_ID
    cube.keyframe_insert("location", frame=1)
    cube.location.x = 2.0
    cube.keyframe_insert("location", frame=24)

    bpy.ops.object.light_add(type="SUN", location=(4.0, -4.0, 8.0))

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.location = (7.0, -7.0, 5.0)
    cam.rotation_euler = (1.1, 0.0, 0.785)

    scene.render.resolution_x, scene.render.resolution_y = 320, 180


def render(engine: str, out_dir: str) -> dict:
    scene = bpy.context.scene
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.device = "CPU"
        scene.cycles.samples = 8
    scene.frame_set(12)
    path = os.path.join(out_dir, f"{engine.lower()}.png")
    scene.render.filepath = path
    start = time.time()
    bpy.ops.render.render(write_still=True)
    ok = os.path.isfile(path) and os.path.getsize(path) > 0
    return {"engine": engine, "ok": ok, "seconds": round(time.time() - start, 2), "path": path}


def read_location_keys(obj) -> dict[int, list[tuple[float, float]]]:
    """Read location keyframes using the Blender 5 layered-action API."""
    ad = obj.animation_data
    channelbag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    keys: dict[int, list[tuple[float, float]]] = {}
    for fc in channelbag.fcurves:
        if fc.data_path == "location":
            keys[fc.array_index] = [(k.co[0], k.co[1]) for k in fc.keyframe_points]
    return keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/smoke")
    parser.add_argument("--engines", default="CYCLES,BLENDER_WORKBENCH")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    result: dict = {"blender": bpy.app.version_string, "renders": [], "checks": {}}
    build_scene()

    for engine in [e.strip() for e in args.engines.split(",") if e.strip()]:
        try:
            result["renders"].append(render(engine, out_dir))
        except Exception as exc:  # report and continue with other engines
            result["renders"].append({"engine": engine, "ok": False, "error": str(exc)})

    blend_path = os.path.join(out_dir, "smoke.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    cube = next((o for o in bpy.data.objects if o.get("vrecon_id") == OBJ_ID), None)
    result["checks"]["id_survives_reload"] = cube is not None
    if cube is not None:
        keys = read_location_keys(cube)
        x_keys = keys.get(0, [])
        result["checks"]["keyframes_survive_reload"] = x_keys == [(1.0, 0.0), (24.0, 2.0)]
    else:
        result["checks"]["keyframes_survive_reload"] = False

    ok = all(r.get("ok") for r in result["renders"]) and all(result["checks"].values())
    result["ok"] = ok
    print("SMOKE_RESULT " + json.dumps(result))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
