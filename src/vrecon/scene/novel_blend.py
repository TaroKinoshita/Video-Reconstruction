"""Novel views of a .blend as it is: open the original file, put a scoring camera in it, render the object-index pass
(evaluation only; the .blend on disk is not changed). Nothing is imported or rebuilt, so meshes, armatures, shape
keys, modifiers, parents and constraints render exactly as the author made them.

The scoring camera comes from metrics/novel.carry_relative (relative to the .blend's own source camera, scaled by
the cameras' heights above each world's floor). Only the frames each novel view needs are rendered.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from vrecon.schema import Camera

from .blend_import import import_current
from .builder import ID_PROP
from .convert import intrinsics_to_blender, opencv_camera_rotation_to_blender
from .passes import render_passes


def floor_height(obj_id: str | None) -> float:
    """Top of the floor object's world bounding box at the first frame (0 when there is none)."""
    if not obj_id:
        return 0.0
    o = next((x for x in bpy.data.objects if x.get(ID_PROP) == obj_id), None)
    if o is None:
        return 0.0
    bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    return float(max((o.matrix_world @ Vector(c)).z for c in o.bound_box))


def open_scene(blend: str | Path):
    """Open the .blend and read its source camera (OpenCV keys per frame) and object ids."""
    bpy.ops.wm.open_mainfile(filepath=str(Path(blend).resolve()))
    return import_current()


def render_view(cam: Camera, frames: list[int]) -> tuple[np.ndarray, dict[int, str]]:
    """Index images of the open scene from `cam` (its key at each frame) at shot frames `frames`."""
    scene = bpy.context.scene
    intr = cam.intrinsics
    scene.render.resolution_x, scene.render.resolution_y = intr.width, intr.height
    scene.render.resolution_percentage = 100
    data = bpy.data.cameras.new("vrecon_scoring_cam")
    data.sensor_fit = "AUTO"
    lens = intrinsics_to_blender(intr)
    data.sensor_width, data.lens, data.shift_x, data.shift_y = lens.sensor_width_mm, lens.lens_mm, lens.shift_x, lens.shift_y
    obj = bpy.data.objects.new("vrecon_scoring_cam", data)
    scene.collection.objects.link(obj)
    old = scene.camera
    scene.camera = obj
    keys = {k.shot_frame: k for k in cam.keys}
    out, id_map = [], {}
    try:
        for f in frames:
            k = keys[f]
            q = opencv_camera_rotation_to_blender(k.pose.rotation)
            obj.matrix_world = Matrix.Translation(Vector(k.pose.translation)) @ Quaternion(q).to_matrix().to_4x4()
            with tempfile.TemporaryDirectory() as tmp:
                res = render_passes(tmp, shot_frames=[f], shaded=False)
                out.append(np.load(Path(tmp) / "index.npz")["index"][0])
                id_map = res.id_map
    finally:
        scene.camera = old
        bpy.data.objects.remove(obj)
        bpy.data.cameras.remove(data)
    return np.stack(out), id_map
