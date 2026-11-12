"""Write a refined Project back into the .blend it was imported from (blend_import.py), keeping the LLM's
meshes and materials: only object transforms (per frame), object scale for a changed size, and the camera
(pose per frame, focal length) are replaced.

Objects are found the way blend_import assigns ids (`renamed` maps the import id to the project id when the
project renamed set pieces). Parents and constraints are cleared, because the imported poses are world poses.
"""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from vrecon.metrics.poses import camera_pose, object_pose
from vrecon.schema import Project

from .blend_import import _UNFLIP, _assign_ids
from .convert import quat_mul


def _quat(r: np.ndarray):
    return Matrix(r.tolist()).to_quaternion()


def export_to_blend(project: Project, original: Project, src: str | Path, dst: str | Path,
                    renamed: dict[str, str] | None = None) -> list[str]:
    """Open src, apply project (vs original = the imported start) and save as dst. Returns notes."""
    bpy.ops.wm.open_mainfile(filepath=str(Path(src).resolve()))
    scene = bpy.context.scene
    meshes = _assign_ids()
    renamed = renamed or {}
    objs = {o.id: o for o in project.reconstruction.objects}
    orig = {o.id: o for o in original.reconstruction.objects}
    shot = project.shots[0].id
    n = project.source.frame_count
    notes = []
    for vid, bobj in meshes.items():
        pid = renamed.get(vid, vid)
        if pid not in objs:
            notes.append(f"{vid}: not in project, left as is")
            continue
        o, o0 = objs[pid], orig[pid]
        if o.model_dump() == o0.model_dump():
            continue
        grow = np.array(o.size) / np.maximum(np.array(o0.size), 1e-9)
        mw = bobj.matrix_world.copy()
        bobj.parent = None
        bobj.matrix_world = mw
        for c in list(bobj.constraints):
            bobj.constraints.remove(c)
        bobj.animation_data_clear()
        bobj.rotation_mode = "QUATERNION"
        frames = range(n) if o.keys else [0]
        for i in frames:
            r, t, s = object_pose(project, o, shot, i)
            bobj.location = Vector(t.tolist())
            bobj.rotation_quaternion = _quat(r)
            bobj.scale = Vector((s * grow).tolist())
            if o.keys:
                f = scene.frame_start + i
                for path in ("location", "rotation_quaternion", "scale"):
                    bobj.keyframe_insert(path, frame=f)
        notes.append(f"{vid}: transforms replaced" + (f", size x{np.round(grow, 3).tolist()}" if np.any(abs(grow - 1) > 1e-6) else ""))

    cam, cam0 = project.reconstruction.cameras[0], original.reconstruction.cameras[0]
    if cam.model_dump() != cam0.model_dump():
        bc = scene.camera
        mw = bc.matrix_world.copy()
        bc.parent = None
        bc.matrix_world = mw
        for c in list(bc.constraints):
            bc.constraints.remove(c)
        bc.animation_data_clear()
        if bc.data.animation_data:
            bc.data.animation_data_clear()
        bc.rotation_mode = "QUATERNION"
        w = scene.render.resolution_x
        bc.data.sensor_fit = "HORIZONTAL"
        for i in range(n):
            r, c, k = camera_pose(cam, i)
            q_cv = tuple(_quat(r))
            bc.location = Vector(c.tolist())
            bc.rotation_quaternion = Quaternion(quat_mul(q_cv, _UNFLIP))
            bc.data.lens = float(k[0, 0]) * bc.data.sensor_width / w
            f = scene.frame_start + i
            bc.keyframe_insert("location", frame=f)
            bc.keyframe_insert("rotation_quaternion", frame=f)
            bc.data.keyframe_insert("lens", frame=f)
        notes.append("camera: pose and focal length replaced")
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(dst).resolve()), copy=True)
    return notes
