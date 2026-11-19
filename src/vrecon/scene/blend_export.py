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


def _source_of(o) -> str | None:
    """The object a rule-made copy came from (run/complete notes: "copy of X", "mirror of X", "repeat of X")."""
    import re

    for e in o.evidence:
        m = re.search(r"(?:copy|mirror|repeat) of ([A-Za-z0-9_\-]+)", e.note or "")
        if m:
            return m.group(1).rstrip(",")
    return None


def _add_new_objects(project, original, meshes, renamed, scene, shot, n) -> list[str]:
    """Objects in the project but not in the .blend: a rule-made copy duplicates its source object (same mesh and
    materials, its own transform and size); an object with a mesh file (.npz) gets that mesh. Marked with the
    vrecon id so a later import finds them."""
    from .builder import ID_PROP, _mesh_asset, BuildReport

    notes = []
    have = set(renamed.get(v, v) for v in meshes)
    inv = {renamed.get(v, v): b for v, b in meshes.items()}
    byid = {o.id: o for o in project.reconstruction.objects}
    for o in project.reconstruction.objects:
        if o.id in have:
            continue
        src_id = _source_of(o)
        src = inv.get(src_id) if src_id else None
        if src is not None:
            new = src.copy()
            new.animation_data_clear()
            base = byid.get(src_id) or next((x for x in original.reconstruction.objects if x.id == src_id), None)
            grow = np.array(o.size) / np.maximum(np.array(base.size), 1e-9) if (base is not None and o.size and base.size) else np.ones(3)
        elif o.asset is not None and str(o.asset.path).endswith(".npz"):
            mesh = _mesh_asset(o, None, BuildReport())
            if mesh is None:
                continue
            new = bpy.data.objects.new(o.name or o.id, mesh)
            grow = np.ones(3)
        else:
            notes.append(f"{o.id}: new object with no source or mesh, not added")
            continue
        scene.collection.objects.link(new)
        new.parent = None
        for c in list(new.constraints):
            new.constraints.remove(c)
        new[ID_PROP] = o.id
        new.rotation_mode = "QUATERNION"
        for i in (range(n) if o.keys else [0]):
            r, t, s_ = object_pose(project, o, shot, i)
            new.location = Vector(t.tolist())
            new.rotation_quaternion = _quat(r)
            new.scale = Vector((s_ * grow).tolist())
            if o.keys:
                for path in ("location", "rotation_quaternion", "scale"):
                    new.keyframe_insert(path, frame=scene.frame_start + i)
        notes.append(f"{o.id}: added" + (f" (copy of {src_id})" if src is not None else " (mesh file)"))
    return notes


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

    notes += _add_new_objects(project, original, meshes, renamed, scene, shot, n)

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
