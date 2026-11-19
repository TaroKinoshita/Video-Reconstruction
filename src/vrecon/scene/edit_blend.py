"""Apply one scripted edit (vrecon/edits.py) to a .blend's own data and check the result (H5 editability test).

The same script runs on every condition's output: a .blend built by vrecon from a project, or a .blend an LLM
built. It edits what a person would edit: the target object's keyframes (timing), its mesh (size), and the
camera's location keys (camera path). Success = the read-back after the edit equals `edits.expected_after`
of the read-back before it, within tolerance, with nothing else changed. No manual repair: any exception or
difference is a failure, with the reason recorded.
"""

from __future__ import annotations

from pathlib import Path

import bpy
import mathutils
from bpy_extras import anim_utils

from vrecon import edits

from .blend_import import import_current
from .builder import ID_PROP


def _fcurves(obj):
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return []
    bag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    return list(bag.fcurves) if bag else []


def _by_id(vid: str):
    return next(o for o in bpy.data.objects if o.get(ID_PROP) == vid)


def _apply(edit: str, before, target: str) -> None:
    scene = bpy.context.scene
    obj = _by_id(target)
    if edit == "action_timing":
        fcs = _fcurves(obj)
        if not fcs:
            raise RuntimeError(f"{target} has no keyframes of its own (animated by a parent, constraint or driver)")
        shift = edits.DELAY_S * scene.render.fps / scene.render.fps_base
        for fc in fcs:
            # pin the pose of the clip's first frame before shifting: keys may sit between frames (a 4D model's
            # sample times), and "holds its first pose" means the pose seen in the first frame
            fc.keyframe_points.insert(scene.frame_start, fc.evaluate(scene.frame_start), options={"FAST"})
            for kp in [k for k in fc.keyframe_points if k.co.x < scene.frame_start - 1e-6][::-1]:
                fc.keyframe_points.remove(kp)  # keys before the clip would otherwise be held instead
            fc.update()
            for kp in fc.keyframe_points:
                kp.co.x += shift
                kp.handle_left.x += shift
                kp.handle_right.x += shift
            fc.extrapolation = "CONSTANT"
            fc.update()
    elif edit == "object_size":
        if obj.data.users > 1:
            raise RuntimeError(f"{target}'s mesh is shared by {obj.data.users} objects")
        obj.data.transform(mathutils.Matrix.Diagonal((*edits.SIZE_FACTORS, 1.0)))
        obj.data.update()
    elif edit == "camera_path":
        cam = scene.camera
        if cam.parent is not None or len(cam.constraints):
            raise RuntimeError("the camera is parented or constrained; its location keys do not set its path")
        frames = range(scene.frame_start, scene.frame_end + 1)
        locs = []
        for f in frames:
            scene.frame_set(f)
            locs.append(cam.location.copy())
        for i, (f, loc) in enumerate(zip(frames, locs)):
            cam.location = loc + mathutils.Vector(edits.truck_offset(before, target, i).tolist())
            cam.keyframe_insert("location", frame=f)
        for fc in _fcurves(cam):
            if fc.data_path == "location":
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
    else:
        raise ValueError(edit)


def run_edit(blend: str | Path, edit: str, target: str | None = None) -> dict:
    """Open `blend`, apply `edit`, return {edit, target, success, problems}. Leaves the edited scene open.
    target: the object to edit (default: edits.choose_target, the animated object with the longest path)."""
    bpy.ops.wm.open_mainfile(filepath=str(Path(blend).resolve()))
    try:
        before = import_current()
    except Exception as e:  # noqa: BLE001  (an unreadable scene is a failure of the test, not of the script)
        return {"edit": edit, "target": None, "success": False, "problems": [f"read-back: {e}"]}
    target = target or edits.choose_target(before)
    if target is None:
        return {"edit": edit, "target": None, "success": False, "problems": ["no animated object"]}
    expected = edits.expected_after(before, edit, target)
    try:
        _apply(edit, before, target)
        after = import_current()
    except Exception as e:  # noqa: BLE001
        return {"edit": edit, "target": target, "success": False, "problems": [str(e)]}
    problems = edits.compare(expected, after, edits.scale_of(before, target))
    return {"edit": edit, "target": target, "success": not problems, "problems": problems}


def run_all(blend: str | Path) -> dict:
    rows = [run_edit(blend, e) for e in edits.EDITS]
    return {"blend": str(blend), "edits": rows, "edit_success_share": sum(r["success"] for r in rows) / len(rows)}
