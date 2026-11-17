"""The three scripted edits of the editability test (H5, REVIEW_INSTRUCTIONS F-13 / WP-7 step 4). Pure Python.

The same three edits as phase 1a (evaluation/phase1a/edits.json), stated so they apply to any scene:
    camera_path   the camera trucks sideways (its own right axis) by TRUCK_SHARE x its start distance to the
                  target, linearly over the clip; orientation unchanged
    object_size   the target's proxy size becomes SIZE_FACTORS x (taller and narrower); path and timing unchanged
    action_timing the target's motion starts DELAY_S later (it holds its first pose until then)
The target is the animated object with the longest path (ties: id order); the floor, the walls and set pieces
are never targets. Nothing here reads ground truth.

`expected_after` gives the scene the edit must produce, from a project sampled at every frame (both the
builder's read-back and blend_import give that). `compare` says whether a read-back matches it. The Blender
side (applying the edit to the .blend's own data) is in scene/edit_blend.py.
"""

from __future__ import annotations

import numpy as np

from vrecon.schema import Pose, Project
from vrecon.schema.project import CameraKey, TransformKey

EDITS = ("camera_path", "object_size", "action_timing")
TRUCK_SHARE = 0.2
SIZE_FACTORS = (0.6, 0.6, 1.8)
DELAY_S = 1.0
TOL_POS = 1e-3  # times the camera's start distance to the target
TOL_ROT = 1e-3  # quaternion component
TOL_SIZE = 1e-3  # relative


def _objects(p: Project):
    return p.reconstruction.objects


def _cam(p: Project):
    return p.reconstruction.cameras[0]


def _path_length(obj) -> float:
    if len(obj.keys) < 2:
        return 0.0
    t = np.array([k.pose.translation for k in obj.keys])
    return float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum())


def choose_target(p: Project) -> str | None:
    cands = [o for o in _objects(p) if o.kind != "set" and o.id not in ("obj_floor",) and not o.id.startswith("obj_wall")
             and _path_length(o) > 0]
    if not cands:
        return None
    return sorted(cands, key=lambda o: (-_path_length(o), o.id))[0].id


def _start_pos(obj) -> np.ndarray:
    return np.array((obj.keys[0] if obj.keys else obj.rest).pose.translation, float)


def scale_of(p: Project, target: str) -> float:
    obj = next(o for o in _objects(p) if o.id == target)
    return float(np.linalg.norm(np.array(_cam(p).keys[0].pose.translation) - _start_pos(obj))) or 1.0


def _right_axis(rotation) -> np.ndarray:
    w, x, y, z = rotation
    return np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])  # first matrix column


def truck_offset(p: Project, target: str, frame: int) -> np.ndarray:
    """World offset of the camera at shot frame `frame` under the camera-path edit."""
    cam = _cam(p)
    n = max(p.source.frame_count - 1, 1)
    return _right_axis(cam.keys[0].pose.rotation) * TRUCK_SHARE * scale_of(p, target) * (frame / n)


def expected_after(p: Project, edit: str, target: str) -> Project:
    q = p.model_copy(deep=True)
    if edit == "camera_path":
        cam = _cam(q)
        cam.keys = [CameraKey(shot_frame=k.shot_frame, intrinsics=k.intrinsics,
                              pose=Pose(translation=tuple((np.array(k.pose.translation) + truck_offset(p, target, k.shot_frame)).tolist()),
                                        rotation=k.pose.rotation)) for k in cam.keys]
    elif edit == "object_size":
        obj = next(o for o in _objects(q) if o.id == target)
        obj.size = tuple((np.array(obj.size) * SIZE_FACTORS).tolist())
    elif edit == "action_timing":
        obj = next(o for o in _objects(q) if o.id == target)
        times = [k.time for k in obj.keys]
        new = []
        for k in obj.keys:
            src = k.time - DELAY_S
            j = int(np.argmin([abs(t - src) for t in times])) if src > times[0] else 0
            new.append(TransformKey(time=k.time, pose=obj.keys[j].pose, scale=obj.keys[j].scale))
        obj.keys = new
    else:
        raise ValueError(edit)
    return q


def compare(expected: Project, got: Project, scale: float) -> list[str]:
    """Differences beyond tolerance (empty = the edit landed exactly and nothing else moved)."""
    out = []
    ce, cg = _cam(expected), _cam(got)
    if len(ce.keys) != len(cg.keys):
        out.append(f"camera keys {len(ce.keys)} vs {len(cg.keys)}")
    for a, b in zip(ce.keys, cg.keys):
        if np.abs(np.subtract(a.pose.translation, b.pose.translation)).max() > TOL_POS * scale or \
                np.abs(np.subtract(a.pose.rotation, b.pose.rotation)).max() > TOL_ROT:
            out.append(f"camera frame {a.shot_frame}")
            break
    ge = {o.id: o for o in _objects(got)}
    for o in _objects(expected):
        g = ge.get(o.id)
        if g is None:
            out.append(f"{o.id} missing")
            continue
        if o.size is not None and (g.size is None or np.abs(np.array(g.size) / np.array(o.size) - 1).max() > TOL_SIZE):
            out.append(f"{o.id} size")
        ke = o.keys or [o.rest]
        kg = g.keys or [g.rest] * len(ke)
        if len(ke) == 1:
            ke = ke * len(kg)
        for a, b in zip(ke, kg):
            if np.abs(np.subtract(a.pose.translation, b.pose.translation)).max() > TOL_POS * scale or \
                    np.abs(np.subtract(a.pose.rotation, b.pose.rotation)).max() > TOL_ROT:
                out.append(f"{o.id} pose at t={b.time:.3f}")
                break
    extra = set(ge) - {o.id for o in _objects(expected)}
    out += [f"{i} appeared" for i in sorted(extra)]
    return out
