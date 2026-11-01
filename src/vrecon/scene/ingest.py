"""Turn human edits made in Blender into project changes (PLAN.md 7.2.1). Pure Python, no bpy.

    handoff:  build .blend -> read it back -> save that readout as the baseline (HandoffSnapshot)
    (human edits the .blend)
    ingest:   read edited .blend -> diff against the baseline -> apply only what changed

Initial scope: transforms of objects with a vrecon_id, camera settings, and their keyframes.
Everything else (new/deleted objects, mesh/modifier/material edits, parenting, render settings)
is reported as a warning and not imported.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from vrecon.schema import (
    Camera,
    CameraKey,
    HumanEdit,
    Intrinsics,
    Pose,
    Project,
    Rational,
    SceneObject,
    TransformKey,
)
from vrecon.schema.common import Model, utcnow

from .convert import FRAME_OFFSET, quat_mul

TOL = 1e-5  # values closer than this count as unchanged
_Q_UNFLIP_X = (0.0, -1.0, 0.0, 0.0)  # inverse of the OpenCV->Blender camera flip

Vec3 = tuple[float, float, float]
Quat4 = tuple[float, float, float, float]


# --- readout of a .blend (filled by readback.py) ------------------------------------------------


class Xform(Model):
    t: Vec3
    q: Quat4
    s: Vec3


class Lens(Model):
    lens: float
    shift_x: float
    shift_y: float


class KeyRead(Model):
    frame: float  # Blender frame
    xf: Xform
    lens: Lens | None = None  # cameras only


class ObjectRead(Model):
    vrecon_id: str
    name: str
    type: str  # Blender object type, e.g. MESH, CAMERA
    parent: str | None = None
    static: Xform  # value when not animated
    keys: list[KeyRead] = Field(default_factory=list)  # union of keyed frames, evaluated
    mesh_fingerprint: str | None = None
    modifiers: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    camera_static: Lens | None = None
    camera_sensor_fit: str | None = None
    camera_sensor_width: float | None = None


class SceneRead(Model):
    project_id: str | None
    revision: int | None
    shot_id: str | None
    frame_offset: int | None
    fps_num: int
    fps_base: float
    frame_start: int
    frame_end: int
    resolution_x: int
    resolution_y: int
    camera: str | None  # vrecon_id of the active camera


class Readout(Model):
    scene: SceneRead
    objects: dict[str, ObjectRead]
    unknown_objects: list[str] = Field(default_factory=list)  # objects without vrecon_id


class HandoffSnapshot(Model):
    project_id: str
    revision: int
    shot_id: str
    blend_path: str
    blend_sha256: str
    created_at: datetime = Field(default_factory=utcnow)
    readout: Readout


class IngestResult(Model):
    project: Project
    edits: list[HumanEdit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    status: Literal["no_changes", "applied"]


class IngestConflict(RuntimeError):
    pass


# --- comparisons ----------------------------------------------------------------------------


def _close(a, b) -> bool:
    return all(abs(x - y) <= TOL for x, y in zip(a, b, strict=True))


def _quat_close(a: Quat4, b: Quat4) -> bool:
    return abs(abs(sum(x * y for x, y in zip(a, b))) - 1.0) <= TOL  # q and -q are the same rotation


def _xform_close(a: Xform, b: Xform) -> bool:
    return _close(a.t, b.t) and _quat_close(a.q, b.q) and _close(a.s, b.s)


def _lens_close(a: Lens | None, b: Lens | None) -> bool:
    if a is None or b is None:
        return a is b
    return _close((a.lens, a.shift_x, a.shift_y), (b.lens, b.shift_x, b.shift_y))


def _keys_close(a: list[KeyRead], b: list[KeyRead], with_lens: bool = True) -> bool:
    return len(a) == len(b) and all(
        abs(x.frame - y.frame) <= TOL
        and _xform_close(x.xf, y.xf)
        and (not with_lens or _lens_close(x.lens, y.lens))
        for x, y in zip(a, b)
    )


# --- conversions Blender -> project conventions -----------------------------------------------


def _canonical(q: Quat4) -> Quat4:
    n = math.sqrt(sum(c * c for c in q))
    q = tuple(c / n for c in q)
    return q if q[0] >= 0 else tuple(-c for c in q)


def _camera_rotation_to_opencv(q_blender: Quat4) -> Quat4:
    return _canonical(quat_mul(q_blender, _Q_UNFLIP_X))


def _lens_to_intrinsics(lens: Lens, base: Intrinsics, sensor_width: float) -> Intrinsics:
    s = max(base.width, base.height)
    f = lens.lens * s / sensor_width
    return base.model_copy(
        update={
            "fx": f,
            "fy": f,
            "cx": (base.width - 1) / 2 - lens.shift_x * s,
            "cy": (base.height - 1) / 2 + lens.shift_y * s,
        }
    )


def _frame_to_time(frame: float, start_time: float, fps: Rational) -> float:
    return start_time + (frame - FRAME_OFFSET) * fps.den / fps.num


# --- ingest ---------------------------------------------------------------------------------


def _scope_warnings(base: Readout, new: Readout) -> list[str]:
    w: list[str] = []
    for name in sorted(set(new.unknown_objects) - set(base.unknown_objects)):
        w.append(f"new object {name!r} has no vrecon_id; not imported (adding objects is not supported yet)")
    for vid in sorted(set(base.objects) - set(new.objects)):
        w.append(f"{vid}: deleted in Blender; not imported (deleting is not supported yet)")
    for vid in sorted(set(base.objects) & set(new.objects)):
        b, n = base.objects[vid], new.objects[vid]
        if b.mesh_fingerprint != n.mesh_fingerprint:
            w.append(f"{vid}: shape edited; not imported")
        if b.modifiers != n.modifiers:
            w.append(f"{vid}: modifiers changed; not imported")
        if b.materials != n.materials:
            w.append(f"{vid}: materials changed; not imported")
        if b.parent != n.parent:
            w.append(f"{vid}: parent changed; not imported")
        if b.camera_sensor_fit != n.camera_sensor_fit or b.camera_sensor_width != n.camera_sensor_width:
            w.append(f"{vid}: sensor settings changed; camera intrinsics not imported")
    bs, ns = base.scene, new.scene
    for attr in ("fps_num", "fps_base", "frame_start", "frame_end", "resolution_x", "resolution_y", "camera"):
        if getattr(bs, attr) != getattr(ns, attr):
            w.append(f"scene {attr} changed ({getattr(bs, attr)} -> {getattr(ns, attr)}); not imported")
    return w


def _object_changes(obj: SceneObject, b: ObjectRead, n: ObjectRead, project: Project, shot_id: str):
    """Yield (field, old, new) for a scene object."""
    if b.parent != n.parent:
        return
    if any(o.shot_id == shot_id and o.target_id == obj.id and o.pose for o in project.reconstruction.overrides):
        if not (_xform_close(b.static, n.static) and _keys_close(b.keys, n.keys)):
            yield ("__warning__", None, f"{obj.id}: has a shot override; editing overrides is not supported yet")
        return
    if not b.keys and not n.keys:
        rest = obj.rest
        if not _close(b.static.t, n.static.t):
            yield ("rest.pose.translation", rest.pose.translation, n.static.t)
        if not _quat_close(b.static.q, n.static.q):
            yield ("rest.pose.rotation", rest.pose.rotation, _canonical(n.static.q))
        if not _close(b.static.s, n.static.s):
            yield ("rest.scale", rest.scale, n.static.s)
        return
    if _keys_close(b.keys, n.keys):
        return
    mapping = next((m for m in project.reconstruction.shot_mapping if m.shot_id == shot_id), None)
    start = mapping.scene_time_start if mapping else 0.0
    new_keys = [
        TransformKey(
            time=_frame_to_time(k.frame, start, project.source.fps),
            pose=Pose(translation=k.xf.t, rotation=_canonical(k.xf.q)),
            scale=k.xf.s,
        )
        for k in n.keys
    ]
    yield ("keys", obj.keys, new_keys)


def _camera_changes(cam: Camera, b: ObjectRead, n: ObjectRead):
    """Yield (field, old, new) for a camera.

    Three things can change: the pose (static or keyed), zoom keys (lens/shift keyed on the
    camera data), and the static lens when it is not keyed.
    """
    if b.parent != n.parent:
        return
    sensor_ok = (
        b.camera_sensor_fit == n.camera_sensor_fit == "AUTO" and b.camera_sensor_width == n.camera_sensor_width
    )
    sensor_w = n.camera_sensor_width or 36.0
    zoom_keyed = any(k.lens is not None for k in n.keys)
    zoom_keyed_before = any(k.lens is not None for k in b.keys)

    pose_changed = not (_xform_close(b.static, n.static) and _keys_close(b.keys, n.keys, with_lens=False))
    zoom_changed = sensor_ok and (zoom_keyed != zoom_keyed_before or not _keys_close(b.keys, n.keys))
    static_lens_changed = sensor_ok and not zoom_keyed and not _lens_close(b.camera_static, n.camera_static)

    if pose_changed or zoom_changed:
        if not n.keys:  # static camera: keep the original first frame
            new_keys = [
                CameraKey(
                    shot_frame=cam.keys[0].shot_frame,
                    pose=Pose(translation=n.static.t, rotation=_camera_rotation_to_opencv(n.static.q)),
                )
            ]
        else:
            new_keys = []
            for k in n.keys:
                shot_frame = k.frame - FRAME_OFFSET
                if abs(shot_frame - round(shot_frame)) > TOL or shot_frame < 0:
                    yield (
                        "__warning__",
                        None,
                        f"{cam.id}: key at Blender frame {k.frame} is not a whole frame inside the shot; "
                        "camera keys not imported",
                    )
                    return
                intr = _lens_to_intrinsics(k.lens, cam.intrinsics, sensor_w) if (k.lens and sensor_ok) else None
                new_keys.append(
                    CameraKey(
                        shot_frame=int(round(shot_frame)),
                        pose=Pose(translation=k.xf.t, rotation=_camera_rotation_to_opencv(k.xf.q)),
                        intrinsics=intr,
                    )
                )
        yield ("keys", cam.keys, new_keys)
    if static_lens_changed and n.camera_static is not None:
        yield ("intrinsics", cam.intrinsics, _lens_to_intrinsics(n.camera_static, cam.intrinsics, sensor_w))


def _dump(value: Any) -> Any:
    if isinstance(value, Model):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(v) for v in value]
    if isinstance(value, tuple):
        return list(value)
    return value


def ingest(project: Project, snapshot: HandoffSnapshot, edited: Readout) -> IngestResult:
    """Diff the edited .blend against the handoff baseline and apply human edits to a copy of project."""
    if snapshot.project_id != project.id:
        raise IngestConflict(f"snapshot is for project {snapshot.project_id}, not {project.id}")
    if snapshot.revision != project.revision:
        raise IngestConflict(
            f"project changed since handoff (revision {snapshot.revision} -> {project.revision}); "
            "hand off again before editing"
        )
    sc = edited.scene
    if (sc.project_id, sc.revision, sc.shot_id) != (snapshot.project_id, snapshot.revision, snapshot.shot_id):
        raise IngestConflict("the .blend does not belong to this handoff (project/revision/shot ids differ)")

    base = snapshot.readout
    warnings = _scope_warnings(base, edited)
    data = project.model_copy(deep=True)
    rec = data.reconstruction
    changes: list[tuple[str, str, Any, Any]] = []

    for vid in sorted(set(base.objects) & set(edited.objects)):
        b, n = base.objects[vid], edited.objects[vid]
        obj = next((o for o in rec.objects if o.id == vid), None)
        cam = next((c for c in rec.cameras if c.id == vid), None)
        if obj is not None:
            found = list(_object_changes(obj, b, n, data, snapshot.shot_id))
        elif cam is not None:
            found = list(_camera_changes(cam, b, n))
        else:
            warnings.append(f"{vid}: in the .blend but not in the project; ignored")
            continue
        for field, old, new in found:
            if field == "__warning__":
                warnings.append(new)
            else:
                changes.append((vid, field, old, new))

    if not changes:
        return IngestResult(project=project, warnings=warnings, status="no_changes")

    edits: list[HumanEdit] = []
    for i, (vid, field, old, new) in enumerate(changes, start=1):
        target = next((o for o in rec.objects if o.id == vid), None) or next(c for c in rec.cameras if c.id == vid)
        _set_field(target, field, new)
        edits.append(
            HumanEdit(
                id=f"edit_r{data.revision + 1}_{i:03d}",
                target_id=vid,
                field=field,
                old=_dump(old),
                new=_dump(new),
                source="blender_ingest",
            )
        )
    data.human.edits.extend(edits)
    data.revision += 1
    data = Project.model_validate(data.model_dump())  # full cross-field check
    return IngestResult(project=data, edits=edits, warnings=warnings, status="applied")


def _set_field(target: Model, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    obj = target
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)
