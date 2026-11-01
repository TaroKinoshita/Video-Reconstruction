"""Read the current Blender session (or a .blend) into an ingest.Readout. Calls bpy."""

from __future__ import annotations

import hashlib
from pathlib import Path

import bpy
from bpy_extras import anim_utils

from .builder import ID_PROP
from .ingest import KeyRead, Lens, ObjectRead, Readout, SceneRead, Xform


def _keyed_frames(id_data) -> set[float]:
    ad = getattr(id_data, "animation_data", None)
    if ad is None or ad.action is None:
        return set()
    channelbag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    if channelbag is None:
        return set()
    return {round(kp.co[0], 6) for fc in channelbag.fcurves for kp in fc.keyframe_points}


def _set_frame(scene, frame: float) -> None:
    whole = int(frame // 1)
    scene.frame_set(whole, subframe=frame - whole)


def _xform(obj) -> Xform:
    # matrix_basis works for any rotation mode (quaternion, Euler, axis-angle).
    t, q, s = obj.matrix_basis.decompose()
    return Xform(t=tuple(t), q=tuple(q), s=tuple(s))


def _lens(cam_data) -> Lens:
    return Lens(lens=cam_data.lens, shift_x=cam_data.shift_x, shift_y=cam_data.shift_y)


def _mesh_fingerprint(obj) -> str | None:
    if obj.type != "MESH":
        return None
    mesh = obj.data
    h = hashlib.sha256()
    h.update(f"{len(mesh.vertices)}/{len(mesh.polygons)}".encode())
    for v in mesh.vertices:
        h.update(("%.6f,%.6f,%.6f;" % tuple(v.co)).encode())
    return h.hexdigest()[:16]


def read_current() -> Readout:
    scene = bpy.context.scene
    objects: dict[str, ObjectRead] = {}
    unknown: list[str] = []
    for obj in bpy.data.objects:
        vid = obj.get(ID_PROP)
        if not vid:
            unknown.append(obj.name)
            continue
        frames = _keyed_frames(obj)
        lens_frames = _keyed_frames(obj.data) if obj.type == "CAMERA" else set()
        keys = []
        for f in sorted(frames | lens_frames):
            _set_frame(scene, f)
            keys.append(
                KeyRead(frame=f, xf=_xform(obj), lens=_lens(obj.data) if lens_frames else None)
            )
        scene.frame_set(scene.frame_start)
        objects[vid] = ObjectRead(
            vrecon_id=vid,
            name=obj.name,
            type=obj.type,
            parent=obj.parent.get(ID_PROP, obj.parent.name) if obj.parent else None,
            static=_xform(obj),
            keys=keys,
            mesh_fingerprint=_mesh_fingerprint(obj),
            modifiers=[m.type for m in obj.modifiers],
            materials=[s.material.name if s.material else "" for s in obj.material_slots],
            camera_static=_lens(obj.data) if obj.type == "CAMERA" else None,
            camera_sensor_fit=obj.data.sensor_fit if obj.type == "CAMERA" else None,
            camera_sensor_width=obj.data.sensor_width if obj.type == "CAMERA" else None,
        )
    return Readout(
        scene=SceneRead(
            project_id=scene.get("vrecon_project_id"),
            revision=scene.get("vrecon_revision"),
            shot_id=scene.get("vrecon_shot_id"),
            frame_offset=scene.get("vrecon_frame_offset"),
            fps_num=scene.render.fps,
            fps_base=scene.render.fps_base,
            frame_start=scene.frame_start,
            frame_end=scene.frame_end,
            resolution_x=scene.render.resolution_x,
            resolution_y=scene.render.resolution_y,
            camera=scene.camera.get(ID_PROP) if scene.camera else None,
        ),
        objects=objects,
        unknown_objects=sorted(unknown),
    )


def read_blend(path: str | Path) -> Readout:
    bpy.ops.wm.open_mainfile(filepath=str(Path(path).resolve()))
    return read_current()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
