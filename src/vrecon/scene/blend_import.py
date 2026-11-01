"""Import an arbitrary .blend (e.g. made by an LLM through blender-mcp) so it can be scored.

The scene was not built by vrecon, so:
  - every visible mesh gets a vrecon_id derived from its name (obj_<name>)
  - the active camera and every mesh are sampled at every frame (no assumptions about keyframes)
  - an object's `size` is its local bounding box; its pose is the world transform (parenting and
    constraints included, via matrix_world), so the metrics see exactly what renders
  - Blender camera axes are converted to OpenCV (CONVENTIONS.md 2)

Masks and depth for scoring are rendered from the .blend itself (true geometry, not proxy boxes).
"""

from __future__ import annotations

import re
from pathlib import Path

import bpy
import numpy as np

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
)

from .builder import ID_PROP
from .convert import quat_mul
from .passes import render_passes

_UNFLIP = (0.0, -1.0, 0.0, 0.0)  # Blender camera -> OpenCV camera


def _canon(q):
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    return tuple((q if q[0] >= 0 else -q).tolist())


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "unnamed"


def _intrinsics(cam_obj, scene) -> Intrinsics:
    w, h = scene.render.resolution_x, scene.render.resolution_y
    d = cam_obj.data
    if d.type != "PERSP":
        raise NotImplementedError(f"camera type {d.type} is not supported (perspective only)")
    if abs(scene.render.pixel_aspect_x - scene.render.pixel_aspect_y) > 1e-6:
        raise NotImplementedError("non-square pixel aspect is not supported")
    fit = d.sensor_fit
    if fit == "AUTO":
        fit = "HORIZONTAL" if w >= h else "VERTICAL"
        sensor = d.sensor_width
    else:
        sensor = d.sensor_width if fit == "HORIZONTAL" else d.sensor_height
    f = d.lens * (w if fit == "HORIZONTAL" else h) / sensor
    s = max(w, h)  # Blender's shift is a fraction of the larger image side
    return Intrinsics(width=w, height=h, fx=f, fy=f, cx=(w - 1) / 2 - d.shift_x * s, cy=(h - 1) / 2 + d.shift_y * s)


def _assign_ids() -> dict[str, bpy.types.Object]:
    used: dict[str, bpy.types.Object] = {}
    for obj in sorted(bpy.data.objects, key=lambda o: o.name):
        if obj.type != "MESH" or obj.hide_render:
            continue
        vid = obj.get(ID_PROP) or f"obj_{_slug(obj.name)}"
        base, n = vid, 2
        while vid in used:
            vid, n = f"{base}_{n}", n + 1
        obj[ID_PROP] = vid
        used[vid] = obj
    return used


def import_current(project_id: str = "proj_imported", video_path: str = "video.mp4") -> Project:
    """Sample the open Blender scene into a vrecon Project (one shot = the scene frame range)."""
    scene = bpy.context.scene
    if scene.camera is None:
        raise ValueError("the .blend has no active camera")
    cam = scene.camera
    meshes = _assign_ids()
    start, end = scene.frame_start, scene.frame_end
    n = end - start + 1
    fps = Rational(num=int(scene.render.fps), den=1) if scene.render.fps_base == 1 else \
        Rational(num=round(scene.render.fps * 1001 / scene.render.fps_base), den=1001)

    cam_keys, obj_keys = [], {vid: [] for vid in meshes}
    base_intr = None
    for i, frame in enumerate(range(start, end + 1)):
        scene.frame_set(frame)
        t, q, _ = cam.matrix_world.decompose()
        intr = _intrinsics(cam, scene)
        base_intr = base_intr or intr
        cam_keys.append(CameraKey(
            shot_frame=i, pose=Pose(translation=tuple(t), rotation=_canon(quat_mul(tuple(q), _UNFLIP))),
            intrinsics=None if intr == base_intr else intr))
        for vid, obj in meshes.items():
            t, q, s = obj.matrix_world.decompose()
            obj_keys[vid].append(TransformKey(time=i / (fps.num / fps.den),
                                              pose=Pose(translation=tuple(t), rotation=_canon(q)), scale=tuple(s)))

    objects = []
    for vid, obj in meshes.items():
        corners = np.array([list(c) for c in obj.bound_box])
        size = tuple(np.maximum(corners.max(0) - corners.min(0), 1e-6).tolist())
        keys = obj_keys[vid]
        static = all(k.pose == keys[0].pose and k.scale == keys[0].scale for k in keys)
        objects.append(SceneObject(id=vid, name=obj.name, kind="proxy", size=size,
                                   rest=keys[0] if static else TransformKey(time=0, pose=Pose()),
                                   keys=[] if static else keys))
    w, h = scene.render.resolution_x, scene.render.resolution_y
    return Project(
        id=project_id,
        source=SourceVideo(file=FileRef(id="file_video", path=video_path), fps=fps, frame_count=n, width=w, height=h),
        shots=[Shot(id="shot_001", start=0, end=n, transition_from_previous="first")],
        reconstruction={
            "cameras": [Camera(id="cam_imported", shot_id="shot_001", intrinsics=base_intr, keys=cam_keys)],
            "objects": objects,
            "shot_mapping": [ShotMapping(shot_id="shot_001", scene_time_start=0.0)],
        },
    )


def import_blend(path: str | Path, passes_dir: str | Path, project_id: str = "proj_imported",
                 size: tuple[int, int] | None = None):
    """Open a .blend, import it as a Project, and render its masks/depth into passes_dir.

    size: force the render resolution (e.g. the reference's) when the .blend uses another one with the
    same aspect ratio; the image geometry is unchanged because Blender cameras scale with resolution.
    Returns (project, warnings).
    """
    bpy.ops.wm.open_mainfile(filepath=str(Path(path).resolve()))
    scene = bpy.context.scene
    warnings = []
    if size is not None and (scene.render.resolution_x, scene.render.resolution_y) != tuple(size):
        w0, h0 = scene.render.resolution_x, scene.render.resolution_y
        if abs(w0 / h0 - size[0] / size[1]) > 1e-3:
            raise ValueError(f".blend renders {w0}x{h0}, reference is {size[0]}x{size[1]} (different aspect ratio)")
        warnings.append(f"render size {w0}x{h0} changed to {size[0]}x{size[1]} for scoring")
        scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    project = import_current(project_id)
    render_passes(passes_dir, shaded=False)
    return project, warnings
