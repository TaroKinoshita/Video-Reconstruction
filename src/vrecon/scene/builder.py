"""Build a Blender scene for one shot from project state, render it, save it.

This is the only module (with the rest of vrecon.scene) that calls bpy.
Blender-side conventions:
  - Objects and cameras carry the custom property `vrecon_id`.
  - The scene carries `vrecon_project_id`, `vrecon_revision`, `vrecon_shot_id`,
    `vrecon_frame_offset` and `vrecon_schema_version`.
  - blender_frame = shot_frame + 1; keys use LINEAR interpolation so values between
    keys match straightforward interpolation of the stored data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import bpy
from bpy_extras import anim_utils

from vrecon.schema import Camera, Project, SceneObject, Shot

from .convert import (
    FRAME_OFFSET,
    continuous_quats,
    intrinsics_to_blender,
    opencv_camera_rotation_to_blender,
    scene_time_to_shot_frame,
    shot_frame_to_blender,
)

ID_PROP = "vrecon_id"


@dataclass
class BuildReport:
    shot_id: str
    camera: str
    objects: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _shot(project: Project, shot_id: str | None) -> Shot:
    if shot_id is None:
        if len(project.shots) != 1:
            raise ValueError("project has several shots; pass shot_id")
        return project.shots[0]
    for shot in project.shots:
        if shot.id == shot_id:
            return shot
    raise ValueError(f"unknown shot {shot_id!r}")


def _set_linear(obj: bpy.types.Object) -> None:
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return
    channelbag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    if channelbag is None:
        return
    for fc in channelbag.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def _new_object(name: str, data, vrecon_id: str) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj[ID_PROP] = vrecon_id
    obj.rotation_mode = "QUATERNION"
    return obj


def _proxy_mesh(name: str, size: tuple[float, float, float], cells: int = 1) -> bpy.types.Mesh:
    """Axis-aligned box centred on the origin. cells: grid cells along the longest edge (shorter edges in
    proportion), so per-face data such as coverage (scene/coverage.py) can vary over a large face."""
    import numpy as np

    half = np.array(size, float) / 2
    n = [max(1, round(cells * s / max(max(size), 1e-12))) for s in size]
    verts, faces, index = [], [], {}

    def vid(p):
        key = tuple(np.round(p, 9))
        if key not in index:
            index[key] = len(verts)
            verts.append(tuple(p))
        return index[key]

    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for sign in (-1, 1):
            for i in range(n[u]):
                for j in range(n[v]):
                    quad = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3)
                        p[axis] = sign * half[axis]
                        p[u] = -half[u] + 2 * half[u] * (i + di) / n[u]
                        p[v] = -half[v] + 2 * half[v] * (j + dj) / n[v]
                        quad.append(vid(p))
                    # (u, v) is right-handed about +axis for x and z, left-handed for y
                    outward = sign * (-1 if axis == 1 else 1) > 0
                    faces.append(quad if outward else quad[::-1])
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


PROXY_CELLS = {"set": 32}  # floor / wall planes are large; other proxies use DEFAULT_CELLS
DEFAULT_CELLS = 6


LOOK_ATTR = "look"  # colour attribute with the rough appearance (passes: scene/color_passes.py)
DEFAULT_LOOK = (0.6, 0.6, 0.6)


def set_look(mesh: bpy.types.Mesh, colors) -> None:
    """Per-vertex RGB (0..1) as the mesh's 'look' colour attribute; one colour or one per vertex."""
    import numpy as np

    c = np.asarray(colors, float)
    if c.ndim == 1:
        c = np.tile(c, (len(mesh.vertices), 1))
    attr = mesh.color_attributes.get(LOOK_ATTR) or mesh.color_attributes.new(LOOK_ATTR, "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", np.c_[c, np.ones(len(c))].ravel().tolist())


def _mesh_asset(obj: SceneObject, root: Path | None, report: BuildReport) -> bpy.types.Mesh | None:
    """A mesh file (.npz: verts [V,3], faces [F,3], optional colors [V,3] 0..1), e.g. analysis/background.py."""
    import numpy as np

    path = Path(obj.asset.path)
    if not path.is_absolute():
        path = (root or Path.cwd()) / path
    if not path.is_file():
        report.warnings.append(f"{obj.id}: mesh file {path} not found; using a proxy box")
        return None
    with np.load(path) as d:
        verts, faces = d["verts"], d["faces"]
        colors = d["colors"] if "colors" in d.files else None
    mesh = bpy.data.meshes.new(obj.id)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()
    set_look(mesh, colors if colors is not None and "all" not in obj.look else obj.look.get("all", DEFAULT_LOOK))
    return mesh


def _build_object(project: Project, obj: SceneObject, shot: Shot, report: BuildReport,
                  root: Path | None = None) -> None:
    mesh = None
    if obj.kind == "character" and obj.asset is not None and obj.asset.path.endswith(".npz"):
        from .character import build_character

        if build_character(project, obj, shot, root, report, ID_PROP):
            return
    elif obj.asset is not None and obj.asset.path.endswith(".npz"):
        mesh = _mesh_asset(obj, root, report)
    elif obj.asset is not None:
        report.warnings.append(f"{obj.id}: asset import not implemented yet; using a proxy box")
    if mesh is None:
        size = obj.size or (1.0, 1.0, 1.0)
        mesh = _proxy_mesh(obj.id, size, PROXY_CELLS.get(obj.kind, DEFAULT_CELLS))
        set_look(mesh, obj.look.get("all", DEFAULT_LOOK))
    bobj = _new_object(obj.name or obj.id, mesh, obj.id)

    override = next(
        (o for o in project.reconstruction.overrides if o.shot_id == shot.id and o.target_id == obj.id and o.pose),
        None,
    )
    if override is not None:
        bobj.location = override.pose.translation
        bobj.rotation_quaternion = override.pose.rotation
        report.warnings.append(f"{obj.id}: shot override {override.id} applied (static pose)")
        return

    bobj.location = obj.rest.pose.translation
    bobj.rotation_quaternion = obj.rest.pose.rotation
    bobj.scale = obj.rest.scale
    if not obj.keys:
        return

    mapping = next((m for m in project.reconstruction.shot_mapping if m.shot_id == shot.id), None)
    start_time = mapping.scene_time_start if mapping else 0.0
    if mapping is None and len(project.shots) > 1:
        report.warnings.append(f"{obj.id}: no shot_mapping for {shot.id}; assuming scene_time 0 = shot start")
    quats = continuous_quats([k.pose.rotation for k in obj.keys])
    for key, q in zip(obj.keys, quats):
        frame = shot_frame_to_blender(scene_time_to_shot_frame(key.time, start_time, project.source.fps))
        bobj.location = key.pose.translation
        bobj.rotation_quaternion = q
        bobj.scale = key.scale
        for path in ("location", "rotation_quaternion", "scale"):
            bobj.keyframe_insert(path, frame=frame)
    _set_linear(bobj)


def _build_camera(cam: Camera, report: BuildReport) -> bpy.types.Object:
    data = bpy.data.cameras.new(cam.id)
    data.sensor_fit = "AUTO"
    bcam = _new_object(cam.id, data, cam.id)
    scene = bpy.context.scene
    scene.render.resolution_x = cam.intrinsics.width
    scene.render.resolution_y = cam.intrinsics.height
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0

    zooming = any(k.intrinsics is not None for k in cam.keys)
    quats = continuous_quats([opencv_camera_rotation_to_blender(k.pose.rotation) for k in cam.keys])
    for key, q in zip(cam.keys, quats):
        intr = key.intrinsics or cam.intrinsics
        if (intr.width, intr.height) != (cam.intrinsics.width, cam.intrinsics.height):
            raise ValueError(f"{cam.id}: image size cannot change within a shot")
        lens = intrinsics_to_blender(intr)
        data.sensor_width = lens.sensor_width_mm
        data.lens = lens.lens_mm
        data.shift_x = lens.shift_x
        data.shift_y = lens.shift_y
        bcam.location = key.pose.translation
        bcam.rotation_quaternion = q
        if len(cam.keys) > 1:
            frame = shot_frame_to_blender(key.shot_frame)
            bcam.keyframe_insert("location", frame=frame)
            bcam.keyframe_insert("rotation_quaternion", frame=frame)
            if zooming:
                for path in ("lens", "shift_x", "shift_y"):
                    data.keyframe_insert(path, frame=frame)
    if len(cam.keys) > 1:
        _set_linear(bcam)
        if zooming:
            # camera data has its own action
            ad = data.animation_data
            channelbag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
            for fc in channelbag.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
    return bcam


def build_scene(project: Project, shot_id: str | None = None, root: str | os.PathLike[str] | None = None) -> BuildReport:
    """Replace the current Blender session with a scene for one shot. root: folder that asset paths are
    relative to (the project file's folder); default the current directory."""
    shot = _shot(project, shot_id)
    cams = [c for c in project.reconstruction.cameras if c.shot_id == shot.id]
    if len(cams) != 1:
        raise ValueError(f"shot {shot.id} needs exactly one camera, found {len(cams)}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    fps = project.source.fps
    scene.render.fps = fps.num
    scene.render.fps_base = fps.den
    scene.frame_start = FRAME_OFFSET
    scene.frame_end = FRAME_OFFSET + (shot.end - shot.start) - 1
    scene["vrecon_project_id"] = project.id
    scene["vrecon_revision"] = project.revision
    scene["vrecon_shot_id"] = shot.id
    scene["vrecon_frame_offset"] = FRAME_OFFSET
    scene["vrecon_schema_version"] = project.schema_version

    report = BuildReport(shot_id=shot.id, camera=cams[0].id)
    scene.camera = _build_camera(cams[0], report)
    for obj in project.reconstruction.objects:
        _build_object(project, obj, shot, report, Path(root) if root is not None else None)
        report.objects.append(obj.id)

    # Neutral lighting so proxies read clearly; not part of the reconstruction.
    sun = bpy.data.objects.new("vrecon_light", bpy.data.lights.new("vrecon_light", "SUN"))
    sun.rotation_euler = (0.8, 0.2, 0.6)
    scene.collection.objects.link(sun)
    world = bpy.data.worlds.new("vrecon_world")
    world.color = (0.05, 0.05, 0.05)
    scene.world = world
    return report


def set_cycles_device(scene, device: str | None = None) -> str:
    """Point Cycles at a device and return what it uses, e.g. "CPU" or "OPTIX: <gpu name>".

    `device` is cpu, optix, cuda or auto (first GPU backend found, else CPU). When it is None, the
    VRECON_RENDER_DEVICE environment variable decides, and without that, auto.
    """
    choice = (device or os.environ.get("VRECON_RENDER_DEVICE") or "auto").lower()
    if choice not in ("cpu", "optix", "cuda", "auto"):
        raise ValueError(f"unknown render device {choice!r}: use cpu, optix, cuda or auto")
    if choice != "cpu":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA") if choice == "auto" else (choice.upper(),):
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                return f"{backend}: {', '.join(d.name for d in gpus)}"
        if choice != "auto":  # asked for this GPU backend explicitly: do not fall back silently
            raise RuntimeError(f"no {choice.upper()} device found")
    scene.cycles.device = "CPU"
    return "CPU"


def render_frames(
    out_dir: str | os.PathLike[str],
    shot_frames: list[int],
    engine: str = "CYCLES",
    resolution_percentage: int = 100,
    samples: int = 8,
) -> list[Path]:
    """Render shot frames of the current scene to PNG files named by shot frame."""
    scene = bpy.context.scene
    scene.render.engine = engine
    if engine == "CYCLES":
        set_cycles_device(scene)
        scene.cycles.samples = samples
    scene.render.resolution_percentage = resolution_percentage
    scene.render.image_settings.file_format = "PNG"
    out = Path(out_dir).resolve()  # Blender resolves relative output paths against the .blend, not the cwd
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in shot_frames:
        scene.frame_set(int(shot_frame_to_blender(f)))
        path = out / f"frame_{f:06d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths.append(path)
    return paths


def save_blend(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path.resolve()))
    return path
