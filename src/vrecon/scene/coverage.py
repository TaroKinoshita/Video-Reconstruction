"""Which faces of the scene the reference video has seen (completion plan step 3).

Per face of every mesh with a vrecon_id, over sampled source frames: inside the source camera's picture, facing it,
and not hidden (a ray from the camera to the face centre hits this face first). The best over the frames is kept:

    2 observed   seen with the face turned less than 70 deg away from the camera
    1 oblique    seen, but only at 70-89 deg (stretched, little detail: e.g. a floor seen from standing height)
    0 unseen     never seen: its shape and look here are not from the video

A face that the clay renders fine can still be unseen; the coverage pass shows that from any camera (new camera
moves included): green / yellow / red. This is about the reconstruction's faces, so it inherits its errors: check
the source view first (a misplaced proxy has the wrong faces "seen").
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from vrecon.schema import Camera, Project

from .builder import ID_PROP, build_scene
from .convert import shot_frame_to_blender

COVERAGE_ATTR = "vrecon_coverage"
COLORS = {0: (0.85, 0.1, 0.1), 1: (1.0, 0.8, 0.0), 2: (0.1, 0.75, 0.25)}
COS_OBSERVED, COS_OBLIQUE = math.cos(math.radians(70)), math.cos(math.radians(89))  # a floor seen from 1.7 m is 75-85 deg


def _meshes():
    return sorted((o for o in bpy.data.objects if o.type == "MESH" and o.get(ID_PROP)), key=lambda o: o[ID_PROP])


def compute_coverage(shot_frames: list[int]) -> dict[str, np.ndarray]:
    """Per object id: uint8 [polygons] coverage class, from the current scene's camera over the frames."""
    scene = bpy.context.scene
    cam = scene.camera
    out = {o[ID_PROP]: np.zeros(len(o.data.polygons), np.uint8) for o in _meshes()}
    w, h = scene.render.resolution_x, scene.render.resolution_y
    for f in shot_frames:
        scene.frame_set(int(shot_frame_to_blender(f)))
        dg = bpy.context.evaluated_depsgraph_get()
        cm = cam.matrix_world
        origin = np.array(cm.translation)
        # world -> camera (Blender camera: looks along -Z)
        w2c = np.array(cm.inverted())
        cdata = cam.data
        frame = [np.array(v) for v in cdata.view_frame(scene=scene)]  # corners at z = -d in camera space
        xs, ys = [v[0] for v in frame], [v[1] for v in frame]
        d = -frame[0][2]
        for o in _meshes():
            ev = o.evaluated_get(dg)
            me = ev.to_mesh()
            n = len(me.polygons)
            centres = np.empty(n * 3)
            normals = np.empty(n * 3)
            me.polygons.foreach_get("center", centres)
            me.polygons.foreach_get("normal", normals)
            ev.to_mesh_clear()
            mw = np.array(ev.matrix_world)
            c = centres.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
            nm = normals.reshape(-1, 3) @ np.linalg.inv(mw[:3, :3])
            nm /= np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-12)
            pc = c @ w2c[:3, :3].T + w2c[:3, 3]
            front = pc[:, 2] < 0
            u = np.where(front, pc[:, 0] * d / np.maximum(-pc[:, 2], 1e-9), np.inf)
            v = np.where(front, pc[:, 1] * d / np.maximum(-pc[:, 2], 1e-9), np.inf)
            inside = front & (u >= min(xs)) & (u <= max(xs)) & (v >= min(ys)) & (v <= max(ys))
            to_cam = origin - c
            dist = np.linalg.norm(to_cam, axis=1)
            cos = (nm * to_cam).sum(1) / np.maximum(dist, 1e-12)
            cand = np.nonzero(inside & (cos >= COS_OBLIQUE))[0]
            cls = out[o[ID_PROP]]
            for i in cand:
                q = 2 if cos[i] >= COS_OBSERVED else 1
                if cls[i] >= q:
                    continue
                direction = Vector(c[i] - origin) / dist[i]
                hit, loc, _, index, hobj, _ = scene.ray_cast(dg, Vector(origin), direction, distance=dist[i] * 1.001)
                if hit and hobj.original == o and (index == i or (np.array(loc) - c[i]) @ (np.array(loc) - c[i]) < (1e-3 * dist[i]) ** 2):
                    cls[i] = q
    return out


def apply_coverage(coverage: dict[str, np.ndarray]) -> None:
    """Write the classes as a per-face colour attribute on the current scene's meshes (same topology)."""
    for o in _meshes():
        cls = coverage.get(o[ID_PROP])
        me = o.data
        if cls is None or len(cls) != len(me.polygons):
            cls = np.zeros(len(me.polygons), np.uint8)
        attr = me.color_attributes.get(COVERAGE_ATTR) or me.color_attributes.new(COVERAGE_ATTR, "FLOAT_COLOR", "CORNER")
        loop_poly = np.empty(len(me.loops), int)
        for p in me.polygons:
            loop_poly[p.loop_start:p.loop_start + p.loop_total] = p.index
        cols = np.array([COLORS[k] for k in range(3)])[cls[loop_poly]]
        attr.data.foreach_set("color", np.c_[cols, np.ones(len(cols))].ravel().tolist())


def coverage_summary(coverage: dict[str, np.ndarray]) -> dict[str, dict]:
    return {k: {"faces": int(len(v)), "observed": float((v == 2).mean()) if len(v) else 0.0,
                "oblique": float((v == 1).mean()) if len(v) else 0.0, "unseen": float((v == 0).mean()) if len(v) else 0.0}
            for k, v in coverage.items()}


def render_coverage(project: Project, out_dir: str | Path, source_frames: list[int], view: Camera | None = None,
                    view_frames: list[int] | None = None, root: str | Path | None = None) -> tuple[list[Path], dict]:
    """Coverage from the project's camera over source_frames, rendered from `view` (default: the same camera)."""
    from .color_passes import render_color_pass

    build_scene(project, root=root)
    cov = compute_coverage(source_frames)
    if view is not None:
        p = project.model_copy(deep=True)
        p.reconstruction.cameras = [view.model_copy(update={"id": project.reconstruction.cameras[0].id,
                                                            "shot_id": project.reconstruction.cameras[0].shot_id})]
        build_scene(p, root=root)
    apply_coverage(cov)
    paths = render_color_pass("coverage", out_dir, view_frames or source_frames, {})
    return paths, coverage_summary(cov)


__all__ = ["COLORS", "COVERAGE_ATTR", "apply_coverage", "compute_coverage", "coverage_summary", "render_coverage"]
