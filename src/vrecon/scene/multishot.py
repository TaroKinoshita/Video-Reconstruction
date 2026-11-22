"""Build a multi-cut .blend from scripts/multishot_prep.py output: one camera per cut, switched by timeline markers.

Each cut is its own world (parented under an empty, placed SPACING metres apart so the cuts never overlap); the
camera is keyed every frame; the body is a stick figure (22 joints, Skin modifier) keyed by shape keys; the
background is the 4RC point cloud with source colours. Timeline frame f+1 = source frame f.
"""

from __future__ import annotations

import json
from pathlib import Path

import bpy
import numpy as np

SPACING = 500.0
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def _points_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("PointColour")
    mat.use_nodes = True
    nt = mat.node_tree
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"
    emit = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
    return mat


def _points(name: str, pts: np.ndarray, col: np.ndarray, mat, parent, radius: float = 0.03) -> None:
    me = bpy.data.meshes.new(name)
    me.from_pydata(pts.tolist(), [], [])
    a = me.attributes.new("Col", "FLOAT_COLOR", "POINT")
    a.data.foreach_set("color", np.concatenate([col, np.ones((len(col), 1))], 1).astype(np.float32).ravel())
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.parent = parent
    ng = bpy.data.node_groups.new(name + "_pts", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    m2p.inputs["Radius"].default_value = radius
    sm = ng.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = mat
    ng.links.new(gi.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(m2p.outputs["Points"], sm.inputs["Geometry"])
    ng.links.new(sm.outputs["Geometry"], go.inputs[0])
    ob.modifiers.new("points", "NODES").node_group = ng


def _body(name: str, joints: np.ndarray, start: int, parent) -> None:
    edges = [(i, p) for i, p in enumerate(PARENTS) if p >= 0]
    me = bpy.data.meshes.new(name)
    me.from_pydata(joints[0].tolist(), edges, [])
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.parent = parent
    ob.shape_key_add(name="Basis")
    for t in range(len(joints)):
        kb = ob.shape_key_add(name=f"f{t}")
        kb.data.foreach_set("co", joints[t].astype(np.float32).ravel())
        for f, v in ((t - 1, 0.0), (t, 1.0), (t + 1, 0.0)):
            kb.value = v
            kb.keyframe_insert("value", frame=start + f)
    body_mat = bpy.data.materials.get("Body") or bpy.data.materials.new("Body")
    body_mat.use_nodes = True
    nt = body_mat.node_tree
    if "Emission" not in nt.nodes:
        e = nt.nodes.new("ShaderNodeEmission")
        e.inputs["Color"].default_value = (1.0, 0.35, 0.05, 1)
        nt.links.new(e.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
    me.materials.append(body_mat)
    sk = ob.modifiers.new("skin", "SKIN")
    for v in me.skin_vertices[0].data:
        v.radius = (0.035, 0.035)
    ob.modifiers.new("smooth", "SUBSURF").levels = 1
    del sk


def _flat(name: str, rgb) -> bpy.types.Material:
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    e = m.node_tree.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*rgb, 1)
    m.node_tree.links.new(e.outputs["Emission"], m.node_tree.nodes["Material Output"].inputs["Surface"])
    return m


def _fill(k: int, s: dict, floor: np.ndarray, cam0: np.ndarray, parent) -> None:
    """Backdrop sphere (seen from inside) around the cut's first camera; the floor plane when one was fitted."""
    from mathutils import Vector

    bpy.ops.mesh.primitive_uv_sphere_add(radius=300.0, location=tuple(cam0))
    sp = bpy.context.active_object
    sp.name = f"Backdrop{k:02d}"
    sp.data.materials.append(_flat(f"Backdrop{k:02d}", s["backdrop_rgb"]))
    sp.parent = parent
    if len(floor):
        bpy.ops.mesh.primitive_plane_add(size=60.0, location=tuple(floor[:3]))
        pl = bpy.context.active_object
        pl.name = f"Floor{k:02d}"
        pl.rotation_mode = "QUATERNION"
        pl.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(tuple(floor[3:6])))
        pl.data.materials.append(_flat(f"Floor{k:02d}", [float(v) for v in floor[6:9]]))
        pl.parent = parent


def _only_during(ob, start: int, frames: int) -> None:
    """visible only in timeline frames [start, start + frames)."""
    for f, hide in ((1, True), (start, False), (start + frames, True)):
        ob.hide_render = ob.hide_viewport = hide
        ob.keyframe_insert("hide_render", frame=f)
        ob.keyframe_insert("hide_viewport", frame=f)


def build(prep: Path, out: Path) -> dict:
    z = np.load(prep)
    info = json.loads(str(z["info"]))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    W, H = int(z["W"]), int(z["H"])
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.fps, sc.render.fps_base = 24000, 1001
    mat = _points_material()
    ends = []
    for k, s in enumerate(info["shots"]):
        a, b = s["source_frames"]
        start = a + 1
        root = bpy.data.objects.new(f"Shot{k:02d}", None)
        bpy.context.collection.objects.link(root)
        root.location = (s.get("slot", k) * SPACING, 0, 0)  # joined cuts share slot 0
        cd = bpy.data.cameras.new(f"Cam{k:02d}")
        cd.sensor_fit, cd.sensor_width, cd.lens = "HORIZONTAL", 36.0, s["lens_35mm"]
        cam = bpy.data.objects.new(f"Cam{k:02d}", cd)
        bpy.context.collection.objects.link(cam)
        cam.parent = root
        cam.rotation_mode = "QUATERNION"
        from mathutils import Matrix

        for t, m in enumerate(z[f"s{k}_cam"]):
            cam.matrix_local = Matrix(m.tolist())
            cam.keyframe_insert("location", frame=start + t)
            cam.keyframe_insert("rotation_quaternion", frame=start + t)
        mk = sc.timeline_markers.new(f"shot{k:02d}", frame=start)
        mk.camera = cam
        if len(z[f"s{k}_pts"]):
            _points(f"Background{k:02d}", z[f"s{k}_pts"], z[f"s{k}_col"], mat, root, s.get("point_radius", 0.03))
            if s.get("points_only_during_cut"):
                _only_during(bpy.data.objects[f"Background{k:02d}"], start, s["frames"])
        if not s.get("shared_fill_from"):
            _fill(k, s, z[f"s{k}_floor"], z[f"s{k}_cam"][0, :3, 3], root)
        if len(z[f"s{k}_body"]):
            _body(f"Body{k:02d}", z[f"s{k}_body"], start, root)
            if s.get("slot", k) != k or "slot" in s:  # in a shared world: the body only during its own cut
                _only_during(bpy.data.objects[f"Body{k:02d}"], start, s["frames"])
        ends.append(start + s["frames"] - 1)
    sc.frame_start, sc.frame_end = 1, max(ends)
    sc.camera = bpy.data.objects["Cam00"]
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    return {"blend": str(out), "frames": sc.frame_end, "shots": len(info["shots"])}


def render_check(out_dir: Path, every: int = 12) -> int:
    """Workbench renders (half size) of every `every`-th frame plus each cut's first frame; the marker camera."""
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    sc.eevee.taa_render_samples = 4
    sc.render.resolution_percentage = 50
    sc.render.image_settings.file_format = "PNG"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(set(range(1, sc.frame_end + 1, every)) | {m.frame for m in sc.timeline_markers})
    for f in frames:
        sc.frame_set(f)
        sc.render.filepath = str(out_dir / f"{f:04d}.png")
        bpy.ops.render.render(write_still=True)
    return len(frames)


def render_top(path: Path, frame: int, extent: float = 16.0) -> None:
    """Orthographic view from above of the shared world (slot 0), every body shown, for checking the layout."""
    sc = bpy.context.scene
    for ob in bpy.data.objects:
        if ob.name.startswith("Backdrop"):
            ob.hide_render = True
        if ob.name.startswith("Body") and ob.animation_data:
            ob.animation_data.action = None
            ob.hide_render = False
    cd = bpy.data.cameras.new("Top")
    cd.type, cd.ortho_scale = "ORTHO", extent
    top = bpy.data.objects.new("Top", cd)
    bpy.context.collection.objects.link(top)
    top.location = (0, 0, 30)
    sc.camera = top
    for m in list(sc.timeline_markers):
        sc.timeline_markers.remove(m)
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x = sc.render.resolution_y = 900
    sc.render.resolution_percentage = 100
    sc.frame_set(frame)
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
