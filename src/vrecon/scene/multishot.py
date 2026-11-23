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


NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
         "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
# semantic proxy colours (scripts/export_clay_blend.SEMANTIC): the V2V prompt's legend names them
SEMANTIC = {"person": (0.95, 0.55, 0.15), "ground": (0.45, 0.6, 0.4), "structure": (0.55, 0.55, 0.75),
            "sky": (0.8, 0.88, 0.95), "held": (0.15, 0.85, 0.95), "prop": (0.9, 0.85, 0.3)}


from .mannequin import dressed, frustum as _frustum, mannequin as _mannequin, open_skirt  # noqa: E402  (pure numpy, also used by the close-up search)


def _body_mannequin(name: str, joints: np.ndarray, start: int, parent, garment=None) -> None:
    v0, faces = _mannequin(joints[0], garment)
    me = bpy.data.meshes.new(name)
    me.from_pydata(v0.tolist(), [], faces)
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.parent = parent
    ob.shape_key_add(name="Basis")
    for t in range(len(joints)):
        kb = ob.shape_key_add(name=f"f{t}")
        kb.data.foreach_set("co", _mannequin(joints[t], garment)[0].astype(np.float32).ravel())
        for f, v in ((t - 1, 0.0), (t, 1.0), (t + 1, 0.0)):
            kb.value = v
            kb.keyframe_insert("value", frame=start + f)
    _label(ob, "person")


def _body_smpl(name: str, verts: np.ndarray, faces: np.ndarray, start: int, parent, joints=None, dress=None) -> None:
    """a person as the SMPL surface fitted to the measured joints (gpu/observe/fit_smpl.py), dressed as measured
    (scripts/multishot_dress.py, scene/mannequin.dressed): one mesh, one shape key per frame, smooth shading."""
    v = np.asarray(verts, np.float32)
    if dress is not None and joints is not None and np.any(np.asarray(dress) > 0):
        out = [dressed(v[t], faces, joints[min(t, len(joints) - 1)], dress) for t in range(len(v))]
        faces = out[0][1]
        v = np.stack([o[0] for o in out]).astype(np.float32)
    me = bpy.data.meshes.new(name)
    me.from_pydata(v[0].tolist(), [], np.asarray(faces).tolist())
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.parent = parent
    ob.shape_key_add(name="Basis")
    for t in range(len(v)):
        kb = ob.shape_key_add(name=f"f{t}")
        kb.data.foreach_set("co", v[t].ravel())
        for f, val in ((t - 1, 0.0), (t, 1.0), (t + 1, 0.0)):
            kb.value = val
            kb.keyframe_insert("value", frame=start + f)
    _label(ob, "person")


SMPL_JOINTS = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
               "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
               "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
               "left_hand", "right_hand"]


def _aa_to_quat(aa: np.ndarray) -> np.ndarray:
    """axis-angle [..., 3] -> quaternion (w, x, y, z) [..., 4]."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    ax = aa / np.maximum(th, 1e-12)
    return np.concatenate([np.cos(th / 2), ax * np.sin(th / 2)], -1)


def _person_rig(ident: str, z, cuts: list[tuple[int, int]], parent) -> None:
    """one person = one SMPL rig for all the cuts it appears in (gpu/observe/fit_person.py: one shared body).
    Rig_<id>: an armature whose bones are SMPL's 24 joints at the person's rest joints, every bone along +Y with no
    roll (rest orientation = identity, so a pose bone's rotation is SMPL's local joint rotation as it is), the
    pelvis bone's location the root shift. Body_<id>: the rest surface (with the measured cloth layer and open coat,
    scripts/multishot_dress.py) skinned with SMPL's weights. One action holds the motion of every cut at its
    timeline frames; the body is shown only during those cuts. Editing the rest body or a bone changes every cut."""
    rest_v = z[f"person_{ident}_rest_verts"].astype(float)
    rest_j = z[f"person_{ident}_rest_joints"].astype(float)
    W = z[f"person_{ident}_weights"]
    parents = z["smpl_parents"]
    faces = np.asarray(z["smpl_faces"])
    dress = [z[f"s{k}_dress"] for k, _ in cuts if f"s{k}_dress" in z]
    off = float(np.median([d[0] for d in dress])) if dress else 0.0
    hem = max([float(d[1]) for d in dress], default=0.0)
    length = max([float(d[2]) for d in dress if d[1] > 0], default=0.0)
    v, f = dressed(rest_v, faces, rest_j[:22], (off, 0.0, 0.0))  # the coat's skirt is cloth (_coat_cloth)
    weights = np.asarray(W, np.float32)
    arm = bpy.data.armatures.new(f"Rig_{ident}")
    rig = bpy.data.objects.new(f"Rig_{ident}", arm)
    bpy.context.collection.objects.link(rig)
    rig.parent = parent
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    bones = []
    for i, name in enumerate(SMPL_JOINTS):
        b = arm.edit_bones.new(name)
        b.head = tuple(rest_j[i])
        b.tail = tuple(rest_j[i] + np.array([0, 0.05, 0]))
        b.roll = 0.0
        if parents[i] >= 0:
            b.parent = bones[parents[i]]
        bones.append(b)
    bpy.ops.object.mode_set(mode="OBJECT")
    me = bpy.data.meshes.new(f"Body_{ident}")
    me.from_pydata(v.tolist(), [], np.asarray(f).tolist())
    me.shade_smooth()
    ob = bpy.data.objects.new(f"Body_{ident}", me)
    bpy.context.collection.objects.link(ob)
    ob.parent = rig
    for i, name in enumerate(SMPL_JOINTS):
        g = ob.vertex_groups.new(name=name)
        idx = np.flatnonzero(weights[:, i] > 1e-4)
        for wv in np.unique(np.round(weights[idx, i], 3)):
            g.add(idx[np.round(weights[idx, i], 3) == wv].tolist(), float(wv), "REPLACE")
    ob.modifiers.new("rig", "ARMATURE").object = rig
    _label(ob, "person")
    ob["vrecon_person"] = ident
    rig["vrecon_person"] = ident
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
    shown = []
    for k, start in cuts:
        q = _aa_to_quat(z[f"s{k}_pose"].reshape(-1, 24, 3))
        tr = z[f"s{k}_transl"]
        for t in range(len(q)):
            for i, pb in enumerate(rig.pose.bones):
                pb.rotation_quaternion = tuple(q[t, SMPL_JOINTS.index(pb.name)])
                pb.keyframe_insert("rotation_quaternion", frame=start + t)
            root = rig.pose.bones["pelvis"]
            root.location = tuple(tr[t])  # SMPL rotates about the rest pelvis joint, then adds transl
            root.keyframe_insert("location", frame=start + t)
        shown.append((start, start + len(q) - 1))
    if hem > 0:
        ob.modifiers.new("collision", "COLLISION")
        ob.collision.thickness_outer = 0.015
        for k, start in cuts:
            _coat_cloth(f"Coat_{ident}_cut{k:02d}", rest_j, rig, ob, start, len(z[f"s{k}_pose"]), length, 0.17 + off, hem)
    for o in (ob,):
        o.hide_render = o.hide_viewport = True
        o.keyframe_insert("hide_render", frame=1)
        o.keyframe_insert("hide_viewport", frame=1)
        for a0, a1 in sorted(shown):
            for fr, hide in ((a0, False), (a1 + 1, True)):
                o.hide_render = o.hide_viewport = hide
                o.keyframe_insert("hide_render", frame=fr)
                o.keyframe_insert("hide_viewport", frame=fr)


COAT_RINGS, COAT_SEGMENTS = 14, 28


def _coat_cloth(name, rest_j, rig, body, start, frames, length, r_top, r_hem) -> None:
    """a long coat's skirt as Blender cloth, one per cut (a cut change moves the body at once, which a single cloth
    run could not follow): an open-front flared surface (scene/mannequin.open_skirt) at the rest pose, its waist ring
    pinned and carried by the rig (armature modifier before the cloth), the rest hanging and swinging with the body,
    colliding with it. The simulation covers the cut's frames only and is baked into the .blend; the coat is shown
    only during the cut."""
    v, f = open_skirt(rest_j[:22], length, r_top, r_hem, COAT_SEGMENTS, COAT_RINGS)
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.tolist(), [], [tuple(int(i) for i in q) for q in f])
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.parent = rig
    n = COAT_SEGMENTS
    rows = np.arange(len(v)) // n
    sp = ob.vertex_groups.new(name="spine1")  # the waist follows the spine, the hem its side's hip a little
    sp.add(np.flatnonzero(rows == 0).tolist(), 1.0, "REPLACE")
    pin = ob.vertex_groups.new(name="pin")
    pin.add(np.flatnonzero(rows == 0).tolist(), 1.0, "REPLACE")
    pin.add(np.flatnonzero(rows == 1).tolist(), 0.6, "REPLACE")
    hips = (ob.vertex_groups.new(name="left_hip"), ob.vertex_groups.new(name="right_hip"))
    side = (v - rest_j[0]) @ (rest_j[1] - rest_j[2]) > 0
    for r in range(1, COAT_RINGS + 1):
        u = r / COAT_RINGS
        idx = np.flatnonzero(rows == r)
        sp.add(idx.tolist(), 1.0 - 0.5 * u, "REPLACE")
        hips[0].add(idx[side[idx]].tolist(), 0.5 * u, "REPLACE")
        hips[1].add(idx[~side[idx]].tolist(), 0.5 * u, "REPLACE")
    ob.modifiers.new("rig", "ARMATURE").object = rig
    cl = ob.modifiers.new("cloth", "CLOTH")
    st = cl.settings
    st.quality = 8
    st.mass = 0.4
    st.air_damping = 2.0
    st.tension_stiffness = st.compression_stiffness = 20.0
    st.bending_stiffness = 4.0  # coat cloth: heavier and stiffer than a shirt, fewer small creases
    st.vertex_group_mass = "pin"
    st.pin_stiffness = 1.0
    cl.collision_settings.use_collision = True
    cl.collision_settings.distance_min = 0.015
    cl.collision_settings.use_self_collision = True
    cl.collision_settings.self_distance_min = 0.01
    cache = cl.point_cache
    cache.frame_start, cache.frame_end = start, start + frames - 1
    _label(ob, "person")
    ob["vrecon_coat_of"] = body.name
    _only_during(ob, start, frames)
    sc = bpy.context.scene
    keep = (sc.frame_start, sc.frame_end)
    sc.frame_start, sc.frame_end = min(sc.frame_start, start), max(sc.frame_end, start + frames - 1)
    with bpy.context.temp_override(scene=sc, active_object=ob, object=ob, point_cache=cache):
        bpy.ops.ptcache.bake(bake=True)
    sc.frame_start, sc.frame_end = keep


def _label(ob, cat: str) -> None:
    ob["vrecon_category"] = cat
    ob.color = (*SEMANTIC[cat], 1.0)


def _surface(name: str, pts: np.ndarray, radius: float, parent, cat: str) -> None:
    """points -> a closed surface (points to volume, volume to mesh): solid set pieces for clay renders."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(pts.tolist(), [], [])
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.parent = parent
    ng = bpy.data.node_groups.new(name + "_surf", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    p2v = ng.nodes.new("GeometryNodePointsToVolume")
    v2m = ng.nodes.new("GeometryNodeVolumeToMesh")
    m2p.inputs["Radius"].default_value = radius * 2.0
    p2v.inputs["Resolution Mode"].default_value = "Size"
    p2v.inputs["Voxel Size"].default_value = radius * 1.2
    p2v.inputs["Radius"].default_value = radius * 2.0
    ng.links.new(gi.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(m2p.outputs["Points"], p2v.inputs["Points"])
    ng.links.new(p2v.outputs["Volume"], v2m.inputs["Volume"])
    ng.links.new(v2m.outputs["Mesh"], go.inputs[0])
    ob.modifiers.new("surface", "NODES").node_group = ng
    _label(ob, cat)


def _layout(z, parent) -> None:
    """flat pieces (one mesh per category) and boxes from scripts/multishot_layout.py."""
    from .character import _box

    quads, cats = z["layout_quads"], z["layout_quad_cat"]
    for cat in sorted(set(cats.tolist())):
        q = quads[cats == cat]
        me = bpy.data.meshes.new(f"Layout_{cat}")
        me.from_pydata(q.reshape(-1, 3).tolist(), [], [tuple(range(4 * i, 4 * i + 4)) for i in range(len(q))])
        me.validate()
        ob = bpy.data.objects.new(f"Layout_{cat}", me)
        bpy.context.collection.objects.link(ob)
        ob.parent = parent
        ob.modifiers.new("thick", "SOLIDIFY").thickness = 0.05
        _label(ob, cat)
    for i, b in enumerate(z["layout_boxes"]):
        v, f = _box(b[:3], tuple(b[3:6]))
        me = bpy.data.meshes.new(f"LayoutBox{i:02d}")
        me.from_pydata(v, [], f)
        ob = bpy.data.objects.new(f"LayoutBox{i:02d}", me)
        bpy.context.collection.objects.link(ob)
        ob.parent = parent
        _label(ob, "structure")


def _extra_people(k: int, arr: np.ndarray, start: int, parent) -> None:
    """coarse figures (no limbs: a body and a head, as export_clay_blend's coarse people) where extra people stand;
    keyed place, shown only while seen."""
    from .character import _box, _cylinder

    for pi, track in enumerate(arr):
        h = float(track[:, 2].max())
        if h <= 0:
            continue
        v1, f1 = _cylinder(np.zeros(3), np.array([0, 0, h * 0.86]), 0.19, 12)
        v2, f2 = _box(np.array([0, 0, h * 0.93]), (0.2, 0.22, h * 0.14))
        me = bpy.data.meshes.new(f"Extra{k:02d}_{pi}")
        me.from_pydata(v1 + v2, [], f1 + [tuple(x + len(v1) for x in ff) for ff in f2])
        ob = bpy.data.objects.new(me.name, me)
        bpy.context.collection.objects.link(ob)
        ob.parent = parent
        _label(ob, "person")
        prev = None
        for t, (x, y, _, vis) in enumerate(track):
            ob.location = (x, y, 0.0)
            ob.keyframe_insert("location", frame=start + t)
            hide = bool(vis < 0.5)
            if hide != prev:
                ob.hide_render = ob.hide_viewport = hide
                ob.keyframe_insert("hide_render", frame=start + t)
                ob.keyframe_insert("hide_viewport", frame=start + t)
                prev = hide
        ob.hide_render = ob.hide_viewport = True
        ob.keyframe_insert("hide_render", frame=start + len(track))
        ob.keyframe_insert("hide_viewport", frame=start + len(track))
        if start > 1:
            ob.keyframe_insert("hide_render", frame=1)
            ob.keyframe_insert("hide_viewport", frame=1)


def _objects(k: int, arr: np.ndarray, labels, start: int, parent) -> None:
    """object proxies: a rough shape of the object's kind (scene/props.py) at its size, keyed place, shown only
    while seen. Sizes are (width, height, depth) in the picture's sense; z is up."""
    from .props import prop_mesh

    for oi, (track, lab) in enumerate(zip(arr, labels)):
        vis = track[:, 6] > 0
        if not vis.any():
            continue
        size = tuple(float(x) for x in track[vis][0, 3:6])
        word, _, shape = str(lab).split(":")[1].partition("|")
        v, f = prop_mesh(shape or word, size)  # "gun|box": the box matched the masks better
        me = bpy.data.meshes.new(f"Obj{k:02d}_{oi}_{word}")
        me.from_pydata(v, [], f)
        ob = bpy.data.objects.new(me.name, me)
        bpy.context.collection.objects.link(ob)
        ob.parent = parent
        _label(ob, str(lab).split(":")[0])
        idx = np.flatnonzero(vis)
        prev = None
        for t in range(len(track)):
            j = idx[np.argmin(np.abs(idx - t))]  # hold the nearest seen place
            ob.location = tuple(track[j, :3])
            ob.keyframe_insert("location", frame=start + t)
            hide = not bool(vis[t])
            if hide != prev:
                ob.hide_render = ob.hide_viewport = hide
                ob.keyframe_insert("hide_render", frame=start + t)
                ob.keyframe_insert("hide_viewport", frame=start + t)
                prev = hide
        for fr in ([1] if start > 1 else []) + [start + len(track)]:
            ob.hide_render = ob.hide_viewport = True
            ob.keyframe_insert("hide_render", frame=fr)
            ob.keyframe_insert("hide_viewport", frame=fr)


def _llm_set(pieces: list, parent) -> None:
    """set pieces a model added (scripts/multishot_llm.py): one object each, LLM_<name>, so a check can hide one."""
    from .mannequin import _box

    for p in pieces:
        c, s = np.array(p["center"], float), np.array(p["size"], float)
        if p.get("type") == "plane":
            hx, hy = s[0] / 2, s[1] / 2
            v, f = [(-hx, -hy, 0.0), (hx, -hy, 0.0), (hx, hy, 0.0), (-hx, hy, 0.0)], [(0, 1, 2, 3)]
        else:
            v, f = _box(np.zeros(3), tuple(max(x, 1e-3) for x in s))
        th = np.radians(float(p.get("yaw_deg", 0)))
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        v = [tuple(R @ np.array(x) + c) for x in v]
        me = bpy.data.meshes.new(f"LLM_{p['name']}")
        me.from_pydata(v, [], f)
        ob = bpy.data.objects.new(me.name, me)
        bpy.context.collection.objects.link(ob)
        ob.parent = parent
        if p.get("type") == "plane":
            ob.modifiers.new("thick", "SOLIDIFY").thickness = 0.05
        _label(ob, "ground" if p.get("category") == "ground" else "structure")


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
    sp.data.flip_normals()  # seen from inside: lit like a dome, not its dark back
    sp.data.materials.append(_flat(f"Backdrop{k:02d}", s["backdrop_rgb"]))
    sp.parent = parent
    _label(sp, "sky")
    if len(floor):
        bpy.ops.mesh.primitive_plane_add(size=60.0, location=tuple(floor[:3]))
        pl = bpy.context.active_object
        pl.name = f"Floor{k:02d}"
        pl.rotation_mode = "QUATERNION"
        pl.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(tuple(floor[3:6])))
        pl.data.materials.append(_flat(f"Floor{k:02d}", [float(v) for v in floor[6:9]]))
        pl.parent = parent
        _label(pl, "ground")


def _only_during(ob, start: int, frames: int) -> None:
    """visible only in timeline frames [start, start + frames)."""
    for f, hide in ((1, True), (start, False), (start + frames, True)):
        ob.hide_render = ob.hide_viewport = hide
        ob.keyframe_insert("hide_render", frame=f)
        ob.keyframe_insert("hide_viewport", frame=f)


def build(prep: Path, out: Path, finished: bool = False, evidence_only: bool = False) -> dict:
    """evidence_only (DESIGN_v3): cameras, people, floor and backdrop only, the locked base an agent builds on; no
    set pieces, objects or added pieces from the observations. finished: mannequin bodies, points as labelled surfaces (ground near z = 0 in the shared world, else
    structure), backdrop = sky, floor = ground; colours are the semantic proxy colours (render_semantic)."""
    z = np.load(prep)
    info = json.loads(str(z["info"]))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    W, H = int(z["W"]), int(z["H"])
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.fps, sc.render.fps_base = 24000, 1001
    mat = _points_material()
    ends = []
    rigged: dict[str, list] = {}  # person id -> [(cut, timeline start)] (_person_rig)
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
        if evidence_only:
            pass
        elif finished and "layout_quads" in z and s.get("slot", k) == 0 and not s.get("points_only_during_cut"):
            if not bpy.data.objects.get("Layout_ground") and not bpy.data.objects.get("Layout_structure"):
                _layout(z, root)
        elif finished and s.get("points_only_during_cut") and "layout_quads" in z:
            pass  # a close-up placed in the shared world: its own close points are blurred people and hands, the world's layout is its background
        elif len(z[f"s{k}_pts"]) and finished:
            P = z[f"s{k}_pts"]
            r = max(s.get("point_radius", 0.03), 0.02)
            g = (np.abs(P[:, 2]) < 0.12) if s.get("slot", k) == 0 and "joined" in info else np.zeros(len(P), bool)
            for cat, sel in (("ground", g), ("structure", ~g)):
                if sel.sum() > 50:
                    _surface(f"{cat.title()}{k:02d}", P[sel], r, root, cat)
                    if s.get("points_only_during_cut"):
                        _only_during(bpy.data.objects[f"{cat.title()}{k:02d}"], start, s["frames"])
        elif len(z[f"s{k}_pts"]):
            _points(f"Background{k:02d}", z[f"s{k}_pts"], z[f"s{k}_col"], mat, root, s.get("point_radius", 0.03))
            if s.get("points_only_during_cut"):
                _only_during(bpy.data.objects[f"Background{k:02d}"], start, s["frames"])
        if not s.get("shared_fill_from"):
            _fill(k, s, z[f"s{k}_floor"], z[f"s{k}_cam"][0, :3, 3], root)
        if finished and f"s{k}_person" in z and f"person_{z[f's{k}_person']}_rest_verts" in z:
            rigged.setdefault(str(z[f"s{k}_person"]), []).append((k, start, root))  # drawn once per person after the loop
        elif len(z[f"s{k}_body"]):
            if finished and f"s{k}_smpl" in z:
                _body_smpl(f"Body{k:02d}", z[f"s{k}_smpl"], z["smpl_faces"], start, root, z[f"s{k}_body"],
                           z[f"s{k}_dress"] if f"s{k}_dress" in z else None)
            elif finished:
                _body_mannequin(f"Body{k:02d}", z[f"s{k}_body"], start, root, z[f"s{k}_garment"] if f"s{k}_garment" in z else None)
            else:
                _body(f"Body{k:02d}", z[f"s{k}_body"], start, root)
            if s.get("slot", k) != k or "slot" in s:  # in a shared world: the body only during its own cut
                _only_during(bpy.data.objects[f"Body{k:02d}"], start, s["frames"])
        if finished and not evidence_only and f"s{k}_objects" in z and len(z[f"s{k}_objects"]):
            _objects(k, z[f"s{k}_objects"], z[f"s{k}_object_labels"], start, root)
        if finished and f"s{k}_extra" in z and len(z[f"s{k}_extra"]):
            _extra_people(k, z[f"s{k}_extra"], start, root)
        ends.append(start + s["frames"] - 1)
    for ident, cuts in rigged.items():  # parented to the first cut's root (joined cuts share slot 0)
        _person_rig(ident, z, [(k, st) for k, st, _ in cuts], cuts[0][2])
    if finished and not evidence_only and "llm_set" in z:
        _llm_set(json.loads(str(z["llm_set"])), bpy.data.objects.get("Shot00"))
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


def render_semantic(out_mp4: Path, frames: int, size: tuple[int, int], start: int = 1) -> Path:
    """Semantic clay control (as scripts/export_clay_blend.py --semantic): Workbench, studio light fixed in the
    world, object colours = category colours, cavity and shadows; the marker cameras; H3 sizes are the caller's."""
    import tempfile

    from vrecon.control.core import encode_mp4

    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light, sh.color_type = "STUDIO", "OBJECT"
    sh.show_cavity, sh.show_shadows = True, True
    sh.use_world_space_lighting = True
    sc.display.light_direction = (0.35, -0.45, 0.82)
    sc.render.resolution_x, sc.render.resolution_y = size
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    for cam in (o for o in bpy.data.objects if o.type == "CAMERA"):
        cam.data.clip_end = 1000
    from vrecon.control.core import load_frames

    if out_mp4 is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(frames):
            sc.frame_set(start + i)
            sc.render.filepath = str(Path(td) / f"{i:05d}.png")
            bpy.ops.render.render(write_still=True)
            paths.append(Path(td) / f"{i:05d}.png")
        imgs = load_frames(paths)
        encode_mp4(imgs, out_mp4, fps=24)
    return out_mp4


LABELS = ["none", *SEMANTIC]  # index in the label images: 0 = nothing drawn


def render_labels(out_npz: Path, frames: list[int], size: tuple[int, int], hide: tuple[str, ...] = ()) -> Path:
    """Category label images for checking (scripts/multishot_check.py): Workbench flat, no anti-aliasing, object
    colour = category colour, decoded back to LABELS indices. Objects whose name starts with a `hide` prefix are left
    out (component ablation). Writes labels uint8 [F, H, W], frames [F]."""
    import tempfile

    from vrecon.control.core import load_frames

    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light, sh.color_type = "FLAT", "OBJECT"
    sh.show_cavity = sh.show_shadows = sh.show_specular_highlight = sh.show_object_outline = False
    sh.background_type = "VIEWPORT"
    sh.background_color = (0.0, 0.0, 0.0)
    sc.display.render_aa = "OFF"
    sc.view_settings.view_transform = "Standard"
    sc.render.film_transparent = False
    sc.render.resolution_x, sc.render.resolution_y = size
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    for ob in bpy.data.objects:
        if ob.type == "CAMERA":
            ob.data.clip_end = 1000
        elif hide and ob.name.startswith(tuple(hide)):
            ob.hide_render = True
            for fc in [c for c in _fcurves(ob) if c.data_path == "hide_render"]:
                for kp in fc.keyframe_points:  # keyed visibility (_only_during) would bring it back
                    kp.co[1] = 1.0
    ref = np.array([[0, 0, 0], *[_srgb(c) for c in SEMANTIC.values()]], float)
    out = []
    with tempfile.TemporaryDirectory() as td:
        for f in frames:
            sc.frame_set(f)
            sc.render.filepath = str(Path(td) / f"{f:05d}.png")
            bpy.ops.render.render(write_still=True)
            img = load_frames([Path(td) / f"{f:05d}.png"])[0][..., :3].astype(float)
            d = ((img[..., None, :] - ref) ** 2).sum(-1)
            out.append(d.argmin(-1).astype(np.uint8))
    np.savez_compressed(out_npz, labels=np.array(out), frames=np.array(frames), names=np.array(LABELS))
    return out_npz


def _srgb(c) -> list[float]:
    c = np.asarray(c, float)
    return list(255 * np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055))


def _fcurves(ob):
    from bpy_extras.anim_utils import action_get_channelbag_for_slot

    ad = ob.animation_data
    cb = action_get_channelbag_for_slot(ad.action, ad.action_slot) if ad and ad.action else None
    return list(cb.fcurves) if cb else []
