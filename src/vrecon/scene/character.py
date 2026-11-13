"""Characters (SceneObject kind 'character' with a skeleton file, analysis/from_gvhmr.py) as a Blender armature
plus a simple body mesh: one cylinder per bone (head: a box), each bound rigidly to its bone.

The mesh carries the object's vrecon_id (so depth / index passes and looks treat the person as one object);
the armature carries `vrecon_rig_of` instead, so read-back does not take the rig for an object. Bones are made
with the rest orientation of the world axes (head at the joint, tail +Y, roll 0): the skeleton file's local
rotations, given in the rest frame's axes, then go straight into pose_bone.rotation_quaternion.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from vrecon.analysis.appearance import body_part
from vrecon.schema import Project, SceneObject, Shot

BONE_TAIL = 0.05  # m, along +Y
RADIUS = {"spine": 0.11, "pelvis": 0.11, "hip": 0.08, "knee": 0.06, "ankle": 0.045, "foot": 0.04, "neck": 0.05,
          "collar": 0.05, "shoulder": 0.05, "elbow": 0.04, "wrist": 0.035, "head": 0.05}


def _radius(name: str) -> float:
    return next((r for k, r in RADIUS.items() if k in name), 0.05)


def _cylinder(a: np.ndarray, b: np.ndarray, r: float, n: int = 8):
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return [], []
    z = axis / length
    x = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ring = [x * math.cos(2 * math.pi * i / n) * r + y * math.sin(2 * math.pi * i / n) * r for i in range(n)]
    verts = [tuple(a + v) for v in ring] + [tuple(b + v) for v in ring]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    return verts, faces


def _box(c: np.ndarray, size: tuple[float, float, float]):
    sx, sy, sz = (s / 2 for s in size)
    verts = [tuple(c + np.array([x * sx, y * sy, z * sz])) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return verts, faces


def build_character(project: Project, obj: SceneObject, shot: Shot, root: Path | None, report, id_prop: str) -> bool:
    """Returns False (and warns) when the skeleton file cannot be used; the caller then makes a proxy box."""
    from .convert import continuous_quats, scene_time_to_shot_frame, shot_frame_to_blender

    path = Path(obj.asset.path)
    if not path.is_absolute():
        path = (root or Path.cwd()) / path
    if not path.is_file():
        report.warnings.append(f"{obj.id}: skeleton file {path} not found; using a proxy box")
        return False
    d = np.load(path)
    names = [str(n) for n in d["joint_names"]]
    parents = [int(p) for p in d["parents"]]
    rest = d["rest_joints"].astype(float)
    rot = d["local_rot"].astype(float)
    if len(obj.keys) != len(rot):
        report.warnings.append(f"{obj.id}: {len(obj.keys)} keys but {len(rot)} skeleton frames; using a proxy box")
        return False

    scene = bpy.context.scene
    arm_data = bpy.data.armatures.new(f"{obj.id}_rig")
    arm = bpy.data.objects.new(f"{obj.name or obj.id}_rig", arm_data)
    scene.collection.objects.link(arm)
    arm["vrecon_rig_of"] = obj.id
    sc = float(obj.rest.scale[0])  # skeleton in metres -> scene units (rest.scale, uniform)
    if max(obj.rest.scale) - min(obj.rest.scale) > 1e-9:
        report.warnings.append(f"{obj.id}: non-uniform rest.scale {obj.rest.scale}; using x")
    arm.scale = (sc, sc, sc)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = []
    for j, n in enumerate(names):
        eb = arm_data.edit_bones.new(n)
        eb.head = Vector(rest[j])
        eb.tail = Vector(rest[j] + np.array([0.0, BONE_TAIL, 0.0]))
        eb.roll = 0.0
        ebs.append(eb)
    for j, p in enumerate(parents):
        if p >= 0:
            ebs[j].parent = ebs[p]
            ebs[j].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    from .builder import DEFAULT_LOOK, set_look

    if "body_verts" in d.files:
        # the person's SMPL-X body (gpu/observe/gvhmr_body_mesh.py), skinned on the same joints: Blender's armature
        # deform with these weights is SMPL-X's linear blend skinning (without pose correctives)
        bv, bf, bw = d["body_verts"].astype(float), d["body_faces"].astype(int), d["body_weights"].astype(float)
        mesh = bpy.data.meshes.new(obj.id)
        mesh.from_pydata(bv.tolist(), [], bf.tolist())
        mesh.update()
        for poly in mesh.polygons:
            poly.use_smooth = True
        main = bw.argmax(1)
        from vrecon.analysis.appearance import clothed_part, part_colour

        look = np.array([part_colour(obj.look, clothed_part(names[j]), DEFAULT_LOOK) for j in main], float)
        set_look(mesh, look)
        body = bpy.data.objects.new(obj.name or obj.id, mesh)
        scene.collection.objects.link(body)
        body[id_prop] = obj.id
        for j, n in enumerate(names):
            idx = np.nonzero(bw[:, j] > 1e-4)[0]
            if len(idx) == 0:
                continue
            vg = body.vertex_groups.new(name=n)
            for wv in np.unique(np.round(bw[idx, j], 4)):  # add() takes one weight per call: group equal weights
                sel = idx[np.round(bw[idx, j], 4) == wv]
                vg.add(sel.tolist(), float(wv), "REPLACE")
        body_note = "SMPL-X body"
    else:
        # body mesh: a cylinder from each joint's parent to the joint, bound to the parent bone; a box for the head
        verts, faces, groups = [], [], []
        for j, p in enumerate(parents):
            if p < 0:
                continue
            v, f = _cylinder(rest[p], rest[j], _radius(names[p]))
            base = len(verts)
            verts += v
            faces += [tuple(i + base for i in face) for face in f]
            groups.append((names[p], range(base, base + len(v))))
        h = names.index("head") if "head" in names else None
        if h is not None:
            v, f = _box(rest[h] + np.array([0, 0, 0.09]), (0.16, 0.19, 0.22))
            base = len(verts)
            verts += v
            faces += [tuple(i + base for i in face) for face in f]
            groups.append(("head", range(base, base + len(v))))
        mesh = bpy.data.meshes.new(obj.id)
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        part_of = {bone: body_part(bone) for bone, _ in groups}
        look = np.zeros((len(verts), 3))
        for bone, idx in groups:
            look[list(idx)] = obj.look.get(part_of[bone], obj.look.get("all", DEFAULT_LOOK))
        set_look(mesh, look)
        body = bpy.data.objects.new(obj.name or obj.id, mesh)
        scene.collection.objects.link(body)
        body[id_prop] = obj.id
        for bone, idx in groups:
            vg = body.vertex_groups.get(bone) or body.vertex_groups.new(name=bone)
            vg.add(list(idx), 1.0, "REPLACE")
        body_note = "simple cylinders"
    body.parent = arm
    mod = body.modifiers.new("skeleton", "ARMATURE")
    mod.object = arm

    mapping = next((m for m in project.reconstruction.shot_mapping if m.shot_id == shot.id), None)
    start_time = mapping.scene_time_start if mapping else 0.0
    pbs = [arm.pose.bones[n] for n in names]
    for pb in pbs:
        pb.rotation_mode = "QUATERNION"
    # the root (pelvis) follows the object's keys, so a placement or an edit of the keys moves the whole body;
    # the joints below it follow the skeleton file
    root_j = parents.index(-1)
    quats = [continuous_quats([tuple(q) for q in rot[:, j]]) for j in range(len(names))]
    quats[root_j] = continuous_quats([k.pose.rotation for k in obj.keys])
    for i, key in enumerate(obj.keys):
        frame = shot_frame_to_blender(scene_time_to_shot_frame(key.time, start_time, project.source.fps))
        for j, pb in enumerate(pbs):
            pb.rotation_quaternion = quats[j][i]
            pb.keyframe_insert("rotation_quaternion", frame=frame)
        # the rig object carries the body's scale (scene units per metre, uniform); the root bone's location is in
        # the rig's unscaled space
        pbs[root_j].location = Vector(np.array(key.pose.translation) / sc - rest[root_j])
        pbs[root_j].keyframe_insert("location", frame=frame)
    report.warnings.append(f"{obj.id}: skeleton with {len(names)} joints, {len(obj.keys)} frames (body = {body_note})")
    return True
