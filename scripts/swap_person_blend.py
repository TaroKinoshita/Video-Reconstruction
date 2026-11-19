"""PLAN v2 7.2 parts 3-4 on a .blend: the person a draft made (mannequin, cylinders) is replaced by the observed body
(GVHMR skeleton + SMPL-X), and what that person held stays in the new hand. Generic: nothing per clip.

    python scripts/swap_person_blend.py IN.blend SKELETON.npz OUT.blend [--report OUT.json]

1. Placement: analysis/place_person.place_in_scene through the .blend's own source camera, the floor at the height of
   the draft's floor (obj_floor after run/refine.mark_set, else z = 0): joints on the camera's rays, feet on the floor.
2. The draft's person: objects whose pixels in the source view lie mostly (>= 60 %) inside the observed person's
   joint box, over a few frames, and are not set pieces. They are hidden (hide_render, hide_viewport), not deleted.
3. Held things: visible objects parented (directly or further up) to a hidden part keep their world motion relative
   to the nearest wrist: their origin (where the draft's hand held them) is put on the body's wrist at the first frame,
   then a Child Of constraint to that wrist bone (inverse taken there) makes them follow the new hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("blend", type=Path)
    ap.add_argument("skeleton", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    import bpy

    from vrecon.analysis.place_person import place_in_scene
    from vrecon.run.refine import mark_set
    from vrecon.scene.builder import ID_PROP, BuildReport, _build_object
    from vrecon.scene.novel_blend import open_scene, render_view
    from vrecon.schema import Pose

    rec = open_scene(a.blend)
    probe = rec.model_copy(deep=True)
    renamed = mark_set(probe)
    floor = next((o for o in probe.reconstruction.objects if o.id == "obj_floor"), None)
    floor_z = 0.0
    if floor is not None:
        fid = next(k for k, v in renamed.items() if v == "obj_floor")
        bo = next(o for o in bpy.data.objects if o.get(ID_PROP) == fid)
        from mathutils import Vector

        floor_z = float(max((bo.matrix_world @ Vector(c)).z for c in bo.bound_box))
    cam = rec.reconstruction.cameras[0]
    shifted = cam.model_copy(update={"keys": [k.model_copy(update={"pose": Pose(
        translation=(k.pose.translation[0], k.pose.translation[1], k.pose.translation[2] - floor_z),
        rotation=k.pose.rotation)}) for k in cam.keys]})
    person, rep = place_in_scene(a.skeleton, str(a.skeleton.resolve()), shifted, obj_id="obj_person_observed",
                                 name="person (observed)", source=rec.source)
    person = person.model_copy(update={"keys": [k.model_copy(update={"pose": Pose(
        translation=(k.pose.translation[0], k.pose.translation[1], k.pose.translation[2] + floor_z),
        rotation=k.pose.rotation)}) for k in person.keys]})
    person = person.model_copy(update={"rest": person.keys[0]})

    # the draft's person parts: mostly inside the observed person's joint box in the source view
    sk = np.load(a.skeleton)
    pc, kg = sk["joints_cam"], sk["K"]
    n = min(len(pc), rec.source.frame_count)
    frames = [int(x) for x in np.linspace(0, n - 1, 5)]
    inside, total = {}, {}
    k_s = cam.intrinsics
    for f in frames:
        idx, idmap = render_view(cam, [f])
        img = idx[0]
        uv = (pc[f, :, :2] / pc[f, :, 2:]) * [kg[f][0, 0], kg[f][1, 1]] + [kg[f][0, 2], kg[f][1, 2]]
        uv = uv * [k_s.width / float(sk["width"]), k_s.height / float(sk["height"])]
        lo, hi = uv.min(0), uv.max(0)
        pad = 0.15 * (hi - lo)
        x0, y0 = np.floor(lo - pad).astype(int)
        x1, y1 = np.ceil(hi + pad).astype(int)
        box = np.zeros(img.shape, bool)
        box[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
        for lab, oid in idmap.items():
            m = img == lab
            if m.any():
                total[oid] = total.get(oid, 0) + int(m.sum())
                inside[oid] = inside.get(oid, 0) + int((m & box).sum())
    sets = {o.id for o in probe.reconstruction.objects if o.kind == "set"} | set(renamed)
    byid = {o.get(ID_PROP): o for o in bpy.data.objects if o.get(ID_PROP)}
    sc = bpy.context.scene

    def world_at(o, f):
        sc.frame_set(sc.frame_start + f)
        return o.matrix_world.translation.copy()

    def moves(o) -> bool:
        p0, p1 = world_at(o, frames[0]), world_at(o, frames[-1])
        return (p1 - p0).length > 0.05 or any((world_at(o, f) - p0).length > 0.05 for f in frames[1:-1])

    inside_moving = [byid[p] for p in total if p in byid and p not in sets and inside.get(p, 0) >= 0.6 * total[p]
                     and moves(byid[p])]

    def top(o):
        while o.parent is not None:
            o = o.parent
        return o

    def subtree(o):
        yield o
        for c in o.children:
            yield from subtree(c)

    roots = []
    for o in inside_moving:
        r = top(o)
        if r not in roots:
            roots.append(r)
    person_meshes = [m for r in roots for m in subtree(r) if m.type == "MESH"] or inside_moving
    # build the observed body into this scene
    report = BuildReport(shot_id=rec.shots[0].id, camera="")
    shot = rec.shots[0]
    proj = rec.model_copy(deep=True)
    proj.reconstruction.objects = [person]
    _build_object(proj, person, shot, report, None)
    arm = next(o for o in bpy.data.objects if o.get("vrecon_rig_of") == person.id)
    sc.frame_set(sc.frame_start)
    bpy.context.view_layer.update()
    from mathutils import Vector

    def centre(m):
        return np.mean([list(m.matrix_world @ Vector(c)) for c in m.bound_box], axis=0)

    def seg_dist(c, a, b):
        ab = b - a
        t = np.clip(((c - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
        return float(np.linalg.norm(c - (a + t * ab)))

    far = []
    for r in roots:  # "far" from the draft's OWN bones (segments between its empties / armature bones), as a share
        nodes = [x for x in subtree(r) if x.type != "MESH"]  # of its own height
        segs = []
        for x in nodes:
            if x.parent is not None and x.parent.type != "MESH":
                segs.append((np.array(list(x.parent.matrix_world.translation)), np.array(list(x.matrix_world.translation))))
            if x.type == "ARMATURE":
                segs += [(np.array(list(x.matrix_world @ b.head)), np.array(list(x.matrix_world @ b.tail))) for b in x.pose.bones]
        mesh_r = [m for m in subtree(r) if m.type == "MESH"]
        if not segs or not mesh_r:
            continue
        cs = np.array([centre(m) for m in mesh_r])
        height = float(cs[:, 2].max() - cs[:, 2].min()) or 1.0
        far += [m for m, c in zip(mesh_r, cs) if min(seg_dist(c, a, b) for a, b in segs) > 0.15 * height]

    def ancestors(o):
        out = [o]
        while o.parent is not None:
            o = o.parent
            out.append(o)
        return out

    # held things: per cluster of far meshes under one root, the lowest common ancestor (the prop's own node)
    held_roots = []
    for r in roots:
        fs = [m for m in far if top(m) is r]
        if not fs:
            continue
        common = set(ancestors(fs[0]))
        for m in fs[1:]:
            common &= set(ancestors(m))
        lca = min(common, key=lambda x: -len(ancestors(x)))  # deepest common ancestor
        if lca is r:  # far parts hang from the root itself: each far mesh is its own held thing
            held_roots += fs
        else:
            held_roots.append(lca)
    held_set = {m for h in held_roots for m in subtree(h)}
    hidden = [m for m in person_meshes if m not in held_set]
    for o in hidden:
        o.hide_render = o.hide_viewport = True
    names = [str(x) for x in sk["joint_names"]]
    wrists = [w for w in ("right_wrist", "left_wrist") if w in names]
    for o in held_roots:
        mw = o.matrix_world.copy()
        best = min(wrists, key=lambda w: (arm.matrix_world @ arm.pose.bones[w].head - mw.translation).length)
        o.animation_data_clear()
        o.parent = None
        # the held thing's own origin is where the draft's hand held it: put that point in the new hand
        mw.translation = arm.matrix_world @ arm.pose.bones[best].head
        o.matrix_world = mw
        c = o.constraints.new("CHILD_OF")
        c.target, c.subtarget = arm, best
        c.inverse_matrix = (arm.matrix_world @ arm.pose.bones[best].matrix).inverted()
    parts = [o.name for o in hidden]
    held = held_roots
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.out.resolve()))
    out = {"floor_z": floor_z, "placement": rep, "hidden": parts, "held": [o.name for o in held],
           "warnings": report.warnings}
    if a.report:
        a.report.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("floor_z", "hidden", "held")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
