"""The tool API an agent builds a scene with (docs/DESIGN_v3.md). The agent writes a Python script of these calls;
scripts/v3_run.py opens the locked evidence scene (cameras, people, floor: scene/multishot.build(evidence_only=True)),
runs the script, saves, renders comparison pictures and returns the checks. The script is the scene's editable
source: every run starts from the same evidence, so a run is reproducible and a human can edit the script.

World: metres, Z up, the floor is z = 0; the same world for every cut. Names must be unique.

    part(name, shape, center, size, rot_deg=(0, 0, 0), category="structure", color=None, group=None)
        shape: box | cylinder | plane | sphere | cone | wedge.  size = (x, y, z) extent before rotation
        (cylinder/cone: x = y = diameter, z = height; plane: z ignored).  category: structure | ground | prop | held
    prop(name, kind, center, size, rot_deg=(0, 0, 0), category="prop", color=None, group=None)
        a rough shape of a kind (gun, chair, railing, door, bottle, table, sign, ...; scene/props.py)
    animate(name, keys)      keys: [{"frame": f, "location": [x, y, z], "rot_deg": [rx, ry, rz]}, ...] timeline frames
    visible(name, start, end) only shown in timeline frames [start, end]
    remove(name)
    hide_person(name, reason)  a measured person the agent judges wrong (must give a reason)
    attach(name, joint, person=None, offset=(0, 0, 0), rot_deg=(0, 0, 0))
        a held thing goes with a person's joint in every cut (the person's rig, Rig_<id>; joint = an SMPL joint name
        such as right_wrist). offset / rot_deg are in the joint's frame: SMPL rest axes (x = the person's left, y = up,
        z = forward); in the rest pose the right arm points to -x, the left to +x. A gun (barrel +x, grip -z as made by
        prop) in the right hand pointing along the forearm: rot_deg=(-90, 180, 0); in the left hand: (-90, 0, 0).

Measured cameras, people and the floor cannot be changed: they are the evidence.
"""

from __future__ import annotations

import math

import bpy
import numpy as np

from .mannequin import _box, _cylinder, frustum
from .multishot import SEMANTIC, _label
from .props import prop_mesh

LOCKED_PREFIXES = ("Cam", "Shot", "Body", "Extra", "Floor", "Backdrop")
PART_PREFIX = "P_"
LOG: list[str] = []
HIDDEN_PEOPLE: dict[str, str] = {}


def _root():
    return bpy.data.objects.get("Shot00")


def _mesh(shape: str, size):
    sx, sy, sz = (max(float(v), 1e-3) for v in size)
    if shape == "box":
        return _box(np.zeros(3), (sx, sy, sz))
    if shape == "plane":
        return [(-sx / 2, -sy / 2, 0.0), (sx / 2, -sy / 2, 0.0), (sx / 2, sy / 2, 0.0), (-sx / 2, sy / 2, 0.0)], [(0, 1, 2, 3)]
    if shape == "cylinder":
        v, f = _cylinder(np.array([0, 0, -sz / 2]), np.array([0, 0, sz / 2]), 1.0, 24)
        return [(x * sx / 2, y * sy / 2, z) for x, y, z in v], f
    if shape == "cone":
        v, f = frustum(np.array([0, 0, -sz / 2]), np.array([0, 0, sz / 2]), 1.0, 0.02, 24)
        return [(x * sx / 2, y * sy / 2, z) for x, y, z in v], f
    if shape == "sphere":
        verts, faces = [], []
        n = 8
        for i in range(n):
            z0, z1 = -1 + 2 * i / n, -1 + 2 * (i + 1) / n
            r0, r1 = math.sqrt(max(1 - z0 * z0, 0)), math.sqrt(max(1 - z1 * z1, 0))
            v, f = frustum(np.array([0, 0, z0]), np.array([0, 0, z1]), max(r0, 1e-3), max(r1, 1e-3), 16)
            o = len(verts)
            verts += v
            faces += [tuple(x + o for x in ff) for ff in f]
        return [(x * sx / 2, y * sy / 2, z * sz / 2) for x, y, z in verts], faces
    if shape == "wedge":
        v = [(-sx / 2, -sy / 2, -sz / 2), (sx / 2, -sy / 2, -sz / 2), (sx / 2, sy / 2, -sz / 2), (-sx / 2, sy / 2, -sz / 2),
             (-sx / 2, sy / 2, sz / 2), (sx / 2, sy / 2, sz / 2)]
        return v, [(0, 1, 2, 3), (3, 2, 5, 4), (0, 4, 5, 1), (0, 3, 4), (1, 5, 2)]
    raise ValueError(f"unknown shape {shape!r}: box | cylinder | plane | sphere | cone | wedge")


def _place(name: str, verts, faces, center, rot_deg, category: str, color, group):
    if not name or name.startswith(LOCKED_PREFIXES):
        raise ValueError(f"name {name!r} is empty or reserved")
    full = PART_PREFIX + name
    if bpy.data.objects.get(full):
        raise ValueError(f"{name!r} exists; remove it first or use another name")
    if category not in SEMANTIC:
        raise ValueError(f"category {category!r}: one of {sorted(SEMANTIC)}")
    me = bpy.data.meshes.new(full)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(f) for f in faces])
    ob = bpy.data.objects.new(full, me)
    bpy.context.collection.objects.link(ob)
    ob.parent = _root()
    ob.location = tuple(float(c) for c in center)
    ob.rotation_euler = tuple(math.radians(float(a)) for a in rot_deg)
    _label(ob, category)
    if color is not None:
        ob["vrecon_color"] = [float(c) for c in color]  # appearance for shaded renders; the label colour stays
    if group:
        ob["vrecon_group"] = str(group)
    LOG.append(f"part {name} ({category})")
    return ob


def part(name, shape, center, size, rot_deg=(0, 0, 0), category="structure", color=None, group=None):
    v, f = _mesh(shape, size)
    ob = _place(name, v, f, center, rot_deg, category, color, group)
    if shape == "plane":
        ob.modifiers.new("thick", "SOLIDIFY").thickness = 0.05
    return name


def prop(name, kind, center, size, rot_deg=(0, 0, 0), category="prop", color=None, group=None):
    v, f = prop_mesh(kind, size)
    _place(name, v, f, center, rot_deg, category, color, group)
    return name


def _get(name):
    ob = bpy.data.objects.get(PART_PREFIX + name)
    if ob is None:
        raise ValueError(f"no part {name!r}")
    return ob


def animate(name, keys):
    ob = _get(name)
    for k in keys:
        if "location" in k:
            ob.location = tuple(float(c) for c in k["location"])
            ob.keyframe_insert("location", frame=int(k["frame"]))
        if "rot_deg" in k:
            ob.rotation_euler = tuple(math.radians(float(a)) for a in k["rot_deg"])
            ob.keyframe_insert("rotation_euler", frame=int(k["frame"]))


def visible(name, start, end):
    ob = _get(name)
    for f, hide in ((1, True), (int(start), False), (int(end) + 1, True)):
        if f < 1:
            continue
        ob.hide_render = ob.hide_viewport = hide
        ob.keyframe_insert("hide_render", frame=f)
        ob.keyframe_insert("hide_viewport", frame=f)


def remove(name):
    ob = _get(name)
    bpy.data.objects.remove(ob, do_unlink=True)


def hide_person(name, reason):
    ob = bpy.data.objects.get(name)
    if ob is None or not name.startswith(("Body", "Extra")):
        raise ValueError(f"no measured person {name!r} (Body.. / Extra..)")
    if not str(reason).strip():
        raise ValueError("hide_person needs a reason")
    ob.hide_render = ob.hide_viewport = True
    if ob.animation_data:
        ob.animation_data_clear()
        ob.hide_render = ob.hide_viewport = True
    HIDDEN_PEOPLE[name] = str(reason)


def attach(name, joint, person=None, offset=(0, 0, 0), rot_deg=(0, 0, 0)):
    ob = _get(name)
    rigs = [o for o in bpy.data.objects if o.type == "ARMATURE" and o.name.startswith("Rig_")]
    rig = bpy.data.objects.get(f"Rig_{person}") if person else (rigs[0] if len(rigs) == 1 else None)
    if rig is None:
        raise ValueError(f"no rig for person {person!r}; rigs: {[r.name for r in rigs]}")
    if joint not in rig.data.bones:
        raise ValueError(f"no joint {joint!r}; joints: {[b.name for b in rig.data.bones]}")
    ob.parent = rig
    ob.parent_type = "BONE"
    ob.parent_bone = joint
    bone = rig.data.bones[joint]
    # a bone child sits at the bone's tail; the offset is meant from the joint (the head), so step back along the bone
    ob.location = (float(offset[0]), float(offset[1]) - bone.length, float(offset[2]))
    ob.rotation_euler = tuple(math.radians(float(a)) for a in rot_deg)
    ob["vrecon_attached"] = f"{rig.name}:{joint}"
    LOG.append(f"attach {name} -> {rig.name}:{joint}")
    return name


CONTEXT: dict = {}  # set by scripts/v3_run.py: diag, tag, starts {cut: timeline start}, frames {cut: n}


def _rotations() -> list:
    from mathutils import Euler

    out = []
    for a in (0, 90, 180, 270):
        for b in (0, 90, 180, 270):
            for c in (0, 90, 180, 270):
                m = Euler((math.radians(a), math.radians(b), math.radians(c))).to_matrix()
                if not any(np.allclose(np.array(m), np.array(o[1]), atol=1e-4) for o in out):
                    out.append(((a, b, c), m))
    return out


def fit_held(name, cut, word="gun", step=2):
    """an attached held thing: choose its turn (24 right-angle turns) and how far along the joint it sits, so that its
    outline matches the observed masks of `word` (SAM 3.1, TAG_shotNN.objects.npz) over the cut. Returns a sentence
    with the overlap before and after; the change is kept only if the overlap grows."""
    import cv2
    from mathutils import Matrix

    ob = _get(name)
    if ob.parent_type != "BONE":
        raise ValueError(f"{name!r} is not attached to a joint (attach it first)")
    op = CONTEXT["diag"] / f"{CONTEXT['tag']}_shot{int(cut):02d}.objects.npz"
    if not op.exists():
        return f"fit_held {name}: no object masks for cut {cut}"
    oz = np.load(op, allow_pickle=True)
    import json as _json

    ids = [int(i) for i, w in _json.loads(str(oz["id_map"])).items() if word in w]
    masks = np.isin(oz["masks"], ids)
    h, w = masks.shape[1:]
    camob = bpy.data.objects[f"Cam{int(cut):02d}"]
    f = camob.data.lens / camob.data.sensor_width * w
    sc = bpy.context.scene
    start = CONTEXT["starts"][int(cut)]
    verts = np.array([v.co[:] for v in ob.data.vertices])
    frames = []
    for t in range(0, min(len(masks), CONTEXT["frames"][int(cut)]), step):
        if not masks[t].any():
            continue
        sc.frame_set(start + t)
        if ob.hide_render:
            continue
        base = ob.matrix_world @ ob.matrix_basis.inverted()
        frames.append((t, np.array(base), np.array(camob.matrix_world.inverted())))
    if not frames:
        return f"fit_held {name}: the cut shows no {word} while {name} is visible"

    def score(loc, rot):
        L = np.array(Matrix.Translation(loc) @ rot.to_4x4())
        ious = []
        for t, base, w2c in frames:
            P = (w2c @ base @ L)[:3]
            pc = verts @ P[:, :3].T + P[:, 3]
            if (pc[:, 2] > -0.02).any():
                ious.append(0.0)
                continue
            uv = np.stack([w / 2 + f * pc[:, 0] / -pc[:, 2], h / 2 - f * pc[:, 1] / -pc[:, 2]], 1)
            im = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(im, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
            m = masks[t]
            u = (im.astype(bool) | m).sum()
            ious.append((im.astype(bool) & m).sum() / u if u else 0.0)
        return float(np.mean(ious))

    loc0, rot0 = ob.location.copy(), ob.rotation_euler.to_matrix()
    before = score(loc0, rot0)
    best = (before, loc0, ob.rotation_euler.copy())
    from mathutils import Euler, Vector

    for d in (-0.06, -0.03, 0.0, 0.03, 0.06):
        for ang, m in _rotations():
            loc = Vector((loc0.x + (d if loc0.x >= 0 else -d), loc0.y, loc0.z))
            sco = score(loc, m)
            if sco > best[0]:
                best = (sco, loc, Euler([math.radians(a) for a in ang]))
    if best[0] > before:
        ob.location, ob.rotation_euler = best[1], best[2]
    msg = f"fit_held {name} cut {cut}: outline overlap {before:.2f} -> {max(before, best[0]):.2f} over {len(frames)} frames"
    LOG.append(msg)
    return msg


API = {"part": part, "attach": attach, "fit_held": fit_held, "prop": prop, "animate": animate, "visible": visible, "remove": remove,
       "hide_person": hide_person}


# ---- checks on the parts (the runner adds the picture checks of scripts/multishot_check.py) ----

def parts():
    return [o for o in bpy.data.objects if o.name.startswith(PART_PREFIX)]


def _world_bbox(ob):
    pts = np.array([ob.matrix_world @ v.co for v in ob.data.vertices])
    return pts.min(0), pts.max(0)


def part_checks(sample_every: int = 6) -> list[str]:
    """MAVE-style evidence checks on the parts, as sentences for the agent:
    people inside parts (penetration), parts floating with nothing under them, parts intersecting cameras."""
    sc = bpy.context.scene
    issues = []
    P = parts()
    boxes = {}
    for ob in P:
        if ob.animation_data is None and ob.get("vrecon_category") != "held":  # held things are in a hand, not on a support
            boxes[ob.name] = _world_bbox(ob)
    # floating: a static part whose bottom is above 0.15 m with no other part or the floor under its centre
    for name, (lo, hi) in boxes.items():
        if lo[2] > 0.15:
            c = (lo + hi) / 2
            under = any(n != name and lo2[0] <= c[0] <= hi2[0] and lo2[1] <= c[1] <= hi2[1] and hi2[2] <= lo[2] + 0.1
                        and hi2[2] >= lo[2] - 3.0 for n, (lo2, hi2) in boxes.items())
            if not under and (hi - lo).max() < 30:
                issues.append(f"floating: {name[2:]} starts {lo[2]:.2f} m above the floor with nothing under it")
    # people and cameras inside static parts, over sampled frames
    bodies = [o for o in bpy.data.objects if o.name.startswith("Body") and not o.hide_render]
    cams = [o for o in bpy.data.objects if o.type == "CAMERA"]
    hits = {}
    for f in range(sc.frame_start, sc.frame_end + 1, sample_every):
        sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        pts = []
        for b in bodies:
            if b.hide_render:
                continue
            ev = b.evaluated_get(dg)
            co = np.empty(len(ev.data.vertices) * 3)
            ev.data.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)[::25]
            M = np.array(ev.matrix_world)
            vs = co @ M[:3, :3].T + M[:3, 3]
            pts.append(("person " + b.name, vs))
        for c in cams:
            pts.append(("camera " + c.name, np.array([tuple(c.matrix_world.translation)])))
        for who, vs in pts:
            for name, (lo, hi) in boxes.items():
                inside = ((vs > lo + 0.03) & (vs < hi - 0.03)).all(1).mean()
                if inside > 0.05:
                    hits.setdefault((who, name), []).append(f)
    for (who, name), fr in hits.items():
        issues.append(f"inside: {who} is inside {name[2:]} (frames {fr[0]}..{fr[-1]})")
    return issues


def render_look(paths_by_frame: dict, size=(640, 266)) -> None:
    """flat category colours as scripts/multishot_check.py, but lit: what the agent compares with the film."""
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light, sh.color_type = "STUDIO", "OBJECT"
    sh.show_cavity = sh.show_shadows = True
    sc.render.resolution_x, sc.render.resolution_y = size
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    for ob in bpy.data.objects:
        if ob.type == "CAMERA":
            ob.data.clip_end = 1000
    for f, path in paths_by_frame.items():
        sc.frame_set(int(f))
        sc.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
