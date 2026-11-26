"""Blender-side builder for the MAVEBench small test (run inside Blender 4.5, the evaluator's version, not imported by
vrecon):

    blender -b --factory-startup -P src/vrecon/scene/mave_blend.py -- PREP.npz OUT/scene.blend --size 1920 816 --fps 23.976

PREP.npz comes from scripts/v11_mave_prep.py. Follows MAVEBench METHOD_PROMPT rules as the evaluator reads them:
video frame f = Blender frame f + 1; animated scene.camera; a "People" collection with one sub-collection per person.
Without --people-anim a person has one mesh per keyframe; each is shown (hide_render / hide_viewport keyed, constant)
from halfway after the previous keyframe of that person to halfway before the next (posed only at keyframes).
With --people-anim ANIM.npz (scripts/v11_people_anim.py) each person is one mesh whose shape keys hold the body on
each sampled frame, blended linearly between samples, coloured per vertex from the film.
With --parts PARTS.npz (scripts/v11_parts.py) the world is turned so the ground is level (z up, origin on it) and the
background is built as parts: Ground (height field, film colours), Trunk_*, Post_*, Beam_*, Canopy_* (unit meshes
scaled, colour per object), Backdrop (what is not split yet, film colours), and the "Imagined" collection (trees
outside every camera's view, custom property imagined = 1).
With --staffs STAFFS.npz (scripts/v11_staffs.py) each held staff is one cylinder in the "Props" collection, keyed on
the frames where it is held and hidden elsewhere.
"""
import json
import sys

import bpy
import numpy as np
from mathutils import Matrix

argv = sys.argv[sys.argv.index("--") + 1:]
prep, out = argv[0], argv[1]
size = [int(argv[argv.index("--size") + 1]), int(argv[argv.index("--size") + 2])]
fps = float(argv[argv.index("--fps") + 1])
anim = np.load(argv[argv.index("--people-anim") + 1]) if "--people-anim" in argv else None
pz = np.load(argv[argv.index("--parts") + 1]) if "--parts" in argv else None
parts = json.loads(str(pz["parts"])) if pz is not None else None
GR = np.array(parts["world_R"]) if parts else np.eye(3)
Gt = np.array(parts["world_t"]) if parts else np.zeros(3)


def gw(X: np.ndarray) -> np.ndarray:
    return (np.asarray(X, np.float64) @ GR.T + Gt).astype(np.float32)
d = np.load(prep)
info = json.loads(str(d["info"]))
W0, H0 = info["size"]
W, H = size
f_px = info["f_img"] * W / W0
N = info["frames"]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = W, H, 100
scene.render.fps = round(fps)
scene.render.fps_base = round(fps) / fps
scene.frame_start, scene.frame_end = 1, N


def mesh_object(name: str, V: np.ndarray, F: np.ndarray, collection, color):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.astype(np.int32).ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", (np.arange(len(F)) * 3).astype(np.int32))
    me.update()
    me.validate()
    mat = bpy.data.materials.new(name + "_mat")
    mat.diffuse_color = color
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob


# camera
cam_data = bpy.data.cameras.new("Camera")
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36.0
cam_data.lens = f_px / W * 36.0
cam_data.clip_start, cam_data.clip_end = 0.01, 10000
cam = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes -> Blender camera axes
for f in range(N):
    M = np.eye(4)
    M[:3, :3] = GR @ d["cam_R"][f] @ FLIP
    M[:3, 3] = GR @ d["cam_T"][f] + Gt
    cam.matrix_world = Matrix(M.tolist())
    cam.keyframe_insert("location", frame=f + 1)
    cam.keyframe_insert("rotation_euler", frame=f + 1)

# background
env = bpy.data.collections.new("Environment")
scene.collection.children.link(env)


def film_material(name: str, attr_name: str = "film"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    attr = nt.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = attr_name
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.9
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def object_colour_material(name: str, roughness: float = 0.9):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    info_node = nt.nodes.new("ShaderNodeObjectInfo")
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    nt.links.new(info_node.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def coloured_mesh(name, V, F, C, collection):
    ob = mesh_object(name, V, F, collection, (0.5, 0.5, 0.5, 1))
    col = ob.data.color_attributes.new("film", "FLOAT_COLOR", "POINT")
    col.data.foreach_set("color", np.concatenate([C, np.ones((len(C), 1), np.float32)], 1).ravel())
    ob.data.materials.clear()
    ob.data.materials.append(film_material(name + "_film"))
    return ob


if parts is None:
    mesh_object("Background", d["bg_v"], d["bg_f"], env, (0.5, 0.5, 0.45, 1))
else:
    import bmesh

    def unit_mesh(name, kind):
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        if kind == "cyl":  # radius 1, from z = 0 to z = 1
            bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=1, radius2=1, depth=1,
                                  matrix=Matrix.Translation((0, 0, 0.5)))
        else:
            bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1)
        bm.to_mesh(me)
        bm.free()
        for poly in me.polygons:
            poly.use_smooth = True
        return me

    cyl, blob = unit_mesh("unit_cylinder", "cyl"), unit_mesh("unit_blob", "ico")
    bark, leaves = object_colour_material("bark"), object_colour_material("leaves")
    cyl.materials.append(bark)
    blob.materials.append(leaves)
    ground = coloured_mesh("Ground", pz["ground_v"], pz["ground_f"], pz["ground_col"], env)
    if "ground_tex" in pz.files:  # film texture where seen, cell colours elsewhere
        tex, seen_t = pz["ground_tex"], pz["ground_tex_seen"]
        TH, TW = seen_t.shape
        img = bpy.data.images.new("ground_film", TW, TH, alpha=True)
        rgba = np.concatenate([tex.astype(np.float32) / 255, seen_t[..., None].astype(np.float32)], -1)
        img.pixels.foreach_set(rgba.ravel())
        img.pack()
        x0, y0, tw, th = pz["ground_tex_box"]
        me = ground.data
        uv = me.uv_layers.new(name="UVMap")
        lv = np.zeros(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", lv)
        V = pz["ground_v"]
        uv.data.foreach_set("uv", np.stack([(V[lv, 0] - x0) / tw, (V[lv, 1] - y0) / th], 1).astype(np.float32).ravel())
        nt = me.materials[0].node_tree
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = img
        t.interpolation = "Linear"
        t.extension = "CLIP"
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        vc = next(n for n in nt.nodes if n.type == "VERTEX_COLOR")
        bsdf = nt.nodes["Principled BSDF"]
        nt.links.new(t.outputs["Alpha"], mix.inputs["Factor"])
        nt.links.new(vc.outputs["Color"], mix.inputs[6])
        nt.links.new(t.outputs["Color"], mix.inputs[7])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    coloured_mesh("Backdrop", pz["backdrop_v"], pz["backdrop_f"], pz["backdrop_col"], env)

    def placed(name, me, collection, loc, scale, colour, rot=None, imagined=False):
        ob = bpy.data.objects.new(name, me)
        collection.objects.link(ob)
        ob.location = loc
        ob.scale = scale
        if rot is not None:
            ob.rotation_mode = "QUATERNION"
            ob.rotation_quaternion = rot
        ob.color = (*colour[:3], 1)
        ob["imagined"] = int(imagined)
        return ob

    from mathutils import Vector
    trees = bpy.data.collections.new("Trees")
    env.children.link(trees)
    for k, t in enumerate(parts["trunks"]):
        placed(f"Trunk_{k:03d}", cyl, trees, (t["x"], t["y"], t["base"] - 0.3), (t["r"], t["r"], t["h"] + 0.3), t["colour_lin"])
    set_pieces = bpy.data.collections.new("SetPieces")
    env.children.link(set_pieces)
    for k, t in enumerate(parts["posts"]):
        placed(f"Post_{k:03d}", cyl, set_pieces, (t["x"], t["y"], t["base"] - 0.2), (t["r"], t["r"], t["h"] + 0.2), t["colour_lin"])
    for k, b in enumerate(parts["beams"]):
        a_, b_ = Vector(b["a"]), Vector(b["b"])
        axis = b_ - a_
        placed(f"Beam_{k:03d}", cyl, set_pieces, a_, (b["r"], b["r"], axis.length), b["colour_lin"],
               rot=Vector((0, 0, 1)).rotation_difference(axis.normalized()))
    imagined = bpy.data.collections.new("Imagined")
    scene.collection.children.link(imagined)
    imagined["note"] = "not seen by any camera; placed from the density, sizes and colours of the seen trees"
    for k, t in enumerate(parts["imagined_trees"]):
        placed(f"ImTrunk_{k:03d}", cyl, imagined, (t["x"], t["y"], t["base"] - 0.3), (t["r"], t["r"], t["h"] + 0.3),
               t["colour_lin"], imagined=True)
    n_seen = len(parts["trunks"])
    for k, c in enumerate(parts["canopy"]):
        placed(f"Canopy_{k:03d}", blob, trees if k < n_seen else imagined, c["c"], c["size"], c["colour_lin"],
               imagined=k >= n_seen)
    # far forest wall at the edge of the imagined area: a ring in the seen foliage colour (imagined)
    ring_r, ring_h, seg = parts["radius"], 25.0, 96
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    gv_ = pz["ground_v"]
    cx, cy = gv_[:, 0].mean(), gv_[:, 1].mean()
    ring_v = np.concatenate([np.stack([cx + ring_r * np.cos(ang), cy + ring_r * np.sin(ang), np.full(seg, -2.0)], 1),
                             np.stack([cx + ring_r * np.cos(ang), cy + ring_r * np.sin(ang), np.full(seg, ring_h)], 1)])
    k = np.arange(seg)
    ring_f = np.concatenate([np.stack([k, (k + 1) % seg, seg + (k + 1) % seg], 1), np.stack([k, seg + (k + 1) % seg, seg + k], 1)])
    leaf = np.array(parts["canopy"][0]["colour_lin"]) if parts["canopy"] else np.array([0.05, 0.08, 0.03])
    rng_ = np.random.default_rng(0)
    ring_c = (leaf[None] * rng_.uniform(0.6, 1.0, (2 * seg, 1))).astype(np.float32)
    wall = coloured_mesh("ForestWall", ring_v.astype(np.float32), ring_f.astype(np.int32), ring_c, imagined)
    wall["imagined"] = 1
    world = bpy.data.worlds.new("Sky")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (*parts["sky_colour_lin"], 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    scene.world = world

# people
people = bpy.data.collections.new("People")
scene.collection.children.link(people)
colours = [(0.9, 0.3, 0.2, 1), (0.2, 0.8, 0.3, 1), (0.2, 0.4, 0.9, 1), (0.9, 0.8, 0.2, 1), (0.8, 0.3, 0.8, 1),
           (0.3, 0.8, 0.8, 1)]
by_person = {}


def colour_material(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    attr = nt.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = "film"
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.8
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def animated_person(label: str, collection):
    ts, Vs, F = anim[f"{label}_t"], anim[f"{label}_v"], anim[f"{label}_f"]
    Vs = np.stack([gw(v) for v in Vs]) if parts else Vs
    ob = mesh_object(label, Vs[0], F, collection, (0.7, 0.7, 0.7, 1))
    me = ob.data
    col = me.color_attributes.new("film", "FLOAT_COLOR", "POINT")
    rgba = np.concatenate([anim[f"{label}_col"], np.ones((len(Vs[0]), 1), np.float32)], 1)
    col.data.foreach_set("color", rgba.ravel())
    me.materials.clear()
    me.materials.append(colour_material(label + "_film"))
    for poly in me.polygons:
        poly.use_smooth = True
    ob.shape_key_add(name="Basis", from_mix=False)
    keys = []
    for j, t in enumerate(ts):
        k = ob.shape_key_add(name=f"f{int(t):04d}", from_mix=False)
        k.data.foreach_set("co", Vs[j].ravel())
        keys.append(k)
    for j, (t, k) in enumerate(zip(ts, keys)):
        frames = {int(t) + 1: 1.0}
        if j > 0:
            frames[int(ts[j - 1]) + 1] = 0.0
        else:
            frames[1] = 1.0
        if j < len(ts) - 1:
            frames[int(ts[j + 1]) + 1] = 0.0
        else:
            frames[N] = 1.0
        if j > 1:
            frames[1] = 0.0
        for fr, val in frames.items():
            k.value = val
            k.keyframe_insert("value", frame=fr)
    for fc in me.shape_keys.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    return ob


if anim is not None:
    for label in json.loads(str(anim["labels"])):
        sub = bpy.data.collections.new(label)
        people.children.link(sub)
        animated_person(label, sub)
for n, p in enumerate(info["people"] if anim is None else []):
    by_person.setdefault(p["label"], []).append((p["kf"], n))
for c, (label, items) in enumerate(sorted(by_person.items())):
    sub = bpy.data.collections.new(label)
    people.children.link(sub)
    items.sort()
    for j, (kf, n) in enumerate(items):
        start = 0 if j == 0 else (items[j - 1][0] + kf) // 2 + 1
        end = N - 1 if j == len(items) - 1 else (kf + items[j + 1][0]) // 2
        ob = mesh_object(f"{label}_kf{kf:04d}", gw(d[f"p{n}_v"]), d[f"p{n}_f"], sub, colours[c % len(colours)])
        for frame, hidden in ((1, True), (start + 1, False), (end + 2, True)):
            if frame > N:
                continue
            ob.hide_render = ob.hide_viewport = hidden
            ob.keyframe_insert("hide_render", frame=frame)
            ob.keyframe_insert("hide_viewport", frame=frame)

if "--staffs" in argv:  # held staffs (scripts/v11_staffs.py): keyed per frame, hidden where not held
    from mathutils import Vector
    sz = np.load(argv[argv.index("--staffs") + 1])
    props = bpy.data.collections.new("Props")
    scene.collection.children.link(props)
    wood = object_colour_material("wood")
    for lab in json.loads(str(sz["labels"])):
        me = bpy.data.meshes.new(f"Staff_{lab}")
        import bmesh as _bm
        bm = _bm.new()
        _bm.ops.create_cone(bm, cap_ends=True, segments=8, radius1=1, radius2=1, depth=1,
                            matrix=Matrix.Translation((0, 0, 0.5)))
        bm.to_mesh(me)
        bm.free()
        me.materials.append(wood)
        ob = bpy.data.objects.new(f"Staff_{lab}", me)
        props.objects.link(ob)
        ob.color = (0.22, 0.13, 0.06, 1)
        ob.rotation_mode = "QUATERNION"
        ts, A, B, r = sz[f"{lab}_t"], gw(sz[f"{lab}_a"]), gw(sz[f"{lab}_b"]), float(sz[f"{lab}_r"])
        held = set(int(t) for t in ts)
        for t, pa, pb in zip(ts, A, B):
            axis = Vector(pb) - Vector(pa)
            ob.location = pa
            ob.scale = (r, r, axis.length)
            ob.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(axis.normalized())
            for path in ("location", "scale", "rotation_quaternion"):
                ob.keyframe_insert(path, frame=int(t) + 1)
        for f in range(N):  # visibility switches only
            vis = f in held
            if f == 0 or vis != ((f - 1) in held):
                ob.hide_render = ob.hide_viewport = not vis
                ob.keyframe_insert("hide_render", frame=f + 1)
                ob.keyframe_insert("hide_viewport", frame=f + 1)

light = bpy.data.objects.new("Sun",bpy.data.lights.new("Sun", "SUN"))
light.data.energy = 3.0
light.rotation_euler = (0.6, 0.2, 0.8)
scene.collection.objects.link(light)
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=out)
print("saved", out, "people", {k: len(v) for k, v in by_person.items()}, "lens", cam_data.lens)
