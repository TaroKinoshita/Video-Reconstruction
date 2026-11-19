"""Clay reference video of a .blend as it is, along a camera move, in H3's format (24 fps, sides multiples of 32,
17n+5 frames). For the A/B V2V comparison (PLAN v2 7.2 part 5): the bare draft and the improved .blend get the same
camera, frames and size, so the only difference in the generated videos is the scene.

    python scripts/export_clay_blend.py IN.blend OUT.mp4 --move orbit45 --size 1024x896 --frames 124 [--start 0]

Output frame i shows scene time start + i/24 s (source frame = time x source fps). The camera follows the .blend's
own source camera plus the move (scripts/camera_moves.apply_move, progress i/(frames-1)); the pivot is the middle of
the moving objects at the first frame. Clay: Workbench, studio light, one grey, cavity, shadows.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


PERSON, HELD = (0.95, 0.55, 0.15), (0.15, 0.85, 0.95)
PALETTE = [(0.85, 0.25, 0.25), (0.3, 0.75, 0.3), (0.3, 0.45, 0.95), (0.9, 0.85, 0.25), (0.75, 0.35, 0.85),
           (0.95, 0.6, 0.75), (0.5, 0.85, 0.7), (0.65, 0.5, 0.3), (0.55, 0.6, 0.95), (0.85, 0.7, 0.5),
           (0.2, 0.55, 0.55), (0.6, 0.2, 0.45), (0.95, 0.45, 0.35), (0.45, 0.45, 0.2), (0.35, 0.25, 0.7),
           (0.7, 0.9, 0.35)]


SEMANTIC = {"person": PERSON, "held": HELD, "ground": (0.45, 0.6, 0.4), "structure": (0.55, 0.55, 0.75),
            "furniture": (0.75, 0.55, 0.35), "prop": (0.9, 0.85, 0.3), "vehicle": (0.85, 0.3, 0.3),
            "nature": (0.3, 0.7, 0.35), "sky": (0.8, 0.88, 0.95)}


def coarse_people(cats: dict) -> list:
    """Unobserved people (category person, not an observed body): hide their parts, add a rigid figure (a capsule
    body and a head, no limbs) that follows the parts' bounding box frame by frame. Returns [(figure, parts)]."""
    import bpy
    import bmesh

    def top(o):
        while o.parent is not None:
            o = o.parent
        return o

    groups: dict = {}
    for o in bpy.data.objects:
        if o.type == "MESH" and not o.hide_render and cats.get(top(o).name.split(".")[0]) == "person"                 and not top(o).get("vrecon_rig_of") and o.get("vrecon_id") != "obj_person_observed":
            groups.setdefault(top(o).name, []).append(o)
    out = []
    for name, parts in groups.items():
        me = bpy.data.meshes.new(f"coarse_{name}")
        bm = bmesh.new()
        from mathutils import Matrix

        def cyl(r, d, x, y, z):  # a limb or torso: radius, length, centre (unit height 1.0 = top of the head)
            bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=r, radius2=r, depth=d,
                                  matrix=Matrix.Translation((x, y, z)))

        cyl(0.055, 0.47, -0.07, 0, 0.235)  # legs
        cyl(0.055, 0.47, 0.07, 0, 0.235)
        cyl(0.12, 0.33, 0, 0, 0.63)        # torso
        cyl(0.04, 0.36, -0.17, 0, 0.6)     # arms hanging
        cyl(0.04, 0.36, 0.17, 0, 0.6)
        bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=0.08, matrix=Matrix.Translation((0, 0, 0.9)))
        bm.to_mesh(me)
        bm.free()
        fig = bpy.data.objects.new(f"coarse_{name}", me)
        bpy.context.scene.collection.objects.link(fig)
        fig.color = (*PERSON, 1.0)
        fig["coarse_root"] = name
        for p in parts:
            p.hide_render = True
        out.append((fig, parts))
    return out


def place_figure(fig, parts) -> None:
    """A rigid standing figure (no articulation): feet on the parts' lowest point at their horizontal middle, as tall
    as they are, facing the way the person's top parent faces (its yaw)."""
    import math

    import bpy
    from mathutils import Vector

    pts = [p.matrix_world @ Vector(c) for p in parts for c in p.bound_box]
    lo = Vector((min(v.x for v in pts), min(v.y for v in pts), min(v.z for v in pts)))
    hi = Vector((max(v.x for v in pts), max(v.y for v in pts), max(v.z for v in pts)))
    h = max(hi.z - lo.z, 0.05)
    root = bpy.data.objects.get(fig["coarse_root"])
    yaw = root.matrix_world.to_euler().z if root is not None else 0.0
    fig.location = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z)
    fig.rotation_euler = (0.0, 0.0, yaw if math.isfinite(yaw) else 0.0)
    fig.scale = (h, h, h)


def colour_objects(roles: Path | None, cats: dict | None = None) -> None:
    """One colour per object, the same for the same thing in a bare draft and its improved copy: the person
    (the draft's person parts, or the observed body) one colour, held things another, every other object the colour
    of its top-level parent's name (palette in name order), so its parts read as one thing; copies keep their source's name
    prefix. Floor and walls are light greys (named "floor"/"wall"/"ground" anywhere in the name)."""
    import json

    import bpy

    r = json.loads(roles.read_text(encoding="utf-8")) if roles else {}
    person, held_roots = set(r.get("hidden", [])), set(r.get("held", []))

    def top(o):
        while o.parent is not None:
            o = o.parent
        return o

    def under_held(o):
        while o is not None:
            if o.name in held_roots:
                return True
            o = o.parent
        return False

    def is_person(o):
        return o.name in person or o.get("vrecon_id") == "obj_person_observed" or bool(top(o).get("vrecon_rig_of"))

    def is_person_root(o):  # a top parent whose meshes are all person parts (the draft's mannequin rig)
        ms = [m for m in bpy.data.objects if m.type == "MESH" and top(m) is top(o)]
        return bool(person) and all(m.name in person or under_held(m) for m in ms)

    # palette slots only for ordinary objects, so the person and held things (whose top parents differ between a
    # bare draft and its improved copy) do not shift the other colours
    names = sorted({top(o).name.split(".")[0] for o in bpy.data.objects if o.type == "MESH"
                    and not is_person(o) and not under_held(o) and not is_person_root(o)})
    slot = {n: i for i, n in enumerate(names)}  # in name order: copies (Name.001) share their source's colour
    for o in bpy.data.objects:
        if o.type != "MESH" or o.name.startswith("coarse_"):
            continue
        name = top(o).name.split(".")[0]
        low = (o.name + " " + name).lower()
        if cats is not None and not is_person(o) and not under_held(o):
            c = SEMANTIC.get(cats.get(name, "prop"), SEMANTIC["prop"])
        elif is_person(o):
            c = PERSON
        elif under_held(o):
            c = HELD
        elif any(w in low for w in ("floor", "ground", "court")):
            c = (0.62, 0.62, 0.6)
        elif "wall" in low:
            c = (0.8, 0.8, 0.82)
        else:
            c = PALETTE[slot.get(name, 0) % len(PALETTE)]
        o.color = (*c, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("blend", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--move", default="source")
    ap.add_argument("--size", default="1024x896")
    ap.add_argument("--frames", type=int, default=124)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--colour", action="store_true", help="shaded clay with one colour per object (idclay)")
    ap.add_argument("--semantic", type=Path, help="scripts/categorize_blend.py output: colour by category (fixed "
                    "colours per category, as semantic proxies) instead of per object")
    ap.add_argument("--coarse", action="store_true", help="people with no observation (category person, no observed "
                    "body) become a rigid root-driven figure (no limbs): the generator fills in the motion")
    ap.add_argument("--camera-path", type=Path, help="JSON camera path (per output frame: centre, rotation, and the "
                    "intrinsics): read when it exists, else computed from this .blend and written, so both conditions "
                    "of an A/B are rendered from exactly the same cameras (the improved copy keeps the draft's world)")
    ap.add_argument("--roles", type=Path, help="scripts/swap_person_blend.py report: the draft's person parts and held "
                    "things by name, so a bare draft and its improved copy get the same colours")
    a = ap.parse_args()
    import json

    w, h = (int(x) for x in a.size.split("x"))
    if w % 32 or h % 32 or (a.frames - 5) % 17:
        raise SystemExit("H3 needs sides that are multiples of 32 and 17n+5 frames")
    import bpy
    from mathutils import Matrix, Quaternion, Vector

    from camera_moves import apply_move
    from vrecon.control.core import encode_mp4
    from vrecon.metrics.poses import camera_pose, object_pose
    from vrecon.scene.convert import intrinsics_to_blender, matrix_to_quat, opencv_camera_rotation_to_blender
    from vrecon.scene.novel_blend import open_scene

    rec = open_scene(a.blend)
    cam = rec.reconstruction.cameras[0]
    n_src, fps = rec.source.frame_count, rec.source.fps.value
    shot = rec.shots[0].id
    movers = [o for o in rec.reconstruction.objects if o.keys] or list(rec.reconstruction.objects)
    pivot = np.median(np.array([object_pose(rec, o, shot, 0)[1] for o in movers]), axis=0)
    pivot[2] = 0.0
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light, sh.color_type, sh.single_color = "STUDIO", "SINGLE", (0.8, 0.8, 0.8)
    sh.show_cavity, sh.show_shadows = True, True
    # light fixed in the world (not turning with the camera), so a camera move does not move the shading
    sh.use_world_space_lighting = True
    sc.display.light_direction = (0.35, -0.45, 0.82)
    cats = json.loads(a.semantic.read_text(encoding="utf-8"))["categories"] if a.semantic else None
    coarse = coarse_people(cats) if (a.coarse and cats) else []
    if a.colour or cats:
        colour_objects(a.roles, cats)
        sh.color_type = "OBJECT"
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
    sc.render.image_settings.file_format = "PNG"
    # the source intrinsics scaled to the new height, centred (as export_scene_controls does)
    intr = cam.intrinsics
    k = h / intr.height
    new = intr.model_copy(update={"width": w, "height": h, "fx": intr.fx * k, "fy": intr.fy * k,
                                  "cx": (w - 1) / 2, "cy": (h - 1) / 2})
    lens = intrinsics_to_blender(new)
    data = bpy.data.cameras.new("clay_cam")
    data.sensor_fit = "AUTO"
    data.sensor_width, data.lens, data.shift_x, data.shift_y = lens.sensor_width_mm, lens.lens_mm, lens.shift_x, lens.shift_y
    data.clip_end = 1000
    obj = bpy.data.objects.new("clay_cam", data)
    sc.collection.objects.link(obj)
    sc.camera = obj
    if a.camera_path and a.camera_path.is_file():
        path = json.loads(a.camera_path.read_text(encoding="utf-8"))
        if (path["move"], path["frames_n"], path["start"], path["size"]) != (a.move, a.frames, a.start, [w, h]):
            raise SystemExit(f"{a.camera_path} was made for another move/window/size")
        lens_d = path["lens"]
        data.sensor_width, data.lens, data.shift_x, data.shift_y = (lens_d[k_] for k_ in ("sensor", "lens", "sx", "sy"))
    else:
        path = {"move": a.move, "frames_n": a.frames, "start": a.start, "size": [w, h], "pivot": pivot.tolist(),
                "from": str(a.blend), "lens": {"sensor": data.sensor_width, "lens": data.lens, "sx": data.shift_x,
                                               "sy": data.shift_y}, "frames": []}
        for i in range(a.frames):
            f = min((a.start + i / 24.0) * fps, n_src - 1)
            r, c, _ = camera_pose(cam, f)
            r, c = apply_move(np.asarray(r, float), np.asarray(c, float), a.move, i / max(a.frames - 1, 1), pivot, 1.0)
            path["frames"].append({"src_frame": float(f), "centre": c.tolist(), "rotation": np.asarray(r).tolist()})
        if a.camera_path:
            a.camera_path.parent.mkdir(parents=True, exist_ok=True)
            a.camera_path.write_text(json.dumps(path, indent=1) + "\n", encoding="utf-8")
    imgs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, fr in enumerate(path["frames"]):
            f = fr["src_frame"]
            r, c = np.array(fr["rotation"]), np.array(fr["centre"])
            q = opencv_camera_rotation_to_blender(matrix_to_quat(r.tolist()))
            sc.frame_set(sc.frame_start + int(f), subframe=float(f - int(f)))
            for fig, parts in coarse:
                place_figure(fig, parts)
            obj.matrix_world = Matrix.Translation(Vector(c.tolist())) @ Quaternion(q).to_matrix().to_4x4()
            sc.render.filepath = str(Path(tmp) / "f.png")
            bpy.ops.render.render(write_still=True)
            imgs.append(cv2.imdecode(np.fromfile(str(Path(tmp) / "f.png"), np.uint8), cv2.IMREAD_COLOR)[..., ::-1])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    encode_mp4(np.stack(imgs), a.out, 24)
    print(a.out, len(imgs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
