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


def colour_objects(roles: Path | None) -> None:
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
        if o.type != "MESH":
            continue
        name = top(o).name.split(".")[0]
        low = (o.name + " " + name).lower()
        if is_person(o):
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
    ap.add_argument("--roles", type=Path, help="scripts/swap_person_blend.py report: the draft's person parts and held "
                    "things by name, so a bare draft and its improved copy get the same colours")
    a = ap.parse_args()
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
    if a.colour:
        colour_objects(a.roles)
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
    imgs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(a.frames):
            f = min((a.start + i / 24.0) * fps, n_src - 1)
            r, c, _ = camera_pose(cam, f)
            r, c = apply_move(np.asarray(r, float), np.asarray(c, float), a.move, i / max(a.frames - 1, 1), pivot, 1.0)
            q = opencv_camera_rotation_to_blender(matrix_to_quat(r.tolist()))
            sc.frame_set(sc.frame_start + int(f), subframe=float(f - int(f)))
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
