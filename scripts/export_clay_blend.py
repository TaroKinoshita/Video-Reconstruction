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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("blend", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--move", default="source")
    ap.add_argument("--size", default="1024x896")
    ap.add_argument("--frames", type=int, default=124)
    ap.add_argument("--start", type=float, default=0.0)
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
