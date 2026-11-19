"""Clay views of a .blend as it is, from the source camera and from camera moves (PLAN v2 P1/P3 comparisons).

    python scripts/clay_moves_blend.py OUT.png VIDEO A.blend [B.blend ...] [--moves source,orbit45,dolly-60]
        [--frames 0,0.5,1] [--labels a,b]

Each .blend is opened as it is (no import or rebuild); a camera is added that follows the .blend's own source camera
plus the move (scripts/camera_moves.apply_move, progress t over the clip). The pivot is the middle of the moving
objects at the first frame, on the floor (no character needed); units are the .blend's (an LLM works in metres).
Clay: Workbench, studio light, one grey, cavity on. The sheet: the source video on top, then per .blend one row per
move, columns = the requested frames (shares of the clip).
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


def clay_rows(blend: Path, moves: list[str], shares: list[float], size: tuple[int, int]) -> list[np.ndarray]:
    import bpy
    from mathutils import Matrix, Quaternion, Vector

    from camera_moves import apply_move
    from vrecon.metrics.poses import camera_pose, object_pose
    from vrecon.scene.convert import intrinsics_to_blender, matrix_to_quat, opencv_camera_rotation_to_blender
    from vrecon.scene.novel_blend import open_scene

    rec = open_scene(blend)
    cam = rec.reconstruction.cameras[0]
    n = rec.source.frame_count
    movers = [o for o in rec.reconstruction.objects if o.keys] or list(rec.reconstruction.objects)
    pts = np.array([object_pose(rec, o, rec.shots[0].id, 0)[1] for o in movers])
    pivot = np.median(pts, axis=0)
    pivot[2] = 0.0
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light, sh.color_type, sh.single_color = "STUDIO", "SINGLE", (0.8, 0.8, 0.8)
    sh.show_cavity, sh.show_shadows = True, True
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = cam.intrinsics.width, cam.intrinsics.height, 100
    sc.render.image_settings.file_format = "PNG"
    data = bpy.data.cameras.new("clay_cam")
    data.sensor_fit = "AUTO"
    lens = intrinsics_to_blender(cam.intrinsics)
    data.sensor_width, data.lens, data.shift_x, data.shift_y = lens.sensor_width_mm, lens.lens_mm, lens.shift_x, lens.shift_y
    data.clip_end = 1000
    obj = bpy.data.objects.new("clay_cam", data)
    sc.collection.objects.link(obj)
    sc.camera = obj
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for move in moves:
            row = []
            for s in shares:
                f = int(round(s * (n - 1)))
                r, c, _ = camera_pose(cam, f)
                r, c = apply_move(np.asarray(r, float), np.asarray(c, float), move, f / max(n - 1, 1), pivot, 1.0)
                q = opencv_camera_rotation_to_blender(matrix_to_quat(r.tolist()))
                sc.frame_set(sc.frame_start + f)
                obj.matrix_world = Matrix.Translation(Vector(c.tolist())) @ Quaternion(q).to_matrix().to_4x4()
                sc.render.filepath = str(Path(tmp) / "f.png")
                bpy.ops.render.render(write_still=True)
                im = cv2.imdecode(np.fromfile(str(Path(tmp) / "f.png"), np.uint8), cv2.IMREAD_COLOR)
                im = cv2.resize(im, size)
                cv2.putText(im, f"{blend.parent.parent.parent.name if blend.parent.name == 'output' else blend.stem} {move}",
                            (6, 18), 0, 0.5, (0, 0, 255), 1)
                row.append(im)
            rows.append(np.hstack(row))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("video", type=Path)
    ap.add_argument("blends", nargs="+", type=Path)
    ap.add_argument("--moves", default="source,orbit45,dolly-60")
    ap.add_argument("--frames", default="0,0.5,1")
    a = ap.parse_args()
    from vrecon.control.core import read_video_rgb

    shares = [float(x) for x in a.frames.split(",")]
    v = read_video_rgb(a.video)
    h0, w0 = v.shape[1:3]
    size = (360, int(round(360 * h0 / w0)))
    top = np.hstack([cv2.resize(v[int(round(s * (len(v) - 1)))][..., ::-1], size) for s in shares])
    rows = [top]
    for b in a.blends:
        rows += clay_rows(b, a.moves.split(","), shares, size)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", np.vstack(rows))[1].tofile(str(a.out))
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
