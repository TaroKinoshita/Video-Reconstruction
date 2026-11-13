"""Finish a completed scene and export V2V control videos for camera moves, H3-conformed.

    python scripts/export_scene_controls.py PROJECT.json ROOT OUT_DIR [--moves source,orbit30] [--size 800x704]
        [--frames 243] [--kinds depth,canny,clay,id,look] [--drop obj_background_rest] [--rise-m 0.45]

PROJECT: a completed scene (scripts/complete_scene.py); ROOT: the folder its mesh files are in.
Finish: the stands plane becomes steps (analysis/proxies.stepped_mesh, one row per --rise-m metres), written to
ROOT/stands_steps.npz. OUT_DIR/project.json + scene.blend hold the finished scene (source camera, source timing).

Controls, per camera move, in OUT_DIR/controls_<move>/ (scene/controls.export_controls): the scene re-timed to
24 fps and --frames frames (17n+5; object keys are in seconds, so they keep their timing), the picture --size
(multiples of 32; the source intrinsics scaled to the new height and centred). Moves:
    any scripts/camera_moves.py move: source, orbitN, dollyN, craneN, combined with "+"
--drop leaves objects out of the controls (the background reference mesh: it tears off the source view).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

H3_FPS = 24


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--moves", default="source,orbit30,orbit90,dolly50,crane6")
    ap.add_argument("--size", default="800x704")
    ap.add_argument("--frames", type=int, default=243)
    ap.add_argument("--kinds", default="depth,canny,clay,id,look")
    ap.add_argument("--drop", default="obj_background_rest")
    ap.add_argument("--rise-m", type=float, default=0.45)
    a = ap.parse_args()

    from vrecon.analysis.proxies import stepped_mesh
    from vrecon.control.core import check_h3_constraints
    from vrecon.metrics.poses import _rot, camera_pose
    from vrecon.scene.builder import build_scene, save_blend
    from vrecon.scene.controls import export_controls
    from vrecon.scene.convert import matrix_to_quat
    from vrecon.schema import CameraKey, Evidence, FileRef, Intrinsics, Pose, Project, load_project, save_project

    root, out = Path(a.root).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    p = load_project(a.project)
    place = json.loads((root.parent / "scene" / "placement.json").read_text(encoding="utf-8"))
    s = float(place["scale_units_per_metre"])

    rep = {}
    st = next((o for o in p.reconstruction.objects if o.id == "obj_stands" and o.asset is None), None)
    if st is not None:
        v, f, rows = stepped_mesh(np.asarray(_rot(st.rest.pose.rotation)), st.size, a.rise_m * s)
        np.savez_compressed(root / "stands_steps.npz", verts=v, faces=f)
        st.asset = FileRef(id="file_stands_steps", path="stands_steps.npz", media_type="application/x-vrecon-mesh+npz")
        st.evidence = list(st.evidence) + [Evidence(note=f"rule: stands drawn as {rows} steps of about {a.rise_m} m "
                                                         "around the fitted plane")]
        rep["stands_rows"] = rows
    save_project(p, out / "project.json")
    build_scene(p, root=root)
    save_blend(out / "scene.blend")

    w, h = (int(x) for x in a.size.split("x"))
    src = p.reconstruction.cameras[0]
    fps_src = p.source.fps.num / p.source.fps.den
    ch = next(o for o in p.reconstruction.objects if o.kind == "character")
    pivot = np.array((ch.keys[0] if ch.keys else ch.rest).pose.translation, float)
    pivot[2] = 0.0
    si = src.intrinsics
    k = h / si.height
    intr = Intrinsics(width=w, height=h, fx=si.fx * k, fy=si.fy * k, cx=(w - 1) / 2 + (si.cx - (si.width - 1) / 2) * k,
                      cy=(h - 1) / 2 + (si.cy - (si.height - 1) / 2) * k)
    n = a.frames
    drop = set(filter(None, a.drop.split(",")))
    from camera_moves import move_camera, pivot_of

    for move in a.moves.split(","):
        cam_m = move_camera(src, move, pivot_of(p), s, n, H3_FPS, fps_src, intrinsics=intr)
        d = p.model_dump(mode="json")  # changed together, then validated once (fields are checked on assignment)
        d["shots"] = [{**d["shots"][0], "start": 0, "end": n}]
        d["source"].update({"fps": {"num": H3_FPS, "den": 1}, "frame_count": n, "width": w, "height": h})
        d["reconstruction"]["cameras"] = [cam_m.model_dump(mode="json")]
        d["reconstruction"]["objects"] = [o for o in d["reconstruction"]["objects"] if o["id"] not in drop]
        q = Project.model_validate(d)
        man = export_controls(q, out / f"controls_{move}", kinds=tuple(a.kinds.split(",")), root=root)
        rep[move] = {"h3_problems": check_h3_constraints(w, h, n, H3_FPS), "controls": sorted(man["controls"]),
                     "warnings": man["warnings"]}
        print(move, json.dumps(rep[move]), flush=True)
    (out / "export.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
