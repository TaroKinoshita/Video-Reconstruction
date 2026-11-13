"""Phase A (docs/VENUE_PLAN.md): a synthetic tennis venue with the whole surroundings known, for scoring full-venue
completion. Still scene, metres, Z up, court centre at the origin, court length along Y.

    python scripts/make_venue_clips.py [--out evaluation/venue/ven_01]

Venue (evaluation only; the reconstruction and LLMs never read gt.json or gt_passes):
    floor (ground) 120 x 100 m, court 23.77 x 10.97 m (ITF doubles), net 12.8 m x 1.07 m at y = 0
    stands: north 15 rows, south 12 rows, west 20 rows, east 20 rows split by a 6 m entrance at the net line;
            rows 0.85 m deep, 0.45 m rise; a 1 m barrier wall in front of each stand
    umpire chair and two player benches west of the net; two players (boxes) near the baselines
The source camera stands behind the south baseline at eye height looking north (like a broadcast low camera),
so it sees the north stand, the net, the players and parts of the side stands only.

Writes:
    gt.json                 the full venue (evaluation only)
    meta.json               object categories, what the source sees (evaluation only)
    source.png              textured render of the source view (what a model may look at)
    observed.npz            what the source view gives a reconstruction: per-pixel 3D points with object labels
                            and small depth noise (points, labels, label_names); a stand-in for 4RC + SAM
    gt_passes/<camera>/     index, depth, id_map for each fixed evaluation camera (evaluation only)
Cameras are fixed in evaluation/venue/cameras.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

W, H = 800, 448
ROW_D, ROW_H = 0.85, 0.45
CATEGORIES = {"obj_floor": "floor", "obj_court": "court", "obj_net": "net", "obj_stand": "stands", "obj_barrier": "wall",
              "obj_umpire": "furniture", "obj_bench": "furniture", "obj_player": "person"}


def category(oid: str) -> str:
    return next((c for p, c in CATEGORIES.items() if oid.startswith(p)), "other")


def stepped_block(width: float, rows: int):
    """A solid stand block: one box per row, row i from z = 0 to (i + 1) * ROW_H, rows going +y (away from the
    court), across x. Local origin at the block's front-bottom centre. Returns (verts, faces)."""
    vs, fs = [], []
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for i in range(rows):
        x0, x1, y0, y1, z1 = -width / 2, width / 2, i * ROW_D, (i + 1) * ROW_D, (i + 1) * ROW_H
        b = len(vs)
        for z in (0.0, z1):
            vs += [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        for q in quads:
            fs += [(b + q[0], b + q[1], b + q[2]), (b + q[0], b + q[2], b + q[3])]
    return np.array(vs, float), np.array(fs, int)


def venue(out: Path):
    from vrecon.schema import FileRef, Pose, SceneObject, TransformKey

    def obj(oid, kind, size, t, yaw=0.0, asset=None):
        q = (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))
        return SceneObject(id=oid, kind=kind, size=size, status="human_confirmed",
                           rest=TransformKey(time=0.0, pose=Pose(translation=t, rotation=q)),
                           asset=FileRef(id="file_" + oid.removeprefix("obj_"), path=asset,
                                         media_type="application/x-vrecon-mesh+npz") if asset else None)

    objs = [obj("obj_floor", "set", (120.0, 100.0, 0.02), (0.0, 0.0, -0.01)),
            obj("obj_court", "set", (10.97, 23.77, 0.01), (0.0, 0.0, 0.005)),
            obj("obj_net", "rigid", (12.8, 0.05, 1.07), (0.0, 0.0, 0.535)),
            obj("obj_umpire", "rigid", (1.0, 1.0, 2.2), (-7.2, 0.0, 1.1)),
            obj("obj_bench_a", "rigid", (0.6, 2.0, 0.9), (-7.4, -3.0, 0.45)),
            obj("obj_bench_b", "rigid", (0.6, 2.0, 0.9), (-7.4, 3.0, 0.45)),
            obj("obj_player_a", "rigid", (0.5, 0.3, 1.8), (1.9, -12.6, 0.9)),
            obj("obj_player_b", "rigid", (0.5, 0.3, 1.8), (-1.5, 12.4, 0.9))]
    # stands: (id, rows, width, front edge centre, yaw: the direction rows go away from the court)
    half_x, half_y = 18.3 / 2 + 3.0, 36.6 / 2
    stands = [("obj_stand_north", 15, 2 * half_x + 4, (0.0, half_y, 0.0), 0.0),
              ("obj_stand_south", 12, 2 * half_x + 4, (0.0, -half_y, 0.0), np.pi),
              ("obj_stand_west", 20, 2 * half_y, (-half_x, 0.0, 0.0), np.pi / 2),
              ("obj_stand_east_s", 20, half_y - 3.0, (half_x, -(half_y + 3.0) / 2, 0.0), -np.pi / 2),
              ("obj_stand_east_n", 20, half_y - 3.0, (half_x, (half_y + 3.0) / 2, 0.0), -np.pi / 2)]
    for sid, rows, width, t, yaw in stands:
        v, f = stepped_block(width, rows)
        np.savez_compressed(out / f"{sid}.npz", verts=v, faces=f)
        objs.append(obj(sid, "set", (width, rows * ROW_D, rows * ROW_H), t, yaw, asset=f"{sid}.npz"))
        d = np.array([-np.sin(yaw), np.cos(yaw)])  # rows go along d
        bt = (float(t[0] - 0.3 * d[0]), float(t[1] - 0.3 * d[1]), 0.5)
        objs.append(obj(sid.replace("obj_stand", "obj_barrier"), "set", (width, 0.2, 1.0), bt, yaw))
    return objs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/venue/ven_01")
    a = ap.parse_args()
    import bpy

    from vrecon.scene.builder import build_scene, render_frames
    from vrecon.scene.convert import look_at_opencv
    from vrecon.scene.look import apply_look
    from vrecon.scene.passes import render_passes
    from vrecon.schema import (Camera, CameraKey, FileRef, Intrinsics, Pose, Project, Rational, Shot, ShotMapping,
                               SourceVideo, save_project)

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    objs = venue(out)
    cams = json.loads((out.parent / "cameras.json").read_text(encoding="utf-8"))["ven_01"]

    def camera(c):
        f = c["fx_of_width"] * W
        intr = Intrinsics(width=W, height=H, fx=f, fy=f, cx=(W - 1) / 2, cy=(H - 1) / 2)
        pos, look = tuple(c["position"]), tuple(c["look_at"])
        return Camera(id="cam_001", shot_id="shot_001", intrinsics=intr, status="human_confirmed",
                      keys=[CameraKey(shot_frame=0, pose=Pose(translation=pos, rotation=look_at_opencv(pos, look)))])

    def project(cam):
        return Project(id="proj_ven_01", source=SourceVideo(file=FileRef(id="file_video", path="source.png"),
                                                             fps=Rational(num=24, den=1), frame_count=1, width=W, height=H),
                       shots=[Shot(id="shot_001", start=0, end=1, transition_from_previous="first")],
                       reconstruction={"scale": {"source": "human", "note": "synthetic ground truth, metres"},
                                       "cameras": [cam], "objects": objs,
                                       "shot_mapping": [ShotMapping(shot_id="shot_001", scene_time_start=0.0)]})

    gt = project(camera(cams["source"]))
    save_project(gt, out / "gt.json")
    for name, c in cams.items():
        build_scene(project(camera(c)), root=out)
        render_passes(out / "gt_passes" / name, shot_frames=[0], shaded=False)
        (out / "gt_passes" / name / "camera.json").write_text(
            json.dumps({"camera": camera(c).model_dump(mode="json")}, indent=1) + "\n", encoding="utf-8")

    # what the source view gives: points per pixel (back-projected depth, 1% noise), labelled by object
    build_scene(gt, root=out)
    sp = out / "gt_passes" / "source"
    depth = np.load(sp / "depth.npz")["depth"][0].astype(float)
    index = np.load(sp / "index.npz")["index"][0]
    id_map = {int(k): v for k, v in json.loads((sp / "id_map.json").read_text(encoding="utf-8")).items()}
    from vrecon.metrics.poses import camera_pose

    r, c, k = camera_pose(gt.reconstruction.cameras[0], 0)
    vv, uu = np.nonzero(index > 0)
    z = depth[vv, uu] * (1 + np.random.default_rng(0).normal(0, 0.01, len(vv)))
    pc = np.stack([(uu - k[0, 2]) / k[0, 0] * z, (vv - k[1, 2]) / k[1, 1] * z, z], 1)
    pts = pc @ np.asarray(r).T + np.asarray(c)
    names = sorted(set(id_map.values()))
    lab = np.array([names.index(id_map[int(i)]) for i in index[vv, uu]])
    np.savez_compressed(out / "observed.npz", points=pts.astype(np.float32), labels=lab.astype(np.int16),
                        label_names=np.array(names), pixels=np.stack([uu, vv], 1).astype(np.int16))
    visible = {v: int((index == kk).sum()) for kk, v in id_map.items()}

    apply_look(7)
    bpy.context.scene.cycles.use_denoising = True
    with tempfile.TemporaryDirectory() as tmp:
        png = render_frames(tmp, [0], engine="CYCLES", samples=32)[0]
        (out / "source.png").write_bytes(png.read_bytes())
    meta = {"id": "ven_01", "categories": {o.id: category(o.id) for o in objs},
            "visible_in_source_px": {k: v for k, v in sorted(visible.items()) if v > 0},
            "hidden_from_source": sorted(o.id for o in objs if visible.get(o.id, 0) == 0),
            "note": "evaluation only: reconstruction and LLMs must not read meta.json, gt.json or gt_passes"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"visible": meta["visible_in_source_px"], "hidden": meta["hidden_from_source"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
