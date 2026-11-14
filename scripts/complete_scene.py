"""Plan step 4 on a reconstructed scene: rule-based completion for camera moves (run/complete.py), then coverage
from those moves' last frames to see what got filled.

    python scripts/complete_scene.py PROJECT.json OUT_DIR [--moves orbit30,orbit-30,orbit90,orbit180,dolly50,crane6]
        [--no-enclose] [--step 13]

Rules applied (no judgement, no ground truth):
    enclose   boundary pieces facing the scene's centre (the middle of the people and things in it) are copied
              around it at 90, 180, 270 deg: the sides the source camera never faced (run/complete.enclose)
    extend    every flat set piece without a mesh file that bounds the scene (no person/thing beyond its plane,
              seen from the source camera), copies included, grows in its plane to fill each move's view
              (extend_set_to_view over sampled frames), but never in front of what the source camera saw
              (protect_depth).
The moves (scripts/camera_moves.py) stand for the camera moves a user asks for.
Writes OUT_DIR/project.json (completed; changed and added objects are status "hypothesis" with a "rule:" note),
OUT_DIR/<before|after>/<move>/ coverage renders of each move's last frame, OUT_DIR/completion.json (operations;
per move the share of the picture with nothing in it (black) and showing unseen faces (red), before and after).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def shares(paths) -> dict:
    from vrecon.control.core import load_frames

    fr = load_frames(paths).astype(int)
    empty = (fr.sum(-1) == 0).mean()
    red = ((fr[..., 0] > 200) & (fr[..., 1] < 120) & (fr[..., 2] < 120)).mean()
    return {"empty": round(float(empty), 3), "unseen_red": round(float(red), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("out")
    ap.add_argument("--moves", default="orbit30,orbit-30,orbit90,orbit180,dolly50,crane6",
                    help="camera moves to build for (scripts/camera_moves.py)")
    ap.add_argument("--no-enclose", action="store_true", help="do not copy the boundary pieces around")
    ap.add_argument("--step", type=int, default=13)
    a = ap.parse_args()
    from camera_moves import move_camera

    from vrecon.metrics.poses import camera_pose
    from vrecon.run.complete import enclose, extend_set_to_view
    from vrecon.scene.coverage import render_coverage
    from vrecon.schema import load_project, save_project

    proj_path = Path(a.project).resolve()
    root = proj_path.parent
    p = load_project(proj_path)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cam = p.reconstruction.cameras[0]
    ch = next(o for o in p.reconstruction.objects if o.kind == "character")
    pivot = np.array((ch.keys[0] if ch.keys else ch.rest).pose.translation, float)
    pivot[2] = 0.0
    n = p.source.frame_count
    src = list(range(0, n, a.step))
    from vrecon.analysis.units import units_per_metre

    upm = units_per_metre(root)
    fps = p.source.fps.num / p.source.fps.den
    views = {mv: move_camera(cam, mv, pivot, upm, n, fps, fps, step=a.step) for mv in a.moves.split(",")}

    from vrecon.metrics.poses import _rot

    eye = np.array(camera_pose(cam, 0)[1])

    def boundary(o) -> bool:
        """Only pieces that bound the scene grow: no thing (person, net, chair) lies beyond the piece's plane (on the side away
        from the source camera). A side wall in the middle of the scene, grown sideways, cut through the people
        in front of it (tennis wall pieces, 2026-11-13). The floor always grows."""
        if o.id == "obj_floor":
            return True
        r = np.asarray(_rot(o.rest.pose.rotation))
        n = r[:, int(np.argmin(o.size))]
        c = np.array(o.rest.pose.translation)
        if (eye - c) @ n > 0:
            n = -n  # n points away from the camera
        for x in p.reconstruction.objects:
            if x.kind == "set":  # walls behind walls are fine; what must not be cut is things (people, net, chairs)
                continue
            t = np.array((x.keys[0] if x.keys else x.rest).pose.translation)
            if (t - c) @ n > 0.02 * np.linalg.norm(c - eye):
                return False
        return True

    grow = [o.id for o in p.reconstruction.objects if o.kind == "set" and o.asset is None and boundary(o)]
    skipped = [o.id for o in p.reconstruction.objects if o.kind == "set" and o.asset is None and o.id not in grow]
    # what the source camera saw, as depth: extensions may not stand in front of it (run/complete.protect_depth)
    import tempfile

    from vrecon.scene.builder import build_scene
    from vrecon.scene.passes import render_passes

    build_scene(p, root=root)
    with tempfile.TemporaryDirectory() as tmp:
        render_passes(tmp, shot_frames=src, shaded=False)
        dep = np.load(Path(tmp) / "depth.npz")["depth"]
    pdepth = {f: dep[i] for i, f in enumerate(src)}
    done, ops = p, [{"op": "boundary_check", "grow": grow, "not_boundary": skipped}]
    if not a.no_enclose:
        # the sides the source never faced: boundary pieces facing the scene's centre (the middle of the things in
        # it: people, net, chairs), copied around it
        things = [np.array((x.keys[0] if x.keys else x.rest).pose.translation) for x in p.reconstruction.objects
                  if x.kind != "set"]
        centre = np.mean(things, 0)
        centre[2] = 0.0
        far = []
        for oid in grow:
            if oid == "obj_floor":
                continue
            o = next(x for x in p.reconstruction.objects if x.id == oid)
            r = np.asarray(_rot(o.rest.pose.rotation))
            nn = r[:, int(np.argmin(o.size))]
            c = np.array(o.rest.pose.translation)
            to_c = (centre - c) / np.linalg.norm(centre - c)
            if abs(nn @ to_c) > np.cos(np.radians(45)):
                far.append(oid)
        contain = things + [np.array(camera_pose(v, f)[1]) for v in views.values() for f in (0, n - 1)]
        contain += [np.array(camera_pose(cam, 0)[1])]
        done, r = enclose(done, centre, far, contain=contain, margin=2.0 * upm)
        # a copy may not show in front of what the source camera saw: pushed outwards in steps, else dropped
        from vrecon.run.complete import box_face_points, covers_source

        kept, moved, dropped = [], {}, []
        for nid in r["new"]:
            o = next(x for x in done.reconstruction.objects if x.id == nid)
            t0 = np.array(o.rest.pose.translation)
            u = np.r_[(t0 - centre)[:2] / np.linalg.norm((t0 - centre)[:2]), 0.0]
            ok = False
            for k in range(9):
                t = t0 + u * k * 2.0 * upm
                o.rest = o.rest.model_copy(update={"pose": o.rest.pose.model_copy(update={"translation": tuple(map(float, t))})})
                if not covers_source(box_face_points(o, 48), cam, src, pdepth, empty_is_seen=True).any():
                    ok = True
                    break
            if ok:
                kept.append(nid)
                if k:
                    moved[nid] = round(k * 2.0, 1)
            else:
                dropped.append(nid)
        done.reconstruction.objects = [x for x in done.reconstruction.objects if x.id not in dropped]
        r.update({"new": kept, "pushed_out_m": moved, "dropped_covering_source": dropped})
        ops.append(r)
        grow = grow + r["new"]
    for name, v in views.items():
        for o in [o for o in done.reconstruction.objects if o.id in grow]:
            done, r = extend_set_to_view(done, o.id, v, frames=src, protect=cam, protect_frames=src, protect_depth=pdepth,
                                         empty_is_seen="_encl" in o.id)
            ops.append({"view": name, **r})
    # final check by rendering the source view: an assumed piece that now covers something built there (by more
    # than 0.1% of the pixels: edges) is grown again with empty source pixels counted as seen; if it still covers,
    # it is put back to how it was before completion, or dropped if it is a copy.
    # The analytic checks above miss overlaps between pieces grown in different moves.
    def ids_at(proj):
        build_scene(proj, root=root)
        with tempfile.TemporaryDirectory() as tmp:
            render_passes(tmp, shot_frames=chk, shaded=False)
            idx = np.load(Path(tmp) / "index.npz")["index"]
            mp = {int(k): v for k, v in json.loads((Path(tmp) / "id_map.json").read_text(encoding="utf-8")).items()}
        lut = np.array([mp.get(i, "") for i in range(int(idx.max()) + 1)], dtype=object)
        return lut[idx]

    chk = [src[0], src[len(src) // 2], src[-1]]
    before_ids = ids_at(p)
    repaired = []
    strict_tried = set()
    for _ in range(8):
        after_ids = ids_at(done)
        bad = (before_ids != "") & (after_ids != before_ids)
        names, counts = np.unique(after_ids[bad], return_counts=True)
        culprits = [nm for nm, c in zip(names, counts) if c > 0.001 * bad.size and nm != ""]  # 0.1% of the pixels
        culprits = [nm for nm in culprits if next(o for o in done.reconstruction.objects if o.id == nm).status == "hypothesis"]
        if not culprits:
            break
        orig = {o.id: o for o in p.reconstruction.objects}
        objs, regrow = [], []
        for o in done.reconstruction.objects:
            if o.id in culprits:
                px = int(counts[list(names).index(o.id)])
                if o.id in orig and o.id not in strict_tried:
                    strict_tried.add(o.id)
                    regrow.append(o.id)
                    repaired.append({"id": o.id, "pixels": px, "action": "regrown strictly"})
                    objs.append(orig[o.id])
                    continue
                repaired.append({"id": o.id, "pixels": px, "action": "reverted" if o.id in orig else "dropped"})
                if o.id in orig:
                    objs.append(orig[o.id])
                continue
            objs.append(o)
        done.reconstruction.objects = objs
        for oid in regrow:
            for v in views.values():
                done, _ = extend_set_to_view(done, oid, v, frames=src, protect=cam, protect_frames=src,
                                             protect_depth=pdepth, empty_is_seen=True)
    ops.append({"op": "source_view_repair", "repaired": repaired})
    save_project(done, out / "project.json")

    res = {"ops": ops, "views": {}}
    for name, v in views.items():
        row = {}
        for tag, proj in (("before", p), ("after", done)):
            paths, _ = render_coverage(proj, out / tag / name, src, view=v, view_frames=[n - 1], root=root)
            row[tag] = shares(paths)
        res["views"][name] = row
    (out / "completion.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res["views"], indent=1))
    print(json.dumps(ops[0]))
    print(json.dumps([o for o in ops if o.get("op") == "enclose"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
