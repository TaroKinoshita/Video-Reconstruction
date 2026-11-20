"""Let a draft's other people follow where they are observed (PLAN v2 8, bystanders): each person-category object
of the .blend that is not the observed body is matched to an observed person mask (the label it overlaps most in the
source view over a few frames); per frame, the mask's lowest middle point (the feet) is cast from the source camera
onto the floor, and the object's whole hierarchy is moved there (horizontally, keeping its height). Frames without
the mask keep the last position. A person whose mask hardly moves (< MOVE m over the clip) is left as it is.

    python scripts/follow_masks_blend.py IN.blend CATEGORIES.json MASKS.npz OUT.blend [--report OUT.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MOVE = 0.3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("blend", type=Path)
    ap.add_argument("categories", type=Path)
    ap.add_argument("masks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    import bpy
    import cv2
    from mathutils import Vector

    from vrecon.metrics.poses import camera_pose
    from vrecon.scene.novel_blend import open_scene, render_view

    cats = json.loads(a.categories.read_text(encoding="utf-8"))["categories"]
    rec = open_scene(a.blend)
    cam = rec.reconstruction.cameras[0]
    W, H = cam.intrinsics.width, cam.intrinsics.height
    sc = bpy.context.scene
    z = np.load(a.masks)
    idmap = {int(k): v for k, v in json.loads(str(z["id_map"])).items()}
    people_labels = [k for k, v in idmap.items() if v.startswith("obs")]
    m = z["masks"]
    t = min(len(m), rec.source.frame_count)

    def top(o):
        while o.parent is not None:
            o = o.parent
        return o

    roots = {}
    for o in bpy.data.objects:
        if o.type == "MESH" and not o.hide_render and cats.get(top(o).name.split(".")[0]) == "person" \
                and not top(o).get("vrecon_rig_of") and o.get("vrecon_id") != "obj_person_observed":
            roots.setdefault(top(o).name, []).append(o.get("vrecon_id"))
    frames = [int(x) for x in np.linspace(0, t - 1, 5)]
    votes = {r: {} for r in roots}
    body_ids = {o.get("vrecon_id") for o in bpy.data.objects if o.type == "MESH" and (
        o.get("vrecon_id") == "obj_person_observed" or top(o).get("vrecon_rig_of"))}
    body_votes = {}
    for f in frames:
        idx, imap = render_view(cam, [f])
        mm = m[f] if m[f].shape == (H, W) else cv2.resize(m[f].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        bpx = np.isin(idx[0], [k for k, v in imap.items() if v in body_ids])
        for lab in people_labels:
            body_votes[lab] = body_votes.get(lab, 0) + int((bpx & (mm == lab)).sum())
        for r, ids in roots.items():
            px = np.isin(idx[0], [k for k, v in imap.items() if v in ids])
            for lab in people_labels:
                ov = int((px & (mm == lab)).sum())
                if ov:
                    votes[r][lab] = votes[r].get(lab, 0) + ov
    report = {}
    # the observed body's own mask is not another person's
    used = {max(body_votes, key=body_votes.get)} if body_votes and max(body_votes.values()) > 0 else set()
    for r, v in sorted(votes.items(), key=lambda kv: -max(kv[1].values(), default=0)):
        lab = max((k for k in v if k not in used), key=lambda k: v[k], default=None)
        if lab is None:
            report[r] = "no observed person"
            continue
        used.add(lab)
        feet = np.full((t, 3), np.nan)
        for f in range(t):
            mk = m[f] == lab
            if mk.shape != (H, W):
                mk = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            if mk.sum() < 20:
                continue
            ys, xs = np.nonzero(mk)
            low = ys >= np.quantile(ys, 0.97)
            u, vv = float(np.median(xs[low])), float(ys.max())
            rr, c, k = camera_pose(cam, f)
            ray = np.asarray(rr) @ np.array([(u - k[0, 2]) / k[0, 0], (vv - k[1, 2]) / k[1, 1], 1.0])
            if ray[2] >= -1e-6:
                continue
            feet[f] = np.asarray(c) + ray * (-np.asarray(c)[2] / ray[2])
        ok = np.isfinite(feet[:, 0])
        if ok.sum() < 3:
            report[r] = f"{idmap[lab]}: too few frames"
            continue
        pts = feet[ok]
        if float(np.linalg.norm(pts.max(0) - pts.min(0))) < MOVE:
            report[r] = f"{idmap[lab]}: does not move"
            continue
        obj = bpy.data.objects[r]
        sc.frame_set(sc.frame_start)
        start = np.array(list(obj.matrix_world.translation))
        first = pts[0]
        last = first
        for f in range(t):
            if ok[f]:
                last = feet[f]
            p = start + np.array([last[0] - first[0], last[1] - first[1], 0.0])
            mw = obj.matrix_world.copy()
            mw.translation = Vector(p.tolist())
            obj.matrix_world = mw
            obj.keyframe_insert("location", frame=sc.frame_start + f)
        report[r] = f"{idmap[lab]}: follows ({int(ok.sum())} frames)"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.out.resolve()))
    if a.report:
        a.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
