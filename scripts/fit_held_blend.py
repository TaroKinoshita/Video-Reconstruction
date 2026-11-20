"""Fit a held stick-like thing (racket, bat, stick) in a .blend to its observed masks, frame by frame
(PLAN v2 8 / racket): the grip stays at the observed hand, the direction comes from the silhouette.

    python scripts/fit_held_blend.py IN.blend MASKS.npz LABEL OUT.blend [--object Racket] [--bone right_wrist]
        [--reach 0.08] [--report OUT.json]

Generic: analysis/props.held_poses (the tool the tennis props used) with
  grips   the hand bone's head moved `reach` metres along the bone, per source frame (the observed body's rig)
  size    the held object's own extents (longest = length, then width, thickness), from the .blend
  masks   the observed masks of LABEL (resized to the source camera)
then analysis/prop_fit.smooth_dirs over time. The holding hand may change (people switch hands): it switches only when the
current hand is far from the mask and the other is nearer for half a second in a row. The object (the held root, e.g. the draft's "Racket" with its parts) gets
one key per frame: its longest local axis along the fitted direction, its middle axis along the fitted side, the end
nearest its origin at the grip. Frames with no mask turn smoothly between fitted ones (as fit_stick does). Any Child Of
constraint is removed (the keys replace it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("blend", type=Path)
    ap.add_argument("masks", type=Path)
    ap.add_argument("label")
    ap.add_argument("out", type=Path)
    ap.add_argument("--object", default=None, help="held object name (default: the one with a Child Of to the rig)")
    ap.add_argument("--bone", default=None, help="hand bone (default: the Child Of's subtarget)")
    ap.add_argument("--reach", type=float, default=0.08)
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    import bpy
    import cv2
    from mathutils import Matrix, Vector

    from vrecon.analysis import props as P
    from vrecon.analysis.prop_fit import smooth_dirs
    from vrecon.metrics.poses import camera_pose
    from vrecon.scene.novel_blend import open_scene

    rec = open_scene(a.blend)
    cam = rec.reconstruction.cameras[0]
    n = rec.source.frame_count
    W, H = cam.intrinsics.width, cam.intrinsics.height
    sc = bpy.context.scene
    held = bpy.data.objects.get(a.object) if a.object else next(
        o for o in bpy.data.objects if any(c.type == "CHILD_OF" for c in o.constraints))
    con = next((c for c in held.constraints if c.type == "CHILD_OF"), None)
    arm = con.target if con else next(o for o in bpy.data.objects if o.get("vrecon_rig_of"))
    bone = a.bone or (con.subtarget if con else "right_wrist")

    # the held thing's geometry in its own frame: extents and which end is at its origin (the grip)
    def subtree(o):
        yield o
        for c in o.children:
            yield from subtree(c)

    inv = held.matrix_world.inverted()
    pts = np.array([list(inv @ (m.matrix_world @ Vector(c))) for m in subtree(held) if m.type == "MESH" for c in m.bound_box])
    lo, hi = pts.min(0), pts.max(0)
    ext = hi - lo
    axes = list(np.argsort(-ext))  # longest, middle, thinnest local axis
    L, Wd, T = (float(ext[i]) * float(held.matrix_world.to_scale()[i]) for i in axes)
    long_ax = axes[0]
    grip_end = lo[long_ax] if abs(lo[long_ax]) < abs(hi[long_ax]) else hi[long_ax]
    sign = 1.0 if grip_end == lo[long_ax] else -1.0  # local direction from the grip to the tip

    z = np.load(a.masks)
    idmap = {v: int(k) for k, v in json.loads(str(z["id_map"])).items()}
    raw = z["masks"] == idmap[a.label]
    masks = np.stack([cv2.resize(raw[f].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                      if raw[f].shape != (H, W) else raw[f] for f in range(min(n, len(raw)))])
    t = len(masks)
    names = [b.name for b in arm.pose.bones]
    hands = [x for x in ("right_wrist", "left_wrist") if x in names] if a.bone is None else [bone]
    per_hand = {h_: np.zeros((t, 3)) for h_ in hands}
    cams = []
    for f in range(t):
        sc.frame_set(sc.frame_start + f)
        for h_ in hands:
            pb = arm.pose.bones[h_]
            head = arm.matrix_world @ pb.head
            d = np.array(list(head - arm.matrix_world @ pb.parent.head)) if pb.parent is not None else np.array([0, 0, -1.0])
            d /= np.linalg.norm(d) + 1e-9
            per_hand[h_][f] = np.array(list(head)) + d * a.reach  # along the forearm, into the palm
        r, c, k = camera_pose(cam, f)
        cams.append((np.asarray(r), np.asarray(c), np.asarray(k)))
    # which hand holds it, per frame: the hand whose grip pixel is nearest the mask (a person may switch hands, as a
    # tennis player does before serving); a running median over 9 frames keeps it from flickering
    from vrecon.analysis.prop_fit import project_pts

    dist = np.full((t, len(hands)), np.nan)
    for f in range(t):
        if masks[f].sum() < 8:
            continue
        ys, xs = np.nonzero(masks[f])
        px = np.c_[xs, ys]
        for j, h_ in enumerate(hands):
            dist[f, j] = float(np.min(np.linalg.norm(px - project_pts(per_hand[h_][f], cams[f]), axis=1)))
    # start with the hand nearest the first observed frames; switch only when the current hand is far from the mask
    # (> FAR px) and the other hand is nearer for SWITCH frames in a row (a racket raised past the other hand is
    # not a change of hands)
    FAR, SWITCH = 40.0, int(round(0.5 * rec.source.fps.value))
    obs = np.isfinite(dist).all(1)
    first = np.nonzero(obs)[0][:10]
    cur = int(np.argmin(np.median(dist[first], axis=0))) if len(first) else 0
    choice, run = np.zeros(t, int), 0
    for f in range(t):
        if obs[f] and len(hands) > 1:
            other = 1 - cur
            run = run + 1 if (dist[f, cur] > FAR and dist[f, other] < dist[f, cur]) else 0
            if run >= SWITCH:
                cur, run = other, 0
        choice[f] = cur
    grips = np.stack([per_hand[hands[choice[f]]][f] for f in range(t)])
    bone = ",".join(sorted({hands[c_] for c_ in choice}))
    tips, sides, how, score = P.held_poses(masks, grips, cams, L, Wd, T, size=(W, H))
    dirs = smooth_dirs((tips - grips) / np.linalg.norm(tips - grips, axis=1, keepdims=True), 1.5)
    sides = smooth_dirs(sides, 1.5)
    seen = [f for f in range(t) if np.all(np.isfinite(dirs[f])) and np.all(np.isfinite(sides[f]))]
    if not seen:
        raise SystemExit("no frame of the mask could be fitted")
    for c in list(held.constraints):
        held.constraints.remove(c)
    held.animation_data_clear()
    held.parent = None
    held.rotation_mode = "QUATERNION"
    for f in range(t):
        before = max([x for x in seen if x <= f], default=None)
        after = min([x for x in seen if x >= f], default=None)
        if before is not None and after is not None and before != after:
            u = (f - before) / (after - before)
            d, sd = P._slerp_dir(dirs[before], dirs[after], u), P._slerp_dir(sides[before], sides[after], u)
        else:
            g = before if before is not None else after
            d, sd = dirs[g], sides[g]
        d = d / np.linalg.norm(d)
        sd = sd - (sd @ d) * d
        sd /= np.linalg.norm(sd) + 1e-9
        nrm = np.cross(d, sd)
        rot = np.zeros((3, 3))
        rot[:, axes[0]] = d * sign
        rot[:, axes[1]] = sd
        rot[:, axes[2]] = nrm if np.linalg.det(np.c_[rot[:, 0], rot[:, 1], rot[:, 2]] + np.eye(3) * 0) >= 0 else nrm
        if np.linalg.det(rot) < 0:
            rot[:, axes[2]] *= -1
        scale = held.matrix_world.to_scale()
        m = Matrix(rot.tolist()).to_4x4()
        grip_local = np.zeros(3)
        grip_local[long_ax] = grip_end
        origin = grips[f] - rot @ (grip_local * np.array(list(scale)))
        held.matrix_world = Matrix.Translation(Vector(origin.tolist())) @ m @ Matrix.Diagonal((*scale, 1.0))
        held.keyframe_insert("location", frame=sc.frame_start + f)
        held.keyframe_insert("rotation_quaternion", frame=sc.frame_start + f)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.out.resolve()))
    rep = {"object": held.name, "bone": bone, "size_LWT": [L, Wd, T], "frames": {k: int((how == k).sum()) for k in
           ("sharp", "swept", "blurred", "unseen")}, "iou_median": float(np.median(score[how != "unseen"]))
           if (how != "unseen").any() else None}
    if a.report:
        a.report.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
