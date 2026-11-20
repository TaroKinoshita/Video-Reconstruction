"""Add a thrown / tossed ball to a .blend from its observed masks (PLAN v2 8, tennis toss): held while it is by a
hand, ballistic in the air, using the same tools the tennis props used (analysis/prop_fit.tracklets,
fit_sphere_track), now with the .blend's own camera and the observed body's hands.

    python scripts/fit_ball_blend.py IN.blend MASKS.npz OUT.blend [--labels a,b] [--prop "tennis ball"]
        [--name Ball_Observed] [--report OUT.json]

Generic: the ball's size from analysis/prop_table (by --prop), the contacts are the observed body's wrists moved the
table's reach along the forearm. Tracklets never by a hand are other balls alike and are left out. The ball is keyed
per frame where it was placed and scaled to zero elsewhere (not invented where nothing places it).
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
    ap.add_argument("out", type=Path)
    ap.add_argument("--labels", default="")
    ap.add_argument("--prop", default="tennis ball")
    ap.add_argument("--name", default="Ball_Observed")
    ap.add_argument("--one", action="store_true", help="one ball in play: join the tracklets into one path")
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    import bpy
    import bmesh  # after bpy (bmesh ships with it)
    import cv2

    from vrecon.analysis.prop_fit import fit_sphere_track, project_pts, tracklets
    from vrecon.analysis.prop_table import PROPS
    from vrecon.metrics.poses import camera_pose
    from vrecon.scene.novel_blend import open_scene

    entry = PROPS[a.prop]
    diameter, reach = float(entry["size"][0]), float(entry["reach"])
    rec = open_scene(a.blend)
    cam = rec.reconstruction.cameras[0]
    W, H = cam.intrinsics.width, cam.intrinsics.height
    fps = rec.source.fps.value
    sc = bpy.context.scene
    z = np.load(a.masks)
    idmap = {v: int(k) for k, v in json.loads(str(z["id_map"])).items()}
    want = [idmap[x] for x in a.labels.split(",") if x] or list(idmap.values())
    raw = np.isin(z["masks"], want)
    t = min(len(raw), rec.source.frame_count)
    mask = np.stack([cv2.resize(raw[f].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                     if raw[f].shape != (H, W) else raw[f] for f in range(t)])
    arm = next(o for o in bpy.data.objects if o.get("vrecon_rig_of"))
    hands = [h for h in ("right_wrist", "left_wrist") if h in arm.pose.bones]
    contacts = {h: np.zeros((t, 3)) for h in hands}
    cams = []
    for f in range(t):
        sc.frame_set(sc.frame_start + f)
        for h in hands:
            pb = arm.pose.bones[h]
            head = arm.matrix_world @ pb.head
            d = np.array(list(head - arm.matrix_world @ pb.parent.head))
            d /= np.linalg.norm(d) + 1e-9
            contacts[h][f] = np.array(list(head)) + d * reach
        r, c, k = camera_pose(cam, f)
        cams.append((np.asarray(r), np.asarray(c), np.asarray(k)))
    times = np.arange(t) / fps
    pos = np.full((t, 3), np.nan)
    trs = tracklets(mask)
    if a.one:
        # a blob that stays within its own width while no hand holds it (the hand only passing by) is a thing in the scene that
        # looks like a ball (a lamp, a post top), not the ball in play
        def hand_px(tr):
            return float(np.median([min(float(np.linalg.norm(project_pts(contacts[h][f], cams[f])[0] - tr[f][0]))
                                        for h in hands) for f in tr]))
        trs = [tr for tr in trs if hand_px(tr) < 35.0
               or float(np.ptp(np.array([tr[f][0] for f in tr]), axis=0).max()) > float(np.median([tr[f][1] for f in tr]))]
        # one ball in play (a serve, a throw): the tracklets are pieces of one path; per frame the blob nearest the
        # previous position (a tracklet that starts near a hand wins at the start)
        merged = {}
        for f in range(t):
            cand = [tr[f] for tr in trs if f in tr]
            if not cand:
                continue
            if merged:
                last = merged[max(merged)][0]
                merged[f] = min(cand, key=lambda cw: float(np.linalg.norm(cw[0] - last)))
            else:
                merged[f] = max(cand, key=lambda cw: cw[1])
        trs = [merged] if merged else []
    kept, segs_all = 0, []
    for tr in trs:
        p, held, segs = fit_sphere_track(tr, t, contacts, cams, times, fps, diameter, 1.0, auto_split=a.one)
        if not held and not a.one:  # never by a hand: another ball alike
            continue
        kept += 1
        fill = np.isnan(pos[:, 0]) & np.isfinite(p[:, 0])
        pos[fill] = p[fill]
        segs_all += segs
    # the ball object
    me = bpy.data.meshes.new(a.name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=diameter / 2)
    bm.to_mesh(me)
    bm.free()
    ball = bpy.data.objects.new(a.name, me)
    sc.collection.objects.link(ball)
    for f in range(t):
        ok = bool(np.all(np.isfinite(pos[f])))
        ball.location = pos[f].tolist() if ok else (0.0, 0.0, -100.0)
        ball.scale = (1.0, 1.0, 1.0) if ok else (0.0, 0.0, 0.0)
        ball.keyframe_insert("location", frame=sc.frame_start + f)
        ball.keyframe_insert("scale", frame=sc.frame_start + f)
    from bpy_extras import anim_utils

    bag = anim_utils.action_get_channelbag_for_slot(ball.animation_data.action, ball.animation_data.action_slot)
    for fc in bag.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT" if fc.data_path == "scale" else "LINEAR"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.out.resolve()))
    rep = {"tracklets_kept": kept, "frames_placed": int(np.isfinite(pos[:, 0]).sum()), "frames": t,
           "segments": [{k: v for k, v in s.items() if k in ("frames", "model", "contacts", "rms_px", "kept")} for s in segs_all]}
    if a.report:
        a.report.write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("tracklets_kept", "frames_placed", "frames")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
