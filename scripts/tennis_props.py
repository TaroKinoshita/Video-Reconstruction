"""Racket and ball into the tennis scene from SAM masks (gpu/observe/run_sam3.py --movers "tennis racket"
--set "tennis ball=obj_ball"), the scene camera and the placed GVHMR player (analysis/props.py tools).

    python scripts/tennis_props.py PROJECT.json MASKS.npz OUT_DIR [--person obj_person_01] [--racket-id obs_0]

Racket: held at the wrist nearest the racket mask (per frame, in pixels), grip a little beyond the wrist along
the forearm; length 0.68 m, head width 0.27 m, thickness 0.03 m (a standard racket's size, a box stand-in).
Ball: 0.067 m (standard); fitted per segment (still / ballistic / straight); a segment whose fit is poor
(rms over --max-rms px) or that sits apart from its neighbours is reported and left out (SAM's "tennis ball"
label also lands on other balls lying on the court). Writes OUT_DIR/project.json (the scene plus obj_racket and
obj_ball, status estimated, per-frame evidence of how each pose was found), props.json and a check sheet
(masks vs the new objects' outlines through the source camera).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from vrecon.analysis import props as P
from vrecon.control.core import read_video_rgb
from vrecon.metrics.poses import camera_pose
from vrecon.scene.convert import matrix_to_quat, quat_to_matrix
from vrecon.schema import Evidence, Pose, SceneObject, TransformKey, load_project, save_project

RACKET = (0.68, 0.27, 0.03)
BALL = 0.067
G = 9.81
GRIP_BEYOND_WRIST = 0.08  # m along the forearm: the hand holds the handle's end


def _project(pts, cam):
    R, c, K = cam
    pc = (np.atleast_2d(pts) - c) @ R
    return pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def wrists(project, person_id, skel_path, camera):
    from vrecon.analysis.from_gvhmr import load_skeleton

    obj = next(o for o in project.reconstruction.objects if o.id == person_id)
    s = obj.keys[0].scale[0]
    d = load_skeleton(skel_path)
    names = list(d["joint_names"])
    out = {}
    from vrecon.analysis.rig_fk import posed_joints

    joints = posed_joints(obj, d)  # the hands where the scene draws them (not the GVHMR ray points)
    for side in ("left", "right"):
        w, e = names.index(f"{side}_wrist"), names.index(f"{side}_elbow")
        fore = joints[:, w] - joints[:, e]
        fore /= np.linalg.norm(fore, axis=1, keepdims=True) + 1e-9
        out[side] = joints[:, w] + fore * GRIP_BEYOND_WRIST * s
    return out, s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("masks")
    ap.add_argument("out")
    ap.add_argument("--person", default="obj_person_01")
    ap.add_argument("--racket-id", default=None, help="label of the racket track (default: the one that is not obj_ball)")
    ap.add_argument("--max-rms", type=float, default=3.0)
    ap.add_argument("--link", type=float, default=60.0, help="px a ball may move between frames in one tracklet")
    ap.add_argument("--ball-masks", default=None, help="masks for the ball from scripts/props_recut.py merge "
                    "(every label in it is a ball candidate); default: the obj_ball label of MASKS")
    ap.add_argument("--one-track", action="store_true", help="all ball blobs are one ball (anchored recut)")
    ap.add_argument("--one-ball", action="store_true", help="the kept tracklets are one ball (one in play)")
    ap.add_argument("--balls", action="store_true", help="also put the fitted ball tracklets in the scene "
                    "(off by default: on tennis they are other balls and blobs, not the served ball)")
    ap.add_argument("--hand-px", type=float, default=50.0, help="ball blob this close to a hand: in the hand")
    ap.add_argument("--with-hand-px", type=float, default=25.0, help="px/frame a held ball may move against the hand")
    ap.add_argument("--hit-m", type=float, default=0.35, help="ball this close to the racket head: the hit")
    ap.add_argument("--after-hit", default="2.0,5.8,0.03", help="where the hit ball goes (hypothesis; court metres, "
                    "default the service box across the net from a server left of the centre mark)")
    ap.add_argument("--stop-y", type=float, default=18.0, help="|y| (court metres) where the hit ball stops (the "
                    "back of the venue, hypothesis)")
    ap.add_argument("--hit-speed", type=float, default=45.0, help="m/s after the hit (a serve, hypothesis)")
    ap.add_argument("--min-track", type=int, default=5, help="frames a tracklet needs to count")
    ap.add_argument("--gap", type=int, default=3, help="frames a tracklet may go unseen")
    ap.add_argument("--video", default="runs/human/tennis/tennis.mp4", help="only for the check sheet")
    a = ap.parse_args()
    proj_path = Path(a.project)
    project = load_project(proj_path)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.masks, allow_pickle=True)
    lab = z["masks"]
    id_map = json.loads(str(z["id_map"]))
    ball_v = int(next(k for k, v in id_map.items() if v == "obj_ball"))
    rk_v = int(next(k for k, v in id_map.items() if (v == a.racket_id if a.racket_id else v != "obj_ball")))
    t_n = lab.shape[0]
    fps = project.source.fps.num / project.source.fps.den
    cam = project.reconstruction.cameras[0]
    cams = [camera_pose(cam, f) for f in range(t_n)]
    size = (lab.shape[2], lab.shape[1])
    person = next(o for o in project.reconstruction.objects if o.id == a.person)
    hands, s = wrists(project, a.person, proj_path.parent / person.asset.path, cam)
    rep = {"scale_units_per_metre": s, "fps": fps}

    # ---- racket: which hand (nearest the mask, by pixels, over the frames it is seen)
    rm = lab == rk_v
    near = {k: [] for k in hands}
    for f in range(t_n):
        if rm[f].sum() < 8:
            continue
        ys, xs = np.nonzero(rm[f])
        pts = np.c_[xs, ys]
        for k, h in hands.items():
            near[k].append(float(np.min(np.linalg.norm(pts - _project(h[f], cams[f]), axis=1))))
    hand = min(near, key=lambda k: np.median(near[k]) if near[k] else 1e9)
    rep["racket_hand"] = {"chosen": hand, "median_px_to_mask": {k: (round(float(np.median(v)), 1) if v else None)
                                                                 for k, v in near.items()}}
    grips = hands[hand]
    L, W, T = (x * s for x in RACKET)
    tips, sides, how, score = P.held_poses(rm, grips, cams, L, W, T, shutter=0.5, size=size)
    rep["racket_frames"] = {k: int((how == k).sum()) for k in ("sharp", "swept", "blurred", "unseen")}
    rep["racket_iou_median"] = round(float(np.median(score[how != "unseen"])), 3)
    # unseen frames: hold the nearest seen pose relative to the hand (marked)
    seen = [f for f in range(t_n) if np.all(np.isfinite(tips[f]))]
    rkeys, rhow = [], []
    for f in range(t_n):
        g = min(seen, key=lambda x: abs(x - f)) if seen else None
        if g is None:
            break
        d = tips[g] - grips[g]
        d /= np.linalg.norm(d)
        sd = sides[g] - (sides[g] @ d) * d
        sd /= np.linalg.norm(sd) + 1e-9
        rot = np.c_[d, sd, np.cross(d, sd)]
        centre = grips[f] + d * L / 2
        rkeys.append(TransformKey(time=f / fps, pose=Pose(translation=tuple(float(x) for x in centre),
                                                           rotation=tuple(matrix_to_quat(rot.tolist())))))
        rhow.append(how[f] if g == f else f"held from frame {g}")

    video_all = read_video_rgb(a.video)
    times = P.frame_times(video_all, fps)
    rep["repeated_frames"] = int(np.sum(np.diff(times) < 1e-9))
    # ---- ball: SAM's one "ball" label holds several balls (in the hand, lying on the court): split it into
    # blobs, link blobs frame to frame into tracklets (nearest within --link px), fit each tracklet on its own
    if a.ball_masks:
        zb = np.load(a.ball_masks, allow_pickle=True)["masks"]
        bm = np.zeros_like(lab, bool)
        bm[:len(zb)] = zb[:len(lab)] > 0
    else:
        bm = lab == ball_v
    tracks = []  # each: {frame: (centre, width, blob mask)}
    live = []
    for f in range(t_n):
        n, cc, stats, cents = cv2.connectedComponentsWithStats(bm[f].astype(np.uint8))
        blobs = [(cents[i], i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]
        used, nxt = set(), []
        for tr in live:
            last = tr[max(tr)][0]
            if f - max(tr) > a.gap:
                continue
            cand = [(np.linalg.norm(c - last), c, i) for c, i in blobs if i not in used]
            if cand:
                dist, c, i = min(cand, key=lambda x: x[0])
                if dist < a.link:
                    st = P.streak(cc == i)
                    tr[f] = (np.array(st["centre"]), st["width"], i)
                    used.add(i)
            nxt.append(tr)
        for c, i in blobs:
            if i not in used:
                st = P.streak(cc == i)
                tr = {f: (np.array(st["centre"]), st["width"], i)}
                tracks.append(tr)
                nxt.append(tr)
        live = nxt
    tracks = [t for t in tracks if len(t) >= a.min_track]
    if a.ball_masks and a.one_track:  # an anchored recut holds one ball: keep it as one track
        tracks = [{f: v for t in tracks for f, v in t.items()}] if tracks else []
    balls, info = [], []
    for k, tr in enumerate(sorted(tracks, key=lambda t: min(t))):
        cen = np.full((t_n, 2), np.nan)
        wid = np.full(t_n, np.nan)
        for f, (c, w, _) in tr.items():
            cen[f], wid[f] = c, w
        # state (docs/PROPS_PLAN.md): held where the blob stays by a hand (the ball on its ray at the hand's depth),
        # free elsewhere (fitted on the rays; the hand position at a release frame anchors the depth)
        held = {}
        fs = sorted(tr)
        dist = {}
        for f in fs:
            near_h = min(hands, key=lambda k_: np.linalg.norm(_project(hands[k_][f], cams[f])[0] - cen[f]))
            dist[f] = (near_h, float(np.linalg.norm(_project(hands[near_h][f], cams[f])[0] - cen[f])))
        # release: the first frame from which the ball stays away from every hand (a toss, a throw); before it the
        # ball is in the hand where it was by one (a frame or two of the blob reaching past the hand don't count)
        rel = next((i for i in range(len(fs)) if all(dist[g][1] > a.hand_px for g in fs[i:])), len(fs))
        for f in fs[:rel]:
            if dist[f][1] < 2 * a.hand_px:
                R, c, K = cams[f]
                hn = dist[f][0]
                ray = P.pixel_ray(cen[f], R, c, K)
                held[f] = (hn, c + ray * float((hands[hn][f] - c) @ ray))
        splits = [fs[rel - 1]] if 0 < rel < len(fs) else []  # the release frame starts the free part (anchor)
        free_cen = cen.copy()
        anchors, vpriors = {}, {}
        for f in held:
            if f + 1 in tr and f + 1 not in held:
                R, c, K = cams[f]
                anchors[f] = (float((held[f][1] - c) @ P.pixel_ray(cen[f], R, c, K)), 0.1 * s)
                hn = held[f][0]
                f0, f1 = max(f - 2, 0), min(f + 1, t_n - 1)  # the hand's velocity about the release
                hv = (hands[hn][f1] - hands[hn][f0]) / max(times[f1] - times[f0], 1e-3)
                vpriors[f] = (hv, 3.0 * s)  # loose: the hand slows as it lets go
            else:
                free_cen[f] = np.nan
        # free = under gravity (or resting): no straight-line model, which only a fast shot far from anything needs
        pos, bhow, segs = P.ball_fit(free_cen, cams, fps, widths=wid, diameter=BALL * s, anchors=anchors,
                                     models=("still", "ballistic"), splits=splits, times=times,
                                     vpriors=vpriors)
        for f, (hn, p_) in held.items():
            pos[f] = p_
        segs.append({"frames": [min(held), max(held)] if held else [0, -1], "model": "held",
                     "hands": sorted({hn for hn, _ in held.values()}), "held_frames": len(held), "rms_px": 0.0,
                     "anchors": 0, "v": [0, 0, 0]})
        for sg in segs:
            lo, hi = sg["frames"]
            sg["track"] = k
            sg["kept"] = bool(sg["rms_px"] <= a.max_rms)
            if not sg["kept"]:
                for f in range(lo, hi + 1):
                    if f not in held:
                        pos[f] = np.nan
            mid = (lo + hi) // 2
            sg["mid_xyz"] = [round(float(x), 2) for x in pos[mid]] if np.all(np.isfinite(pos[mid])) else None
            info.append(sg)
        # a prop's free path starts or ends at a contact: a tracklet never by a hand is another object alike
        # (a ball lying elsewhere, a yellow sign) and is left out
        if not held:
            for sg in segs:
                sg["kept"] = False
                sg["why_left_out"] = "never by a hand"
            continue
        if np.any(np.all(np.isfinite(pos), 1)):
            balls.append((k, pos, [sg for sg in segs if sg["kept"]]))
    rep["ball_tracklets"] = len(tracks)
    rep["ball_segments"] = info
    rep["balls_placed"] = [{"track": k, "frames": int(np.all(np.isfinite(p_), 1).sum())} for k, p_, _ in balls]

    # ---- scene
    objs = [o for o in project.reconstruction.objects if o.id != "obj_racket" and not o.id.startswith("obj_ball")]
    objs.append(SceneObject(
        id="obj_racket", name="racket", kind="rigid", size=(L, W, T), keys=rkeys, status="estimated",
        look={"all": (0.15, 0.15, 0.18)},
        evidence=[Evidence(note=f"SAM 'tennis racket' mask; held at the {hand} wrist (GVHMR) +{GRIP_BEYOND_WRIST} m; "
                                "pose fitted to the silhouette (analysis/props.held_poses); standard racket size")]
        + [Evidence(source_frames=[f], note=h) for f, h in enumerate(rhow) if h not in ("sharp",)][:400]))
    if balls and a.one_ball:
        merged = np.full((t_n, 3), np.nan)
        segs_all = []
        for k, pos_k, kept in balls:
            ok = np.all(np.isfinite(pos_k), 1) & ~np.all(np.isfinite(merged), 1)
            merged[ok] = pos_k[ok]
            segs_all += kept
        # unseen after a free flight (docs/PROPS_PLAN.md part 4): physics carries it on until a contact. Here the
        # contact is the racket head passing the path (the serve); after it, a hypothesis for where it goes
        # (--after-hit target and speed: the rules of the game, not observed), marked hypothesis
        flight_end = None
        for sg in [x for x in segs_all if x.get("model") == "ballistic" and "fit" in x]:
            prm = {**sg["fit"], "p0": np.array(sg["fit"]["p0"]), "v": np.array(sg["fit"]["v_exact"])}
            f = sg["frames"][1] + 1
            hit = None
            while f < t_n and not np.all(np.isfinite(merged[f])):
                pball = P.trajectory_at(prm, times[f])
                rot = np.array(quat_to_matrix(rkeys[f].pose.rotation))
                head = np.array(rkeys[f].pose.translation) + rot[:, 0] * L * 0.25
                if np.linalg.norm(pball - head) < a.hit_m * s:
                    hit = f
                    break
                if pball[2] < BALL * s / 2 or times[f] - prm["t0"] > 2.0:
                    break
                merged[f] = pball
                f += 1
            how_hit = "racket head within --hit-m of the path"
            if hit is None:
                # the toss's depth and the racket's pose in the swing (mask lost there) are both uncertain, so the
                # path may miss the racket: then take the swing's highest racket-head frame as the contact
                # (hypothesis) and bend the unseen part of the toss to meet it (a ballistic arc through the last
                # seen point and the contact point)
                lo_f, hi_f = sg["frames"][1] + 1, min(sg["frames"][1] + int(1.5 * fps), t_n - 1)
                heads = []
                for g_ in range(lo_f, hi_f + 1):
                    rot = np.array(quat_to_matrix(rkeys[g_].pose.rotation))
                    heads.append(np.array(rkeys[g_].pose.translation) + rot[:, 0] * L * 0.25)
                heads = np.array(heads)
                hit = lo_f + int(np.argmax(heads[:, 2]))
                p_last = merged[sg["frames"][1]]
                p_c = heads[hit - lo_f]
                dt = times[hit] - times[sg["frames"][1]]
                v0 = (p_c - p_last + np.array([0, 0, G * s * dt * dt / 2])) / max(dt, 1e-3)
                arc = {"t0": times[sg["frames"][1]], "p0": p_last, "v": v0, "model": "ballistic", "g": G * s}
                for g_ in range(lo_f, hit):
                    merged[g_] = P.trajectory_at(arc, times[g_])
                for g_ in range(hit, t_n):
                    merged[g_] = np.nan
                head = p_c
                f = hit
                how_hit = "no contact found; the swing's highest racket-head frame (hypothesis)"
            sg["extended_to"] = f - 1
            segs_all.append({"frames": [sg["frames"][1] + 1, f - 1], "model": "unseen toss: physics to the contact "
                             f"({how_hit})", "rms_px": 0.0})
            if hit is not None:
                p_hit = np.array(head)
                tgt = np.array([float(x) for x in a.after_hit.split(",")]) * s
                tof = np.linalg.norm(tgt - p_hit) / (a.hit_speed * s)
                v0 = (tgt - p_hit + np.array([0, 0, G * s * tof * tof / 2])) / tof
                pr = {"t0": times[hit], "p0": p_hit, "v": v0, "model": "ballistic", "g": G * s}
                g2 = hit
                rest = None
                while g2 < t_n:
                    if rest is not None:  # stopped at the back of the venue (hypothesis): lies there
                        merged[g2] = rest
                        g2 += 1
                        continue
                    q = P.trajectory_at(pr, times[g2])
                    if abs(q[1]) > a.stop_y * s:
                        rest = np.r_[q[0], np.sign(q[1]) * a.stop_y * s, BALL * s / 2]
                        q = rest
                    if q[2] < BALL * s / 2:  # bounce (restitution 0.7), then on
                        tb = times[g2]
                        vb = pr["v"] - np.array([0, 0, pr["g"] * (tb - pr["t0"])])
                        pr = {"t0": tb, "p0": np.r_[q[:2], BALL * s / 2], "v": vb * [0.9, 0.9, -0.7], "model": "ballistic", "g": G * s}
                        q = pr["p0"]
                    merged[g2] = q
                    g2 += 1
                flight_end = g2
                segs_all.append({"frames": [hit, t_n - 1], "model": f"after the hit: towards {a.after_hit} m at "
                                 f"{a.hit_speed} m/s (hypothesis: the rules of the game, not observed)", "rms_px": 0.0,
                                 "hit_frame": int(hit)})
            rep["ball_hit_frame"] = None if hit is None else int(hit)
        # unseen otherwise: in the hand that held it (a hypothesis, marked), so the ball neither hangs nor vanishes
        held_hand = "left"
        for f in range(t_n):
            if not np.all(np.isfinite(merged[f])):
                merged[f] = hands[held_hand][f]
        segs_all.append({"frames": [0, t_n - 1], "model": f"other unseen frames: in the {held_hand} hand (hypothesis)",
                         "rms_px": 0.0})
        balls = [(0, merged, segs_all)]
    for k, pos_k, kept in (balls if a.balls else []):
        bkeys = [TransformKey(time=f / fps, pose=Pose(translation=tuple(float(x) for x in pos_k[f])))
                 for f in range(t_n) if np.all(np.isfinite(pos_k[f]))]
        objs.append(SceneObject(
            id="obj_ball" if a.one_ball else f"obj_ball_{k:02d}", name="ball" if a.one_ball else f"ball {k}", kind="rigid", size=(BALL * s,) * 3, keys=bkeys,
            status="estimated", look={"all": (0.85, 0.95, 0.2)},
            evidence=[Evidence(note="SAM 'tennis ball' mask, one tracklet; path fitted per segment (still / ballistic / "
                                    "straight) on the camera rays, depth from the standard size; spin not observed; "
                                    "shown only in the frames it was seen")]
            + [Evidence(source_frames=list(range(sg["frames"][0], sg["frames"][1] + 1)),
                        note=f"{sg['model']}, rms {sg['rms_px']} px") for sg in kept]))
    project = project.model_copy(deep=True)
    project.reconstruction.objects = objs
    for f in [*proj_path.parent.glob("*.npz"), *proj_path.parent.glob("units.json")]:
        shutil.copyfile(f, out / f.name)
    save_project(project, out / "project.json")

    # ---- check sheet: masks (filled) and the new objects' outlines through the source camera
    video = read_video_rgb(a.video) if Path(a.video).exists() else None
    tiles = []
    for f in np.linspace(0, t_n - 1, 8).astype(int):
        im = video[f].copy() if video is not None else np.zeros((size[1], size[0], 3), np.uint8)
        im[rm[f]] = (im[rm[f]] * 0.5 + np.array([0, 128, 128])).astype(np.uint8)
        im[bm[f]] = (im[bm[f]] * 0.5 + np.array([128, 0, 0])).astype(np.uint8)
        if f < len(rkeys):
            k = rkeys[f]
            d = tips[min(seen, key=lambda x: abs(x - f))] - grips[min(seen, key=lambda x: abs(x - f))]
            sil = P.silhouette(P._box_corners(grips[f], grips[f] + d / np.linalg.norm(d) * L, sides[min(seen, key=lambda x: abs(x - f))], L, W, T), cams[f], size)
            cnt, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, cnt, -1, (255, 255, 0), 2)
        for k, pos_k, _ in balls:
            if np.all(np.isfinite(pos_k[f])):
                uv = _project(pos_k[f], cams[f])[0]
                cv2.circle(im, (int(uv[0]), int(uv[1])), 8, (255, 0, 0), 2)
        cv2.putText(im, f"{f} {rhow[f] if f < len(rhow) else ''}", (8, 24), 0, 0.7, (255, 255, 255), 2)
        tiles.append(cv2.resize(im, (406, 360)))
    sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])[..., ::-1]
    ok, buf = cv2.imencode(".png", sheet)
    buf.tofile(str(out / "props_check.png"))
    txt = json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    (out / "props.json").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
