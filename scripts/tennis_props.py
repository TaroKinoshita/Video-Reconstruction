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
from vrecon.analysis.place_person import _rays
from vrecon.control.core import read_video_rgb
from vrecon.metrics.poses import camera_pose
from vrecon.scene.convert import matrix_to_quat
from vrecon.schema import Evidence, Pose, SceneObject, TransformKey, load_project, save_project

RACKET = (0.68, 0.27, 0.03)
BALL = 0.067
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
    for side in ("left", "right"):
        w, e = names.index(f"{side}_wrist"), names.index(f"{side}_elbow")
        pts = []
        for r, c, q in _rays(d, camera, 0):
            pw = c + (q * s) @ r.T
            fore = pw[w] - pw[e]
            pts.append(pw[w] + fore / (np.linalg.norm(fore) + 1e-9) * GRIP_BEYOND_WRIST * s)
        out[side] = np.array(pts)
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
    ap.add_argument("--balls", action="store_true", help="also put the fitted ball tracklets in the scene "
                    "(off by default: on tennis they are other balls and blobs, not the served ball)")
    ap.add_argument("--hand-px", type=float, default=30.0, help="ball blob this close to a hand: in the hand")
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

    # ---- ball: SAM's one "ball" label holds several balls (in the hand, lying on the court): split it into
    # blobs, link blobs frame to frame into tracklets (nearest within --link px), fit each tracklet on its own
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
    tracks = [t for t in tracks if len(t) >= 5]
    balls, info = [], []
    for k, tr in enumerate(sorted(tracks, key=lambda t: min(t))):
        cen = np.full((t_n, 2), np.nan)
        wid = np.full(t_n, np.nan)
        for f, (c, w, _) in tr.items():
            cen[f], wid[f] = c, w
        # hypothesis checked by the fit: where the ball's blob touches a hand, the ball is in that hand
        anchors = {}
        for f in tr:
            R, c, K = cams[f]
            for h in hands.values():
                if np.linalg.norm(_project(h[f], cams[f])[0] - cen[f]) < a.hand_px:
                    anchors[f] = (float((h[f] - c) @ P.pixel_ray(cen[f], R, c, K)), 0.15 * s)
        pos, bhow, segs = P.ball_fit(cen, cams, fps, widths=wid, diameter=BALL * s, anchors=anchors)
        for sg in segs:
            lo, hi = sg["frames"]
            sg["track"] = k
            sg["kept"] = bool(sg["rms_px"] <= a.max_rms)
            if not sg["kept"]:
                pos[lo:hi + 1] = np.nan
            mid = (lo + hi) // 2
            sg["mid_xyz"] = [round(float(x), 2) for x in pos[mid]] if np.all(np.isfinite(pos[mid])) else None
            info.append(sg)
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
    for k, pos_k, kept in (balls if a.balls else []):
        bkeys = [TransformKey(time=f / fps, pose=Pose(translation=tuple(float(x) for x in pos_k[f])))
                 for f in range(t_n) if np.all(np.isfinite(pos_k[f]))]
        objs.append(SceneObject(
            id=f"obj_ball_{k:02d}", name=f"ball {k}", kind="rigid", size=(BALL * s,) * 3, keys=bkeys,
            status="estimated", look={"all": (0.85, 0.95, 0.2)},
            evidence=[Evidence(note="SAM 'tennis ball' mask, one tracklet; path fitted per segment (still / ballistic / "
                                    "straight) on the camera rays, depth from the standard size; spin not observed; "
                                    "shown only in the frames it was seen")]
            + [Evidence(source_frames=list(range(sg["frames"][0], sg["frames"][1] + 1)),
                        note=f"{sg['model']}, rms {sg['rms_px']} px") for sg in kept]))
    project = project.model_copy(deep=True)
    project.reconstruction.objects = objs
    for f in proj_path.parent.glob("*.npz"):
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
