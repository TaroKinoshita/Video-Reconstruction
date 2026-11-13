"""Score the prop tools (analysis/props.py) on the synthetic prop clips (scripts/make_prop_clips.py).

    python scripts/eval_props.py evaluation/props/prop_01 [--out runs/props/prop_01.json]

Inputs to the tools: the video only for the masks (colour thresholds: the red paddle, the yellow ball - a
segmenter stand-in that sees the blur as it is) and, as stand-ins for what the scene already has, the camera
(from gt.json) and the grip point (the paddle's pivot from gt.json plus 2 cm noise, standing in for a wrist joint).
Ground truth is read only to score. Reports: ball position error per frame (m) with and without the blur end
points, the found hit frame; paddle direction error (deg) on sharp / blurred frames, sweep fit vs plain
interpolation between sharp frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from vrecon.analysis import props as P
from vrecon.control.core import read_video_rgb
from vrecon.metrics.poses import camera_pose, object_pose
from vrecon.schema import load_project


def colour_masks(video, lo_hsv, hi_hsv, lo2=None, hi2=None, follow=False):
    """Largest blob per frame; with follow=True the blob nearest the last one kept (one object of several alike,
    as a tracking segmenter would keep its identity)."""
    out, last, lost = [], None, False
    for fr in video:
        hsv = cv2.cvtColor(fr, cv2.COLOR_RGB2HSV)
        m = cv2.inRange(hsv, np.array(lo_hsv), np.array(hi_hsv))
        if lo2 is not None:
            m |= cv2.inRange(hsv, np.array(lo2), np.array(hi2))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros_like(m, bool)
        if follow and lost:
            n = 0  # once lost, stays lost: this stand-in cannot re-identify it
        if n > 1:
            big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            if follow and last is not None:
                cen = np.array([np.argwhere(lab == i).mean(0)[::-1] for i in range(1, n)])
                near = 1 + int(np.argmin(np.linalg.norm(cen - last, axis=1)))
                if np.linalg.norm(cen[near - 1] - last) >= 80:
                    lost = True
                    out.append(keep)  # lost it (left the view): another blob alike is not the same object
                    continue
                big = near
            keep = lab == big
            last = np.argwhere(keep).mean(0)[::-1]
        out.append(keep)
    return np.array(out)


def ang(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.clip)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    gt = load_project(d / "gt.json")
    video = read_video_rgb(str(d / "video.mp4"))
    t_n, fps = len(video), meta["fps"]
    shutter = meta.get("motion_blur", {}).get("shutter_frames", 0.0)
    cam = gt.reconstruction.cameras[0]
    cams = [camera_pose(cam, f) for f in range(t_n)]
    objs = {o.id: o for o in gt.reconstruction.objects}
    size = (video.shape[2], video.shape[1])
    rep = {"clip": d.name, "stand_ins": "camera from gt.json; grip = paddle pivot from gt.json + 2 cm noise"}

    # ---- ball (yellow)
    bm = colour_masks(video, (20, 35, 120), (45, 255, 255), follow=True)
    cen = np.full((t_n, 2), np.nan)
    ends = np.full((t_n, 2, 2), np.nan)
    wid = np.full(t_n, np.nan)
    for f in range(t_n):
        if bm[f].sum() >= 2:
            s = P.streak(bm[f])
            cen[f] = s["centre"]
            ends[f] = s["ends"]
            wid[f] = s["width"]
    gt_ball = np.array([object_pose(gt, objs["obj_ball"], "shot_001", f)[1] for f in range(t_n)])
    size_kw = {"widths": wid, "diameter": objs["obj_ball"].size[0]}
    for name, kw in (("centres only", {}), ("centres + size", size_kw),
                     ("size + blur ends", {**size_kw, "ends": ends, "shutter": shutter})):
        pos, how, info = P.ball_fit(cen, cams, fps, **kw)
        ok = np.all(np.isfinite(pos), 1)
        err = np.linalg.norm(pos[ok] - gt_ball[ok], axis=1)
        rep[f"ball {name}"] = {"frames_placed": int(ok.sum()), "seen": int(np.isfinite(cen[:, 0]).sum()),
                               "error_m_median": round(float(np.median(err)), 3) if len(err) else None,
                               "error_m_p90": round(float(np.percentile(err, 90)), 3) if len(err) else None,
                               "segments": [{**s, "error_m_median": round(float(np.median(np.linalg.norm(
                                   pos[s["frames"][0]:s["frames"][1] + 1] - gt_ball[s["frames"][0]:s["frames"][1] + 1],
                                   axis=1))), 3)} for s in info]}
    rep["ball splits found"] = P.split_at_changes(cen)
    rep["ball events (gt)"] = [e for e in meta["prop_events"] if e["object"] == "obj_ball"]

    # ---- paddle (red)
    pm = colour_masks(video, (0, 60, 70), (10, 255, 255), (165, 60, 70), (180, 255, 255))
    rng = np.random.default_rng(0)
    pad = objs["obj_paddle"]
    L, Wd, Th = pad.size
    gt_dir, gt_grip = [], []
    for f in range(t_n):
        R, tr, _ = object_pose(gt, pad, "shot_001", f)
        ax = R[:, 0]
        gt_dir.append(ax)
        gt_grip.append(tr - ax * L / 2)
    gt_dir, gt_grip = np.array(gt_dir), np.array(gt_grip)
    grips = gt_grip + rng.normal(0, 0.02 / np.sqrt(3), gt_grip.shape)
    rep["sharp frames (paddle)"] = P.sharp_frames(video, pm, top=5)
    tips, sides, how, score = P.held_poses(pm, grips, cams, L, Wd, Th, shutter=shutter, size=size)
    est = tips - grips
    errs = {k: [] for k in ("sharp", "swept", "interpolated (no sweep)")}
    sharp = [f for f in range(t_n) if how[f] == "sharp"]
    for f in range(t_n):
        if how[f] in ("sharp", "swept") and np.all(np.isfinite(est[f])):
            errs[how[f]].append(ang(est[f], gt_dir[f]))
        if how[f] == "swept":
            lo = max(s for s in sharp if s < f)
            hi = min(s for s in sharp if s > f)
            lin = P._slerp_dir(est[lo], est[hi], (f - lo) / (hi - lo))
            errs["interpolated (no sweep)"].append(ang(lin, gt_dir[f]))
    rep["paddle"] = {"frames": {k: int((how == k).sum()) for k in ("sharp", "swept", "blurred", "unseen")},
                     **{f"dir_error_deg {k}": (round(float(np.median(v)), 1) if v else None) for k, v in errs.items()},
                     "swept frames": [int(f) for f in range(t_n) if how[f] == "swept"]}
    txt = json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
