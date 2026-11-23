"""Which camera does the picture support, segment by segment (docs/PLAN_v4.md phase 2)?

    python scripts/diagnose_camera.py SCENE.npz DIAG TAG CUT OUT.json [--seg 21] [--gap 3]

Evidence independent of the 3D estimators: background corners tracked through the cut (Lucas-Kanade, frame to
frame, forward-backward check, re-seeded every RESEED frames), outside the person mask (dilated). Tracks are split
by id: two thirds "fit" (used to estimate a camera), one third "check" (only used to score).

For frame pairs (t, t+gap) the relative motion of each camera candidate is scored on the check tracks with the
Sampson distance (epipolar error, pixels in a 1280-wide picture; translation scale does not matter):
  scene        the scene's camera (s<k>_cam, with its lens)
  picture      the relative pose estimated from the fit tracks alone (essential matrix, scene lens): what the
               picture itself supports, the floor for any camera with that lens
  lens x s     the same, with the focal length scaled (0.8, 1.25): is the lens wrong?
Per segment of --seg frames: medians, how many check correspondences, and which candidate the picture supports.
Also: the scene error at frames that are 4RC samples (multishot_prep.sampled) against frames interpolated between
them, and the person's joint pixel error (kp2d) per segment, to tell camera problems from body problems.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
RESEED = 8
REF_W = 1280.0
CV = np.diag([1.0, -1.0, -1.0])  # Blender camera axes <-> OpenCV camera axes


def tracks(video, pmask, w, h):
    gray = [cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY) for f in video]
    free = []
    for t in range(len(video)):
        if pmask is None or t >= len(pmask):
            free.append(np.ones((h, w), np.uint8))
            continue
        big = cv2.resize((pmask[t] > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        free.append((cv2.dilate(big, np.ones((25, 25), np.uint8)) == 0).astype(np.uint8))
    pos: dict[int, dict[int, np.ndarray]] = {}
    ids, cur, nid = [], np.zeros((0, 1, 2), np.float32), 0
    for t in range(len(gray)):
        if t > 0 and len(cur):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[t - 1], gray[t], cur, None, winSize=(21, 21), maxLevel=3)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t - 1], nxt, None, winSize=(21, 21), maxLevel=3)
            xy = np.clip(nxt[:, 0].round().astype(int), 0, [w - 1, h - 1])
            ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((back - cur)[:, 0], axis=1) < 0.7) \
                & (free[t][xy[:, 1], xy[:, 0]] == 1)
            cur = nxt[ok]
            ids = [i for i, o in zip(ids, ok) if o]
        if t % RESEED == 0:
            m = free[t].copy() * 255
            for p in cur[:, 0]:
                cv2.circle(m, (int(p[0]), int(p[1])), 10, 0, -1)
            new = cv2.goodFeaturesToTrack(gray[t], 500 - len(cur), 0.01, 10, mask=m) if len(cur) < 500 else None
            if new is not None:
                cur = np.concatenate([cur, new.astype(np.float32)])
                ids += list(range(nid, nid + len(new)))
                nid += len(new)
        for i, p in zip(ids, cur[:, 0]):
            pos.setdefault(i, {})[t] = p.copy()
    return pos


def pairs(pos, a, b, sel):
    P = [(v[a], v[b]) for i, v in pos.items() if sel(i) and a in v and b in v]
    if not P:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.array([p[0] for p in P]), np.array([p[1] for p in P])


def rel_pose(cam, a, b):
    """Blender c2w at frames a, b -> OpenCV R, t mapping camera-a coordinates to camera-b coordinates."""
    Ra, Rb = cam[a, :3, :3] @ CV, cam[b, :3, :3] @ CV
    return Rb.T @ Ra, Rb.T @ (cam[a, :3, 3] - cam[b, :3, 3])


def sampson(R, t, K, x1, x2):
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Ki = np.linalg.inv(K)
    F = Ki.T @ tx @ R @ Ki
    p1 = np.c_[x1, np.ones(len(x1))]
    p2 = np.c_[x2, np.ones(len(x2))]
    Fx1, Ftx2 = p1 @ F.T, p2 @ F
    num = np.sum(p2 * Fx1, 1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def homography_err(R, K, x1, x2):
    """pure rotation (no translation): x2 ~ K R K^-1 x1."""
    H = K @ R @ np.linalg.inv(K)
    p = np.c_[x1, np.ones(len(x1))] @ H.T
    return np.linalg.norm(p[:, :2] / p[:, 2:] - x2, axis=1)


def score(R, t, K, x1, x2):
    if np.linalg.norm(t) < 1e-6:
        return homography_err(R, K, x1, x2)
    return sampson(R, t / np.linalg.norm(t), K, x1, x2)


def main() -> int:
    from multishot_prep import sampled
    from vrecon.control.core import read_video_rgb

    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("diag", type=Path)
    ap.add_argument("tag")
    ap.add_argument("cut", type=int)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seg", type=int, default=21)
    ap.add_argument("--gap", type=int, default=3)
    # "at sample": the pair starts within 1 frame of a 4RC sample; "between": 3 or more frames from any
    a = ap.parse_args()
    k = a.cut
    z = np.load(a.scene)
    info = json.loads(str(z["info"]))
    cam = z[f"s{k}_cam"]
    lens = info["shots"][k]["lens_35mm"]
    video = read_video_rgb(a.diag / f"{a.tag}_shot{k:02d}.mp4")
    T, h, w = min(len(video), len(cam)), video.shape[1], video.shape[2]
    mp = a.diag / f"{a.tag}_shot{k:02d}.masks.npz"
    pos = tracks(video[:T], np.load(mp)["masks"] if mp.exists() else None, w, h)
    s = REF_W / w
    rec = np.array([(i, t, *xy) for i, v in pos.items() for t, xy in v.items()], np.float32)
    np.savez_compressed(a.out.with_suffix(".tracks.npz"), tracks=rec, w=w, h=h)  # id, frame, x, y (refit_camera.py)

    def K_of(scale=1.0):
        f = lens / 36.0 * w * scale
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])

    fit_sel, chk_sel = (lambda i: i % 3 != 0), (lambda i: i % 3 == 0)
    samples = set(sampled(T))
    rows = []
    for t0 in range(0, T - a.gap):
        t1 = t0 + a.gap
        x1c, x2c = pairs(pos, t0, t1, chk_sel)
        x1f, x2f = pairs(pos, t0, t1, fit_sel)
        if len(x1c) < 8 or len(x1f) < 15:
            rows.append({"t": t0, "n_check": int(len(x1c))})
            continue
        near = lambda f: min(abs(f - q) for q in samples)  # noqa: E731
        r = {"t": t0, "n_check": int(len(x1c)), "at_sample": near(t0) <= 1, "between": near(t0) >= 3}
        R, tt = rel_pose(cam, t0, t1)
        r["scene"] = float(np.median(score(R, tt, K_of(), x1c, x2c))) * s
        for name, sc in (("picture", 1.0), ("lens_x0.8", 0.8), ("lens_x1.25", 1.25)):
            K = K_of(sc)
            E, inl = cv2.findEssentialMat(x1f, x2f, K, cv2.RANSAC, 0.999, 1.0)
            if E is None or E.shape != (3, 3):
                continue
            _, Rp, tp, _ = cv2.recoverPose(E, x1f, x2f, K)
            r[name] = float(np.median(score(Rp, tp[:, 0], K, x1c, x2c))) * s
        rows.append(r)
    # the person's joint error per frame (kp2d against the scene body)
    kp_p = a.diag / f"kp2d_{a.tag}{k:02d}.npz"
    body = z[f"s{k}_body"]
    jerr = np.full(T, np.nan)
    if kp_p.exists() and len(body):
        from measure_v4 import PAIRS, project
        kp = np.load(kp_p)["kp2d"]
        for t in range(min(T, len(body), len(kp))):
            e = []
            for pairs_ in PAIRS.values():
                for sj, cj in pairs_:
                    if kp[t, cj, 2] >= 0.5:
                        uv, d = project(body[t, sj][None], cam[t], lens, w, h)
                        if d[0] > 0.05:
                            e.append(np.linalg.norm(uv[0] - kp[t, cj, :2]) * s)
            if e:
                jerr[t] = np.median(e)
    segs = []
    for a0 in range(0, T, a.seg):
        rr = [r for r in rows if a0 <= r["t"] < a0 + a.seg and "scene" in r]
        seg = {"frames": [a0, min(a0 + a.seg, T) - 1], "pairs": len(rr),
               "n_check": int(sum(r["n_check"] for r in rr))}
        for c in ("scene", "picture", "lens_x0.8", "lens_x1.25"):
            v = [r[c] for r in rr if c in r]
            seg[c] = round(float(np.median(v)), 2) if v else None
        je = jerr[a0:a0 + a.seg]
        seg["person_px"] = round(float(np.nanmedian(je)), 1) if np.isfinite(je).any() else None
        if seg["scene"] is not None and seg["picture"] is not None:
            gap_ = seg["scene"] - seg["picture"]
            seg["verdict"] = ("scene camera agrees with the picture" if gap_ < 0.5 else
                              "scene camera disagrees with the picture" if gap_ > 2 else "small disagreement")
        else:
            seg["verdict"] = "not enough background evidence"
        segs.append(seg)
    at = [r["scene"] for r in rows if r.get("at_sample") and "scene" in r]
    btw = [r["scene"] for r in rows if "scene" in r and r.get("between")]
    allv = [r["scene"] for r in rows if "scene" in r]
    rep = {"overall_scene": round(float(np.median(allv)), 3) if allv else None,
           "overall_picture": round(float(np.median([r["picture"] for r in rows if "picture" in r])), 3) if allv else None,
           "check_pairs": len(allv), "cut": k, "frames": T, "tracks": len(pos), "gap": a.gap, "lens_35mm": lens,
           "scene_at_4rc_samples": round(float(np.median(at)), 2) if at else None,
           "scene_between_samples": round(float(np.median(btw)), 2) if btw else None,
           "segments": segs}
    a.out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    md = [f"cut {k}: {T} frames, {len(pos)} background tracks, pairs {a.gap} frames apart; Sampson px (1280 wide), "
          f"check tracks only", f"scene camera at 4RC sample pairs {rep['scene_at_4rc_samples']}, between samples "
          f"{rep['scene_between_samples']}", "",
          "| frames | check pts | scene | picture | lens x0.8 | lens x1.25 | person px | verdict |", "|---|---|---|---|---|---|---|---|"]
    for g in segs:
        md.append(f"| {g['frames'][0]}-{g['frames'][1]} | {g['n_check']} | {g['scene']} | {g['picture']} | "
                  f"{g['lens_x0.8']} | {g['lens_x1.25']} | {g['person_px']} | {g['verdict']} |")
    a.out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
