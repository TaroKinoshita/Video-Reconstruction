"""The measures of docs/PLAN_v4.md, all against the picture (not against another estimator):

    python scripts/measure_v4.py SCENE.npz DIAG_DIR TAG OUT.json

  joints_px      per cut: the scene's body joints projected through the scene's camera vs. the 2D keypoints of the
                 video (scripts/multishot_kp2d.py, ViTPose, confidence >= KP_CONF): median and 95th percentile
                 pixel error (1280-wide picture), per part (shoulders, elbows, wrists, hips, knees, ankles), and how
                 many joints were compared. Also "still": the share of frames whose drawn pose equals the previous
                 frame's (a frozen pose).
  bones          per person (info["who"] identity): median length of each bone per cut, and the largest ratio
                 between cuts (1.0 = the same body in every cut).
  background_px  per cut: corners tracked in the video outside the person mask (Lucas-Kanade, forward,
                 with a backward check), each lifted to the nearest scene point seen at its first frame, then
                 projected through the scene camera in every later frame: median and 95th percentile pixel error.
                 Low = the camera (and the set) hold still where the film holds still.
Writes OUT.json and OUT.md (a table).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
KP_CONF = 0.5
# SMPL joint -> COCO-17 keypoint (both: the person's own left/right)
PAIRS = {"shoulders": [(16, 5), (17, 6)], "elbows": [(18, 7), (19, 8)], "wrists": [(20, 9), (21, 10)],
         "hips": [(1, 11), (2, 12)], "knees": [(4, 13), (5, 14)], "ankles": [(7, 15), (8, 16)]}
BONES = {"shoulder_width": (16, 17), "upper_arm": (16, 18), "forearm": (18, 20), "hip_width": (1, 2), "thigh": (1, 4),
         "shin": (4, 7), "spine": (0, 12), "neck_head": (12, 15)}
REF_W = 1280


def project(p: np.ndarray, cam: np.ndarray, lens: float, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """world points [N,3] -> pixels [N,2] and depth [N] (Blender camera: looks down -Z, 36 mm horizontal sensor)."""
    w2c = np.linalg.inv(cam)
    pc = p @ w2c[:3, :3].T + w2c[:3, 3]
    d = -pc[:, 2]
    f = lens / 36.0 * w
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([w / 2 + f * pc[:, 0] / d, h / 2 - f * pc[:, 1] / d], 1)
    return uv, d


def stats(e) -> dict:
    e = np.asarray(e, float)
    if not len(e):
        return {"n": 0}
    return {"n": int(len(e)), "median": round(float(np.median(e)), 1), "p95": round(float(np.percentile(e, 95)), 1)}


def joints_px(body, cam, lens, kp, w, h) -> dict:
    s = REF_W / w
    out, allerr = {}, []
    for part, pairs in PAIRS.items():
        err = []
        for t in range(min(len(body), len(kp), len(cam))):
            for sj, cj in pairs:
                if kp[t, cj, 2] < KP_CONF:
                    continue
                uv, d = project(body[t, sj][None], cam[t], lens, w, h)
                if d[0] <= 0.05:
                    continue
                err.append(float(np.linalg.norm(uv[0] - kp[t, cj, :2])) * s)
        out[part] = stats(err)
        allerr += err
    out["all"] = stats(allerr)
    still = [np.allclose(body[t], body[t - 1], atol=1e-5) for t in range(1, len(body))]
    out["still_share"] = round(float(np.mean(still)), 2) if still else None
    return out


def person_free(m, t, w, h):
    """1 where frame t has no person (dilated mask), as a full-size uint8 picture."""
    if m is None or t >= len(m):
        return np.ones((h, w), np.uint8)
    big = cv2.resize((m[t] > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (cv2.dilate(big, np.ones((21, 21), np.uint8)) == 0).astype(np.uint8)


def background_px(video, pmask, cam, lens, pts, w, h) -> dict:
    """corners outside the person (per frame; SAM's object words include walls and floors, so only the person
    and things moving with it are left out)."""
    T = min(len(video), len(cam))
    if T < 3 or not len(pts):
        return {"n": 0}
    s = REF_W / w
    gray = [cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY) for f in video[:T]]
    keep = person_free(pmask, 0, w, h)
    p0 = cv2.goodFeaturesToTrack(gray[0], 400, 0.01, 12, mask=keep * 255)
    if p0 is None:
        return {"n": 0}
    p0 = p0.reshape(-1, 2)
    # lift: the nearest scene point in front, among those projecting within 3 px of the corner at frame 0
    uv, d = project(pts, cam[0], lens, w, h)
    ok = (d > 0.1) & np.isfinite(uv).all(1)
    uv, d, P = uv[ok], d[ok], pts[ok]
    anchors, track_pts = [], []
    for c in p0:
        near = np.flatnonzero(np.linalg.norm(uv - c, axis=1) < 4.0)
        if len(near):
            anchors.append(P[near[np.argmin(d[near])]])
            track_pts.append(c)
    if not anchors:
        return {"n": 0, "note": "no scene point under the tracked corners"}
    A = np.array(anchors)
    cur = np.array(track_pts, np.float32).reshape(-1, 1, 2)
    alive = np.ones(len(A), bool)
    err = []
    for t in range(1, T):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[t - 1], gray[t], cur, None, winSize=(21, 21), maxLevel=3)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t - 1], nxt, None, winSize=(21, 21), maxLevel=3)
        good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((back - cur)[:, 0], axis=1) < 1.0)
        alive &= good
        cur = nxt
        q, dd = project(A, cam[t], lens, w, h)
        free = person_free(pmask, t, w, h)
        xy = np.clip(cur[:, 0].round().astype(int), 0, [w - 1, h - 1])
        sel = alive & (dd > 0.1) & (free[xy[:, 1], xy[:, 0]] == 1)
        err += (np.linalg.norm(q[sel] - cur[sel, 0], axis=1) * s).tolist()
    r = stats(err)
    r["corners"] = int(len(A))
    r["alive_at_end"] = int(alive.sum())
    return r


def main() -> int:
    from vrecon.control.core import read_video_rgb

    src, diag, tag, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    z = np.load(src)
    info = json.loads(str(z["info"]))
    who = info.get("who", {})
    rep = {"scene": str(src), "cuts": {}, "bones": {}}
    lengths: dict[str, dict] = {}
    for k, s in enumerate(info["shots"]):
        cam, body, lens = z[f"s{k}_cam"], z[f"s{k}_body"], s["lens_35mm"]
        vid = diag / f"{tag}_shot{k:02d}.mp4"
        if not vid.exists():
            continue
        video = read_video_rgb(vid)
        h, w = video.shape[1:3]
        r = {}
        kp_p = diag / f"kp2d_{tag}{k:02d}.npz"
        if len(body) and kp_p.exists():
            r["joints_px"] = joints_px(body, cam, lens, np.load(kp_p)["kp2d"], w, h)
        mp = diag / f"{tag}_shot{k:02d}.masks.npz"
        pmask = np.load(mp)["masks"] if mp.exists() else None
        r["background_px"] = background_px(video, pmask, cam, lens, z[f"s{k}_pts"], w, h)
        if len(body):
            ident = who.get(str(k), {}).get("identity", f"cut{k}")
            lengths.setdefault(ident, {})[k] = {b: round(float(np.median(np.linalg.norm(body[:, i] - body[:, j], axis=-1))), 3)
                                                for b, (i, j) in BONES.items()}
        rep["cuts"][k] = r
        print(k, json.dumps(r), flush=True)
    for ident, per in lengths.items():
        ratio = {b: round(max(v[b] for v in per.values()) / max(min(v[b] for v in per.values()), 1e-6), 2) for b in BONES} if len(per) > 1 else {}
        rep["bones"][ident] = {"cuts": per, "max_ratio_between_cuts": ratio}
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    md = ["| cut | joints px median / p95 (n) | wrists px median | still pose share | background px median / p95 (n) |",
          "|---|---|---|---|---|"]
    for k, r in rep["cuts"].items():
        j, b = r.get("joints_px", {}), r.get("background_px", {})
        ja = j.get("all", {})
        md.append(f"| {k} | {ja.get('median', '-')} / {ja.get('p95', '-')} ({ja.get('n', 0)}) | "
                  f"{j.get('wrists', {}).get('median', '-')} | {j.get('still_share', '-')} | "
                  f"{b.get('median', '-')} / {b.get('p95', '-')} ({b.get('n', 0)}) |")
    md += ["", "| person | " + " | ".join(BONES) + " |", "|---|" + "---|" * len(BONES)]
    for ident, v in rep["bones"].items():
        md.append(f"| {ident} | " + " | ".join(str(v["max_ratio_between_cuts"].get(b, "-")) for b in BONES) + " |")
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
