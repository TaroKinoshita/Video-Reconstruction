"""Place close-up cuts in the joined world (after scripts/multishot_join.py).

    python scripts/multishot_closeups.py JOINED.npz DIAG_DIR PREFIX OUT.npz

A close-up has no keyframes in the joint 4RC run (too little shared view), but it shows part of a person who is
in the wide cuts. Per close-up k (a cut still in its own slot):
  who      the joined cut whose person looks most alike: colour histogram (hue/sat) of the masked pixels, nearest
           in time on ties
  pose     that person's body at the wide cut's frame nearest in time (the close-up shows the same moment or near)
  camera   searched: look-at joint x azimuth x elevation x distance (+ the close-up's own lens), scoring the
           overlap (IoU) of the body's thick stick silhouette with the close-up mask at its middle frame; then
           the cut's own camera motion is kept relative to that pose
  scale    the found distance / the 4RC depth of the mask: the close-up's own points and motion in metres
The overlap is reported; below MIN_IOU the cut is left where it was (not placed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrecon.control.core import read_video_rgb  # noqa: E402

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
RADIUS = {15: 0.11, 12: 0.06, 9: 0.13, 6: 0.13, 3: 0.13, 0: 0.13}  # head, neck, torso; limbs 0.06
MIN_IOU = 0.6  # a looser fit is as likely a wrong mask (e.g. the floor) as a person
GW, GH = 160, 68  # search grid size


def hist(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    px = rgb[mask].astype(float) / 255.0
    if len(px) == 0:
        return np.zeros(64)
    mx, mn = px.max(1), px.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    h, _, _ = np.histogram2d(mx, sat, bins=8, range=[[0, 1], [0, 1]])
    return (h / h.sum()).ravel()


def silhouette(joints: np.ndarray, cam: np.ndarray, lens: float, w: int, h: int) -> np.ndarray:
    """body as spheres sampled along the bones, projected by a Blender-axes camera (4x4 c2w)."""
    pts, rad = [], []
    for j, p in enumerate(PARENTS):
        if p < 0:
            continue
        r = RADIUS.get(p, 0.06) if p in RADIUS else RADIUS.get(j, 0.06)
        for u in np.linspace(0, 1, 6):
            pts.append(joints[p] * (1 - u) + joints[j] * u)
            rad.append(r)
    pts, rad = np.array(pts), np.array(rad)
    w2c = np.linalg.inv(cam)
    pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = -pc[:, 2]
    ok = z > 0.05
    f = lens / 36.0 * w
    u = w / 2 + f * pc[ok, 0] / z[ok]
    v = h / 2 - f * pc[ok, 1] / z[ok]
    rr = f * rad[ok] / z[ok]
    sil = np.zeros((h, w), bool)
    for a, b, r in zip(u, v, rr):
        x0, x1 = max(int(a - r), 0), min(int(a + r) + 1, w)
        y0, y1 = max(int(b - r), 0), min(int(b + r) + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        sil[y0:y1, x0:x1] |= (xx - a) ** 2 + (yy - b) ** 2 < r * r
    return sil


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, -fwd, eye
    return m


def main() -> int:
    src, diag, prefix, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    z = dict(np.load(src))
    info = json.loads(str(z["info"]))
    shots = info["shots"]
    wide = [k for k, s in enumerate(shots) if s.get("slot") == 0 and len(z[f"s{k}_body"])]
    report = {}
    looks = {}
    for k in wide:
        rgb = read_video_rgb(diag / f"{prefix}{k:02d}.mp4")
        m = np.load(diag / f"{prefix}{k:02d}.masks.npz")["masks"]
        t = len(m) // 2
        mm = m[t] > 0
        if mm.shape != rgb.shape[1:3]:
            mm = mm[np.arange(rgb.shape[1]) * mm.shape[0] // rgb.shape[1]][:, np.arange(rgb.shape[2]) * mm.shape[1] // rgb.shape[2]]
        looks[k] = hist(rgb[t], mm)
    AZ, EL, DIST = np.arange(0, 360, 20), np.array([-30, -10, 10, 30, 60]), np.array([0.3, 0.5, 0.8, 1.2, 2.0])
    fits = {}
    for k, s in enumerate(shots):
        if s.get("slot") == 0:
            continue
        rgb = read_video_rgb(diag / f"{prefix}{k:02d}.mp4")
        masks = np.load(diag / f"{prefix}{k:02d}.masks.npz")["masks"]
        t = len(masks) // 2
        mk = masks[t] > 0
        if mk.mean() < 0.01:
            report[k] = {"placed": False, "why": "no person mask"}
            continue
        hk = hist(rgb[t], mk)
        a0 = s["source_frames"][0] + t

        def time_gap(c):
            a, b = shots[c]["source_frames"]
            return 0 if a <= a0 < b else min(abs(a0 - a), abs(a0 - b))

        sim = {c: float(np.minimum(hk, looks[c]).sum()) for c in wide}
        who = max(wide, key=lambda c: (round(sim[c], 2), -time_gap(c)))
        a, b = shots[who]["source_frames"]
        tf = int(np.clip(a0 - a, 0, len(z[f"s{who}_body"]) - 1))
        body = z[f"s{who}_body"][tf]
        small = mk[np.arange(GH) * mk.shape[0] // GH][:, np.arange(GW) * mk.shape[1] // GW]
        S = np.zeros((22, len(AZ), len(EL), len(DIST)))
        for j in range(22):
            for ia, az in enumerate(np.radians(AZ)):
                for ie, el in enumerate(np.radians(EL)):
                    for i_d, d in enumerate(DIST):
                        eye = body[j] + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
                        sil = silhouette(body, look_at(eye, body[j]), s["lens_35mm"], GW, GH)
                        S[j, ia, ie, i_d] = (sil & small).sum() / max((sil | small).sum(), 1)
        fits[k] = {"who": who, "sim": sim, "tf": tf, "body": body, "S": S, "t": t}
    # neighbouring close-ups of the same person (only a short transition between them) are one set-up: they share
    # the viewing direction (azimuth, elevation); each keeps its own look-at joint and distance
    groups, cur = [], []
    for k in sorted(fits):
        if cur and fits[k]["who"] == fits[cur[-1]]["who"] and                 shots[k]["source_frames"][0] - shots[cur[-1]]["source_frames"][1] <= 3:
            cur.append(k)
        else:
            if cur:
                groups.append(cur)
            cur = [k]
    if cur:
        groups.append(cur)
    choice = {}
    for g in groups:
        view = sum(fits[k]["S"].max(axis=(0, 3)) for k in g)  # [az, el]
        ia, ie = np.unravel_index(np.argmax(view), view.shape)
        for k in g:
            sub = fits[k]["S"][:, ia, ie, :]
            j, i_d = np.unravel_index(np.argmax(sub), sub.shape)
            choice[k] = (int(j), float(AZ[ia]), float(EL[ie]), float(DIST[i_d]), float(sub[j, i_d]), g)
    for k, (j, az, el, d, iou, g) in choice.items():
        f = fits[k]
        t, body, who, tf = f["t"], f["body"], f["who"], f["tf"]
        report[k] = {"who": who, "similarity": f["sim"], "body_frame_of_cut": tf, "iou": iou,
                     "iou_alone": float(f["S"].max()), "shares_view_with": [c for c in g if c != k],
                     "look_at_joint": j, "azimuth_deg": az, "elevation_deg": el, "distance_m": d}
        if iou < MIN_IOU:
            report[k]["placed"] = False
            continue
        azr, elr = np.radians(az), np.radians(el)
        eye = body[j] + d * np.array([np.cos(elr) * np.cos(azr), np.cos(elr) * np.sin(azr), np.sin(elr)])
        P = look_at(eye, body[j])  # world pose of the close-up's middle-frame camera
        cam = z[f"s{k}_cam"]  # own (unitless) path: rescale so the subject sits at distance d
        own_d = np.median(np.linalg.norm(z[f"s{k}_pts"] - cam[t, :3, 3], axis=1)) if len(z[f"s{k}_pts"]) else d
        sc = d / max(own_d, 1e-6)
        rel = np.linalg.inv(cam[t])
        new = np.empty_like(cam)
        for i in range(len(cam)):
            r = rel @ cam[i]
            r[:3, 3] *= sc
            new[i] = P @ r
        pts = z[f"s{k}_pts"]
        if len(pts):
            pr = (pts - cam[t, :3, 3]) @ cam[t, :3, :3] * sc  # in the middle camera's frame, metres
            z[f"s{k}_pts"] = pr @ P[:3, :3].T + P[:3, 3]
        z[f"s{k}_cam"] = new
        shots[k]["point_radius"] = shots[k].get("point_radius", 0.03) * sc
        z[f"s{k}_body"] = np.repeat(body[None], len(new), 0)  # the matched person, held, shown during this cut
        shots[k]["points_only_during_cut"] = True  # its points are mostly the person up close: not a shared set
        shots[k]["slot"] = 0
        shots[k]["shared_fill_from"] = info["joined"].get("floor_from_cut", 0)
        report[k]["placed"] = True
        report[k]["scale"] = float(sc)
    info["closeups"] = report
    z["info"] = json.dumps(info)
    np.savez_compressed(out, **z)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
