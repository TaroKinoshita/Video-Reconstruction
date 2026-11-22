"""Place close-up cuts in the joined world (after scripts/multishot_join.py).

    python scripts/multishot_closeups.py JOINED.npz DIAG_DIR PREFIX OUT.npz

A close-up has no keyframes in the joint 4RC run (too little shared view), but it shows part of a person who is
in the wide cuts. Per close-up k (a cut still in its own slot):
  who      the joined cut whose person looks most alike: colour histogram (hue/sat) of the masked pixels, nearest
           in time on ties
  pose     that person's body at the wide cut's frame nearest in time (the close-up shows the same moment or near)
  camera   searched: look-at joint x azimuth x elevation x distance (+ the close-up's own lens), scoring the
           overlap (IoU) of the drawn mannequin's silhouette (scene/mannequin.py) with the close-up mask at its middle frame; then
           the cut's own camera motion is kept relative to that pose
  own pose a FAB body of the close-up itself (DIAG/fab_PREFIXNN.npz, gpu/observe/run_fab.py) replaces the
           borrowed pose's shape when present (pose_onto); place and size stay the matched body's
  arms     the borrowed pose's arms are refit to the close-up mask (refine_arms): arms move between cuts
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
from vrecon.scene.mannequin import mannequin, silhouette as mannequin_silhouette  # noqa: E402

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


def silhouette(joints: np.ndarray, cam: np.ndarray, lens: float, w: int, h: int, mesh=None) -> np.ndarray:
    """the drawn mannequin's silhouette (scene/mannequin.py): the search fits the shape the render shows."""
    return mannequin_silhouette(joints, cam, lens, w, h, mesh=mesh)


ARMS = ((16, 18, 20), (17, 19, 21))  # shoulder, elbow, wrist
STEPS_DEG = (80, 40, 20, 10)


def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _pose(body: np.ndarray, q: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """q [2 arms x (upper yaw, upper pitch, fore yaw, fore pitch)] in degrees, about the camera's up and right axes
    (the moves the picture can show)."""
    b = body.copy()
    up, right = cam[:3, 1], cam[:3, 0]
    for (sh, el, wr), (a1, a2, a3, a4) in zip(ARMS, q.reshape(2, 4)):
        R = _rot(up, a1) @ _rot(right, a2)
        b[el] = b[sh] + R @ (b[el] - b[sh])
        b[wr] = b[sh] + R @ (b[wr] - b[sh])
        R2 = _rot(up, a3) @ _rot(right, a4)
        b[wr] = b[el] + R2 @ (b[wr] - b[el])
    return b


def refine_arms(body: np.ndarray, cam: np.ndarray, lens: float, mask: np.ndarray, hint) -> tuple[np.ndarray, dict]:
    """A close-up borrows the pose of a wide cut, but arms move (a gun raised at the camera). Pattern search over
    both arms' upper and fore rotations for the mannequin silhouette's IoU with the close-up mask, plus a bonus when
    a wrist falls inside the box of an object seen in the cut (things are held in hands). Kept only if it helps."""
    def score(b):
        sil = silhouette(b, cam, lens, GW, GH)
        iou = (sil & mask).sum() / max((sil | mask).sum(), 1)
        bonus = 0.0
        if hint is not None:
            uv = _project(b[[20, 21]], cam, lens / 36.0 * GW, GW, GH)
            x0, y0, x1, y1 = hint
            c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            dist = np.linalg.norm(uv - c, axis=1).min() / np.hypot(GW, GH)  # nearest wrist to the thing, 0..~1
            bonus = 0.2 * (1 - min(dist * 2, 1.0))  # a slope the search can follow, not only a hit
        return iou + bonus, iou

    q = np.zeros(8)
    best, iou0 = score(body)
    start = best
    for step in STEPS_DEG:
        improved = True
        while improved:
            improved = False
            for i in range(8):
                for sgn in (1, -1):
                    c = q.copy()
                    c[i] += sgn * step
                    if abs(c[i]) > 160:
                        continue
                    sc, _ = score(_pose(body, c, cam))
                    if sc > best + 1e-4:
                        q, best, improved = c, sc, True
    out = _pose(body, q, cam)
    _, iou1 = score(out)
    rep = {"angles_deg": q.tolist(), "score_before": float(start), "score_after": float(best), "iou_before": float(iou0),
           "iou_after": float(iou1)}
    ok = best > start + 0.01 and iou1 >= iou0 - 0.02  # the hand may move to the thing, the outline must not get worse
    return (out, rep) if ok else (body, {**rep, "kept": False})


def _project(p: np.ndarray, cam: np.ndarray, f: float, W: int, H: int) -> np.ndarray:
    w2c = np.linalg.inv(cam)
    pc = p @ w2c[:3, :3].T + w2c[:3, 3]
    z = np.maximum(-pc[:, 2], 1e-3)
    return np.stack([W / 2 + f * pc[:, 0] / z, H / 2 - f * pc[:, 1] / z], 1)


TORSO = [0, 1, 2, 3, 6, 9, 12, 16, 17]


def pose_onto(own: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """A pose seen in the close-up (its own camera frame, its own size) put where the matched person stands: the
    torso joints are fitted to the matched body's torso (rotation, translation, one scale), then the whole pose
    follows. The pose's shape (arms, head) is the close-up's, its place and size the wide cut's."""
    a, b = own[TORSO], ref[TORSO]
    ca, cb = a.mean(0), b.mean(0)
    A, B = a - ca, b - cb
    sc = np.sqrt((B ** 2).sum() / max((A ** 2).sum(), 1e-12))
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    R = (U @ np.diag([1, 1, d]) @ Vt).T
    return (own - ca) @ R.T * sc + cb


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
        fp = diag / f"fab_{prefix}{k:02d}.npz"
        own_pose = False
        if fp.exists():  # the close-up's own pose (gpu/observe/run_fab.py): its shape, the matched body's place and size
            fz = np.load(fp)
            if fz["ok"].any():
                t_ok = np.flatnonzero(fz["ok"])
                body = pose_onto(fz["joints"][t_ok[np.argmin(np.abs(t_ok - t))]], body)
                own_pose = True
        small = mk[np.arange(GH) * mk.shape[0] // GH][:, np.arange(GW) * mk.shape[1] // GW]
        mesh = mannequin(body)
        S = np.zeros((22, len(AZ), len(EL), len(DIST)))
        for j in range(22):
            for ia, az in enumerate(np.radians(AZ)):
                for ie, el in enumerate(np.radians(EL)):
                    for i_d, d in enumerate(DIST):
                        eye = body[j] + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
                        sil = silhouette(body, look_at(eye, body[j]), s["lens_35mm"], GW, GH, mesh)
                        S[j, ia, ie, i_d] = (sil & small).sum() / max((sil | small).sum(), 1)
        hint = None
        op = diag / f"{prefix}{k:02d}.objects.npz"
        if op.exists():
            om = np.load(op)["masks"]
            om = om[min(t, len(om) - 1)] > 0
            if om.any():
                ys, xs = np.nonzero(om[np.arange(GH) * om.shape[0] // GH][:, np.arange(GW) * om.shape[1] // GW])
                if len(xs):
                    hint = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        fits[k] = {"own_pose": own_pose, "who": who, "sim": sim, "tf": tf, "body": body, "S": S, "t": t, "small": small, "hint": hint}
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
        report[k] = {"own_pose": f["own_pose"], "who": who, "similarity": f["sim"], "body_frame_of_cut": tf, "iou": iou,
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
        body, pose_rep = refine_arms(body, new[t], shots[k]["lens_35mm"], f["small"], f["hint"])
        report[k]["pose"] = pose_rep
        z[f"s{k}_body"] = np.repeat(body[None], len(new), 0)  # the matched person, arms refit, shown during this cut
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
