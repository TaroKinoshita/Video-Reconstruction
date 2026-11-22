"""People GVHMR did not follow (it follows one person per cut): a coarse figure where each extra mask track stands.

    python scripts/multishot_people.py IN.npz DIAG_DIR PREFIX OUT.npz

Per cut in the shared world (slot 0, its floor is z = 0) and per mask track:
  main     a track whose box overlaps the projected box of the cut's GVHMR body (mean box IoU > MAIN_IOU) is that
           body: skipped
  merge    extra tracks that are never on screen together and whose boxes overlap where one ends and the next
           starts are one person (a track that SAM lost and found again)
  place    per frame with the mask's lowest row inside the picture (feet seen): the ray through the bottom
           centre meets the floor -> where the person stands; the ray through the top centre, at that spot -> height.
           Positions are median-smoothed; frames with the feet cut off keep the last place.
  check    a height outside 1.0-2.3 m is not a standing person (reported; the figure gets the median height)
Writes s<k>_extra [P, T, 4] (x, y, height, visible) and info["people"].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MAIN_IOU = 0.3


def box(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None


def box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / max(u, 1)


def project(p: np.ndarray, cam: np.ndarray, f: float, W: int, H: int) -> np.ndarray:
    w2c = np.linalg.inv(cam)
    pc = p @ w2c[:3, :3].T + w2c[:3, 3]
    z = np.maximum(-pc[:, 2], 1e-3)
    return np.stack([W / 2 + f * pc[:, 0] / z, H / 2 - f * pc[:, 1] / z], 1)


def ray(u: float, v: float, cam: np.ndarray, f: float, W: int, H: int):
    d = cam[:3, :3] @ np.array([(u - W / 2) / f, -(v - H / 2) / f, -1.0])
    return cam[:3, 3], d / np.linalg.norm(d)


def seg_hits_quad(o: np.ndarray, e: np.ndarray, q: np.ndarray) -> bool:
    """does the segment o->e cross the (planar, convex) quad q[4,3]?"""
    n = np.cross(q[1] - q[0], q[3] - q[0])
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return False
    n /= nn
    a, b = (o - q[0]) @ n, (e - q[0]) @ n
    if a * b > 0 or abs(a - b) < 1e-9:
        return False
    x = o + (e - o) * (a / (a - b))
    for i in range(4):
        if np.cross(q[(i + 1) % 4] - q[i], x - q[i]) @ n < 0:
            return False
    return True


def carve(z: dict, info: dict, every: int = 8, margin: float = 0.3) -> dict:
    """A camera sees its people: no set piece stands between them. Rays from each shared-world camera (every
    `every`-th frame) to the seen people (body joints, extra figures' middle line), stopping `margin` short,
    remove every layout quad they cross."""
    quads = z.get("layout_quads", np.zeros((0, 4, 3)))
    if not len(quads):
        return {"removed": 0}
    lo, hi = quads.min(1), quads.max(1)
    keep = np.ones(len(quads), bool)
    for k, s in enumerate(info["shots"]):
        if s.get("slot") != 0 or s.get("points_only_during_cut"):
            continue
        cam, body, extra = z[f"s{k}_cam"], z[f"s{k}_body"], z.get(f"s{k}_extra", np.zeros((0, 0, 4)))
        for t in range(0, len(cam), every):
            o = cam[t, :3, 3]
            targets = list(body[t][[0, 6, 12, 15, 7, 8, 20, 21]]) if t < len(body) else []
            for pp in extra:
                if t < pp.shape[0] and pp[t, 3] > 0.5:
                    targets += [np.array([pp[t, 0], pp[t, 1], hz * pp[t, 2]]) for hz in (0.2, 0.5, 0.9)]
            for e in targets:
                d = e - o
                L = np.linalg.norm(d)
                if L <= margin:
                    continue
                e2 = o + d * (L - margin) / L
                bl, bh = np.minimum(o, e2), np.maximum(o, e2)
                cand = np.flatnonzero(keep & (hi >= bl).all(1) & (lo <= bh).all(1))
                for qi in cand:
                    if seg_hits_quad(o, e2, quads[qi]):
                        keep[qi] = False
    for key in ("layout_quads", "layout_quad_cat", "layout_quad_col"):
        z[key] = z[key][keep]
    return {"removed": int((~keep).sum()), "kept": int(keep.sum())}


def main() -> int:
    src, diag, prefix, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    z = dict(np.load(src))
    info = json.loads(str(z["info"]))
    W, H = int(z["W"]), int(z["H"])
    rep = {}
    for k, s in enumerate(info["shots"]):
        z[f"s{k}_extra"] = np.zeros((0, s["frames"], 4))
        mp = diag / f"{prefix}{k:02d}.masks.npz"
        if s.get("slot") != 0 or s.get("points_only_during_cut") or not mp.exists():
            continue
        masks = np.load(mp)["masks"]
        sx, sy = W / masks.shape[2], H / masks.shape[1]
        cam, body = z[f"s{k}_cam"], z[f"s{k}_body"]
        f = s["lens_35mm"] / 36.0 * W
        T = min(len(masks), len(cam))
        tracks = {}
        for i in (x for x in np.unique(masks) if x):
            bx = [box(masks[t] == i) for t in range(T)]
            tracks[int(i)] = [None if b is None else (b[0] * sx, b[1] * sy, (b[2] + 1) * sx, (b[3] + 1) * sy) for b in bx]
        main_ids = []
        if len(body):
            for i, bx in tracks.items():
                ious = []
                for t in range(min(T, len(body))):
                    if bx[t] is None:
                        continue
                    uv = project(body[t], cam[t], f, W, H)
                    ious.append(box_iou(bx[t], (*uv.min(0), *uv.max(0))))
                if ious and np.mean(ious) > MAIN_IOU:
                    main_ids.append(i)
        extra = [i for i in tracks if i not in main_ids]
        # one person per chain of tracks that never share a frame and meet where one ends and the next starts
        people = []
        for i in sorted(extra, key=lambda i: next(t for t, b in enumerate(tracks[i]) if b is not None)):
            on = {t for t, b in enumerate(tracks[i]) if b is not None}
            first = tracks[i][min(on)]
            for p in people:
                p_on = set().union(*({t for t, b in enumerate(tracks[j]) if b is not None} for j in p))
                last = tracks[p[-1]][max(t for t, b in enumerate(tracks[p[-1]]) if b is not None)]
                if not (on & p_on) and box_iou(first, last) > 0.2:
                    p.append(i)
                    break
            else:
                people.append([i])
        arr = np.zeros((len(people), s["frames"], 4))
        r = {"main_tracks": main_ids, "people": []}
        for pi, p in enumerate(people):
            spots, heights, vis = {}, [], np.zeros(T, bool)
            for t in range(T):
                b = next((tracks[j][t] for j in p if tracks[j][t] is not None), None)
                if b is None:
                    continue
                vis[t] = True
                if b[3] >= H - 2:  # feet cut off by the frame
                    continue
                o, d = ray((b[0] + b[2]) / 2, b[3], cam[t], f, W, H)
                if d[2] >= -1e-3:
                    continue
                g = o + d * (-o[2] / d[2])
                if np.linalg.norm(g - o) > 80:
                    continue
                spots[t] = g[:2]
                o2, d2 = ray((b[0] + b[2]) / 2, b[1], cam[t], f, W, H)
                hz = d2[:2] @ d2[:2]  # point on the top ray nearest the vertical line at the spot
                lam = ((g[:2] - o2[:2]) @ d2[:2]) / max(hz, 1e-9)
                heights.append(float(o2[2] + lam * d2[2]))
            if not spots:
                r["people"].append({"tracks": p, "placed": False, "why": "feet never seen"})
                continue
            ts = np.array(sorted(spots))
            xy = np.array([spots[t] for t in ts])
            full = np.stack([np.interp(np.arange(T), ts, xy[:, 0]), np.interp(np.arange(T), ts, xy[:, 1])], 1)
            k5 = 5
            sm = np.array([np.median(full[max(0, t - k5):t + k5 + 1], 0) for t in range(T)])
            h = float(np.median(heights))
            ok = 1.0 <= h <= 2.3
            arr[pi, :T, :2] = sm
            arr[pi, :T, 2] = h if ok else 1.75
            arr[pi, :T, 3] = vis
            r["people"].append({"tracks": p, "placed": True, "height_m": h, "height_plausible": ok,
                                "frames_seen": int(vis.sum()), "frames_feet_seen": len(spots),
                                "spot_first": sm[ts[0]].round(2).tolist()})
        z[f"s{k}_extra"] = arr
        rep[k] = r
    rep["carved"] = carve(z, info)
    info["people"] = rep
    z["info"] = json.dumps(info)
    np.savez_compressed(out, **z)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
