"""Held staffs: a thin straight object seen in a SAM mask is a staff on the frames where it touches a person's HAND
(body-mesh vertices near the hand keypoints, projected), so beams, branches and masks that only cross a body drop
out. Pure numpy/cv2.

  python scripts/v11_staffs.py PREP.npz PEOPLE_ANIM.npz STICKS.masks.npz BODY_DIR OUT.npz [--touch 4] [--gap 12]

Per stick label:
- grip: the hand (left or right of one person) with most vertices touching the mask; the grip point is that hand's
  vertex centre (from the moving body, so it is defined on every frame);
- length: one per label (a staff keeps its length): the 80th percentile of the seen lengths in the picture, back-
  projected at the hand's depth, so a staff seen at a slant is not taken as shorter;
- tilt: with the length known, the far end lies on the ray through its pixel at that distance from the grip; of the
  two solutions the one nearer the grip's depth is taken (the picture cannot tell toward / away); the near end
  continues the line through the grip. Picture-exact on both ends when the mask is whole.
- gaps up to --gap frames with the same hand are filled: direction slerped, grip from the hand on that frame.
Output per label "<label>_t", "<label>_a", "<label>_b" world ends (PREP space), "<label>_r"; "labels".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v11_mave_prep import read_ply  # noqa: E402

HANDS = {"right": range(21, 42), "left": range(42, 63)}  # MHR70 hand keypoints incl. wrist (41 right, 62 left)


def hand_vertices(body: Path, radius: float = 0.09) -> dict[str, np.ndarray]:
    """Mesh vertex ids near each hand's keypoints (same topology on every body file)."""
    js = next(body.glob("f*_obs_*.json"))
    meta = json.loads(js.read_text(encoding="utf-8"))["metadata"]["people"][0]
    V, _ = read_ply(js.with_name(js.stem + "_0.ply"))
    V = V * [1, -1, -1]
    K = np.array(meta["keypoints_3d"])
    out = {}
    for h, ids in HANDS.items():
        d = np.linalg.norm(V[:, None] - K[list(ids)][None], axis=2).min(1)
        out[h] = np.nonzero(d < radius)[0]
    return out


def body_at(anim, label: str, f: int, max_gap: int = 12):
    ts = anim[f"{label}_t"]
    if f < ts[0] or f > ts[-1]:
        return None
    j = int(np.searchsorted(ts, f))
    if ts[j] == f:
        return anim[f"{label}_v"][j]
    if ts[j] - ts[j - 1] > max_gap:
        return None
    w = (f - ts[j - 1]) / (ts[j] - ts[j - 1])
    return (1 - w) * anim[f"{label}_v"][j - 1] + w * anim[f"{label}_v"][j]


def slerp_dir(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    c = float(np.clip(a @ b, -1, 1))
    th = np.arccos(c)
    if th < 1e-4:
        return a
    return (np.sin((1 - w) * th) * a + np.sin(w * th) * b) / np.sin(th)


def main() -> int:
    ap = argparse.ArgumentParser()
    for n in ("prep", "anim", "sticks", "body", "out"):
        ap.add_argument(n, type=Path)
    ap.add_argument("--touch", type=int, default=4, help="pixels: how near hand and staff mask must come")
    ap.add_argument("--min-px", type=int, default=40)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--gap", type=int, default=12, help="frames: longest gap filled from the hand's motion")
    ap.add_argument("--max-len", type=float, default=2.5, help="metres; longer is not a hand-held staff")
    a = ap.parse_args()
    prep = np.load(a.prep)
    info = json.loads(str(prep["info"]))
    W, H = info["size"]
    f_img = info["f_img"]
    cam_R, cam_T = prep["cam_R"], prep["cam_T"]
    anim = dict(np.load(a.anim))  # load once: an npz re-reads (and unzips) on every access
    people = json.loads(str(anim["labels"]))
    hv = hand_vertices(a.body)
    sd = np.load(a.sticks)
    masks, id_map = sd["masks"], json.loads(str(sd["id_map"]))
    ker = np.ones((2 * a.touch + 1,) * 2, np.uint8)

    def project(Xw, f):
        Xc = (Xw - cam_T[f]) @ cam_R[f]
        return Xc, f_img * Xc[:, 0] / Xc[:, 2] + W / 2, f_img * Xc[:, 1] / Xc[:, 2] + H / 2

    def grip_at(p, h, f):
        X = body_at(anim, p, f, a.gap)
        return None if X is None else X[hv[h]].mean(0)

    out, report = {}, {}
    for num, lab in id_map.items():
        obs = {}  # f -> (person, hand, grip_c, end_far_px, end_near_px, len_far_px, len_px)
        for f in range(len(masks)):
            m = masks[f] == int(num)
            if m.sum() < a.min_px:
                continue
            n_cc, cc = cv2.connectedComponents(m.astype(np.uint8))
            if n_cc > 2:  # keep the largest piece
                m = cc == 1 + np.argmax(np.bincount(cc[cc > 0])[1:])
            near = cv2.dilate(m.astype(np.uint8), ker).astype(bool)
            best = None
            for p in people:
                X = body_at(anim, p, f, a.gap)
                if X is None:
                    continue
                for h, ids in hv.items():
                    Xc, u, v = project(X[ids], f)
                    ok = (Xc[:, 2] > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                    hits = int(near[v[ok].astype(int), u[ok].astype(int)].sum())
                    if hits >= a.min_hits and (best is None or hits > best[0]):
                        best = (hits, p, h)
            if best is None:
                continue
            _, p, h = best
            ys, xs = np.nonzero(m)
            P = np.stack([xs, ys], 1).astype(float)
            c = P.mean(0)
            ev, evec = np.linalg.eigh(np.cov((P - c).T))
            d = evec[:, 1]
            s = (P - c) @ d
            thick = 4 * np.sqrt(max(ev[0], 0.25))
            if np.ptp(s) < 4 * thick:  # not long and thin
                continue
            g = grip_at(p, h, f)
            gc, gu, gv = project(g[None], f)
            sg = (np.array([gu[0], gv[0]]) - c) @ d
            e_lo, e_hi = c + s.min() * d, c + s.max() * d
            far, nearp = (e_hi, e_lo) if s.max() - sg >= sg - s.min() else (e_lo, e_hi)
            obs[f] = dict(p=p, h=h, gc=gc[0], far=far, near=nearp, lfar=float(np.linalg.norm(far - [gu[0], gv[0]])),
                          lpx=float(np.ptp(s)), r=thick / 2 * gc[0, 2] / f_img)
        if len(obs) < 5:
            report[lab] = {"frames": len(obs), "kept": False}
            continue
        # one length per staff: seen lengths back-projected at the grip depth (a slant only shortens them)
        L = float(np.percentile([o["lpx"] * o["gc"][2] / f_img for o in obs.values()], 80))
        L = min(L, a.max_len)
        res = {}
        for f, o in obs.items():
            G = o["gc"]
            frac_far = np.clip(o["lfar"] / max(o["lpx"], 1e-6), 0.5, 1.0)
            s_far = L * frac_far
            ray = np.array([(o["far"][0] - W / 2) / f_img, (o["far"][1] - H / 2) / f_img, 1.0])
            ray /= np.linalg.norm(ray)
            b = ray @ G
            disc = b * b - (G @ G - s_far ** 2)
            ts_ = [b] if disc <= 0 else [b - np.sqrt(disc), b + np.sqrt(disc)]
            Xf = min((t * ray for t in ts_ if t > 0.1), key=lambda X: abs(X[2] - G[2]), default=b * ray)
            u = (Xf - G) / max(np.linalg.norm(Xf - G), 1e-6)
            Xn = G - u * (L - s_far)
            Rw = cam_R[f]
            res[f] = dict(p=o["p"], h=o["h"], dirw=u @ Rw.T, frac=float(frac_far), r=o["r"])
        # fill gaps with the same hand; grip from the moving hand on every frame
        fs = sorted(res)
        frames = []
        for i, f in enumerate(fs):
            frames.append((f, res[f]))
            if i + 1 < len(fs):
                f2 = fs[i + 1]
                o1, o2 = res[f], res[f2]
                if 1 < f2 - f <= a.gap and (o1["p"], o1["h"]) == (o2["p"], o2["h"]):
                    for k in range(f + 1, f2):
                        w = (k - f) / (f2 - f)
                        frames.append((k, dict(p=o1["p"], h=o1["h"], dirw=slerp_dir(o1["dirw"], o2["dirw"], w),
                                               frac=(1 - w) * o1["frac"] + w * o2["frac"], r=o1["r"])))
        T, A, B = [], [], []
        for f, o in frames:
            G = grip_at(o["p"], o["h"], f)
            if G is None:
                continue
            dw = o["dirw"] / np.linalg.norm(o["dirw"])
            T.append(f)
            B.append(G + dw * L * o["frac"])
            A.append(G - dw * L * (1 - o["frac"]))
        out[f"{lab}_t"] = np.array(T, np.int32)
        out[f"{lab}_a"] = np.array(A, np.float32)
        out[f"{lab}_b"] = np.array(B, np.float32)
        out[f"{lab}_r"] = np.float32(np.clip(np.median([o["r"] for o in obs.values()]), 0.015, 0.035))
        report[lab] = {"seen_frames": len(obs), "frames_with_fill": len(T), "first": int(min(T)), "last": int(max(T)),
                       "length_m": round(L, 2), "holders": sorted({f"{o['p']}:{o['h']}" for o in res.values()})}
    out["labels"] = np.array(json.dumps(sorted(k[:-2] for k in out if k.endswith("_t"))))
    np.savez_compressed(a.out, **out)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
