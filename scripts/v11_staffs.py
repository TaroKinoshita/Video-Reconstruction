"""Held staffs: a thin straight object seen in a SAM mask is kept as a staff on frames where it touches a person's
hand region (the projected body mesh near the mask), so beams, branches and other wrong masks drop out. Pure numpy/cv2.

  python scripts/v11_staffs.py PREP.npz PEOPLE_ANIM.npz STICKS.masks.npz OUT.npz [--touch 6] [--min-px 40]

Per frame and stick label: the mask's 2D principal axis gives the ends; the staff is put at the depth of the body
vertices that touch the mask (median camera depth), i.e. assumed facing the camera at the hand's depth. Unsure in
depth, exact in the picture. Output: per label "<label>_t" frames, "<label>_a"/"<label>_b" world ends (PREP space),
"<label>_r" radius; "labels". Frames without a held staff are left out (the builder hides the staff there).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def body_at(anim, label: str, f: int):
    ts = anim[f"{label}_t"]
    if f < ts[0] or f > ts[-1]:
        return None
    j = int(np.searchsorted(ts, f))
    if ts[j] == f:
        return anim[f"{label}_v"][j]
    w = (f - ts[j - 1]) / (ts[j] - ts[j - 1])
    if ts[j] - ts[j - 1] > 12:
        return None
    return (1 - w) * anim[f"{label}_v"][j - 1] + w * anim[f"{label}_v"][j]


def main() -> int:
    ap = argparse.ArgumentParser()
    for n in ("prep", "anim", "sticks", "out"):
        ap.add_argument(n, type=Path)
    ap.add_argument("--touch", type=int, default=6, help="pixels: how near body and staff mask must come")
    ap.add_argument("--min-px", type=int, default=40)
    ap.add_argument("--max-len", type=float, default=2.5, help="metres; longer is not a hand-held staff")
    a = ap.parse_args()
    prep = np.load(a.prep)
    info = json.loads(str(prep["info"]))
    W, H = info["size"]
    f_img = info["f_img"]
    cam_R, cam_T = prep["cam_R"], prep["cam_T"]
    anim = dict(np.load(a.anim))  # load once: an npz re-reads (and unzips) on every access
    people = json.loads(str(anim["labels"]))
    sd = np.load(a.sticks)
    masks, id_map = sd["masks"], json.loads(str(sd["id_map"]))
    out, report = {}, {}
    ker = np.ones((2 * a.touch + 1,) * 2, np.uint8)
    for num, lab in id_map.items():
        T, A, B, R = [], [], [], []
        for f in range(len(masks)):
            m = masks[f] == int(num)
            if m.sum() < a.min_px:
                continue
            near = cv2.dilate(m.astype(np.uint8), ker).astype(bool)
            best = None
            for p in people:
                X = body_at(anim, p, f)
                if X is None:
                    continue
                Xc = (X - cam_T[f]) @ cam_R[f]
                z = Xc[:, 2]
                ok = z > 0.1
                u = np.round(f_img * Xc[ok, 0] / z[ok] + W / 2).astype(int)
                v = np.round(f_img * Xc[ok, 1] / z[ok] + H / 2).astype(int)
                ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                hit = np.zeros(ok.sum(), bool)
                hit[ins] = near[v[ins], u[ins]]
                if hit.sum() >= 5 and (best is None or hit.sum() > best[0]):
                    best = (int(hit.sum()), float(np.median(z[ok][hit])))
            if best is None:
                continue
            ys, xs = np.nonzero(m)
            P = np.stack([xs, ys], 1).astype(float)
            c = P.mean(0)
            ev, evec = np.linalg.eigh(np.cov((P - c).T))
            d = evec[:, 1]
            s = (P - c) @ d
            thick = 4 * np.sqrt(max(ev[0], 0.25))
            if np.ptp(s) < 4 * thick:  # not long and thin
                continue
            zc = best[1]
            ends = []
            for e in (s.min(), s.max()):
                px = c + e * d
                Xc = np.array([(px[0] - W / 2) * zc / f_img, (px[1] - H / 2) * zc / f_img, zc])
                ends.append(Xc @ cam_R[f].T + cam_T[f])
            if np.linalg.norm(ends[1] - ends[0]) > a.max_len:
                continue
            T.append(f)
            A.append(ends[0])
            B.append(ends[1])
            R.append(thick / 2 * zc / f_img)
        if len(T) < 5:
            report[lab] = {"frames": len(T), "kept": False}
            continue
        out[f"{lab}_t"] = np.array(T, np.int32)
        out[f"{lab}_a"] = np.array(A, np.float32)
        out[f"{lab}_b"] = np.array(B, np.float32)
        out[f"{lab}_r"] = np.float32(np.clip(np.median(R), 0.015, 0.04))
        L = np.linalg.norm(np.array(B) - np.array(A), axis=1)
        report[lab] = {"frames": len(T), "kept": True, "first": T[0], "last": T[-1],
                       "length_m_median": round(float(np.median(L)), 2), "radius_m": round(float(out[f'{lab}_r']), 3)}
    out["labels"] = np.array(json.dumps(sorted(k[:-2] for k in out if k.endswith("_t"))))
    np.savez_compressed(a.out, **out)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
