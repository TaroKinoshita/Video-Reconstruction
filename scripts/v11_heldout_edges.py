"""Compare static structure between a source cut and a render of the same cut: edges outside people only.

For each frame: Canny edges of both images, people (SAM masks of both, dilated) removed, then
precision = share of render edges within TOL px of a source edge, recall = the reverse, F = harmonic mean.
Writes per-frame numbers (JSON) and an overlay video (red = source only, green = render only, yellow = both, grey = masked).
This is a floor check for "is the structure in the right place", not a judgement of quality.

  python scripts/v11_heldout_edges.py SRC.mp4 SRC.masks.npz REN.mp4 REN.masks.npz OUT_PREFIX [--tol-frac 0.015]
Paths must be ASCII (cv2 file functions); the script is meant for scratch copies.
"""
from __future__ import annotations
import argparse, json
import cv2
import numpy as np

def frames(p):
    c = cv2.VideoCapture(p); out = []
    while True:
        ok, f = c.read()
        if not ok: return out
        out.append(f)

MIN_LEN = 0.04

def edges(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0.8)
    e = cv2.Canny(g, 15, 45)
    # keep long edge pieces only (structure lines), drop texture speckle
    n, lab, st, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = np.maximum(st[1:, 2], st[1:, 3]) >= MIN_LEN * img.shape[1]
    return keep[lab]

def main():
    ap = argparse.ArgumentParser()
    for a in ("src", "src_masks", "ren", "ren_masks", "out"): ap.add_argument(a)
    ap.add_argument("--tol-frac", type=float, default=0.015, help="tolerance as a share of image width")
    ap.add_argument("--dilate-frac", type=float, default=0.02, help="grow people masks by this share of width")
    a = ap.parse_args()
    S, R = frames(a.src), frames(a.ren)
    ms = np.load(a.src_masks)["masks"] > 0; mr = np.load(a.ren_masks)["masks"] > 0
    n = min(len(S), len(R), len(ms), len(mr)); H, W = S[0].shape[:2]
    tol = max(1, round(a.tol_frac * W)); dil = max(1, round(a.dilate_frac * W))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1))
    rows, vis = [], []
    for i in range(n):
        people = cv2.dilate((ms[i] | mr[i]).astype(np.uint8), k) > 0
        es, er = edges(S[i]) & ~people, edges(R[i]) & ~people
        ds = cv2.distanceTransform((~es).astype(np.uint8), cv2.DIST_L2, 3)
        dr = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
        p = float((ds[er] <= tol).mean()) if er.any() else float("nan")
        r = float((dr[es] <= tol).mean()) if es.any() else float("nan")
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        rows.append({"frame": i, "precision": p, "recall": r, "f": f, "static_share": float((~people).mean())})
        v = np.zeros((H, W, 3), np.uint8); v[people] = (60, 60, 60)
        v[es] = (0, 0, 255); v[er] = (0, 255, 0); v[es & er] = (0, 255, 255)
        vis.append(np.hstack([S[i], R[i], v]))
    mean = {k: float(np.nanmean([r[k] for r in rows])) for k in ("precision", "recall", "f", "static_share")}
    json.dump({"tol_px": tol, "mean": mean, "frames": rows}, open(a.out + ".json", "w", encoding="utf-8"), indent=1)
    h, w = vis[0].shape[:2]
    wr = cv2.VideoWriter(a.out + "_overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24000 / 1001, (w, h))
    for v in vis: wr.write(v)
    wr.release()
    print(a.out, {k: round(v, 3) for k, v in mean.items()})

if __name__ == "__main__":
    main()
