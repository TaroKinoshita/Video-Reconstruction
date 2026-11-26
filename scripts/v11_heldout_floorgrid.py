"""Floor pattern check for a held-out cut: direction and spacing of the floor's joint lines (grout), source vs render.

Region overlap cannot see a floor that fills the frame (scripts/v11_heldout_regions.py), so this measures the
pattern instead: inside the SAM 3 'floor' region (people removed), long straight lines (Hough on Canny),
grouped into two direction families (median direction error) and the mean distance between the two line drawings
(% of width). The gap between joints is not reported: with ~2 joints in view it was unstable (controls failed).
Controls from the source itself: zoom by 4% / 10% (position error), rotate by 5 / 15 degrees (direction error).

  python scripts/v11_heldout_floorgrid.py SRC.mp4 SRC.regions.npz REN.mp4 REN.regions.npz OUT_PREFIX
ASCII paths only (cv2).
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

def floor_mask(path):
    d = np.load(path); ids = {v: int(k) for k, v in json.loads(str(d["id_map"])).items()}
    return d["masks"] == ids["floor"]

def lines(img, floor):
    W = img.shape[1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fl = cv2.erode(floor.astype(np.uint8), k) > 0
    # joints are thin dark valleys: black-hat (closing minus image) picks them, fine texture stays weak after blur
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    kk = max(5, int(0.015 * W) | 1)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (kk, kk)))
    vals = bh[fl]
    t = np.percentile(vals, 93) if vals.size else 255
    e = ((bh > t) & fl).astype(np.uint8) * 255
    L = cv2.HoughLinesP(e, 1, np.pi / 360, threshold=80, minLineLength=int(0.12 * W), maxLineGap=int(0.02 * W))
    return [] if L is None else [tuple(map(float, l[0])) for l in L]

def families(L):
    """two direction families (angles mod 180); per family: direction (deg) and median gap (px)."""
    if len(L) < 2: return []
    ang = np.array([np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180 for x1, y1, x2, y2 in L])
    ln = np.array([np.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in L])
    hist = np.zeros(180)
    for a, w in zip(ang, ln):
        for d in range(-4, 5): hist[int(round(a + d)) % 180] += w * np.exp(-d * d / 8)
    p1 = int(np.argmax(hist)); h2 = hist.copy()
    for d in range(-30, 31): h2[(p1 + d) % 180] = 0
    out = []
    for p in (p1, int(np.argmax(h2))):
        sel = np.abs((ang - p + 90) % 180 - 90) < 6
        if sel.sum() < 2: continue
        th = np.radians(p); nrm = np.array([-np.sin(th), np.cos(th)])
        mid = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for (x1, y1, x2, y2), s in zip(L, sel) if s])
        off = np.sort(mid @ nrm); merged = [off[0]]
        for o in off[1:]:
            if o - merged[-1] > 8: merged.append(o)
        gaps = np.diff(merged)
        out.append({"dir": float(p), "gap": float(np.median(gaps)) if len(gaps) else None, "n": int(sel.sum())})
    return out

def summary(F, M):
    return [families(lines(f, m)) for f, m in zip(F, M)]

def line_pos_err(F1, M1, F2, M2):
    """median over frames of the symmetric mean distance between the two joint-line drawings, as % of width."""
    out = []
    for f1, m1, f2, m2 in zip(F1, M1, F2, M2):
        H, W = f1.shape[:2]; d = []
        for L in (lines(f1, m1), lines(f2, m2)):
            im = np.zeros((H, W), np.uint8)
            for x1, y1, x2, y2 in L: cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
            d.append(im)
        if not (d[0].any() and d[1].any()): continue
        dt = [cv2.distanceTransform((x == 0).astype(np.uint8), cv2.DIST_L2, 3) for x in d]
        out.append(100 * (dt[1][d[0] > 0].mean() + dt[0][d[1] > 0].mean()) / 2 / W)
    return float(np.median(out)) if out else None

def pair(a, b):
    """match families by direction; return mean |direction difference| and mean |log gap ratio|."""
    da, dg = [], []
    for fa, fb in zip(a, b):
        for x in fa:
            if not fb: continue
            y = min(fb, key=lambda y: abs((x["dir"] - y["dir"] + 90) % 180 - 90))
            da.append(abs((x["dir"] - y["dir"] + 90) % 180 - 90))
            if x["gap"] and y["gap"]: dg.append(abs(np.log(y["gap"] / x["gap"])))
    return {"dir_err_deg": float(np.median(da)) if da else None,
            "gap_err_pct": float(100 * (np.exp(np.median(dg)) - 1)) if dg else None, "pairs": len(da)}

def zoom(f, frac, nearest=False):
    H, W = f.shape[:2]; h, w = round(H * (1 - frac)), round(W * (1 - frac))
    return cv2.resize(f[:h, :w], (W, H), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)

def rot(f, deg, nearest=False):
    H, W = f.shape[:2]; R = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
    return cv2.warpAffine(f, R, (W, H), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)

def main():
    ap = argparse.ArgumentParser()
    for a in ("src", "src_reg", "ren", "ren_reg", "out"): ap.add_argument(a)
    a = ap.parse_args()
    S, R = frames(a.src), frames(a.ren); ms, mr = floor_mask(a.src_reg), floor_mask(a.ren_reg)
    n = min(len(S), len(R), len(ms), len(mr)); S, R, ms, mr = S[:n], R[:n], ms[:n], mr[:n]
    fs = summary(S, ms)
    m8 = [m.astype(np.uint8) for m in ms]
    cands = {"render": (R, list(mr))}
    for z in (0.04, 0.10):
        cands[f"zoom_{z}"] = ([zoom(f, z) for f in S], [zoom(m, z, True) > 0 for m in m8])
    for d in (5, 15):
        cands[f"rotate_{d}"] = ([rot(f, d) for f in S], [rot(m, d, True) > 0 for m in m8])
    res = {}
    for k, (F2, M2) in cands.items():
        r = pair(fs, summary(F2, M2)); r.pop("gap_err_pct", None)
        r["line_pos_err_pct_w"] = line_pos_err(S, ms, F2, M2); res[k] = r
    json.dump({"results": res, "src_families": fs}, open(a.out + ".json", "w", encoding="utf-8"), indent=1)
    for k, v in res.items(): print(k.ljust(10), v)
    # picture: middle frame, detected lines drawn
    i = n // 2; pics = []
    for img, m in ((S[i], ms[i]), (R[i], mr[i])):
        v = img.copy()
        for x1, y1, x2, y2 in lines(img, m): cv2.line(v, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        pics.append(v)
    cv2.imwrite(a.out + "_lines.jpg", np.hstack(pics))

if __name__ == "__main__":
    main()
