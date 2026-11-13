"""Phase B: calibrate a tennis clip's camera from the court lines (analysis/court.py), check it, write it.

    python scripts/calibrate_court.py VIDEO MASKS.npz OUT_DIR [--step 2] [--overlays 0,60,120,180,240,300]
        [--top 250] [--points points.json]

1. First frame: white line pixels (masked things removed), long lines, best match to the ITF court (two lines
   across, two along; pixels of court lines landing on white). Of a match and its 180 deg twin, the camera behind
   the near baseline is kept. Manual mode (--points): a JSON list of {"frame": f, "px": [u, v], "court": [X, Y]
   or [X, Y, Z]} replaces the automatic match for the frames it covers (click intersections).
2. Every --step frames: detected lines matched to the previous frame's projected court lines; their
   intersections give a camera (free: focal, rotation, centre).
3. A broadcast camera stays in place and pans (and may zoom): the centre is the median of the free centres and
   every frame is refitted with it. If the per-frame focal lengths show no trend (drift over the clip < 2% and
   scatter < 3%), one focal length (their median) is used and only the rotation changes per frame (tennis: no
   zoom, a 0.9 deg pan; the free fits had traded focal length against distance, 2065-2209 px).
Checks (calibration.json): per frame the share of projected court-line pixels on white (support), reprojection
error, focal length; overlays (court lines red, net top yellow) for --overlays frames.
Writes OUT_DIR/calibration.json: per frame {frame, focal_px, R, t} (OpenCV axes, court frame in metres) and the
centre; overlay_XXXXXX.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("masks")
    ap.add_argument("out")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--overlays", default="0,60,120,180,240,300")
    ap.add_argument("--top", type=int, default=250, help="ignore rows above this (stands, signs)")
    ap.add_argument("--points", default=None)
    a = ap.parse_args()
    import cv2

    from vrecon.analysis.court import (COURT_LINES, NET_CENTRE_H, NET_POST_H, NET_POST_X, camera_fixed_centre,
                                       camera_from_points, detect_lines, line_support, match_court, match_near,
                                       project, white_lines)
    from vrecon.control.core import read_video_rgb

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    video = read_video_rgb(a.video)
    t_n, h, w = video.shape[:3]
    masks = np.load(a.masks)["masks"]
    manual = json.loads(Path(a.points).read_text(encoding="utf-8")) if a.points else []

    def prep(f):
        m = masks[f] if masks.shape[1:] == (h, w) else cv2.resize(masks[f], (w, h), interpolation=cv2.INTER_NEAREST)
        ex = cv2.dilate((m > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        valid = ~ex
        valid[:a.top] = False
        return white_lines(video[f], ex, top=a.top), valid

    def hom(K, R, t):
        return K @ np.c_[R[:, 0], R[:, 1], t]

    frames = sorted(set(range(0, t_n, a.step)) | {t_n - 1})
    free, how = {}, {}
    H_prev = None
    for f in frames:
        white, valid = prep(f)
        pts = [p for p in manual if p["frame"] == f]
        if len(pts) >= 4:
            img, court, how[f] = np.array([p["px"] for p in pts], float), np.array([p["court"] for p in pts], float), "manual"
        elif H_prev is None:
            best, cands = match_court(detect_lines(white), white, w, h, valid=valid)
            if best is None:
                continue
            top = [c for c in cands if c["score"] >= best["score"] - 1e-6]
            best = min(top, key=lambda c: c["centre_from_h"][1])
            img, court, how[f] = best["img"], best["court"], "auto match " + json.dumps(best["assign"])
        else:
            img, court, names = match_near(detect_lines(white), H_prev, w, h)
            how[f] = "tracked " + ",".join(names)
            if len(img) < 4:
                continue
        cam = camera_from_points(img, court, w, h)
        if cam is None:
            continue
        K, R, t, fo, err = cam
        free[f] = {"img": img, "court": court, "centre": -R.T @ t, "f": fo, "err": err}
        H_prev = hom(K, R, t)

    centres = np.array([v["centre"] for v in free.values()])
    C = np.median(centres, 0)
    res = {"court_model": "ITF doubles 23.77 x 10.97 m, origin at the centre, X across, Y along, Z up (metres)",
           "centre_m": C.round(4).tolist(), "free_centre_spread_m": float(np.percentile(np.linalg.norm(centres - C, axis=1), 90)),
           "width": w, "height": h, "frames": []}
    overlays = {int(x) for x in a.overlays.split(",")}
    fits = {f: camera_fixed_centre(v["img"], v["court"], C, w, h) for f, v in free.items()}
    fn = np.array(list(fits))
    fo_all = np.array([x[3] for x in fits.values()])
    slope = np.polyfit(fn, fo_all, 1)[0] if len(fn) > 2 else 0.0
    drift = abs(slope * (fn.max() - fn.min())) / np.median(fo_all)
    scatter = float(np.std(fo_all - np.polyval(np.polyfit(fn, fo_all, 1), fn)) / np.median(fo_all)) if len(fn) > 2 else 0.0
    zoom = bool(drift > 0.02 or scatter > 0.03)
    res["focal"] = {"zoom": zoom, "drift_share": round(float(drift), 4), "scatter_share": round(scatter, 4),
                    "median_px": round(float(np.median(fo_all)), 2)}
    for f, v in free.items():
        K, R, t, fo, err = (fits[f] if zoom else
                            camera_fixed_centre(v["img"], v["court"], C, w, h, focal=float(np.median(fo_all))))
        white, valid = prep(f)
        sup = line_support(hom(K, R, t), white, valid=valid)
        inside = [(s, n) for s, n in sup.values() if s is not None and n > 20]
        support = float(sum(s * n for s, n in inside) / max(sum(n for _, n in inside), 1))
        res["frames"].append({"frame": f, "how": how[f], "focal_px": round(fo, 2), "reprojection_px": round(err, 2),
                              "free_focal_px": round(v["f"], 1), "support": round(support, 3), "points": len(v["img"]),
                              "R": R.tolist(), "t": t.tolist()})
        if f in overlays:
            o = video[f].copy()
            for nm, (p0, p1) in COURT_LINES.items():
                X = np.c_[np.linspace(p0[0], p1[0], 400), np.linspace(p0[1], p1[1], 400), np.zeros(400)]
                X = X[(X @ R.T + t)[:, 2] > 0.1]
                uv = project(K, R, t, X)
                for p, q in zip(uv[:-1], uv[1:]):
                    cv2.line(o, tuple(map(int, p)), tuple(map(int, q)), (255, 40, 40), 1)
            xs = np.linspace(-NET_POST_X, NET_POST_X, 60)
            net = np.c_[xs, np.zeros(60), NET_CENTRE_H + (NET_POST_H - NET_CENTRE_H) * (xs / NET_POST_X) ** 2]
            uv = project(K, R, t, net)
            for p, q in zip(uv[:-1], uv[1:]):
                cv2.line(o, tuple(map(int, p)), tuple(map(int, q)), (255, 230, 0), 2)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
            buf.tofile(out / f"overlay_{f:06d}.png")
    fr = res["frames"]
    res["summary"] = {"frames_calibrated": len(fr), "frames_tried": len(frames),
                      "support_min": min(x["support"] for x in fr), "support_median": float(np.median([x["support"] for x in fr])),
                      "reprojection_px_max": max(x["reprojection_px"] for x in fr),
                      "focal_px_range": [min(x["focal_px"] for x in fr), max(x["focal_px"] for x in fr)]}
    (out / "calibration.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"centre_m": res["centre_m"], "free_centre_spread_m": round(res["free_centre_spread_m"], 3),
                      **res["summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
