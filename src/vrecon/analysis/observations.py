"""What the source view says about the venue, in court metres: a short summary per observed thing, the input
for laying out a venue (by hand or by an LLM) and for fitting the layout's numbers. Pure numpy.

summarize(points, labels, names, camera_centre) -> {"camera": ..., "things": {name: summary}}
    per thing (points within 15% of the median distance from the camera: mask edges bleed): points, centroid, extent (5th-95th percentile box), z range, distance range from the court centre,
    and for large things a horizontal line fit (direction, the front edge nearest the court centre, its ends).
Names are the segmenter's labels (obj_net, obj_stands, obs_0 ...); they are hints, not categories.
"""

from __future__ import annotations

import numpy as np


def _line(p2: np.ndarray):
    c = p2.mean(0)
    d = np.linalg.svd(p2 - c, full_matrices=False)[2][0]
    s = (p2 - c) @ d
    a, b = np.percentile(s, [3, 97])
    return c, d, c + a * d, c + b * d


def summarize(points: np.ndarray, labels: np.ndarray, names: list[str], camera_centre) -> dict:
    out = {"frame": "court: metres, origin at the court centre, X across, Y along (away from the camera is +Y), Z up",
           "camera_centre": [round(float(v), 2) for v in camera_centre], "things": {}}
    cam = np.asarray(camera_centre, float)
    for i, nm in enumerate(names):
        p = points[labels == i]
        if len(p) < 30:
            continue
        # mask edges pick up points of what lies behind or in front: keep the band around the median distance
        # from the camera (as analysis/proxies.near_band)
        dc = np.linalg.norm(p - cam, axis=1)
        med = np.median(dc)
        q = p[np.abs(dc - med) < 0.15 * med]
        p = q if len(q) >= 30 else p
        lo, hi = np.percentile(p, [5, 95], axis=0)
        r = np.linalg.norm(p[:, :2], axis=1)
        s = {"points": int(len(p)), "centroid": np.round(p.mean(0), 2).tolist(),
             "extent_min": np.round(lo, 2).tolist(), "extent_max": np.round(hi, 2).tolist(),
             "z_range": [round(float(np.percentile(p[:, 2], 2)), 2), round(float(np.percentile(p[:, 2], 98)), 2)],
             "distance_from_centre": [round(float(np.percentile(r, 5)), 2), round(float(np.percentile(r, 95)), 2)]}
        if np.ptp(p[:, :2], 0).max() > 2.0:
            c, d, e0, e1 = _line(p[:, :2])
            n = np.array([-d[1], d[0]])
            if n @ c < 0:
                n = -n  # pointing away from the court centre
            front = float(np.percentile((p[:, :2] - c) @ n, 5))
            s["line"] = {"direction": np.round(d, 3).tolist(), "outward_normal": np.round(n, 3).tolist(),
                         "front_edge_from": np.round(e0 + front * n, 2).tolist(), "front_edge_to": np.round(e1 + front * n, 2).tolist(),
                         "depth_behind_front": round(float(np.percentile((p[:, :2] - c) @ n, 95) - front), 2)}
        out["things"][nm] = s
    return out


__all__ = ["summarize"]
