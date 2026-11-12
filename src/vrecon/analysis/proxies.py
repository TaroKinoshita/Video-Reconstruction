"""Simple stand-ins (proxies) for masked set pieces, fitted to 4RC points in the aligned world (floor z = 0). Pure numpy.

    plane_proxy    a thin box in the points' main plane (stands, signs, walls; may be tilted)
    board_proxy    an upright thin box (a net): upright_box with the thin side kept thin
    box_proxy      an upright box (chairs, tables): upright_box, the side away from the camera deepened

Each returns (centre, quaternion wxyz, size, report). Edges use quantiles of the points, because mask edges pick up
points of what lies behind.
"""

from __future__ import annotations

import numpy as np

from vrecon.analysis.from_4d import _deepen, _yaw_quat, upright_box
from vrecon.scene.convert import matrix_to_quat


def near_band(p: np.ndarray, eye: np.ndarray, rel: float = 0.15) -> np.ndarray:
    """Points whose distance from the camera is within rel of the median: drops what a mask edge picked up from
    behind (or in front of) the thing."""
    d = np.linalg.norm(p - eye, axis=1)
    md = np.median(d)
    return p[np.abs(d - md) < rel * md]


def plane_proxy(p: np.ndarray, lo: float = 0.02, hi: float = 0.98, thickness: float | None = None,
                eye: np.ndarray | None = None):
    """A plane facing the camera (stands, signs, walls): depth along the camera's horizontal view direction is
    regressed on (sideways, height), so a sloped plane comes out tilted back. A free least-squares plane lies
    down instead: the stands are wide and 4RC gives them little depth or height spread. Without eye: free plane."""
    c = p.mean(0)
    if eye is None:
        n = np.linalg.svd((p - c)[:: max(1, len(p) // 200_000)], full_matrices=False)[2][-1]
    else:
        f = c - eye
        f = np.array([f[0], f[1], 0.0]) / np.linalg.norm(f[:2])
        r = np.cross(f, [0.0, 0.0, 1.0])
        X = np.c_[(p - c) @ r, p[:, 2] - c[2], np.ones(len(p))]
        (ka, kb, _), *_ = np.linalg.lstsq(X, (p - c) @ f, rcond=None)
        n = f - ka * r - kb * np.array([0.0, 0.0, 1.0])
        n /= np.linalg.norm(n)
    inl = np.ones(len(p), bool)
    q = p
    if n[2] < 0:
        n = -n
    # in-plane axes: one horizontal (along the plane), one up the slope
    a = np.cross([0.0, 0.0, 1.0], n)
    a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else np.array([1.0, 0.0, 0.0])
    b = np.cross(n, a)
    c = q.mean(0)
    la, lb = (q - c) @ a, (q - c) @ b
    (a0, b0), (a1, b1) = np.quantile(np.c_[la, lb], [lo, hi], axis=0)
    centre = c + a * (a0 + a1) / 2 + b * (b0 + b1) / 2
    ext = (float(a1 - a0), float(b1 - b0))
    t = thickness or 0.01 * max(ext)
    r = np.c_[a, b, n]
    return centre, matrix_to_quat(r.tolist()), (ext[0], ext[1], t), {
        "normal": n.round(4).tolist(), "tilt_deg": float(np.degrees(np.arccos(abs(n[2])))),
        "inliers": float(inl.mean()), "points": int(len(p))}


def board_proxy(p: np.ndarray, thickness: float | None = None):
    centre, yaw, size = upright_box(p)
    t = thickness or 0.01 * size[0]
    return centre, _yaw_quat(yaw), (size[0], t, size[2]), {"points": int(len(p)), "fitted_depth": size[1]}


def box_proxy(p: np.ndarray):
    centre, yaw, size = upright_box(p)
    cxy, size = _deepen(np.asarray(centre[:2]), yaw, size)
    return np.array([cxy[0], cxy[1], centre[2]]), _yaw_quat(yaw), tuple(float(v) for v in size), {"points": int(len(p))}


FITS = {"plane": plane_proxy, "board": board_proxy, "box": box_proxy}

__all__ = ["FITS", "ground_box", "near_band", "seated_figure", "standing_figure", "stepped_mesh", "board_proxy", "box_proxy", "plane_proxy"]


def _ray(r, c, k, u, v):
    d = r @ np.array([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], 1.0])
    return c, d


def ground_box(mask: np.ndarray, r: np.ndarray, c: np.ndarray, k: np.ndarray, depth_ratio: float = 1.0,
               distance: float | None = None):
    """An upright box on the floor (z = 0) from one frame's mask and camera only: the mask's lowest pixels are
    where it touches the floor, so the ray through them meets z = 0 at the foot point; height and width come from
    the mask's top and sides at that distance. Used for small or far things (people, chairs), whose 4RC points
    are few and smeared in depth. Returns (centre, yaw, size) or None (no floor contact visible: mask touches the
    picture's bottom edge, or the ray misses the floor).
    distance: put the foot point at this horizontal distance along the ray instead (things above the floor, e.g.
    spectators in the stands, whose feet ray never meets z = 0); the box then stands at that height."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20 or ys.max() >= mask.shape[0] - 2:
        return None
    bottom = ys >= ys.max() - max(2, 0.03 * (ys.max() - ys.min()))
    u0, v0 = float(np.median(xs[bottom])), float(ys.max()) + 0.5
    o, d = _ray(r, c, k, u0, v0)
    if distance is not None:
        foot = o + d * (distance / np.linalg.norm(d[:2]))
    elif d[2] >= -1e-9:
        return None
    else:
        foot = o + d * (-o[2] / d[2])
    view = foot - o
    fwd = np.array([view[0], view[1], 0.0]) / np.linalg.norm(view[:2])
    side = np.cross(fwd, [0.0, 0.0, 1.0])

    def on_plane(u, v):  # ray hit on the vertical plane through the foot, facing the camera
        o2, d2 = _ray(r, c, k, u, v)
        t = ((foot - o2) @ fwd) / (d2 @ fwd)
        return o2 + t * d2

    top = on_plane(u0, float(ys.min()) - 0.5)[2]
    lo, hi = (on_plane(float(x), v0) @ side for x in np.quantile(xs, [0.02, 0.98]))
    width = abs(hi - lo)
    centre = foot + side * ((lo + hi) / 2 - foot @ side) + fwd * (width * depth_ratio / 2)
    centre[2] = (top + foot[2]) / 2
    top = top - foot[2]
    yaw = float(np.arctan2(side[1], side[0]))
    return centre, yaw, (float(width), float(width * depth_ratio), float(max(top, 1e-3)))


def standing_figure(width: float, depth: float, height: float):
    """A blocky standing person (head, torso, arms, legs) filling a box of this size, centred at the origin
    (z from -height/2 to height/2), facing -y. For people too small or too cut off for a pose estimate: it tells
    a generator "a person stands here" better than a box. Returns (verts [V,3], faces [F,3])."""
    H = height
    sw = min(width, 0.36 * H)  # across the arms
    d = min(depth, 0.16 * H) if depth > 0 else 0.12 * H
    lw, aw = 0.11 * H, 0.08 * H
    parts = [  # (x0, x1, z0, z1, share of the depth); z from the feet
        (-sw / 2 + aw, sw / 2 - aw, 0.47 * H, 0.82 * H, 1.0),       # torso
        (-0.065 * H, 0.065 * H, 0.84 * H, 1.0 * H, 0.9),           # head
        (-sw / 2, -sw / 2 + aw, 0.45 * H, 0.81 * H, 0.7),           # arms
        (sw / 2 - aw, sw / 2, 0.45 * H, 0.81 * H, 0.7),
        (-sw / 2 + aw, -sw / 2 + aw + lw, 0.0, 0.48 * H, 0.8),      # legs
        (sw / 2 - aw - lw, sw / 2 - aw, 0.0, 0.48 * H, 0.8),
    ]
    vs, fs = [], []
    quad = [(0, 1, 2), (0, 2, 3)]
    box_faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for x0, x1, z0, z1, ds in parts:
        y0, y1 = -d * ds / 2, d * ds / 2
        base = len(vs)
        for z in (z0, z1):
            vs += [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        for f in box_faces:
            fs += [(base + f[a], base + f[b], base + f[c]) for a, b, c in quad]
    v = np.array(vs, float)
    v[:, 2] -= H / 2
    return v, np.array(fs, int)


def _boxes(parts, height: float):
    """Mesh of axis-aligned boxes (x0, x1, y0, y1, z0, z1), z from the feet, shifted to centre on the origin."""
    vs, fs = [], []
    box_faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for x0, x1, y0, y1, z0, z1 in parts:
        base = len(vs)
        for z in (z0, z1):
            vs += [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        for f in box_faces:
            fs += [(base + f[0], base + f[1], base + f[2]), (base + f[0], base + f[2], base + f[3])]
    v = np.array(vs, float)
    v[:, 2] -= height / 2
    return v, np.array(fs, int)


def seated_figure(width: float, depth: float, height: float):
    """A blocky seated person (thighs forward along -y, lower legs down) filling a box of this size, centred at the
    origin. For people whose mask is not tall and thin (mask height / width under 2: ball kids, spectators)."""
    H = height
    sw = min(width, 0.42 * H)
    aw, lw = 0.1 * H, 0.13 * H
    d = max(depth, 0.45 * H)
    tb = 0.16 * H  # torso thickness
    y_back, y_front = d / 2, d / 2 - tb
    knee = -d / 2
    parts = [
        (-sw / 2 + aw, sw / 2 - aw, y_front, y_back, 0.40 * H, 0.80 * H),        # torso
        (-0.09 * H, 0.09 * H, y_front, y_back, 0.81 * H, 1.0 * H),             # head
        (-sw / 2, -sw / 2 + aw, y_front, y_back, 0.42 * H, 0.78 * H),           # arms
        (sw / 2 - aw, sw / 2, y_front, y_back, 0.42 * H, 0.78 * H),
        (-sw / 2 + aw, -sw / 2 + aw + lw, knee, y_back, 0.30 * H, 0.42 * H),    # thighs
        (sw / 2 - aw - lw, sw / 2 - aw, knee, y_back, 0.30 * H, 0.42 * H),
        (-sw / 2 + aw, -sw / 2 + aw + lw, knee, knee + lw, 0.0, 0.32 * H),      # lower legs
        (sw / 2 - aw - lw, sw / 2 - aw, knee, knee + lw, 0.0, 0.32 * H),
    ]
    return _boxes(parts, H)


def stepped_mesh(rotation_matrix: np.ndarray, size, rise: float):
    """Stands as steps: a sawtooth (vertical riser, then tread) whose average is the fitted plane, in the object's
    local axes (a = across, b = up the slope, n = normal, as plane_proxy sets them), extruded across a.
    rise: wanted height of one row (the count is rounded). Returns (verts, faces, rows)."""
    r = np.asarray(rotation_matrix, float)
    W, L = float(size[0]), float(size[1])
    b = r[:, 1]
    height = abs(L * b[2])
    rows = max(3, int(round(height / rise))) if rise > 0 else 3
    step = L / rows
    up = np.array([0.0, 0.0, 1.0]) * step * b[2]
    pts = []
    for i in range(rows):
        s0 = (-L / 2 + i * step) * b
        pts += [s0, s0 + up]
    pts.append((L / 2) * b)
    loc = np.array(pts) @ r  # world offsets -> local (a, b, n)
    loc[:, 2] -= loc[:, 2].mean()  # sawtooth centred on the plane
    vs = []
    for x in (-W / 2, W / 2):
        q = loc.copy()
        q[:, 0] = x
        vs.append(q)
    v = np.concatenate(vs)
    k = len(loc)
    f = []
    for i in range(k - 1):
        f += [(i, i + 1, k + i + 1), (i, k + i + 1, k + i)]
    return v, np.array(f, int), rows
