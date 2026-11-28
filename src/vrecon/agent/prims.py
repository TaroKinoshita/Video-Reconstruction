"""Editable primitives as in MAVE eq. 8: type, centre, dimensions, orientation, class, name.

A primitive is a dict:
  {"name": "Trunk_000", "type": "box|cylinder|plane|ellipsoid", "center": [x, y, z], "size": [sx, sy, sz],
   "rotation": [rx, ry, rz], "class": "trunk", "colour": [r, g, b], "imagined": false}
World axes: z up, metres. size = full extents in the primitive's own axes (cylinder: diameter x, diameter y,
height along its own z; plane: sz is ignored and taken as PLANE_T). rotation = Blender XYZ Euler in degrees
(R = Rz @ Ry @ Rx). colour = linear RGB, used only for rendering.
"""
from __future__ import annotations

import numpy as np

TYPES = ("box", "cylinder", "plane", "ellipsoid")
PLANE_T = 0.02


def rot(p: dict) -> np.ndarray:
    rx, ry, rz = np.radians(p.get("rotation", [0, 0, 0]))
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def euler_from(R: np.ndarray) -> list[float]:
    """Blender XYZ Euler (degrees) of a rotation matrix."""
    ry = np.arcsin(np.clip(-R[2, 0], -1, 1))
    if abs(R[2, 0]) < 0.9999:
        rx, rz = np.arctan2(R[2, 1], R[2, 2]), np.arctan2(R[1, 0], R[0, 0])
    else:
        rx, rz = np.arctan2(-R[1, 2], R[1, 1]), 0.0
    return [float(np.degrees(v)) for v in (rx, ry, rz)]


def half(p: dict) -> np.ndarray:
    h = np.abs(np.asarray(p["size"], float)) / 2
    if p["type"] == "plane":
        h[2] = PLANE_T / 2
    return np.maximum(h, 1e-3)


def corners(p: dict) -> np.ndarray:
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    return (s * half(p)) @ rot(p).T + np.asarray(p["center"], float)


def aabb(p: dict) -> tuple[np.ndarray, np.ndarray]:
    c = corners(p)
    return c.min(0), c.max(0)


def intersect(p: dict, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Ray parameter t of the first hit (inf where missed) for rays o + t d (o [3] or [n, 3], d [n, 3]).
    A ray starting inside the primitive hits at t = 0."""
    R, h, c = rot(p), half(p), np.asarray(p["center"], float)
    ol = ((np.broadcast_to(o, d.shape) - c) @ R) / h
    dl = (d @ R) / h
    t = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        if p["type"] in ("box", "plane"):
            t1, t2 = (-1 - ol) / dl, (1 - ol) / dl
            tn = np.nanmax(np.minimum(t1, t2), 1)
            tf = np.nanmin(np.maximum(t1, t2), 1)
        elif p["type"] == "cylinder":
            a = dl[:, 0] ** 2 + dl[:, 1] ** 2
            b = 2 * (ol[:, 0] * dl[:, 0] + ol[:, 1] * dl[:, 1])
            cc = ol[:, 0] ** 2 + ol[:, 1] ** 2 - 1
            disc = b * b - 4 * a * cc
            sq = np.sqrt(np.maximum(disc, 0))
            r0, r1 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
            par = a < 1e-12
            r0 = np.where(par, np.where(cc <= 0, -np.inf, np.inf), np.where(disc >= 0, r0, np.inf))
            r1 = np.where(par, np.where(cc <= 0, np.inf, -np.inf), np.where(disc >= 0, r1, -np.inf))
            z1, z2 = (-1 - ol[:, 2]) / dl[:, 2], (1 - ol[:, 2]) / dl[:, 2]
            zpar = np.abs(dl[:, 2]) < 1e-12
            zn = np.where(zpar, np.where(np.abs(ol[:, 2]) <= 1, -np.inf, np.inf), np.minimum(z1, z2))
            zf = np.where(zpar, np.where(np.abs(ol[:, 2]) <= 1, np.inf, -np.inf), np.maximum(z1, z2))
            tn, tf = np.maximum(r0, zn), np.minimum(r1, zf)
        else:  # ellipsoid
            a = (dl ** 2).sum(1)
            b = 2 * (ol * dl).sum(1)
            cc = (ol ** 2).sum(1) - 1
            disc = b * b - 4 * a * cc
            sq = np.sqrt(np.maximum(disc, 0))
            tn = np.where(disc >= 0, (-b - sq) / (2 * a), np.inf)
            tf = np.where(disc >= 0, (-b + sq) / (2 * a), -np.inf)
    hit = (tn <= tf) & (tf > 0)
    t[hit] = np.maximum(tn[hit], 0.0)
    return t


def sdf(p: dict, X: np.ndarray) -> np.ndarray:
    """Signed distance (metres, negative inside; exact for boxes, close for cylinders and ellipsoids)."""
    R, h, c = rot(p), half(p), np.asarray(p["center"], float)
    q = (X - c) @ R
    if p["type"] in ("box", "plane"):
        e = np.abs(q) - h
        return np.linalg.norm(np.maximum(e, 0), axis=1) + np.minimum(e.max(1), 0)
    if p["type"] == "cylinder":
        r = np.hypot(q[:, 0] / h[0], q[:, 1] / h[1]) * min(h[0], h[1]) - min(h[0], h[1])
        e = np.stack([r, np.abs(q[:, 2]) - h[2]], 1)
        return np.linalg.norm(np.maximum(e, 0), axis=1) + np.minimum(e.max(1), 0)
    return (np.linalg.norm(q / h, axis=1) - 1) * h.min()


def top_at(p: dict, xy: np.ndarray) -> np.ndarray:
    """Height of the primitive's upper surface above each (x, y) (nan where it is not above it)."""
    o = np.concatenate([xy, np.full((len(xy), 1), 1e4)], 1)
    t = intersect(p, o, np.tile([0.0, 0.0, -1.0], (len(xy), 1)))
    return np.where(np.isfinite(t), 1e4 - t, np.nan)


def from_parts(parts: dict) -> list[dict]:
    """The v11 parts (scripts/v11_parts.py) as primitives, with the names the .blend builder gives them."""
    out = []

    def add(name, typ, c, s, colour, cls, imagined=False, rotation=(0, 0, 0)):
        out.append({"name": name, "type": typ, "center": [float(v) for v in c], "size": [float(v) for v in s],
                    "rotation": [float(v) for v in rotation], "class": cls, "colour": [float(v) for v in colour],
                    "imagined": bool(imagined)})

    for k, t in enumerate(parts["trunks"]):
        b, top = t["base"] - 0.3, t["base"] + t["h"]
        add(f"Trunk_{k:03d}", "cylinder", (t["x"], t["y"], (b + top) / 2), (2 * t["r"], 2 * t["r"], top - b),
            t["colour_lin"], "trunk")
    for k, t in enumerate(parts["posts"]):
        b, top = t["base"] - 0.2, t["base"] + t["h"]
        add(f"Post_{k:03d}", "cylinder", (t["x"], t["y"], (b + top) / 2), (2 * t["r"], 2 * t["r"], top - b),
            t["colour_lin"], "post")
    for k, bm in enumerate(parts["beams"]):
        a, b = np.asarray(bm["a"], float), np.asarray(bm["b"], float)
        z = (b - a) / max(np.linalg.norm(b - a), 1e-6)
        x = np.cross([0, 0, 1.0], z) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z], 1)
        add(f"Beam_{k:03d}", "cylinder", (a + b) / 2, (2 * bm["r"], 2 * bm["r"], np.linalg.norm(b - a)),
            bm["colour_lin"], "beam", rotation=euler_from(R))
    for k, t in enumerate(parts["imagined_trees"]):
        b, top = t["base"] - 0.3, t["base"] + t["h"]
        add(f"ImTrunk_{k:03d}", "cylinder", (t["x"], t["y"], (b + top) / 2), (2 * t["r"], 2 * t["r"], top - b),
            t["colour_lin"], "trunk", imagined=True)
    n_seen = len(parts["trunks"])
    for k, c in enumerate(parts["canopy"]):
        add(f"Canopy_{k:03d}", "ellipsoid", c["c"], 2 * np.asarray(c["size"], float), c["colour_lin"], "foliage",
            imagined=k >= n_seen)
    return out


def check(p: dict) -> list[str]:
    """Problems with one primitive dict (empty when it is well formed)."""
    probs = []
    if p.get("type") not in TYPES:
        probs.append(f"type must be one of {TYPES}")
    for key, n in (("center", 3), ("size", 3)):
        v = p.get(key)
        if not (isinstance(v, (list, tuple)) and len(v) == n and all(isinstance(x, (int, float)) for x in v)):
            probs.append(f"{key} must be {n} numbers")
    r = p.get("rotation", [0, 0, 0])
    if not (isinstance(r, (list, tuple)) and len(r) == 3):
        probs.append("rotation must be 3 numbers (degrees)")
    if not probs and min(abs(float(s)) for s in p["size"][:2]) <= 0:
        probs.append("size must be positive")
    if not isinstance(p.get("name"), str) or not p["name"]:
        probs.append("name must be a non-empty string")
    return probs
