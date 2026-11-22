"""The clay mannequin's geometry, pure numpy (no bpy): shared by the Blender builder (scene/multishot.py,
scene/character.py) and the close-up camera search, so the search fits exactly the shape that is drawn."""

from __future__ import annotations

import math

import numpy as np

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
         "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
RADIUS = {"spine": 0.11, "pelvis": 0.11, "hip": 0.08, "knee": 0.06, "ankle": 0.045, "foot": 0.04, "neck": 0.05,
          "collar": 0.05, "shoulder": 0.05, "elbow": 0.04, "wrist": 0.035, "head": 0.05}


def _radius(name: str) -> float:
    return next((r for k, r in RADIUS.items() if k in name), 0.05)


def _cylinder(a: np.ndarray, b: np.ndarray, r: float, n: int = 8):
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return [], []
    z = axis / length
    x = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ring = [x * math.cos(2 * math.pi * i / n) * r + y * math.sin(2 * math.pi * i / n) * r for i in range(n)]
    verts = [tuple(a + v) for v in ring] + [tuple(b + v) for v in ring]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    return verts, faces


def _box(c: np.ndarray, size: tuple[float, float, float]):
    sx, sy, sz = (s / 2 for s in size)
    verts = [tuple(c + np.array([x * sx, y * sy, z * sz])) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return verts, faces



def frustum(a: np.ndarray, b: np.ndarray, ra: float, rb: float, n: int = 16):
    """closed cone frustum from a (radius ra) to b (radius rb)."""
    import math

    axis = b - a
    L = max(float(np.linalg.norm(axis)), 1e-6)
    zz = axis / L
    x = np.cross(zz, [0, 0, 1.0]) if abs(zz[2]) < 0.9 else np.cross(zz, [1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(zz, x)
    ring = [x * math.cos(2 * math.pi * i / n) + y * math.sin(2 * math.pi * i / n) for i in range(n)]
    verts = [tuple(a + r * ra) for r in ring] + [tuple(b + r * rb) for r in ring]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    return verts, faces


def mannequin(j: np.ndarray, garment=None):
    """one closed cylinder per bone (radius by joint, as scene/character.py), a box head; fixed topology."""
    verts, faces = [], []
    for i, p in enumerate(PARENTS):
        if p < 0:
            continue
        a, b = j[p], j[i]
        if np.linalg.norm(b - a) < 1e-4:
            b = a + np.array([0, 0, 1e-3])
        v, f = _cylinder(a, b, _radius(NAMES[p]) * 1.15)
        o = len(verts)
        verts += v
        faces += [tuple(x + o for x in ff) for ff in f]
    v, f = _box(j[15] + (j[15] - j[12]) * 0.6, (0.17, 0.2, 0.24))
    o = len(verts)
    verts += v
    faces += [tuple(x + o for x in ff) for ff in f]
    if garment is not None and garment[0] > 0:  # loose clothing: waist down, flared (multishot_clothing.py)
        top = j[3]
        bottom = top + (0.5 * (j[7] + j[8]) - top) * float(garment[1])
        v, f = frustum(top, bottom, 0.17, float(garment[0]))
        o = len(verts)
        verts += v
        faces += [tuple(x + o for x in ff) for ff in f]
    return np.array(verts), faces



def silhouette(joints: np.ndarray, cam: np.ndarray, lens: float, w: int, h: int, garment=None,
               near: float = 0.02, mesh=None) -> np.ndarray:
    """the mannequin's silhouette from a Blender-axes camera (4x4 c2w, horizontal 36 mm sensor): every face filled.
    Faces reaching behind the near plane are clipped there (close-ups put the camera inside the reach of an arm)."""
    import cv2

    v, faces = mesh if mesh is not None else mannequin(joints, garment)
    w2c = np.linalg.inv(cam)
    pc = v @ w2c[:3, :3].T + w2c[:3, 3]
    f = lens / 36.0 * w
    sil = np.zeros((h, w), np.uint8)
    for face in faces:
        poly = pc[list(face)]
        z = -poly[:, 2]
        if (z < near).all():
            continue
        if (z < near).any():  # clip the polygon against z = near
            out = []
            for i in range(len(poly)):
                p, q = poly[i], poly[(i + 1) % len(poly)]
                zp, zq = -p[2], -q[2]
                if zp >= near:
                    out.append(p)
                if (zp >= near) != (zq >= near):
                    t = (near - zp) / (zq - zp)
                    out.append(p + t * (q - p))
            poly = np.array(out)
            z = -poly[:, 2]
        uv = np.stack([w / 2 + f * poly[:, 0] / z, h / 2 - f * poly[:, 1] / z], 1)
        uv = np.clip(uv, -4 * w, 5 * w)
        cv2.fillPoly(sil, [np.round(uv * 4).astype(np.int32)], 1, shift=2)
    return sil.astype(bool)
