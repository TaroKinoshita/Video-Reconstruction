"""Diagnostic pictures for the agent, ray cast in numpy (no Blender): primitives shaded by class, the kept ground,
people as points, the backdrop as points, the camera path; names written on the larger primitives.

views: "source" (the shot's camera at a frame, beside the film frame), "orbit" (around the scene, azimuth /
elevation in degrees), "top" (straight down, orthographic).
"""
from __future__ import annotations

import cv2
import numpy as np

from vrecon.agent import prims

CLASS_RGB = {"trunk": (120, 85, 55), "post": (150, 110, 70), "beam": (170, 130, 80), "foliage": (60, 120, 50),
             "wall": (180, 180, 190), "support": (150, 140, 120), "floor": (150, 140, 120), "ceiling": (200, 200, 210),
             "rock": (130, 130, 125), "prop": (200, 160, 60)}
PEOPLE_RGB = [(230, 70, 60), (60, 200, 80), (70, 110, 240), (230, 200, 50), (200, 80, 200), (80, 200, 200)]


class View:
    def __init__(self, R: np.ndarray, T: np.ndarray, f: float, w: int, h: int, ortho: float | None = None):
        self.R, self.T, self.f, self.w, self.h, self.ortho = R, T, f, w, h, ortho

    def project(self, X: np.ndarray):
        c = (X - self.T) @ self.R
        if self.ortho:
            return c[:, 0] / self.ortho + self.w / 2, c[:, 1] / self.ortho + self.h / 2, c[:, 2]
        z = np.maximum(c[:, 2], 1e-6)
        return self.f * c[:, 0] / z + self.w / 2, self.f * c[:, 1] / z + self.h / 2, c[:, 2]

    def rays(self, us, vs):
        if self.ortho:
            o = np.stack([(us + 0.5 - self.w / 2) * self.ortho, (vs + 0.5 - self.h / 2) * self.ortho,
                          np.zeros_like(us)], 1) @ self.R.T + self.T
            return o, np.tile(self.R[:, 2], (len(us), 1))
        d = np.stack([(us + 0.5 - self.w / 2) / self.f, (vs + 0.5 - self.h / 2) / self.f, np.ones_like(us)], 1)
        return self.T, d @ self.R.T

    def box(self, p):
        u, v, z = self.project(prims.corners(p))
        if (z <= 0.05).all():
            return None
        if (z <= 0.05).any():
            return 0, 0, self.w, self.h
        u0, u1 = int(max(np.floor(u.min()), 0)), int(min(np.ceil(u.max()) + 1, self.w))
        v0, v1 = int(max(np.floor(v.min()), 0)), int(min(np.ceil(v.max()) + 1, self.h))
        return None if u0 >= u1 or v0 >= v1 else (u0, v0, u1, v1)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)  # OpenCV: x right, y down, z forward


def _normal(p, X):
    e = 0.01
    g = np.stack([prims.sdf(p, X + d) - prims.sdf(p, X - d) for d in np.eye(3) * e], 1)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)


def draw(view: View, scene: dict, ground=None, people_pts=None, backdrop=None, cam_path=None, cam_now=None,
         label_names=(), max_labels=14, far=120.0, hide=()) -> np.ndarray:
    w, h = view.w, view.h
    zb = np.full((h, w), np.inf)
    img = np.zeros((h, w, 3), float)
    img[:] = (150, 175, 200)  # sky
    light = np.array([0.4, -0.3, 0.85])
    light /= np.linalg.norm(light)
    area = {}
    eye = view.T if not view.ortho else None
    for p in scene["primitives"]:
        if p.get("class") in hide or (eye is not None and prims.sdf(p, eye[None])[0] < 0):
            continue  # hidden class, or the viewpoint is inside it
        b = view.box(p)
        if b is None:
            continue
        u0, v0, u1, v1 = b
        vs, us = np.mgrid[v0:v1, u0:u1]
        us, vs = us.ravel().astype(float), vs.ravel().astype(float)
        o, d = view.rays(us, vs)
        t = prims.intersect(p, o, d)
        ui, vi = us.astype(int), vs.astype(int)
        hit = np.isfinite(t) & (t < zb[vi, ui])
        if not hit.any():
            continue
        oo = o[hit] if o.ndim == 2 else o
        X = oo + t[hit, None] * d[hit]
        n = _normal(p, X)
        shade = 0.45 + 0.55 * np.clip(n @ light, 0, 1)
        rgb = np.array(CLASS_RGB.get(p.get("class"), (170, 170, 170)), float)
        if p.get("imagined"):
            rgb = rgb * 0.7 + 40
        zb[vi[hit], ui[hit]] = t[hit]
        img[vi[hit], ui[hit]] = rgb[None] * shade[:, None]
        area[p["name"]] = (int(hit.sum()), float(us[hit].mean()), float(vs[hit].mean()))
    if ground is not None:
        vs, us = np.mgrid[0:h, 0:w]
        us, vs = us.ravel().astype(float), vs.ravel().astype(float)
        o, d = view.rays(us, vs)
        if view.ortho:
            X = o.copy()
            X[:, 2] = ground(o[:, 0], o[:, 1])
            t = (o[:, 2] - X[:, 2]) / np.maximum(-d[:, 2], 1e-6)
        else:
            ts = np.geomspace(0.3, far, 220)
            t = np.full(len(d), np.inf)
            for a in range(0, len(d), 6000):  # in chunks: rays x steps
                P = o[None] + ts[:, None, None] * d[None, a:a + 6000]
                below = P[..., 2] < ground(P[..., 0], P[..., 1])
                first = np.where(below.any(0), below.argmax(0), -1)
                t[a:a + 6000] = np.where(first >= 0, ts[np.maximum(first, 0)], np.inf)
        ui, vi = us.astype(int), vs.astype(int)
        hit = np.isfinite(t) & (t < zb[vi, ui])
        X = (o[hit] if o.ndim == 2 else o) + t[hit, None] * d[hit]
        e = 0.3
        gx = ground(X[:, 0] + e, X[:, 1]) - ground(X[:, 0] - e, X[:, 1])
        gy = ground(X[:, 0], X[:, 1] + e) - ground(X[:, 0], X[:, 1] - e)
        n = np.stack([-gx, -gy, np.full(len(X), 2 * e)], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        shade = 0.5 + 0.5 * np.clip(n @ light, 0, 1)
        checker = ((np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2) * 12
        zb[vi[hit], ui[hit]] = t[hit]
        img[vi[hit], ui[hit]] = (np.array([120, 110, 85.0])[None] + checker[:, None]) * shade[:, None]

    def splat(X, rgb, r=1):
        u, v, z = view.project(X)
        ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z = u[ok].astype(int), v[ok].astype(int), z[ok]
        vis = z < zb[v, u] + 0.05
        for du in range(-r + 1, r):
            for dv in range(-r + 1, r):
                uu, vv = np.clip(u[vis] + du, 0, w - 1), np.clip(v[vis] + dv, 0, h - 1)
                img[vv, uu] = rgb

    if backdrop is not None:
        splat(backdrop, (95, 120, 90))
    for k, X in enumerate(people_pts or []):
        if X is not None:
            splat(X, PEOPLE_RGB[k % len(PEOPLE_RGB)], r=2)
    out = np.clip(img, 0, 255).astype(np.uint8)
    if cam_path is not None and len(cam_path) > 1:
        u, v, z = view.project(cam_path)
        pts = np.stack([u, v], 1)[z > 0.05].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(out, [pts], False, (255, 230, 0), 1, cv2.LINE_AA)
    if cam_now is not None:
        u, v, z = view.project(cam_now[None])
        if z[0] > 0.05:
            cv2.circle(out, (int(u[0]), int(v[0])), 4, (255, 230, 0), -1)
    names = sorted(area, key=lambda n: -area[n][0])[:max_labels]
    names += [n for n in label_names if n in area and n not in names]
    for n in names:
        _, cu, cv = area[n]
        col = (255, 80, 80) if n in label_names else (255, 255, 255)
        (tw, th), _ = cv2.getTextSize(n, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
        x0, y0 = int(np.clip(cu - tw / 2, 0, w - tw - 2)), int(np.clip(cv, th + 2, h - 2))
        cv2.rectangle(out, (x0 - 1, y0 - th - 2), (x0 + tw + 1, y0 + 2), (0, 0, 0), -1)
        cv2.putText(out, n, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1, cv2.LINE_AA)
    return out
