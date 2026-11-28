"""Up direction from several cues (MAVE eq. 3):

  u = argmin_{|u|=1}  sum_{a in ground, ceiling} rho(angle(u, n_a)) / s_a^2  +  angle(u, v_cam)^2 / s_cam^2
                      + l_w * sum_w arcsin|u . n_w|  +  l_body * rho(angle(u, v_body))

n_ground / n_ceiling: normals of the support and ceiling surfaces (oriented up), n_w: normals of vertical
surfaces, v_cam: the cameras' up axis, v_body: people's upright direction on frames where they stand. rho is a
robust (Cauchy) penalty. The paper gives no values for the scales; the defaults below are this project's choice.
"""
from __future__ import annotations

import numpy as np

S_GROUND, S_CEIL, S_CAM = np.radians(5), np.radians(5), np.radians(30)
L_WALL, L_BODY, RHO_C = 1.0, 1.0, np.radians(10)


def _rho(a: np.ndarray) -> np.ndarray:
    return RHO_C ** 2 * np.log1p((a / RHO_C) ** 2)


def _ang(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(v @ u, -1, 1))


def cost(u: np.ndarray, n_ground, n_ceil, n_wall, v_cam, v_body) -> float:
    c = 0.0
    if len(n_ground):
        c += _rho(_ang(u, n_ground)).mean() / S_GROUND ** 2
    if len(n_ceil):
        c += _rho(_ang(u, n_ceil)).mean() / S_CEIL ** 2
    if len(v_cam):
        c += (_ang(u, v_cam) ** 2).mean() / S_CAM ** 2
    if len(n_wall):
        c += L_WALL * np.arcsin(np.clip(np.abs(n_wall @ u), 0, 1)).mean() / RHO_C
    if len(v_body):
        c += L_BODY * _rho(_ang(u, v_body)).mean() / RHO_C ** 2
    return float(c)


def estimate(n_ground, n_ceil, n_wall, v_cam, v_body, start=(0, 0, 1.0), cone_deg: float = 30) -> dict:
    """Search the sphere cap of cone_deg around start, coarse to fine. Each cue set is an [n, 3] array of unit
    vectors (ceiling normals pointing up, i.e. flipped from facing the camera)."""
    arrs = [np.asarray(x, float).reshape(-1, 3) for x in (n_ground, n_ceil, n_wall, v_cam, v_body)]
    z = np.asarray(start, float) / np.linalg.norm(start)
    x = np.cross(z, [1.0, 0, 0]) if abs(z[0]) < 0.9 else np.cross(z, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    best, best_c, span = z, cost(z, *arrs), np.radians(cone_deg)
    centre = (0.0, 0.0)
    for _ in range(6):
        g = np.linspace(-span, span, 15)
        for a in g + centre[0]:
            for b in g + centre[1]:
                u = np.cos(np.hypot(a, b)) * z + np.sin(np.hypot(a, b)) * (a * x + b * y) / max(np.hypot(a, b), 1e-12)
                c = cost(u, *arrs)
                if c < best_c:
                    best, best_c, centre = u, c, (a, b)
        span /= 4
    return {"up": best, "cost": best_c, "tilt_from_start_deg": float(np.degrees(np.arccos(np.clip(best @ z, -1, 1))))}


def cues_from(ev: dict, people: dict | None = None, max_pts: int = 20000) -> tuple:
    """The cues from evidence.npz (with Q_n) and bodies {label: {"v": [T, V, 3]}}, in its world."""
    rng = np.random.default_rng(0)
    n, cls = ev["Q_n"].astype(np.float64), ev["Q_cls"]

    def pick(m):
        idx = np.where(m)[0]
        return n[rng.choice(idx, min(len(idx), max_pts), replace=False)] if len(idx) else np.zeros((0, 3))

    ng = pick(cls == 0)
    nc = -pick(cls == 2)  # ceilings face down towards the camera: flipped to point up
    nw = pick(cls == 1)
    vc = -ev["cam_R"][:, :, 1]  # OpenCV y is down
    vb = []
    for P in (people or {}).values():
        V = P["v"]
        ext = V[:, :, 2].max(1) - V[:, :, 2].min(1)
        for j in np.where(ext > np.percentile(ext, 70))[0][::3]:  # the most upright frames
            z = V[j][:, 2]
            lo, hi = V[j][z < np.percentile(z, 10)].mean(0), V[j][z > np.percentile(z, 90)].mean(0)
            d = hi - lo
            vb.append(d / np.linalg.norm(d))
    return ng, nc, nw, vc, np.array(vb).reshape(-1, 3)
