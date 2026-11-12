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

__all__ = ["FITS", "near_band", "board_proxy", "box_proxy", "plane_proxy"]
