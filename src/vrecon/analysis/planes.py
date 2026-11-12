"""Split leftover background points into flat pieces (walls, signs, ledges), so the scene is made of named parts
that rules can extend, instead of a mesh glued to the source view (which tears when the camera moves). Pure numpy.

split_planes: repeated RANSAC (from_4d.fit_plane) on the points not yet taken; each plane's inliers are split into
connected patches on a grid in the plane (two coplanar signs stay two pieces); patches large enough become a
thin box (centre, rotation, size). What no plane takes is returned as the residual (kept only as a reference).
"""

from __future__ import annotations

import numpy as np

from vrecon.analysis.from_4d import fit_plane
from vrecon.scene.convert import matrix_to_quat


def _components(occ: np.ndarray) -> np.ndarray:
    """4-connected labels of a boolean grid (0 = empty)."""
    lab = np.zeros(occ.shape, int)
    n = 0
    for i, j in zip(*np.nonzero(occ)):
        if lab[i, j]:
            continue
        n += 1
        stack = [(i, j)]
        lab[i, j] = n
        while stack:
            a, b = stack.pop()
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, y = a + da, b + db
                if 0 <= x < occ.shape[0] and 0 <= y < occ.shape[1] and occ[x, y] and not lab[x, y]:
                    lab[x, y] = n
                    stack.append((x, y))
    return lab


def fit_vertical_plane(p: np.ndarray, tol: float, iters: int = 400, seed: int = 0):
    """RANSAC for a vertical plane (wall-like): two points plus the up axis. Returns (unit normal, d)."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        i, j = rng.choice(len(p), 2, replace=False)
        h = p[j, :2] - p[i, :2]
        if np.linalg.norm(h) < 1e-9:
            continue
        n = np.array([-h[1], h[0], 0.0]) / np.linalg.norm(h)
        cnt = int((np.abs((p - p[i]) @ n) < tol).sum())
        if cnt > best_n:
            best, best_n = (n, -p[i] @ n), cnt
    n, d = best
    inl = np.abs(p @ n + d) < tol
    # refine: horizontal least squares line through the inliers' (x, y)
    q = p[inl, :2]
    c = q.mean(0)
    v = np.linalg.svd(q - c, full_matrices=False)[2][0]
    n = np.array([-v[1], v[0], 0.0])
    return n, -np.r_[c, 0.0] @ n


def plane_axes(n: np.ndarray):
    """In-plane axes: a horizontal, b = n x a (up the plane for a standing plane)."""
    if n[2] < 0:
        n = -n
    a = np.cross([0.0, 0.0, 1.0], n)
    a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else np.array([1.0, 0.0, 0.0])
    return n, a, np.cross(n, a)


def split_planes(p: np.ndarray, tol: float, cell: float, min_points: int = 150, max_planes: int = 12,
                 thickness: float | None = None, seed: int = 0, vertical: bool = True):
    """Returns (pieces, residual mask). Each piece: dict(centre, rotation (wxyz), size, normal, points, index).
    vertical: walls and signs only. Free planes on the tennis background were mostly horizontal slices through a
    wall that 4RC spreads in depth (2026-11-13)."""
    free = np.ones(len(p), bool)
    pieces = []
    for k in range(max_planes):
        idx = np.nonzero(free)[0]
        if len(idx) < min_points:
            break
        n, d = fit_vertical_plane(p[idx], tol, seed=seed + k) if vertical else fit_plane(p[idx], tol=tol, seed=seed + k)[:2]
        inl = idx[np.abs(p[idx] @ n + d) < tol]
        if len(inl) < min_points:
            break
        free[inl] = False  # taken by this plane whether or not a patch is kept (otherwise RANSAC finds it again)
        n, a, b = plane_axes(n)
        uv = np.c_[p[inl] @ a, p[inl] @ b]
        g = np.floor((uv - uv.min(0)) / cell).astype(int)
        occ = np.zeros(g.max(0) + 1, bool)
        occ[g[:, 0], g[:, 1]] = True
        lab = _components(occ)[g[:, 0], g[:, 1]]
        for c in np.unique(lab):
            sel = inl[lab == c]
            if len(sel) < min_points:
                free[sel] = True  # small patch: back to the pool (may belong to a later plane), else residual
                continue
            q = p[sel]
            ctr = q.mean(0)
            la, lb = (q - ctr) @ a, (q - ctr) @ b
            (a0, b0), (a1, b1) = np.quantile(np.c_[la, lb], [0.02, 0.98], axis=0)
            centre = ctr + a * (a0 + a1) / 2 + b * (b0 + b1) / 2
            ext = (float(a1 - a0), float(b1 - b0))
            pieces.append({"centre": centre, "rotation": matrix_to_quat(np.c_[a, b, n].tolist()),
                           "size": (ext[0], ext[1], thickness or tol), "normal": n, "points": int(len(sel)),
                           "index": sel, "tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(n[2])))))})
    pieces = _merge_coplanar(p, pieces, tol, thickness)
    taken = np.zeros(len(p), bool)
    for pc in pieces:
        taken[pc["index"]] = True
    return pieces, ~taken


def _piece(p: np.ndarray, sel: np.ndarray, n: np.ndarray, tol: float, thickness: float | None) -> dict:
    n, a, b = plane_axes(n)
    q = p[sel]
    ctr = q.mean(0)
    la, lb = (q - ctr) @ a, (q - ctr) @ b
    (a0, b0), (a1, b1) = np.quantile(np.c_[la, lb], [0.02, 0.98], axis=0)
    return {"centre": ctr + a * (a0 + a1) / 2 + b * (b0 + b1) / 2, "rotation": matrix_to_quat(np.c_[a, b, n].tolist()),
            "size": (float(a1 - a0), float(b1 - b0), thickness or tol), "normal": n, "points": int(len(sel)),
            "index": sel, "tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(n[2])))))}


def _merge_coplanar(p: np.ndarray, pieces: list[dict], tol: float, thickness: float | None, max_deg: float = 8.0) -> list[dict]:
    """Pieces of one wall found by different RANSAC rounds (same plane, a gap between) are merged: two coplanar
    boxes z-fight wherever an extension makes them overlap (tennis, 2026-11-13)."""
    merged = True
    while merged:
        merged = False
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                ni, nj = pieces[i]["normal"], pieces[j]["normal"]
                if abs(ni @ nj) < np.cos(np.radians(max_deg)):
                    continue
                if abs((pieces[j]["centre"] - pieces[i]["centre"]) @ ni) > 2 * tol:
                    continue
                sel = np.r_[pieces[i]["index"], pieces[j]["index"]]
                n = ni * pieces[i]["points"] + np.sign(ni @ nj) * nj * pieces[j]["points"]
                pieces[i] = _piece(p, sel, n / np.linalg.norm(n), tol, thickness)
                pieces.pop(j)
                merged = True
                break
            if merged:
                break
    return pieces


__all__ = ["plane_axes", "split_planes"]
