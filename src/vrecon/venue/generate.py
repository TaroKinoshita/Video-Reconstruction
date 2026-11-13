"""Meshes for venue parts, in court metres, from two shared operations. Pure numpy.

    boxes_mesh(boxes)          oriented boxes (centre, half sizes along (u, v, up), u direction) -> one mesh
    rows_along(front, ...)     a stand: rows of a step profile behind each segment of a front line, climbing
                               away from the court centre, split by aisles; also used (one row) for walls
    canopy_along(front, ...)   a roof slab behind a front line at a height
The same operations make stands for any sport or a theatre: only the front line and the numbers change.
"""

from __future__ import annotations

import numpy as np

QUADS = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def boxes_mesh(boxes) -> tuple[np.ndarray, np.ndarray]:
    """boxes: iterable of (centre xyz, (hu, hv, hz), u direction (2D)). Returns (verts, faces)."""
    vs, fs = [], []
    for c, (hu, hv, hz), u in boxes:
        u = np.array([u[0], u[1], 0.0]) / np.hypot(u[0], u[1])
        v = np.array([-u[1], u[0], 0.0])
        z = np.array([0.0, 0.0, 1.0])
        c = np.asarray(c, float)
        b = len(vs)
        for sz in (-1, 1):
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                vs.append(c + su * hu * u + sv * hv * v + sz * hz * z)
        for q in QUADS:
            fs += [(b + q[0], b + q[1], b + q[2]), (b + q[0], b + q[2], b + q[3])]
    return (np.array(vs, float) if vs else np.zeros((0, 3))), (np.array(fs, int) if fs else np.zeros((0, 3), int))


def _outward(p0, p1, centre=(0.0, 0.0)):
    d = np.asarray(p1, float) - np.asarray(p0, float)
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    mid = (np.asarray(p0, float) + np.asarray(p1, float)) / 2
    return n if n @ (mid - np.asarray(centre)) > 0 else -n


def _pieces(p0, p1, aisles):
    """Sub-segments of p0-p1 with aisle gaps (every `every` m, `width` wide)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    if not aisles or aisles.get("every", 0) <= 0 or L <= aisles["every"]:
        return [(p0, p1)]
    n = max(1, int(round(L / aisles["every"])))
    w = float(aisles.get("width", 1.2))
    step = L / n
    d = (p1 - p0) / L
    out = []
    for i in range(n):
        a = i * step + (w / 2 if i > 0 else 0.0)
        b = (i + 1) * step - (w / 2 if i < n - 1 else 0.0)
        if b > a:
            out.append((p0 + a * d, p0 + b * d))
    return out


def rows_along(front, rows: int, row_depth: float, rise: float, base_height: float, aisles=None,
               centre=(0.0, 0.0), colour=None):
    """A stand behind a front line: for each segment, rows of boxes; row i spans i..i+1 row depths behind the
    front (away from `centre`) and rises from the ground to base_height + (i + 1) * rise."""
    boxes = []
    for p0, p1 in zip(front[:-1], front[1:]):
        n = _outward(p0, p1, centre)
        for a, b in _pieces(p0, p1, aisles):
            u = b - a
            hu = np.linalg.norm(u) / 2
            mid = (a + b) / 2
            for i in range(int(rows)):
                top = base_height + (i + 1) * rise
                c2 = mid + n * (i + 0.5) * row_depth
                boxes.append(((c2[0], c2[1], top / 2), (hu, row_depth / 2, top / 2), u))
    v, f = boxes_mesh(boxes)
    if colour is None:
        return v, f
    # seat tops lighter, risers darker: the steps read in flat-colour renders (V2V controls, judges)
    c = np.asarray(colour, float)
    top = np.zeros(len(v), bool)
    for b in range(len(v) // 8):
        top[b * 8 + 4: b * 8 + 8] = True
    cols = np.where(top[:, None], np.clip(c * 1.35, 0, 1), c * 0.55)
    return v, f, cols


def court_lines_boxes(width: float = 0.05):
    """The ITF court's white lines as thin boxes on the ground (2 mm above the court surface)."""
    from vrecon.analysis.court import COURT_LINES

    boxes = []
    for (x0, y0), (x1, y1) in COURT_LINES.values():
        a, b = np.array([x0, y0]), np.array([x1, y1])
        m = (a + b) / 2
        boxes.append(((m[0], m[1], 0.012), (np.linalg.norm(b - a) / 2 + width / 2, width / 2, 0.002), b - a))
    return boxes_mesh(boxes)


def wall_along(line, height: float, thickness: float, centre=(0.0, 0.0)):
    boxes = []
    for p0, p1 in zip(line[:-1], line[1:]):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        n = _outward(p0, p1, centre)
        mid = (p0 + p1) / 2 + n * thickness / 2
        boxes.append(((mid[0], mid[1], height / 2), (np.linalg.norm(p1 - p0) / 2, thickness / 2, height / 2), p1 - p0))
    return boxes_mesh(boxes)


def canopy_along(front, depth: float, height: float, thickness: float = 0.4, centre=(0.0, 0.0)):
    boxes = []
    for p0, p1 in zip(front[:-1], front[1:]):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        n = _outward(p0, p1, centre)
        mid = (p0 + p1) / 2 + n * depth / 2
        boxes.append(((mid[0], mid[1], height), (np.linalg.norm(p1 - p0) / 2, depth / 2, thickness / 2), p1 - p0))
    return boxes_mesh(boxes)


__all__ = ["boxes_mesh", "canopy_along", "court_lines_boxes", "rows_along", "wall_along"]
