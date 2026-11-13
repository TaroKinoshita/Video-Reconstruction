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
    """(seat blocks, aisle gaps) along p0-p1: aisles `width` wide every `every` m. Returns two lists of (a, b)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    if not aisles or aisles.get("every", 0) <= 0 or L <= aisles["every"]:
        return [(p0, p1)], []
    n = max(1, int(round(L / aisles["every"])))
    w = float(aisles.get("width", 1.4))
    step = L / n
    d = (p1 - p0) / L
    blocks, gaps = [], []
    for i in range(n):
        lo = i * step + (w / 2 if i > 0 else 0.0)
        hi = (i + 1) * step - (w / 2 if i < n - 1 else 0.0)
        if hi > lo:
            blocks.append((p0 + lo * d, p0 + hi * d))
        if i < n - 1:
            gaps.append((p0 + ((i + 1) * step - w / 2) * d, p0 + ((i + 1) * step + w / 2) * d))
    return blocks, gaps


SEAT_TOP, SEAT_RISER, STAIR = 1.35, 0.55, (0.85, 0.83, 0.78)  # stairs light: they read as ways in between blocks


def _two_tone(boxes_kinds, colour):
    """Mesh + per-vertex colours from (box, kind) pairs: seats light on top and dark on the risers, stairs grey."""
    v, f = boxes_mesh([bx for bx, _ in boxes_kinds])
    if colour is None:
        return v, f
    c = np.asarray(colour, float)
    cols = np.zeros((len(v), 3))
    for i, (_, kind) in enumerate(boxes_kinds):
        base = np.asarray(STAIR, float) if kind == "stair" else c
        cols[i * 8: i * 8 + 4] = base * SEAT_RISER
        cols[i * 8 + 4: i * 8 + 8] = base if kind == "stair" else np.clip(base * SEAT_TOP, 0, 1)
    return v, f, cols


def _row_boxes(a, b, n, rows, row_depth, rise, base_height, kind="seat"):
    u = b - a
    hu = np.linalg.norm(u) / 2
    mid = (a + b) / 2
    out = []
    for i in range(int(rows)):
        if kind == "stair":  # two steps per row, half a rise each: a stairway people walk up
            for h, k in ((0.25, 0.5), (0.75, 1.0)):
                top = base_height + (i + k) * rise
                c2 = mid + n * (i + h) * row_depth
                out.append((((c2[0], c2[1], top / 2), (hu, row_depth / 4, top / 2), u), "stair"))
        else:
            top = base_height + (i + 1) * rise
            c2 = mid + n * (i + 0.5) * row_depth
            out.append((((c2[0], c2[1], top / 2), (hu, row_depth / 2, top / 2), u), "seat"))
    return out


def rows_along(front, rows: int, row_depth: float, rise: float, base_height: float, aisles=None,
               centre=(0.0, 0.0), colour=None):
    """A stand behind a front line: for each segment, rows of boxes; row i spans i..i+1 row depths behind the
    front (away from `centre`) and rises from the ground to base_height + (i + 1) * rise. Aisles are stairways
    (two half-rise steps per row, grey), not gaps: a gap read as a hole, not a way in. Where the front line
    turns, the outer wedge is filled with a fan of rows (corner_fan)."""
    out = []
    pts = [np.asarray(q, float) for q in front]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        n = _outward(p0, p1, centre)
        blocks, gaps = _pieces(p0, p1, aisles)
        for a, b in blocks:
            out += _row_boxes(a, b, n, rows, row_depth, rise, base_height)
        for a, b in gaps:
            out += _row_boxes(a, b, n, rows, row_depth, rise, base_height, kind="stair")
    for prev, here, nxt in zip(pts[:-2], pts[1:-1], pts[2:]):
        out += corner_fan(here, _outward(prev, here, centre), _outward(here, nxt, centre), rows, row_depth, rise,
                          base_height)
    return _two_tone(out, colour)


def corner_fan(corner, n1, n2, rows, row_depth, rise, base_height, step_deg: float = 7.5):
    """Rows in the wedge between two stands meeting at `corner` with outward normals n1 and n2: for each small
    angle step, a box per row at the row's distance from the corner (a stadium's curved corner, in pieces)."""
    a1, a2 = np.arctan2(n1[1], n1[0]), np.arctan2(n2[1], n2[0])
    da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
    if abs(da) < np.radians(2):
        return []
    k = max(1, int(np.ceil(abs(np.degrees(da)) / step_deg)))
    out = []
    for j in range(k):
        ang = a1 + (j + 0.5) * da / k
        dirv = np.array([np.cos(ang), np.sin(ang)])
        tang = np.array([-dirv[1], dirv[0]])
        for i in range(int(rows)):
            r = (i + 0.5) * row_depth
            w = 2 * (r + row_depth / 2) * np.tan(abs(da) / k / 2) + 0.05  # outer edge, slightly overlapping
            top = base_height + (i + 1) * rise
            c2 = np.asarray(corner, float) + dirv * r
            out.append((((c2[0], c2[1], top / 2), (w / 2, row_depth / 2, top / 2), tang), "seat"))
    return out


def join_corners(stands: list, max_gap: float = 12.0, centre=(0.0, 0.0)) -> list:
    """Stands of one venue meet at corners: where one stand's front line end and another's lie within max_gap of
    the point where their (extended) front lines cross, beyond both ends, both lines are extended to that point
    and a corner fan is returned for it (the shallower profile of the two). Modifies the stands' fronts."""
    corners = []
    done = set()
    for i, s1 in enumerate(stands):
        for j, s2 in enumerate(stands):
            if i >= j:
                continue
            f1, f2 = np.asarray(s1["front"], float), np.asarray(s2["front"], float)
            best = None
            for e1 in (0, -1):
                for e2 in (0, -1):
                    a, da = f1[e1], f1[e1] - f1[1 if e1 == 0 else -2]
                    b, db = f2[e2], f2[e2] - f2[1 if e2 == 0 else -2]
                    da, db = da / np.linalg.norm(da), db / np.linalg.norm(db)
                    M = np.array([da, -db]).T
                    if abs(np.linalg.det(M)) < 1e-3:
                        continue
                    t = np.linalg.solve(M, b - a)
                    if abs(t[0]) > max_gap or abs(t[1]) > max_gap:  # beyond the ends: extend; inside: trim
                        continue
                    if best is None or np.abs(t).sum() < best[0]:
                        best = (np.abs(t).sum(), e1, e2, a + t[0] * da)
            if best is None or (i, j) in done:
                continue
            _, e1, e2, P = best
            f1[e1], f2[e2] = P, P
            s1["front"], s2["front"] = f1.round(3).tolist(), f2.round(3).tolist()
            n1 = _outward(*(f1[[0, 1]] if e1 == 0 else f1[[-2, -1]]), centre)
            n2 = _outward(*(f2[[0, 1]] if e2 == 0 else f2[[-2, -1]]), centre)
            corners.append({"id": f"corner_{s1['id']}_{s2['id']}", "corner": P.tolist(), "n1": n1.tolist(), "n2": n2.tolist(),
                            "rows": int(min(s1["rows"], s2["rows"])), "row_depth": float(min(s1["row_depth"], s2["row_depth"])),
                            "rise": float(max(s1["rise"], s2["rise"])), "base_height": float(max(s1["base_height"], s2["base_height"])),
                            "status": "hypothesis" if "hypothesis" in (s1["status"], s2["status"]) else s1["status"]})
            done.add((i, j))
    return corners


CLOTHES = [(0.85, 0.85, 0.85), (0.15, 0.2, 0.45), (0.75, 0.2, 0.2), (0.95, 0.85, 0.3), (0.2, 0.5, 0.3),
           (0.1, 0.1, 0.12), (0.9, 0.55, 0.25), (0.45, 0.6, 0.85), (0.6, 0.6, 0.62)]
HEAD = (0.55, 0.42, 0.33)


def crowd_along(front, rows: int, row_depth: float, rise: float, base_height: float, aisles=None,
                density: float = 0.3, seat: float = 0.55, seed: int = 0, centre=(0.0, 0.0)):
    """Seated spectators as simple figures (body box + head box) on a stand's seat rows (not on the stairs):
    one seat every `seat` metres, each taken with probability `density`. Clothing colours vary; a hypothesis
    unless the picture shows the crowd. Returns (verts, faces, colours)."""
    rng = np.random.default_rng(seed)
    out = []
    pts = [np.asarray(q, float) for q in front]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        n = _outward(p0, p1, centre)
        blocks, _ = _pieces(p0, p1, aisles)
        for a, b in blocks:
            L = float(np.linalg.norm(b - a))
            if L < seat:
                continue
            u = (b - a) / L
            k = int(L // seat)
            for i in range(int(rows)):
                top = base_height + (i + 1) * rise
                for j in range(k):
                    if rng.random() > density:
                        continue
                    c2 = a + u * (j + 0.5) * L / k + n * (i + 0.62) * row_depth
                    col = CLOTHES[rng.integers(len(CLOTHES))]
                    out.append((((c2[0], c2[1], top + 0.3), (0.2, 0.14, 0.3), u), col))
                    out.append((((c2[0], c2[1], top + 0.72), (0.1, 0.1, 0.12), u), HEAD))
    v, f = boxes_mesh([bx for bx, _ in out])
    cols = np.repeat(np.array([c for _, c in out], float), 8, axis=0) if out else np.zeros((0, 3))
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


__all__ = ["boxes_mesh", "canopy_along", "corner_fan", "court_lines_boxes", "crowd_along", "join_corners", "rows_along", "wall_along"]
