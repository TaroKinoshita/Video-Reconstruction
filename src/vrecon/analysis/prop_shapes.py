"""Simple prop meshes in the prop's own axes (x along its length, centred; y across; z its thickness), written as
the scene's mesh files (.npz: verts, faces, colors). A box reads as a board to an image or video model (the fal
look check drew the racket box as a black board, 2026-11-14); a frame with a handle reads as a racket.

    shape "sphere"  UV sphere
    shape "racket"  handle, throat and an oval frame with a lighter string bed (tennis, badminton, squash)
    shape "paddle"  handle and a round blade (table tennis)
    shape "bat"     tapered round bar (baseball, cricket)
    shape "box"     the old box
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _box(c, size, col):
    c, (sx, sy, sz) = np.asarray(c, float), size
    v = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (-sz / 2, sz / 2)]) + c
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4),
         (1, 5, 7), (1, 7, 3)]
    return v, np.array(f), np.tile(col, (8, 1))


def _join(parts):
    vs, fs, cs, n = [], [], [], 0
    for v, f, c in parts:
        vs.append(v)
        fs.append(f + n)
        cs.append(c)
        n += len(v)
    return np.concatenate(vs), np.concatenate(fs), np.concatenate(cs)


def sphere(d: float, col=(0.85, 0.95, 0.2), n: int = 12):
    v, f = [], []
    for i in range(n + 1):
        th = np.pi * i / n
        for j in range(2 * n):
            ph = np.pi * j / n
            v.append([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    for i in range(n):
        for j in range(2 * n):
            a, b = i * 2 * n + j, i * 2 * n + (j + 1) % (2 * n)
            c, e = a + 2 * n, b + 2 * n
            f += [(a, c, e), (a, e, b)]
    v = np.array(v) * d / 2
    return v, np.array(f), np.tile(col, (len(v), 1))


def _ring(cx, a, b, w, t, col, n=36):
    """Elliptical tube of band width w and thickness t, centred at (cx, 0) in the x-y plane."""
    v, f = [], []
    for i in range(n):
        u = 2 * np.pi * i / n
        for r in (0.0, 1.0):
            for z in (-t / 2, t / 2):
                k = 1 - r * w / min(a, b)
                v.append([cx + a * k * np.cos(u), b * k * np.sin(u), z])
    for i in range(n):
        j = (i + 1) % n
        q = [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
        p = [4 * j, 4 * j + 1, 4 * j + 2, 4 * j + 3]
        for s0, s1 in ((0, 1), (1, 3), (3, 2), (2, 0)):  # outer, top, inner, bottom faces
            f += [(q[s0], p[s0], p[s1]), (q[s0], p[s1], q[s1])]
    return np.array(v), np.array(f), np.tile(col, (len(v), 1))


def _disc(cx, a, b, col, n=36, z=0.0):
    v = [[cx, 0, z]] + [[cx + a * np.cos(2 * np.pi * i / n), b * np.sin(2 * np.pi * i / n), z] for i in range(n)]
    f = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)] + [(0, 1 + (i + 1) % n, 1 + i) for i in range(n)]
    return np.array(v), np.array(f), np.tile(col, (len(v), 1))


def racket(length: float, width: float, thick: float, col=(0.15, 0.15, 0.18), strings=(0.8, 0.8, 0.75)):
    head = 0.46 * length
    cx = length / 2 - head / 2
    handle = 0.30 * length
    hw = max(0.035, width * 0.13)
    parts = [
        _box((-length / 2 + handle / 2, 0, 0), (handle, hw, hw * 0.85), col),  # handle
        _ring(cx, head / 2, width / 2, 0.012 + 0.03 * width, thick, col),     # frame
        _disc(cx, head / 2 * 0.93, width / 2 * 0.9, strings),                 # string bed
    ]
    # throat: two bars from the handle's end to the frame's sides
    x0, x1 = -length / 2 + handle, cx - head / 2 * 0.75
    for sgn in (-1, 1):
        mid = np.array([(x0 + x1) / 2, sgn * width * 0.18, 0])
        ln = np.hypot(x1 - x0, width * 0.36)
        v, f, c = _box((0, 0, 0), (ln, hw * 0.6, thick * 0.8), col)
        ang = np.arctan2(sgn * width * 0.36, x1 - x0)
        rot = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        parts.append((v @ rot.T + mid, f, c))
    return _join(parts)


def paddle(length: float, width: float, thick: float, col=(0.7, 0.1, 0.1), handle_col=(0.6, 0.45, 0.3)):
    handle = 0.38 * length
    blade = length - handle
    return _join([_box((-length / 2 + handle / 2, 0, 0), (handle, 0.03, 0.025), handle_col),
                  _disc(length / 2 - blade / 2, blade / 2, width / 2, col, z=thick / 2),
                  _disc(length / 2 - blade / 2, blade / 2, width / 2, col, z=-thick / 2)])


def bat(length: float, width: float, thick: float, col=(0.7, 0.55, 0.35), n: int = 12):
    v, f = [], []
    for i, x in enumerate(np.linspace(-length / 2, length / 2, 8)):
        r = (0.35 + 0.65 * (i / 7) ** 2) * width / 2
        for j in range(n):
            u = 2 * np.pi * j / n
            v.append([x, r * np.cos(u), r * np.sin(u)])
    for i in range(7):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            f += [(a, a + n, b + n), (a, b + n, b)]
    return np.array(v), np.array(f), np.tile(col, (len(v), 1))


def prop_mesh(entry: dict, scale: float = 1.0):
    """(verts, faces, colors) for a prop table entry, in scene units."""
    size = [x * scale for x in entry["size"]]
    col = tuple(entry.get("colour", (0.5, 0.5, 0.5)))
    kind = entry.get("mesh", "sphere" if entry["shape"] == "sphere" else "box")
    if kind == "sphere":
        return sphere(size[0], col)
    if kind == "racket":
        return racket(*size, col=col)
    if kind == "paddle":
        return paddle(*size, col=col)
    if kind == "bat":
        return bat(*size, col=col)
    return _box((0, 0, 0), size, col)


def write_mesh(path: Path, mesh) -> None:
    v, f, c = mesh
    np.savez_compressed(path, verts=v.astype(np.float32), faces=f.astype(np.int32), colors=c.astype(np.float32))


__all__ = ["bat", "paddle", "prop_mesh", "racket", "sphere", "write_mesh"]
