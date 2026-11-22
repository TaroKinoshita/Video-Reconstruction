"""Rough object shapes by kind, pure numpy (no bpy): a gun reads as a gun, a chair as a chair, at the size the
observation gave. One fixed table for every video (the vocabulary of gpu/observe/find_objects_sam3.py); words it
does not know stay a box. Every shape fills the box [-w/2, w/2] x [-d/2, d/2] x [-h/2, h/2] (x = width across the
picture, y = depth, z = up), so placement and the checks are unchanged."""

from __future__ import annotations

import numpy as np

from .mannequin import _box, _cylinder, frustum

LONG = {"gun", "knife", "bat", "pipe", "rope", "racket", "umbrella", "guitar", "microphone"}


def _add(verts, faces, part):
    v, f = part
    o = len(verts)
    verts += [tuple(x) for x in v]
    faces += [tuple(i + o for i in ff) for ff in f]


def _bx(c, s):
    return _box(np.asarray(c, float), tuple(max(float(x), 1e-3) for x in s))


def _cyl(a, b, r):
    return _cylinder(np.asarray(a, float), np.asarray(b, float), max(float(r), 1e-3), 12)


def prop_mesh(word: str, size) -> tuple[list, list]:
    w, h, d = (max(float(x), 1e-3) for x in size)
    word = word.lower()
    v, f = [], []
    if word == "gun":
        # barrel along the longer of width / depth, grip down at the back
        ax = np.array([1.0, 0, 0]) if w >= d else np.array([0, 1.0, 0])
        L, T = max(w, d), min(w, d)
        bs = ax * L + (1 - ax) * T
        _add(v, f, _bx([0, 0, h / 2 - 0.2 * h], [bs[0], bs[1], 0.4 * h]))
        _add(v, f, _bx(-ax * 0.3 * L + [0, 0, -0.2 * h], [*(ax * 0.3 * L + (1 - ax) * T)[:2], 0.6 * h]))
    elif word in LONG:
        ax = np.argmax([w, d, h])
        e = np.zeros(3)
        e[ax] = [w, d, h][ax] / 2
        r = 0.5 * min(x for i, x in enumerate((w, d, h)) if i != ax)
        _add(v, f, _cyl(-e, e, r))
        if word in ("racket", "umbrella"):
            _add(v, f, frustum(e * 0.2, e, r, 2.5 * r, 12))
    elif word in ("bottle",):
        r = min(w, d) / 2
        _add(v, f, _cyl([0, 0, -h / 2], [0, 0, 0.2 * h], r))
        _add(v, f, frustum(np.array([0, 0, 0.2 * h]), np.array([0, 0, h / 2]), r, 0.35 * r, 12))
    elif word in ("cup", "bucket", "barrel", "trash can", "lamp", "plant", "traffic cone", "helmet", "hat"):
        r = min(w, d) / 2
        top = {"traffic cone": 0.15, "lamp": 0.6, "plant": 1.2, "bucket": 1.15, "cup": 1.1}.get(word, 1.0)
        _add(v, f, frustum(np.array([0, 0, -h / 2]), np.array([0, 0, h / 2]), r * (1 / max(top, 1)), r * min(top, 1) if top < 1 else r, 12))
    elif word == "ball":
        r = min(w, d, h) / 2
        for z0, z1 in ((-1, -0.6), (-0.6, 0), (0, 0.6), (0.6, 1)):
            r0, r1 = (np.sqrt(max(1 - z * z, 0.05)) for z in (z0, z1))
            _add(v, f, frustum(np.array([0, 0, z0 * r]), np.array([0, 0, z1 * r]), r0 * r, r1 * r, 12))
    elif word in ("chair", "table", "bench", "bed", "sofa"):
        top_z = {"chair": 0.0, "table": h / 2 - 0.04 * h, "bench": 0.0, "bed": 0.0, "sofa": -0.1 * h}[word]
        th = 0.08 * h
        _add(v, f, _bx([0, 0, top_z], [w, d, th]))
        leg = 0.06 * min(w, d) + 0.01
        for sx in (-1, 1):
            for sy in (-1, 1):
                _add(v, f, _bx([sx * (w / 2 - leg / 2), sy * (d / 2 - leg / 2), (top_z - h / 2) / 2],
                               [leg, leg, top_z + h / 2]))
        if word in ("chair", "sofa"):
            _add(v, f, _bx([0, d / 2 - 0.06 * d, (top_z + h / 2) / 2], [w, 0.12 * d, h / 2 - top_z]))
    elif word in ("door", "television", "laptop", "flag", "clock"):
        t = min(d, 0.06 * max(w, h) + 0.02)
        _add(v, f, _bx([0, 0, 0], [w, t, h]))
    elif word == "sign":
        t = min(d, 0.04)
        _add(v, f, _bx([0, 0, h / 4], [w, t, h / 2]))
        _add(v, f, _bx([0, 0, -h / 4], [0.05 * w + 0.02, t, h / 2]))
    elif word in ("railing", "ladder"):
        n = max(2, int(round(max(w, d) / 1.2)) + 1)
        ax = np.array([1.0, 0, 0]) if w >= d else np.array([0, 1.0, 0])
        L = max(w, d)
        bar = 0.04 * h + 0.02
        _add(v, f, _bx([0, 0, h / 2 - bar / 2], [*(ax * L + (1 - ax) * bar)[:2], bar]))
        _add(v, f, _bx([0, 0, 0], [*(ax * L + (1 - ax) * bar)[:2], bar]))
        for x in np.linspace(-L / 2 + bar / 2, L / 2 - bar / 2, n):
            _add(v, f, _bx(ax * x, [bar, bar, h]))
    else:  # box, crate, bag, suitcase, car, debris, ... and every unknown word
        _add(v, f, _bx([0, 0, 0], [w, d, h]))
    return v, f
