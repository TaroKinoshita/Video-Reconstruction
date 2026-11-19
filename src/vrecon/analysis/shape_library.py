"""Category shapes for objects the scene only knows as a box (scene-independent; no clip-specific values).

A box tells an image or video model nothing about what it is: the racket box was drawn as a board, a chair box next
to a person was drawn as a second person (tennis, 2026-11-14 and 11-16). Here every object gets a category from what
the scene already records about it, and a simple template mesh of that category that fills its box exactly, so the
size, place and motion found by the reconstruction stay as they are; only the outline inside the box changes.

    category    from the object's label words (SAM prompt word, name, id) through CATEGORIES' synonyms;
                kind "character" or a person word -> person (standing / seated from the box's proportions)
    template    a generator per category, (verts, faces) centred on the origin, x across, y depth, z up,
                scaled to the object's size; front faces -y (the builder's convention for figures)
    unknown     categories without a template keep their box and are listed in the report (planned: a shape
                extruded from the object's own mask, which needs no category at all)

Extending: add a row to CATEGORIES (synonyms + generator name), or pass extra rows from a JSON file
({"category": {"synonyms": [...], "template": "chair"}}); a new template is one function in TEMPLATES.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .proxies import seated_figure, standing_figure

# ---------------------------------------------------------------- parts


def _boxes(parts):
    """Axis-aligned boxes (x0, x1, y0, y1, z0, z1) in a unit cube [-.5, .5]^3 -> verts, faces."""
    vs, fs = [], []
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for x0, x1, y0, y1, z0, z1 in parts:
        b = len(vs)
        for z in (z0, z1):
            vs += [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        for f in faces:
            fs += [(b + f[0], b + f[1], b + f[2]), (b + f[0], b + f[2], b + f[3])]
    return np.array(vs, float), np.array(fs, int)


def _cylinder(r, z0, z1, n=16, cx=0.0, cy=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]
    v = np.vstack([np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)], [[cx, cy, z0], [cx, cy, z1]]])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return v, np.array(f)


def _join(parts):
    vs, fs, n = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + n)
        n += len(v)
    return np.vstack(vs), np.vstack(fs)


# ---------------------------------------------------------------- templates (unit box, then scaled)

def _chair():
    leg = 0.08
    return _boxes([(-0.5, 0.5, -0.5, 0.5, -0.02, 0.06),                       # seat at about 45% of the height
                   (-0.5, 0.5, 0.4, 0.5, 0.06, 0.5),                         # back rest
                   (-0.5, -0.5 + leg, -0.5, -0.5 + leg, -0.5, -0.02),         # legs
                   (0.5 - leg, 0.5, -0.5, -0.5 + leg, -0.5, -0.02),
                   (-0.5, -0.5 + leg, 0.5 - leg, 0.5, -0.5, -0.02),
                   (0.5 - leg, 0.5, 0.5 - leg, 0.5, -0.5, -0.02)])


def _bench():
    leg = 0.06
    return _boxes([(-0.5, 0.5, -0.5, 0.5, 0.3, 0.5),                          # long seat, legs at the ends
                   (-0.48, -0.48 + leg, -0.45, 0.45, -0.5, 0.3),
                   (0.48 - leg, 0.48, -0.45, 0.45, -0.5, 0.3)])


def _table():
    leg = 0.07
    return _boxes([(-0.5, 0.5, -0.5, 0.5, 0.42, 0.5),
                   (-0.5, -0.5 + leg, -0.5, -0.5 + leg, -0.5, 0.42), (0.5 - leg, 0.5, -0.5, -0.5 + leg, -0.5, 0.42),
                   (-0.5, -0.5 + leg, 0.5 - leg, 0.5, -0.5, 0.42), (0.5 - leg, 0.5, 0.5 - leg, 0.5, -0.5, 0.42)])


def _bag():
    return _boxes([(-0.5, 0.5, -0.4, 0.4, -0.5, 0.3),                         # body
                   (-0.25, -0.15, -0.05, 0.05, 0.3, 0.5), (0.15, 0.25, -0.05, 0.05, 0.3, 0.5),  # handle posts
                   (-0.25, 0.25, -0.05, 0.05, 0.42, 0.5)])                    # handle


def _crate():
    return _boxes([(-0.5, 0.5, -0.5, 0.5, -0.5, -0.4), (-0.5, 0.5, -0.5, -0.42, -0.4, 0.5),
                   (-0.5, 0.5, 0.42, 0.5, -0.4, 0.5), (-0.5, -0.42, -0.42, 0.42, -0.4, 0.5),
                   (0.42, 0.5, -0.42, 0.42, -0.4, 0.5)])                       # open box


def _pole():
    return _cylinder(0.5, -0.5, 0.5)


def _ball():
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    rings = [(-0.5, 0.0)] + [(0.5 * np.sin(t - np.pi / 2), 0.5 * np.cos(t - np.pi / 2)) for t in np.linspace(0, np.pi, 9)[1:-1]] + [(0.5, 0.0)]
    v = []
    for z, r in rings:
        v += [[r * np.cos(b), r * np.sin(b), z] for b in a] if r > 0 else [[0, 0, z]]
    v = np.array(v)
    f, n = [], len(a)
    for i in range(n):
        j = (i + 1) % n
        f.append([0, 1 + j, 1 + i])
    for k in range(len(rings) - 3):
        p, q = 1 + k * n, 1 + (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            f += [[p + i, p + j, q + j], [p + i, q + j, q + i]]
    top, last = len(v) - 1, 1 + (len(rings) - 3) * n
    for i in range(n):
        j = (i + 1) % n
        f.append([top, last + i, last + j])
    return v, np.array(f)


def _figure_standing():
    v, f = standing_figure(1.0, 1.0, 1.0)
    return v, f


def _figure_seated():
    v, f = seated_figure(1.0, 1.0, 1.0)
    return v, f


TEMPLATES = {"chair": _chair, "bench": _bench, "table": _table, "bag": _bag, "crate": _crate, "pole": _pole,
             "ball": _ball, "person_standing": _figure_standing, "person_seated": _figure_seated}

CATEGORIES: dict[str, dict] = {
    "person": {"synonyms": ["person", "people", "player", "man", "woman", "child", "kid", "human", "umpire", "referee",
                            "spectator", "judge", "coach"], "template": "person"},
    "chair": {"synonyms": ["chair", "seat", "stool", "armchair"], "template": "chair"},
    "bench": {"synonyms": ["bench", "sofa", "couch", "pew"], "template": "bench"},
    "table": {"synonyms": ["table", "desk", "counter"], "template": "table"},
    "bag": {"synonyms": ["bag", "backpack", "suitcase", "case", "luggage", "handbag"], "template": "bag"},
    "crate": {"synonyms": ["crate", "basket", "bin", "box", "container", "cooler"], "template": "crate"},
    "pole": {"synonyms": ["pole", "post", "bottle", "can", "cone", "lamp", "umbrella", "column", "pillar"],
             "template": "pole"},
    "ball": {"synonyms": ["ball", "sphere", "globe"], "template": "ball"},
}


def load_extra(path: str | Path | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def categorize(obj, label: str = "", extra: dict | None = None) -> str | None:
    """Category of a scene object from its label words (most specific first: label, name, id)."""
    table = {**CATEGORIES, **(extra or {})}
    if obj.kind == "character":
        return "person"
    for text in (label, obj.name or "", obj.id):
        words = set(re.findall(r"[a-z]+", text.lower()))
        for cat, row in table.items():
            if words & set(row["synonyms"]):
                return cat
    return None


def template_mesh(category: str, size, extra: dict | None = None):
    """(verts, faces) filling a box of this size, centred on the origin; None when the category has no template."""
    table = {**CATEGORIES, **(extra or {})}
    name = table.get(category, {}).get("template")
    sx, sy, sz = (float(x) for x in size)
    if name == "person":
        name = "person_standing" if sz >= 1.8 * max(sx, sy) else "person_seated"
    if name not in TEMPLATES:
        return None
    v, f = TEMPLATES[name]()
    lo, hi = v.min(0), v.max(0)
    v = (v - (lo + hi) / 2) / np.maximum(hi - lo, 1e-9) * np.array([sx, sy, sz])
    return v, f


def apply_templates(project, out_dir: str | Path, labels: dict[str, str] | None = None, extra: dict | None = None,
                    replace_assets: bool = False):
    """Give every box-only object a category template mesh (written to out_dir/<id>.npz, referenced by name).
    Characters with a skeleton, meshes already made (unless replace_assets) and set pieces are left alone.
    Returns (project copy, report {object id: category or "unknown"/"kept"})."""
    from vrecon.schema import Evidence, FileRef

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = project.model_copy(deep=True)
    report = {}
    for o in p.reconstruction.objects:
        if o.kind in ("set", "character") or not o.size:
            continue
        if p.is_locked(o.id, "asset") or p.is_locked(o.id, "size"):
            report[o.id] = "kept (locked by a human)"
            continue
        if o.asset is not None and not replace_assets:
            report[o.id] = "kept (has a mesh)"
            continue
        cat = categorize(o, (labels or {}).get(o.id, ""), extra)
        mesh = template_mesh(cat, o.size, extra) if cat else None
        if mesh is None:
            report[o.id] = "unknown" if cat is None else f"{cat} (no template)"
            continue
        v, f = mesh
        name = f"{o.id}_template.npz"
        np.savez_compressed(out / name, verts=v.astype(np.float32), faces=f.astype(np.int32))
        o.asset = FileRef(id=f"file_{o.id[4:]}_template", path=name, media_type="application/x-vrecon-mesh+npz")
        o.evidence = list(o.evidence) + [Evidence(kind="rule", note=f"shape: {cat} template (analysis/shape_library) "
                                                                     f"filling the fitted box; outline assumed")]
        report[o.id] = cat
    return p, report


__all__ = ["CATEGORIES", "TEMPLATES", "apply_templates", "categorize", "load_extra", "template_mesh"]
