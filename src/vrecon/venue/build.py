"""Venue layout -> scene objects (Project), with the observed moving things kept. Pure python/numpy (meshes are
written as .npz files, the builder in scene/ loads them).

build_venue(layout, base, out_dir, replace)
    base: a Project in the court frame whose camera and observed things (people, chairs...) are kept;
    replace: ids of base objects the venue takes over (the old floor, net, stands, background ...).
    Parts become objects: court -> obj_court (+ obj_floor run-off unless a ground part), net -> obj_net,
    ground -> obj_floor, stand -> obj_stand_<id>, wall -> obj_wall_<id>, box -> obj_<category prefix>_<id>,
    canopy -> obj_roof_<id>. Status: standard/observed -> "estimated", hypothesis -> "assumed"; the part's
    status and id go into the evidence note.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vrecon.schema import Evidence, FileRef, Pose, SceneObject, TransformKey

from .generate import canopy_along, court_lines_boxes, rows_along, wall_along
from .layout import check

BOX_PREFIX = {"furniture": "obj_bench", "person": "obj_player", "other": "obj_thing"}
COURT = (10.97, 23.77)
NET = (12.8, 0.05, 1.07)


def _obj(oid, kind, size, centre, status, note, asset=None, yaw=0.0, look=None):
    q = (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))
    return SceneObject(id=oid, name=oid.removeprefix("obj_"), kind=kind, size=tuple(float(x) for x in size),
                       status="assumed" if status == "hypothesis" else "estimated",
                       rest=TransformKey(time=0.0, pose=Pose(translation=tuple(float(x) for x in centre), rotation=q)),
                       asset=FileRef(id="file_" + oid.removeprefix("obj_"), path=asset,
                                     media_type="application/x-vrecon-mesh+npz") if asset else None,
                       look={"all": tuple(look)} if look else {},
                       evidence=[Evidence(note=f"venue layout part ({status}): {note}")])


def build_venue(layout: dict, base, out_dir: str | Path, replace: set[str], looks: dict | None = None):
    probs = check(layout)
    if probs:
        raise ValueError("layout problems:\n  " + "\n  ".join(probs))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    looks = looks or {}
    objs = []
    has_ground = any(p["type"] == "ground" for p in layout["parts"])

    def mesh(oid, vf):
        v, f = vf[:2]
        if len(vf) > 2:
            np.savez_compressed(out / f"{oid}.npz", verts=v, faces=f, colors=vf[2])
        else:
            np.savez_compressed(out / f"{oid}.npz", verts=v, faces=f)
        ext = np.ptp(v, 0) if len(v) else np.zeros(3)
        return f"{oid}.npz", ext

    # stands that meet at a corner: front lines extended to the crossing and a fan of rows between them
    from .generate import _two_tone, corner_fan, join_corners

    stands = [p for p in layout["parts"] if p["type"] == "stand"]
    for c in (join_corners(stands) if layout.get("join_corners", True) else []):
        oid = f"obj_stand_{c['id']}"
        boxes = corner_fan(np.array(c["corner"]), np.array(c["n1"]), np.array(c["n2"]), c["rows"], c["row_depth"],
                           c["rise"], c["base_height"])
        path, ext = mesh(oid, _two_tone(boxes, looks.get("stands", (0.2, 0.35, 0.65))))
        objs.append(_obj(oid, "set", ext, (0, 0, 0), c["status"], f"{c['id']}: corner fan between stands", asset=path))
    for p in layout["parts"]:
        t, st, pid = p["type"], p["status"], p["id"]
        note = f"{pid}: " + json.dumps({k: v for k, v in p.items() if k not in ("id", "type", "status")})[:300]
        if t == "court":
            objs.append(_obj("obj_court", "set", (COURT[0], COURT[1], 0.01), (0, 0, 0.005), st, note, look=looks.get("court")))
            path, ext = mesh("obj_court_lines", court_lines_boxes())
            objs.append(_obj("obj_court_lines", "set", ext, (0, 0, 0), st, "ITF court lines", asset=path, look=(0.95, 0.95, 0.95)))
            if not has_ground:
                objs.append(_obj("obj_floor", "set", (COURT[0] + 2 * 3.66, COURT[1] + 2 * 6.4, 0.02), (0, 0, -0.01), st,
                                 "run-off (ITF minimum)", look=looks.get("ground")))
        elif t == "net":
            # a mesh of cords (0.1 m squares drawn every 0.3 m) instead of a solid slab: the far court shows through
            from .generate import boxes_mesh

            cords = [((x, 0.0, NET[2] / 2), (0.006, 0.006, NET[2] / 2), (1.0, 0.0)) for x in np.arange(-6.3, 6.31, 0.3)]
            cords += [((0.0, 0.0, z), (NET[0] / 2, 0.006, 0.006), (1.0, 0.0)) for z in np.arange(0.1, NET[2] - 0.05, 0.3)]
            cords += [((x, 0.0, NET[2] / 2), (0.04, 0.04, NET[2] / 2), (1.0, 0.0)) for x in (-NET[0] / 2, NET[0] / 2)]
            path, ext = mesh("obj_net", boxes_mesh(cords))
            objs.append(_obj("obj_net", "rigid", NET, (0, 0, 0), st, note, asset=path, look=(0.2, 0.2, 0.22)))
            objs.append(_obj("obj_net_tape", "rigid", (NET[0], 0.07, 0.07), (0, 0, NET[2] - 0.035), st, "net tape",
                             look=(0.95, 0.95, 0.95)))
        elif t == "ground":
            w, l = p["size"]
            objs.append(_obj("obj_floor", "set", (w, l, 0.02), (*p.get("centre", (0.0, 0.0)), -0.01), st, note,
                             look=looks.get("ground")))
        elif t == "stand":
            oid = f"obj_stand_{pid}"
            path, ext = mesh(oid, rows_along(p["front"], p["rows"], p["row_depth"], p["rise"], p["base_height"], p.get("aisles"),
                                             colour=looks.get("stands", (0.2, 0.35, 0.65))))
            objs.append(_obj(oid, "set", ext, (0, 0, 0), st, note, asset=path))
        elif t == "wall":
            oid = f"obj_wall_{pid}"
            path, ext = mesh(oid, wall_along(p["line"], p["height"], p["thickness"]))
            objs.append(_obj(oid, "set", ext, (0, 0, 0), st, note, asset=path, look=looks.get("wall")))
        elif t == "canopy":
            oid = f"obj_roof_{pid}"
            path, ext = mesh(oid, canopy_along(p["front"], p["depth"], p["height"]))
            objs.append(_obj(oid, "set", ext, (0, 0, 0), st, note, asset=path, look=looks.get("roof")))
        elif t == "box":
            oid = f"{BOX_PREFIX.get(p['category'], 'obj_thing')}_{pid}"
            w, d, h = p["size"]
            objs.append(_obj(oid, "rigid", (w, d, h), (p["centre"][0], p["centre"][1], h / 2), st, note,
                             yaw=np.radians(p.get("yaw_deg", 0.0)), look=looks.get(p["category"])))
    q = base.model_copy(deep=True)
    keep = [o for o in q.reconstruction.objects if o.id not in replace]
    q.reconstruction.objects = keep + objs
    return q


__all__ = ["build_venue"]
