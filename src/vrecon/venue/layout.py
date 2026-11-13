"""Venue layout format. Court frame, metres (origin at the court centre, X across, Y along, Z up).

{
  "site": "short description",
  "assumptions": ["free text: what is assumed and why"],
  "join_corners": true,      stands that meet are joined by a curved corner of rows (a bowl); false keeps
                             corners open (separate stands, as at many smaller venues). A hypothesis either way.
  "crowd": 0.3,              share of seats with a seated spectator (simple figures, always a hypothesis); a stand
                             part may set its own "crowd"; 0 or absent = empty stands
  "parts": [
    {"id": "court",  "type": "court",  "status": "standard"},                 ITF surface (+ run-off as ground)
    {"id": "net",    "type": "net",    "status": "standard"},                 12.8 m, 1.07 m at the posts
    {"id": "ground", "type": "ground", "size": [W, L], "status": ...},       flat ground around everything
    {"id": "north",  "type": "stand",  "front": [[x, y], ...], "rows": 15, "row_depth": 0.85, "rise": 0.45,
         "base_height": 1.0, "aisles": {"every": 10.0, "width": 1.2}, "status": ..., "fit": "obj_stands"},
                rows of seats behind a front line (the court side); they climb away from the court centre
    {"id": "n_wall", "type": "wall",   "line": [[x, y], ...], "height": 1.0, "thickness": 0.3, "status": ...},
    {"id": "umpire", "type": "box",    "category": "furniture"|"person"|"other", "centre": [x, y], "size": [w, d, h],
         "yaw_deg": 0, "status": ...},
    {"id": "roof_n", "type": "canopy", "front": [[x, y], ...], "depth": 8.0, "height": 14.0, "status": ...}
  ]
}
status: "standard" (fixed by the sport's rules), "observed" (the source view shows it; may carry "fit": the
observation label to fit it to), "hypothesis" (not seen: a plausible guess, kept as such in the scene).
"""

from __future__ import annotations

TYPES = {"court", "net", "ground", "stand", "wall", "box", "canopy"}
STATUS = {"standard", "observed", "hypothesis"}
REQUIRED = {"stand": ["front", "rows"], "wall": ["line", "height"], "box": ["centre", "size"], "canopy": ["front", "depth", "height"],
            "ground": ["size"]}
DEFAULTS = {"stand": {"row_depth": 0.85, "rise": 0.45, "base_height": 1.0, "aisles": None},
            "wall": {"thickness": 0.3}, "box": {"yaw_deg": 0.0, "category": "other"}}


def check(layout: dict) -> list[str]:
    """Problems with a layout (empty = fine). Defaults are filled in place."""
    probs = []
    ids = set()
    for i, p in enumerate(layout.get("parts", [])):
        t = p.get("type")
        if t not in TYPES:
            probs.append(f"part {i}: unknown type {t!r}")
            continue
        if p.get("status") not in STATUS:
            probs.append(f"part {p.get('id', i)}: status must be one of {sorted(STATUS)}")
        if not p.get("id") or p["id"] in ids:
            probs.append(f"part {i}: missing or repeated id")
        ids.add(p.get("id"))
        for k in REQUIRED.get(t, []):
            if k not in p:
                probs.append(f"part {p.get('id', i)}: {t} needs {k!r}")
        for k, v in DEFAULTS.get(t, {}).items():
            p.setdefault(k, v)
        for key in ("front", "line"):
            if key in p and (len(p[key]) < 2 or any(len(q) != 2 for q in p[key])):
                probs.append(f"part {p['id']}: {key} needs two or more [x, y] points")
        if t == "stand" and not (1 <= int(p.get("rows", 0)) <= 80):
            probs.append(f"part {p['id']}: rows out of range")
    return probs


__all__ = ["DEFAULTS", "STATUS", "TYPES", "check"]
