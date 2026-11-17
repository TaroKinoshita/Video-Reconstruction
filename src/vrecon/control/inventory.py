"""What a control video shows, as a short scene inventory for a generation prompt. Pure numpy; scene-independent.

The scene already knows every object; the index pass of an export says which of them the camera sees and where.
From that, a sentence like "Exactly 3 people are in the shot: ... Do not add any other person." keeps a generator
from turning an unlabelled shape (a chair drawn as a box) into an extra person. Used by scripts/make_prompts.py.

Classes come from the object's kind and id (no clip-specific table): character / id containing "person" or "player"
-> person; "crowd" -> seated spectators; "chair", "bench", "bag" -> that word; set pieces by id word (court, net,
stand, wall, floor). An object counts as visible when it covers at least MIN_SHARE of the frame in
more than MIN_FRAMES of the frames; its place is the side of the frame its pixels mostly fall on.
"""

from __future__ import annotations

import numpy as np

MIN_SHARE = 0.0008
MIN_FRAMES = 0.1  # share of the frames
PERSON_WORDS = ("person", "player", "human", "man", "woman")
THING_WORDS = ("chair", "bench", "bag", "umpire", "table", "box", "car", "ball", "racket", "bottle")
SET_WORDS = {"court": "the court", "net": "the net", "stand": "the stands", "wall": "the wall", "floor": "the ground",
             "ground": "the ground"}


def _class(obj) -> str:
    oid = obj.id.lower()
    if "crowd" in oid or any("not on the floor" in e.note for e in obj.evidence):
        return "crowd"  # spectators, and people analysis/floor_contact found behind a barrier
    if obj.kind == "character" or any(w in oid for w in PERSON_WORDS):
        return "person"
    for w in THING_WORDS:
        if w in oid:
            return w
    for w in SET_WORDS:
        if w in oid:
            return "set:" + w
    return "thing"


def visible_objects(index: np.ndarray, id_map: dict[int, str], project) -> list[dict]:
    objs = {o.id: o for o in project.reconstruction.objects}
    t_n, h, w = index.shape
    out = []
    for lab, oid in id_map.items():
        m = index == lab
        area = m.reshape(t_n, -1).mean(1)
        seen = area >= MIN_SHARE
        if seen.mean() <= MIN_FRAMES or oid not in objs:
            continue
        xs = np.nonzero(m[seen])[2] / w
        place = "on the left" if np.median(xs) < 0.33 else "on the right" if np.median(xs) > 0.67 else "in the middle"
        mv = m[seen]
        edges = {"left": mv[:, :, 0].any(1).mean(), "right": mv[:, :, -1].any(1).mean(),
                 "bottom": mv[:, -1, :].any(1).mean()}
        cut = [k for k, v in edges.items() if v > 0.5]
        out.append({"id": oid, "class": _class(objs[oid]), "place": place, "frames_share": float(seen.mean()),
                    "area": float(area[seen].mean()), "name": objs[oid].name or oid, "cut_by": cut})
    return sorted(out, key=lambda o: -o["area"])


def inventory_sentence(visible: list[dict], main_person: str | None = None, main_role: str = "the main subject") -> str:
    people = [o for o in visible if o["class"] == "person"]
    parts = []
    for o in people:
        role = main_role if o["id"] == main_person else "a person standing " + o["place"]
        if o.get("cut_by"):
            role += f" (only partly in frame, cut by the {' and '.join(o['cut_by'])} edge)"
        parts.append(role)
    lines = []
    if people:
        n = len(people)
        lines.append(f"Exactly {n} {'person is' if n == 1 else 'people are'} in the shot outside the stands: "
                     + "; ".join(parts) + ". Do not add any other person on the court or beside it.")
    things = [o for o in visible if o["class"] in THING_WORDS and o["class"] not in ("ball", "racket")]
    if things:
        lines.append("Other objects: " + "; ".join(
            f"a {o['class']} {o['place']}" + (f" (cut by the {' and '.join(o['cut_by'])} edge)" if o.get("cut_by") else "")
            for o in things) + " (objects, not people).")
        lines.append("Anything cut by the frame edge is part of an object that continues outside the picture; do not "
                     "turn its visible part into a new object.")
    if any(o["class"] == "crowd" for o in visible):
        lines.append("Spectators only sit in the stands.")
    return " ".join(lines)


__all__ = ["inventory_sentence", "visible_objects"]
