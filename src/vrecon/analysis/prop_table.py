"""Known props (docs/PROPS_PLAN.md, part 5): size, shape and which body parts touch them.

Each entry:
    shape        "sphere" (a ball: held / free / unseen states, fitted with analysis/props.ball_fit) or
                 "stick" (a rigid thing held at one end: analysis/props.held_poses)
    size         metres: sphere (diameter,); stick (length, width, thickness)
    contacts     joints that can hold or strike it (GVHMR / SMPL-X joint names)
    reach        metres beyond the contact joint along its bone where the prop sits (the palm, the handle's end)
    strike_frac  stick only: where along it a ball is struck (from the held end)
    colour       look for the scene (a stand-in; the real colour is not measured here)
    mesh         the stand-in's shape (analysis/prop_shapes.py): sphere, racket, paddle, bat, box
    source       where the numbers come from

Sizes are the sports' regulation sizes or typical ones; anything not listed comes from a JSON file in the same
form (`load_table(extra)`), e.g. written by an LLM from the object's name, and keeps its own "source".
"""

from __future__ import annotations

import json
from pathlib import Path

HANDS = ["left_wrist", "right_wrist"]
FEET = ["left_foot", "right_foot"]

PROPS: dict[str, dict] = {
    "tennis ball": {"shape": "sphere", "size": (0.067,), "contacts": HANDS, "reach": 0.08,
                    "colour": (0.85, 0.95, 0.2), "source": "ITF rules: 6.54-6.86 cm"},
    "tennis racket": {"shape": "stick", "size": (0.68, 0.27, 0.03), "contacts": HANDS, "reach": 0.08,
                      "strike_frac": 0.75, "colour": (0.15, 0.15, 0.18), "mesh": "racket", "source": "ITF rules: up to 73.7 x 31.7 cm; typical"},
    "baseball": {"shape": "sphere", "size": (0.074,), "contacts": HANDS, "reach": 0.08,
                 "colour": (0.95, 0.95, 0.92), "source": "MLB rules: 9-9.25 in circumference"},
    "baseball bat": {"shape": "stick", "size": (0.84, 0.066, 0.066), "contacts": HANDS, "reach": 0.05,
                     "strike_frac": 0.8, "colour": (0.7, 0.55, 0.35), "mesh": "bat", "source": "typical adult bat, 33 in"},
    "soccer ball": {"shape": "sphere", "size": (0.22,), "contacts": HANDS + FEET, "reach": 0.08,
                    "colour": (0.95, 0.95, 0.95), "source": "FIFA size 5: 68-70 cm circumference"},
    "basketball": {"shape": "sphere", "size": (0.24,), "contacts": HANDS, "reach": 0.1,
                   "colour": (0.85, 0.4, 0.1), "source": "FIBA size 7: 74.9-78 cm circumference"},
    "golf ball": {"shape": "sphere", "size": (0.043,), "contacts": HANDS, "reach": 0.08,
                  "colour": (0.97, 0.97, 0.97), "source": "R&A rules: at least 42.67 mm"},
    "table tennis ball": {"shape": "sphere", "size": (0.04,), "contacts": HANDS, "reach": 0.08,
                          "colour": (0.97, 0.97, 0.97), "source": "ITTF rules: 40 mm"},
    "table tennis paddle": {"shape": "stick", "size": (0.26, 0.15, 0.02), "contacts": HANDS, "reach": 0.05,
                            "strike_frac": 0.7, "colour": (0.7, 0.1, 0.1), "mesh": "paddle", "source": "typical"},
    "cup": {"shape": "sphere", "size": (0.09,), "contacts": HANDS, "reach": 0.08,
            "colour": (0.9, 0.9, 0.9), "source": "typical mug, as a ball-sized stand-in"},
    "phone": {"shape": "stick", "size": (0.15, 0.072, 0.008), "contacts": HANDS, "reach": 0.0,
              "strike_frac": 0.5, "colour": (0.1, 0.1, 0.1), "source": "typical smartphone"},
}


def load_table(extra: str | Path | None = None) -> dict[str, dict]:
    """The built-in table, plus (overriding) entries from a JSON file {name: entry}."""
    table = {k: dict(v) for k, v in PROPS.items()}
    if extra:
        for k, v in json.loads(Path(extra).read_text(encoding="utf-8")).items():
            v = dict(v)
            v.setdefault("source", str(extra))
            table[k] = v
    return table


def lookup(name: str, table: dict[str, dict] | None = None) -> dict:
    table = table or PROPS
    key = name.strip().lower()
    if key not in table:
        raise KeyError(f"prop {name!r} not in the table ({', '.join(sorted(table))}); give it in a JSON file "
                       "(shape, size, contacts, reach, colour, source)")
    return {"name": key, **table[key]}


__all__ = ["FEET", "HANDS", "PROPS", "load_table", "lookup"]
