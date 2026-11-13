"""Scene units of a run folder: units.json in the folder or a parent (written by court calibration: metres),
else the person placement's scale (scene/placement.json next to a parent)."""

from __future__ import annotations

import json
from pathlib import Path


def units_per_metre(folder: str | Path) -> float:
    folder = Path(folder).resolve()
    for d in [folder, *folder.parents]:
        if (d / "units.json").exists():
            return float(json.loads((d / "units.json").read_text(encoding="utf-8"))["units_per_metre"])
        if (d / "scene" / "placement.json").exists():
            return float(json.loads((d / "scene" / "placement.json").read_text(encoding="utf-8"))["scale_units_per_metre"])
    raise FileNotFoundError(f"no units.json or scene/placement.json above {folder}")


__all__ = ["units_per_metre"]
