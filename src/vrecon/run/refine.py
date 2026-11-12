"""Option 4 (docs/DIRECTION_OPTIONS.md): refine an LLM-made .blend with the repair loop.

import (scene/blend_import) -> mark the set (flat static meshes; floor/wall get the mask source's labels)
-> run on the masks with the fixed hypothesis order -> write the result back into a copy of the .blend
(scene/blend_export). No ground truth is read here; scoring is the caller's business.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

FLAT = 0.02  # thinnest side / longest side below this: a plane (floor, wall, backdrop)


def mark_set(project) -> dict:
    """Static flat meshes -> kind 'set'; the largest horizontal becomes obj_floor, the largest vertical obj_wall."""
    from vrecon.metrics.poses import _rot

    best = {"obj_floor": (0.0, None), "obj_wall": (0.0, None)}
    for o in project.reconstruction.objects:
        if o.keys:
            continue
        s = np.array(o.size) * np.array(o.rest.scale)
        thin = int(np.argmin(s))
        if s[thin] > FLAT * s.max():
            continue
        o.kind = "set"
        n = np.asarray(_rot(o.rest.pose.rotation))[:, thin]
        area = float(np.prod(np.delete(s, thin)))
        label = "obj_floor" if abs(n[2]) > 0.8 else "obj_wall" if abs(n[2]) < 0.3 else None
        if label and area > best[label][0]:
            best[label] = (area, o)
    renamed = {}
    for label, (_, o) in best.items():
        if o is not None:
            renamed[o.id] = label
            o.id = label
    return renamed


def refine_blend(blend: str | Path, masks: str | Path, run_dir: str | Path, size: tuple[int, int],
                 passes: int = 2, attempts: int = 30, write_back: bool = True, focal: bool = False,
                 tracks: bool = False) -> dict:
    """masks: a clip folder (ground-truth stand-in, as C) or a masks .npz. size: (w, h) of the masks.

    focal and tracks default off: on GPT's B scenes (syn_04) the point-track camera loss, which assumes the
    floor and wall are where the scene puts them, pulled the focal length from 3% to 69% error."""
    import json
    from vrecon.scene.blend_import import import_blend
    from vrecon.schema import save_project

    from . import service

    run = Path(run_dir).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        project, warnings = import_blend(blend, Path(tmp), project_id="proj_b_refine", size=size)
    renamed = mark_set(project)
    run.mkdir(parents=True, exist_ok=False)
    start = run / "start_from_blend.json"
    save_project(project, start)
    save_project(project, run / "start_from_blend.orig.json")  # the run overwrites `start` with each adopted state
    service.init_run(str(start), str(run), str(masks), max_attempts=attempts, focal=focal)
    if not tracks:
        cfg_path = run / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["video"] = None
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    out = service.auto(str(run), passes=passes)
    rec = {"blend": str(Path(blend).resolve()), "import_warnings": warnings, "renamed": renamed,
           "objects": [{"id": o.id, "kind": o.kind, "animated": bool(o.keys)} for o in project.reconstruction.objects],
           "accepted": out["accepted"], "tried": out["tried"], "metrics_before": out["metrics_before"],
           "metrics_after": out["metrics_after"],
           "changes": [a["hypothesis"] for a in out["attempts"] if a["status"] == "accepted"]}
    if write_back:
        from vrecon.scene.blend_export import export_to_blend
        from vrecon.schema import load_project

        rec["refined_blend"] = str(run / "refined.blend")
        rec["export_notes"] = export_to_blend(load_project(start), project, blend, run / "refined.blend", renamed)
    return rec


__all__ = ["FLAT", "mark_set", "refine_blend"]
