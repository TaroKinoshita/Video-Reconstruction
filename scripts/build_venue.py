"""Phase C/D: layout (venue/layout.py) -> fitted to the observations -> scene, for a synthetic clip or tennis.

    python scripts/build_venue.py LAYOUT.json OBSERVATIONS.json OUT_DIR --synthetic CLIP_DIR
    python scripts/build_venue.py LAYOUT.json OBSERVATIONS.json OUT_DIR --tennis COURT_PROXIES_DIR

synthetic: the scene starts from the clip's source camera; observed things the layout does not take (labels
other than court/floor/net/stands/walls) become upright boxes from their observed extent (the players).
tennis: the scene starts from the court-frame reconstruction (scripts/proxies_scene.py --calibration): its people,
figures and chair are kept, its floor, net, stands plane and background mesh are replaced by the venue.
Writes OUT_DIR/project.json (+ part meshes), layout_fitted.json, fit_log.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

VENUE_LABELS = ("obj_court", "obj_floor", "obj_net", "obj_stand", "obj_barrier", "obj_wall", "obj_background")
LOOKS = {"court": (0.25, 0.35, 0.55), "ground": (0.35, 0.5, 0.35), "net": (0.85, 0.85, 0.85), "stands": (0.2, 0.35, 0.65),
         "wall": (0.15, 0.2, 0.4), "roof": (0.6, 0.6, 0.62), "furniture": (0.8, 0.8, 0.75), "other": (0.6, 0.6, 0.6)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("layout")
    ap.add_argument("observations")
    ap.add_argument("out")
    ap.add_argument("--synthetic", default=None)
    ap.add_argument("--tennis", default=None)
    a = ap.parse_args()
    from vrecon.schema import (Camera, FileRef, Pose, Project, Rational, SceneObject, Shot, ShotMapping, SourceVideo,
                               TransformKey, load_project, save_project)
    from vrecon.venue.build import build_venue
    from vrecon.venue.fit import fit_layout

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    layout = json.loads(Path(a.layout).read_text(encoding="utf-8"))
    obs = json.loads(Path(a.observations).read_text(encoding="utf-8"))
    fitted, log = fit_layout(layout, obs)
    looks = dict(LOOKS)
    if a.synthetic:
        clip = Path(a.synthetic)
        cam = Camera.model_validate(json.loads((clip / "gt_passes" / "source" / "camera.json").read_text(encoding="utf-8"))["camera"])
        objs = []
        used = {p.get("fit") for p in fitted["parts"]}
        for nm, o in obs["things"].items():
            if nm in used or nm.startswith(VENUE_LABELS):
                continue
            lo, hi = np.array(o["extent_min"]), np.array(o["extent_max"])
            size = np.maximum(hi - lo, 0.2)
            size[2] = max(o["z_range"][1], 0.3)
            objs.append(SceneObject(id=nm if nm.startswith("obj_") else "obj_" + nm, kind="rigid", size=tuple(map(float, size)),
                                    status="estimated", rest=TransformKey(time=0.0, pose=Pose(
                                        translation=(float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(size[2] / 2))))))
        w, h = cam.intrinsics.width, cam.intrinsics.height
        base = Project(id="proj_venue", source=SourceVideo(file=FileRef(id="file_video", path="source.png"), fps=Rational(num=24, den=1),
                                                            frame_count=1, width=w, height=h),
                       shots=[Shot(id="shot_001", start=0, end=1, transition_from_previous="first")],
                       reconstruction={"scale": {"source": "estimated", "note": "metres"}, "cameras": [cam], "objects": objs,
                                       "shot_mapping": [ShotMapping(shot_id="shot_001", scene_time_start=0.0)]})
        replace = set()
    else:
        src = Path(a.tennis).resolve()
        for f in src.glob("*.npz"):
            if f.name != "background.npz":
                shutil.copyfile(f, out / f.name)
        if (src / "units.json").exists():
            shutil.copyfile(src / "units.json", out / "units.json")
        base = load_project(src / "project.json")
        # the base already holds the observed people (obs_*: characters, figures) and the chair: layout boxes fitted
        # to those labels would stand twice
        dup = [q["id"] for q in fitted["parts"] if q["type"] == "box" and (str(q.get("fit", "")).startswith("obs_")
                                                                           or q.get("fit") == "obj_chair" or q.get("category") == "person")]
        fitted["parts"] = [q for q in fitted["parts"] if q["id"] not in dup]
        log.append({"dropped_duplicates_of_base": dup})
        replace = {o.id for o in base.reconstruction.objects if o.id.startswith(("obj_floor", "obj_net", "obj_stands", "obj_background"))}
        # observed colours where the source gives them
        for o in base.reconstruction.objects:
            if o.id == "obj_stands" and "all" in o.look:
                looks["stands"] = tuple(o.look["all"])
            if o.id == "obj_net" and "all" in o.look:
                looks["net"] = tuple(o.look["all"])
    p = build_venue(fitted, base, out, replace, looks=looks)
    save_project(p, out / "project.json")
    (out / "layout_fitted.json").write_text(json.dumps(fitted, indent=1) + "\n", encoding="utf-8")
    (out / "fit_log.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
