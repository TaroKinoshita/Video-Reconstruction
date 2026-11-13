"""Phase B end: move a reconstruction into the calibrated court frame (analysis/to_court.py) and check it.

    python scripts/scene_to_court.py PROJECT.json ROOT CALIBRATION.json MASKS.npz OUT_DIR [--check 0,120,240]

ROOT: folder of the project's mesh files (copied to OUT_DIR). The calibrated camera replaces the old one.
Check: source-view index pass at --check frames, IoU of each SAM mask with the object that stands for it
(net, chair, stands, people: obj_person_01/02 and obj_person_box_NN), before (old camera, old world) and after.
Writes OUT_DIR/project.json, court.json (similarity, checks), scene.blend.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    for x in ("project", "root", "calibration", "masks", "out"):
        ap.add_argument(x)
    ap.add_argument("--check", default="0,120,240")
    a = ap.parse_args()
    import cv2

    from vrecon.analysis.to_court import court_cameras, similarity_to_court, transform_project
    from vrecon.scene.builder import build_scene, save_blend
    from vrecon.scene.passes import render_passes
    from vrecon.schema import load_project, save_project

    out, root = Path(a.out).resolve(), Path(a.root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for f in root.glob("*.npz"):
        shutil.copyfile(f, out / f.name)
    p = load_project(a.project)
    cal = json.loads(Path(a.calibration).read_text(encoding="utf-8"))
    A, k, T, rep = similarity_to_court(p, cal)
    cam = court_cameras(cal, p.reconstruction.cameras[0].id, p.reconstruction.cameras[0].shot_id)
    q = transform_project(p, A, k, T, cam)
    save_project(q, out / "project.json")

    z = np.load(a.masks)
    masks = z["masks"]
    id_map = {int(kk): v for kk, v in json.loads(str(z["id_map"])).items()}
    chk = [int(x) for x in a.check.split(",")]

    def ious(proj, rt):
        build_scene(proj, root=rt)
        with tempfile.TemporaryDirectory() as t:
            render_passes(t, shot_frames=chk, shaded=False)
            idx = np.load(Path(t) / "index.npz")["index"]
            rm = {int(kk): v for kk, v in json.loads((Path(t) / "id_map.json").read_text(encoding="utf-8")).items()}
        objs = {v for v in rm.values()}
        res = {}
        for kk, lab in id_map.items():
            cands = [lab] if lab in objs else []
            if lab.startswith("obs_"):
                n = lab.removeprefix("obs_").zfill(2)
                cands = [o for o in objs if o.endswith(f"box_{n}")] or []
            if not cands:
                continue
            rk = [r for r, v in rm.items() if v in cands]
            vals = []
            for i, f in enumerate(chk):
                g = cv2.resize(masks[f], (idx.shape[2], idx.shape[1]), interpolation=cv2.INTER_NEAREST) == kk
                r = np.isin(idx[i], rk)
                u = (g | r).sum()
                if g.sum():
                    vals.append(float((g & r).sum() / u) if u else 0.0)
            res[lab] = round(float(np.mean(vals)), 3) if vals else None
        # the posed people (characters) by overlap with the largest people masks
        for o in [o for o in proj.reconstruction.objects if o.kind == "character"]:
            rk = [r for r, v in rm.items() if v == o.id]
            best = 0.0
            for kk, lab in id_map.items():
                if not lab.startswith("obs_"):
                    continue
                vals = []
                for i, f in enumerate(chk):
                    g = cv2.resize(masks[f], (idx.shape[2], idx.shape[1]), interpolation=cv2.INTER_NEAREST) == kk
                    r = np.isin(idx[i], rk)
                    u = (g | r).sum()
                    vals.append(float((g & r).sum() / u) if u else 0.0)
                best = max(best, float(np.mean(vals)))
            res[o.id] = round(best, 3)
        return res

    rep["iou_before"] = ious(p, root)
    rep["iou_after"] = ious(q, out)
    save_blend(out / "scene.blend")
    (out / "court.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
