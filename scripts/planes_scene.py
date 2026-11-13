"""Replace the background mesh with flat pieces (analysis/planes.split_planes), so camera moves can be completed.

    python scripts/planes_scene.py SCENE_DIR OUT_DIR [--tol-cm 19] [--cell-cm 38]

SCENE_DIR: a scripts/proxies_scene.py output (project.json, background.npz, skeleton.npz, figure meshes).
Pieces: vertical planes only (walls, signs). A piece whose bottom is less than half its height above the floor is
stood on the floor (rule "walls stand on the floor": the lower part is hidden behind the net and people, or
cut with the floor band). What no piece takes stays as obj_background_rest, a mesh kept as a reference of what
the source showed (status "observed reference"; leave it out of control videos for new cameras).
Writes OUT_DIR/project.json, background_rest.npz, planes.json, and copies the other mesh files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("out")
    ap.add_argument("--tol-cm", type=float, default=19.0)
    ap.add_argument("--cell-cm", type=float, default=38.0)
    a = ap.parse_args()
    from vrecon.analysis.planes import split_planes
    from vrecon.schema import Evidence, FileRef, Pose, SceneObject, TransformKey, load_project, save_project

    src, out = Path(a.scene).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.npz"):
        if f.name != "background.npz":
            shutil.copyfile(f, out / f.name)
    from vrecon.analysis.units import units_per_metre

    s = units_per_metre(src)
    p = load_project(src / "project.json")
    bg = np.load(src / "background.npz")
    v, faces, cols = bg["verts"], bg["faces"], bg["colors"]
    pieces, rest = split_planes(v, tol=a.tol_cm / 100 * s, cell=a.cell_cm / 100 * s, thickness=0.05 * s)

    objs, rep = [], []
    for i, pc in enumerate(pieces, 1):
        c, size = np.array(pc["centre"], float), list(pc["size"])
        bottom = c[2] - size[1] / 2  # vertical piece: in-plane axis b is up
        note = f"vertical plane through {pc['points']} background points (analysis/planes.split_planes)"
        snapped = bool(0 < bottom < 0.5 * size[1])
        if snapped:
            top = c[2] + size[1] / 2
            size[1], c[2] = top, top / 2
            note += "; rule: stood on the floor (bottom was %.2f m up)" % (bottom / s)
        oid = f"obj_wall_{i:02d}"
        look = tuple(float(x) for x in np.median(cols[pc["index"]], 0))
        objs.append(SceneObject(id=oid, name=f"wall {i}", kind="set", size=tuple(size), status="estimated",
                                look={"all": look}, evidence=[Evidence(note=note)],
                                rest=TransformKey(time=0.0, pose=Pose(translation=tuple(map(float, c)), rotation=pc["rotation"]))))
        rep.append({"id": oid, "points": pc["points"], "size_m": [round(x / s, 2) for x in size[:2]],
                    "normal": np.round(pc["normal"], 2).tolist(), "snapped_to_floor": snapped})

    keep_f = rest[faces].all(1)
    used = np.unique(faces[keep_f])
    remap = -np.ones(len(v), int)
    remap[used] = np.arange(len(used))
    np.savez_compressed(out / "background_rest.npz", verts=v[used], faces=remap[faces[keep_f]], colors=cols[used])
    old = next(o for o in p.reconstruction.objects if o.id == "obj_background")
    rest_obj = old.model_copy(update={
        "id": "obj_background_rest", "name": "background rest (reference)",
        "asset": FileRef(id="file_background_rest", path="background_rest.npz", media_type="application/x-vrecon-mesh+npz"),
        "evidence": [Evidence(note="observed reference: background points no plane took; leave out of control videos for new cameras")]})
    p.reconstruction.objects = [o for o in p.reconstruction.objects if o.id != "obj_background"] + objs + [rest_obj]
    save_project(p, out / "project.json")
    summary = {"pieces": rep, "points": int(len(v)), "residual_share": round(float(rest.mean()), 3)}
    (out / "planes.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
