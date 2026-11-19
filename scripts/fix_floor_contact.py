"""Check every object placed by "floor contact of the SAM mask" against a floor mask of the video and re-place the
ones that do not stand on the floor (analysis/floor_contact.py). Scene-independent; writes a new project folder.

    python scripts/fix_floor_contact.py PROJECT_DIR MASKS.npz FLOOR_MASKS.npz OUT_DIR [--map LABEL=OBJECT ...]
        [--every 15]

MASKS.npz: the objects' masks (run_sam3.py format) on the project's source video; FLOOR_MASKS.npz: one label, the
floor (gpu/h3_control/comfy_sam.py --prompt "floor, ground"). Objects and mask labels are paired by --map, else by
the proxies_scene rule (obs_N -> obj_person_box_NN, obj_X -> obj_X).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

MIN_FARTHER = 1.05  # move an off-floor object only when the surface behind its foot ray is this much farther


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir", type=Path)
    ap.add_argument("masks", type=Path)
    ap.add_argument("floor", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--map", nargs="*", default=[])
    ap.add_argument("--every", type=int, default=15)
    a = ap.parse_args()

    from vrecon.analysis.floor_contact import check, is_floor_piece, surface_depth
    from vrecon.metrics.poses import camera_pose
    from vrecon.scene.builder import build_scene
    from vrecon.scene.passes import render_passes
    from vrecon.schema import Evidence, load_project, save_project

    p = load_project(a.project_dir / "project.json")
    z = np.load(a.masks)
    masks, id_map = z["masks"], json.loads(str(z["id_map"]))
    floor = np.load(a.floor)["masks"] > 0
    by_label = {v: int(k) for k, v in id_map.items()}
    pairs = dict(x.split("=", 1) for x in a.map)
    for lab in by_label:
        pairs.setdefault(lab, "obj_person_box_" + lab.removeprefix("obs_").zfill(2) if lab.startswith("obs_") else lab)
    objs = {o.id: o for o in p.reconstruction.objects}
    cands = {oid: lab for lab, oid in pairs.items()
             if oid in objs and any("floor contact" in e.note for e in objs[oid].evidence)}
    n = min(len(masks), len(floor), p.source.frame_count)
    frames = list(range(0, n, a.every))

    report = {}
    off_floor = []
    for oid, lab in cands.items():
        res = check({f: masks[f] == by_label[lab] for f in frames}, {f: floor[f] for f in frames})
        report[oid] = {"label": lab, **res}
        if not res["on_floor"]:
            off_floor.append(oid)

    dep = {}
    if off_floor:  # where the foot rays first meet the scene once the floor and the candidates are taken out
        q = p.model_copy(deep=True)
        q.reconstruction.objects = [o for o in q.reconstruction.objects if o.id not in cands and not is_floor_piece(o)]
        with tempfile.TemporaryDirectory() as tmp:
            build_scene(q, root=a.project_dir)
            render_passes(tmp, shot_frames=frames, shaded=False)
            depth = np.load(Path(tmp) / "depth.npz")["depth"].astype(np.float32)
        dep = {f: depth[i] for i, f in enumerate(frames)}
    cam = p.reconstruction.cameras[0]
    poses = [camera_pose(cam, f) for f in frames]
    c_ref = np.median([c for _, c, _ in poses], axis=0)
    r_ref = poses[len(poses) // 2][0]

    a.out.mkdir(parents=True, exist_ok=True)
    for f in a.project_dir.glob("*"):
        if f.is_file() and f.name != "project.json":
            shutil.copyfile(f, a.out / f.name)
    p2 = p.model_copy(deep=True)
    for o in p2.reconstruction.objects:
        if o.id not in off_floor:
            continue
        if p2.is_locked(o.id, "rest") or p2.is_locked(o.id, "size"):
            report[o.id].update({"moved": False, "why": "locked by a human"})
            continue
        zs = surface_depth({f: masks[f] == by_label[cands[o.id]] for f in frames}, dep)
        t = np.asarray(o.rest.pose.translation, float)
        bottom = t - np.array([0, 0, o.size[2] / 2])
        z_now = float((bottom - c_ref) @ r_ref[:, 2])
        if zs is None or z_now <= 0:
            report[o.id]["moved"] = False
            continue
        s = zs / z_now
        if s < MIN_FARTHER:  # the surface found is in front: something hides the feet (a net, a rail), the object
            # may still stand on the floor behind it; an off-floor object is behind something, so it only moves back
            report[o.id].update({"moved": False, "why": f"surface in front (x{s:.2f}): an occluder of the feet"})
            continue
        nb = c_ref + s * (bottom - c_ref)
        size = tuple(float(x) * s for x in o.size)
        nt = nb + np.array([0, 0, size[2] / 2])
        o.rest = o.rest.model_copy(update={"pose": o.rest.pose.model_copy(update={"translation": tuple(map(float, nt))})})
        o.size = size
        o.status = "hypothesis"
        o.evidence = list(o.evidence) + [Evidence(kind="rule", note=(
            f"not on the floor: below its mask the floor mask is missing in {report[o.id]['off_share']:.0%} of "
            f"{report[o.id]['frames']} frames; placed where its foot ray meets the scene without the floor "
            f"(depth {zs:.2f} instead of {z_now:.2f}), size x{s:.2f} (analysis/floor_contact)"))]
        report[o.id].update({"moved": True, "scale": s, "from": t.tolist(), "to": nt.tolist()})
    save_project(p2, a.out / "project.json")
    (a.out / "floor_contact.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in r.items()
                          if kk in ("on_floor", "off_share", "frames", "moved", "scale", "to")} for k, r in report.items()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
