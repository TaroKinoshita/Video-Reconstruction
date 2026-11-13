"""Props into a scene, for any prop of analysis/prop_table.py (docs/PROPS_PLAN.md). Replaces tennis_props.py.

    python scripts/fit_props.py PROJECT.json VIDEO OUT_DIR --prop "NAME=MASKS.npz[:LABEL,LABEL]" [--prop ...]
        [--person obj_person_01] [--table extra_props.json] [--after after.json] [--one NAME]

--prop: a prop name from the table and the masks holding it (run_sam3.py or props_recut.py merge output); without
LABELs every label in the file is a candidate. Sticks (shape "stick") are fitted first; spheres then use them as
strikers. --after: per sphere name, the hypothesis for where it goes after a strike, e.g.
    {"tennis ball": {"target": [2.0, 5.8, 0.03], "speed": 45, "stop_abs_y": 18, "why": "a serve into the service box"}}
(the rules of the activity, not observed: an LLM or a person writes it, the file is recorded). --one NAME: the kept
tracklets of that sphere are one object (one in play). Writes OUT_DIR/project.json (scene + obj_<name> objects,
status estimated, per-frame notes), props.json and props_check.png.

Tennis (2026-11-14):
    python scripts/fit_props.py runs/venue/tennis/fal2c/project.json runs/human/tennis/tennis.mp4
        runs/venue/tennis/fal2c_props --prop "tennis racket=runs/human/tennis/props_masks.npz:obs_0"
        --prop "tennis ball=runs/human/tennis/recut/ball_masks.npz" --one "tennis ball" --after tennis_after.json
        --link 180
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from vrecon.analysis import prop_fit as F
from vrecon.analysis import props as P
from vrecon.analysis.prop_table import load_table, lookup
from vrecon.control.core import read_video_rgb
from vrecon.metrics.poses import camera_pose
from vrecon.scene.convert import matrix_to_quat
from vrecon.analysis.prop_shapes import prop_mesh, write_mesh
from vrecon.schema import Evidence, FileRef, Pose, SceneObject, TransformKey, load_project, save_project


def load_mask(spec: str, t_n: int):
    path, _, labels = spec.partition(":")
    z = np.load(path, allow_pickle=True)
    m = z["masks"]
    id_map = json.loads(str(z["id_map"]))
    want = [x for x in labels.split(",") if x] if labels else list(id_map.values())
    nums = [int(k) for k, v in id_map.items() if v in want]
    out = np.zeros((t_n,) + m.shape[1:], bool)
    n = min(t_n, len(m))
    out[:n] = np.isin(m[:n], nums)
    return out, {"file": path, "labels": want}


def obj_id(name: str) -> str:
    return "obj_" + name.replace(" ", "_")


def mesh_asset(entry: dict, s: float, oid: str, out: Path) -> FileRef:
    """The prop's stand-in shape as a mesh file next to the project (a box reads as a board to image models)."""
    name = f"{oid}.npz"
    write_mesh(out / name, prop_mesh(entry, s))
    return FileRef(id=f"file_{oid[4:]}_mesh", path=name, media_type="application/x-vrecon-mesh+npz")


def notes_to_evidence(notes: list) -> list:
    return [Evidence(source_frames=list(range(n["frames"][0], n["frames"][1] + 1)) if n["frames"][1] >= n["frames"][0] else [],
                     note=f"{n['model']}" + (f", rms {n['rms_px']} px" if "rms_px" in n and n["model"] not in ("held",) else ""))
            for n in notes]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--prop", action="append", required=True)
    ap.add_argument("--person", default="obj_person_01")
    ap.add_argument("--table", default=None)
    ap.add_argument("--after", default=None)
    ap.add_argument("--one", action="append", default=[])
    ap.add_argument("--link", type=float, default=60.0)
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--min-track", type=int, default=3)
    ap.add_argument("--hand-px", type=float, default=35.0)
    ap.add_argument("--max-rms", type=float, default=12.0)
    ap.add_argument("--hit-m", type=float, default=0.35)
    a = ap.parse_args()

    proj_path = Path(a.project)
    project = load_project(proj_path)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = load_table(a.table)
    after_all = json.loads(Path(a.after).read_text(encoding="utf-8")) if a.after else {}
    fps = project.source.fps.num / project.source.fps.den
    video = read_video_rgb(a.video)
    t_n = len(video)
    size = (video.shape[2], video.shape[1])
    cam = project.reconstruction.cameras[0]
    cams = [camera_pose(cam, f) for f in range(t_n)]
    times = P.frame_times(video, fps)
    person = next(o for o in project.reconstruction.objects if o.id == a.person)
    from vrecon.analysis.from_gvhmr import load_skeleton

    skel = load_skeleton(proj_path.parent / person.asset.path)
    rep = {"fps": fps, "repeated_frames": int(np.sum(np.diff(times) < 1e-9)), "table": a.table, "after": a.after,
           "props": {}}
    specs = []
    for p in a.prop:
        name, _, mspec = p.partition("=")
        entry = lookup(name, table)
        mask, src = load_mask(mspec, t_n)
        specs.append((entry, mask, src))
    specs.sort(key=lambda e: e[0]["shape"] != "stick")  # sticks first: spheres use them as strikers

    new_objs, strikes, overlays = [], None, []
    for entry, mask, src in specs:
        contacts, s = F.contact_points(person, skel, entry["contacts"], entry["reach"])
        r = {"entry": entry, "masks": src}
        oid = obj_id(entry["name"])
        if entry["shape"] == "stick":
            poses, notes, info = F.fit_stick(mask, contacts, cams, entry, s, size)
            r.update(info)
            strikes = F.strike_points(poses, entry, s)
            keys = [TransformKey(time=f / fps, pose=Pose(translation=tuple(float(x) for x in c),
                                                          rotation=tuple(matrix_to_quat(R.tolist()))))
                    for f, (c, R) in enumerate(poses)]
            ev = [Evidence(note=f"mask {src}; held by {info['contact']} (+{entry['reach']} m); pose from the silhouette "
                                f"(analysis/props.held_poses); size from the table ({entry['source']})")]
            ev += [Evidence(source_frames=[f], note=n) for f, n in enumerate(notes) if n != "sharp"][:400]
            new_objs.append(SceneObject(id=oid, name=entry["name"], kind="rigid", size=tuple(x * s for x in entry["size"]),
                                        asset=mesh_asset(entry, s, oid, out), keys=keys, status="estimated",
                                        evidence=ev))
            overlays.append(("stick", entry, poses, s))
        else:
            d = entry["size"][0] * s
            tracks = F.tracklets(mask, a.link, a.gap, a.min_track)
            kept, segs_all, left_out = [], [], 0
            for tr in tracks:
                pos, held, segs = F.fit_sphere_track(tr, t_n, contacts, cams, times, fps, d, s, a.hand_px, a.max_rms)
                if not held:
                    left_out += 1  # never by a contact: another object alike
                    continue
                kept.append((pos, held))
                segs_all += segs
            r.update({"tracklets": len(tracks), "left_out_never_by_a_contact": left_out, "kept": len(kept)})
            groups = [kept] if entry["name"] in a.one else [[k] for k in kept]
            for gi, grp in enumerate(groups):
                if not grp:
                    continue
                pos = np.full((t_n, 3), np.nan)
                counts = {}
                for p_, held in grp:
                    ok = np.all(np.isfinite(p_), 1) & ~np.all(np.isfinite(pos), 1)
                    pos[ok] = p_[ok]
                    for k in held.values():
                        counts[k] = counts.get(k, 0) + 1
                seen_frames = int(np.all(np.isfinite(pos), 1).sum())
                notes = list(segs_all)
                notes += F.carry_on(pos, segs_all, times, fps, strikes, s, d / 2, a.hit_m, after_all.get(entry["name"]))
                notes += F.fill_unseen(pos, contacts, counts)
                r[f"object_{gi}"] = {"placed_from_masks": seen_frames, "notes": notes}
                keys = [TransformKey(time=f / fps, pose=Pose(translation=tuple(float(x) for x in pos[f])))
                        for f in range(t_n) if np.all(np.isfinite(pos[f]))]
                ev = [Evidence(note=f"mask {src}; held / free states (analysis/prop_fit); size from the table "
                                    f"({entry['source']}); spin not observed")] + notes_to_evidence(notes)
                sid = oid if len(groups) == 1 else f"{oid}_{gi:02d}"
                new_objs.append(SceneObject(id=sid, name=entry["name"], kind="rigid", size=(d, d, d),
                                            asset=mesh_asset(entry, s, sid, out), keys=keys, status="estimated",
                                            evidence=ev))
                overlays.append(("sphere", entry, pos, s))
        rep["props"][entry["name"]] = r

    ids = {o.id for o in new_objs}
    project = project.model_copy(deep=True)
    project.reconstruction.objects = [o for o in project.reconstruction.objects if o.id not in ids] + new_objs
    for f in [*proj_path.parent.glob("*.npz"), *proj_path.parent.glob("units.json")]:
        shutil.copyfile(f, out / f.name)
    save_project(project, out / "project.json")

    tiles = []
    for f in np.linspace(0, t_n - 1, 8).astype(int):
        im = video[f].copy()
        for entry, mask, _ in specs:
            im[mask[f]] = (im[mask[f]] * 0.5 + np.array([0, 128, 128])).astype(np.uint8)
        for kind, entry, data, s in overlays:
            if kind == "stick":
                c, R = data[f]
                L, W, T = (x * s for x in entry["size"])
                grip = c - R[:, 0] * L / 2
                sil = P.silhouette(P._box_corners(grip, grip + R[:, 0] * L, R[:, 1], L, W, T), cams[f], size)
                cnt, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cnt, -1, (255, 255, 0), 2)
            elif np.all(np.isfinite(data[f])):
                uv = F.project_pts(data[f], cams[f])[0]
                cv2.circle(im, (int(uv[0]), int(uv[1])), 8, (255, 0, 0), 2)
        cv2.putText(im, str(f), (8, 24), 0, 0.7, (255, 255, 255), 2)
        tiles.append(cv2.resize(im, (406, 360)))
    ok, buf = cv2.imencode(".png", np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])[..., ::-1])
    buf.tofile(str(out / "props_check.png"))
    txt = json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    (out / "props.json").write_text(txt + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("entry",) and not kk.startswith("object_")}
                      for k, v in rep["props"].items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
