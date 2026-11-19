"""PLAN v2 7.2: improve an LLM's draft .blend for the camera moves a user asks for, without removing anything it made.

    python scripts/improve_blend.py DRAFT.blend OUT_DIR [--moves orbit45,dolly-60] [--steps outside]

Steps (each generic; no per-clip values):
  outside   the space the source camera did not see (part 1):
            - flat static pieces become set pieces (run/refine.mark_set: the largest floor and wall get their roles)
            - every set piece grows in its plane until it fills each move's view (run/complete.extend_set_to_view),
              never covering what the source camera shows (protect = the source camera)
            - the background structure (static objects behind the action: farther from the pivot than every moving
              thing, seen from the source camera, and at least 0.15 x the camera-to-pivot distance across) is copied around the pivot at 90/180/270 deg, pushed out so the
              cameras and the action stay inside (run/complete.enclose); the copies are grown like the rest
Writes OUT_DIR/improved.blend (the draft with the changes written back by scene/blend_export, new objects as copies
of their sources), OUT_DIR/project.json (the improved scene as a Project) and OUT_DIR/improve.json (what was done).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def move_cams(project, moves: list[str], pivot: np.ndarray):
    from camera_moves import move_camera

    cam = project.reconstruction.cameras[0]
    n, fps = project.source.frame_count, project.source.fps.value
    return {m: move_camera(cam, m, pivot, 1.0, n, fps, fps, step=max(n // 6, 1)) for m in moves}


def bake_scale(project):
    """Static objects: the per-axis scale goes into `size` (scale 1), so geometry rules see real dimensions. The
    write-back scales by new size / old size, so the .blend's own scale is kept and multiplied."""
    p = project.model_copy(deep=True)
    for o in p.reconstruction.objects:
        if o.keys or not o.size:
            continue
        s = np.asarray(o.rest.scale, float)
        o.size = tuple(float(x) for x in np.asarray(o.size, float) * s)
        o.rest = o.rest.model_copy(update={"scale": (1.0, 1.0, 1.0)})
    return p


def unbake_scale(project, scales: dict):
    """Back to the .blend's own scale (by id; a copy takes its source's), for the write-back."""
    from vrecon.scene.blend_export import _source_of

    p = project.model_copy(deep=True)
    for o in p.reconstruction.objects:
        sc = scales.get(o.id) or scales.get(_source_of(o) or "")
        if o.keys or not o.size or sc is None:
            continue
        o.size = tuple(float(x) for x in np.asarray(o.size, float) / np.asarray(sc, float))
        o.rest = o.rest.model_copy(update={"scale": tuple(float(x) for x in sc)})
    return p


def pivot_and_movers(project):
    from vrecon.metrics.poses import object_pose

    shot = project.shots[0].id
    movers = [o for o in project.reconstruction.objects if o.keys and o.kind != "set"]
    base = movers or [o for o in project.reconstruction.objects if o.kind != "set"]
    pts = np.array([object_pose(project, o, shot, 0)[1] for o in base]) if base else np.zeros((1, 3))
    pivot = np.median(pts, axis=0)
    pivot[2] = 0.0
    return pivot, movers, pts


def _blocks(o, project, shot, a: np.ndarray, b: np.ndarray, f: int) -> bool:
    """Does the segment a-b pass through the object's box at frame f?"""
    from vrecon.metrics.poses import object_pose

    r, t, sc = object_pose(project, o, shot, f)
    half = 0.5 * np.array(o.size) * np.asarray(sc, float)
    pa, pb = r.T @ (a - t), r.T @ (b - t)
    d = pb - pa
    lo, hi = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if abs(pa[k]) > half[k]:
                return False
            continue
        t1, t2 = sorted(((-half[k] - pa[k]) / d[k], (half[k] - pa[k]) / d[k]))
        lo, hi = max(lo, t1), min(hi, t2)
        if lo > hi:
            return False
    return 0.02 < lo < 0.98 or 0.02 < hi < 0.98


def guard(project, original_sizes: dict, added: list[str], src, cams: dict, frames: list[int], pts) -> tuple:
    """Undo what got in the way. An added copy that covers the source view (run/complete.covers_source) or stands
    between any camera (source or move) and the action is removed; a grown piece that does is put back to its old
    size. The action = the moving things' positions (and the pivot) at those frames."""
    from vrecon.metrics.poses import camera_pose, object_pose
    from vrecon.run.complete import box_face_points, covers_source

    shot = project.shots[0].id
    targets = list(pts)
    removed, reverted = [], []
    keep = []
    for o in project.reconstruction.objects:
        changed = o.id in added or (o.id in original_sizes and tuple(o.size or ()) != tuple(original_sizes[o.id][0] or ()))
        if not changed:
            keep.append(o)
            continue
        bad = bool(o.id in added and covers_source(box_face_points(o), src, frames, None).any())
        for cam in [src, *cams.values()]:
            if bad:
                break
            for f in frames:
                c = np.array(camera_pose(cam, f)[1])
                if any(_blocks(o, project, shot, c, np.asarray(p, float) + np.array([0, 0, 0.3]), f) for p in targets):
                    bad = True
                    break
        if not bad:
            keep.append(o)
        elif o.id in added:
            removed.append(o.id)
        else:
            size, rest = original_sizes[o.id]
            keep.append(o.model_copy(update={"size": size, "rest": rest}))
            reverted.append(o.id)
    project = project.model_copy(deep=True)
    project.reconstruction.objects = keep
    return project, {"removed": removed, "reverted": reverted}


def step_outside(project, moves: list[str], log: dict):
    from vrecon.metrics.poses import camera_pose, object_pose
    from vrecon.run.complete import enclose, extend_set_to_view

    shot = project.shots[0].id
    src = project.reconstruction.cameras[0]
    n = project.source.frame_count
    pivot, movers, pts = pivot_and_movers(project)
    cams = move_cams(project, moves, pivot)
    frames = sorted({0, n // 2, n - 1})
    c_src = np.array(camera_pose(src, 0)[1])
    reach = max(float(np.linalg.norm((p - pivot)[:2])) for p in pts) if len(pts) else 0.0
    # background structure: static, not a set piece, farther from the pivot than the action, on the far side
    bg = []
    for o in project.reconstruction.objects:
        if o.keys or o.kind == "set" or not o.size:
            continue
        t = np.array(object_pose(project, o, shot, 0)[1])
        d = float(np.linalg.norm((t - pivot)[:2]))
        behind = float((t - pivot)[:2] @ (pivot - c_src)[:2]) > 0
        big = max(o.size) * float(np.max(o.rest.scale)) >= 0.15 * float(np.linalg.norm((c_src - pivot)[:2]))
        if d > reach and behind and big:  # structure (bed, stands, shelves), not small props
            bg.append(o.id)
    walls = [o.id for o in project.reconstruction.objects if o.kind == "set" and o.id != "obj_floor"]
    contain = [np.array(camera_pose(c, f)[1]) for c in cams.values() for f in frames] + [c_src] + list(pts)
    original_sizes = {o.id: (o.size, o.rest) for o in project.reconstruction.objects}
    project, rec = enclose(project, pivot, walls + bg, contain=contain, margin=0.3)
    log["enclose"] = rec
    grown = []
    for o in [x for x in project.reconstruction.objects if x.kind == "set" or x.id in rec["new"]]:
        if o.kind != "set" and min(o.size) > 0.05 * max(o.size):
            continue  # only flat pieces grow in their plane
        for m, cam in cams.items():
            try:
                project, r = extend_set_to_view(project, o.id, cam, frames, protect=src, protect_frames=frames)
                grown.append({"object": o.id, "move": m, **{k: v for k, v in r.items() if k in ("factor", "grew", "size")}})
            except Exception as e:  # noqa: BLE001  (a piece the view never meets is left as it is)
                grown.append({"object": o.id, "move": m, "skipped": str(e)[:120]})
    log["extend"] = grown
    project, log["guard"] = guard(project, original_sizes, rec["new"], src, cams, frames, pts)
    log["background"] = bg
    log["pivot"] = pivot.tolist()
    return project


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("draft", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--moves", default="orbit45,dolly-60")
    ap.add_argument("--steps", default="outside")
    a = ap.parse_args()
    from vrecon.run.refine import mark_set
    from vrecon.scene.blend_export import export_to_blend
    from vrecon.scene.blend_import import import_blend
    from vrecon.schema import save_project

    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        project, warnings = import_blend(a.draft, Path(tmp), project_id="proj_improve", mesh_dir=a.out / "meshes")
    scales = {o.id: tuple(o.rest.scale) for o in project.reconstruction.objects if not o.keys}
    renamed = mark_set(project)
    scales = {renamed.get(k, k): v for k, v in scales.items()}
    original = project.model_copy(deep=True)
    project = bake_scale(project)  # rules below read sizes: a cube scaled to a thin wall must look thin to them
    log = {"draft": str(a.draft), "renamed": renamed, "warnings": warnings}
    moves = a.moves.split(",")
    for step in a.steps.split(","):
        if step == "outside":
            project = step_outside(project, moves, log)
    save_project(project, a.out / "project.json")
    out = unbake_scale(project, scales)
    log["export"] = export_to_blend(out, original, a.draft, a.out / "improved.blend", renamed)
    (a.out / "improve.json").write_text(json.dumps(log, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: (v if k != "export" else len(v)) for k, v in log.items() if k in ("renamed", "background", "export")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
