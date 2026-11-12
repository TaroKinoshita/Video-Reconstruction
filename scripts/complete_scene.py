"""Plan step 4 on a reconstructed scene: rule-based completion for camera moves (run/complete.py), then coverage
from those cameras to see what got filled.

    python scripts/complete_scene.py PROJECT.json OUT_DIR [--orbits 30,-30] [--frames 0,120,240] [--step 13]

Rules applied (no judgement, no ground truth): every flat set piece without a mesh file (floor, stands, walls)
that bounds the scene (nothing else beyond its plane, seen from the source camera)
is grown in its plane to fill each new camera's view (extend_set_to_view, all sampled frames), but not into
the source camera's picture (protect=source camera: the extension must not cover what was observed). The cameras are
orbits about the main character (scripts/coverage_scene.orbit), standing for the camera move a user asks for.
Writes OUT_DIR/project.json (completed; changed objects are status "assumed" with a "rule:" note),
OUT_DIR/<before|after>/<view>/ coverage renders, OUT_DIR/completion.json (operations; per view the share of the
picture with nothing in it (black) and the share showing unseen faces (red), before and after).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def shares(paths) -> dict:
    from vrecon.control.core import load_frames

    fr = load_frames(paths).astype(int)
    empty = (fr.sum(-1) == 0).mean()
    red = ((fr[..., 0] > 200) & (fr[..., 1] < 120) & (fr[..., 2] < 120)).mean()
    return {"empty": round(float(empty), 3), "unseen_red": round(float(red), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("out")
    ap.add_argument("--orbits", default="30,-30")
    ap.add_argument("--frames", default="0,120,240")
    ap.add_argument("--step", type=int, default=13)
    a = ap.parse_args()
    from coverage_scene import orbit

    from vrecon.metrics.poses import camera_pose
    from vrecon.run.complete import extend_set_to_view
    from vrecon.scene.coverage import render_coverage
    from vrecon.schema import load_project, save_project

    proj_path = Path(a.project).resolve()
    root = proj_path.parent
    p = load_project(proj_path)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cam = p.reconstruction.cameras[0]
    ch = next(o for o in p.reconstruction.objects if o.kind == "character")
    pivot = np.array((ch.keys[0] if ch.keys else ch.rest).pose.translation, float)
    pivot[2] = 0.0
    n = p.source.frame_count
    src = list(range(0, n, a.step))
    frames = [int(f) for f in a.frames.split(",")]
    views = {f"orbit{int(x):+d}": orbit(cam, float(x), pivot) for x in a.orbits.split(",")}

    from vrecon.metrics.poses import _rot

    eye = np.array(camera_pose(cam, 0)[1])

    def boundary(o) -> bool:
        """Only pieces that bound the scene grow: no thing (person, net, chair) lies beyond the piece's plane (on the side away
        from the source camera). A side wall in the middle of the scene, grown sideways, cut through the people
        in front of it (tennis wall pieces, 2026-11-13). The floor always grows."""
        if o.id == "obj_floor":
            return True
        r = np.asarray(_rot(o.rest.pose.rotation))
        n = r[:, int(np.argmin(o.size))]
        c = np.array(o.rest.pose.translation)
        if (eye - c) @ n > 0:
            n = -n  # n points away from the camera
        for x in p.reconstruction.objects:
            if x.kind == "set":  # walls behind walls are fine; what must not be cut is things (people, net, chairs)
                continue
            t = np.array((x.keys[0] if x.keys else x.rest).pose.translation)
            if (t - c) @ n > 0.02 * np.linalg.norm(c - eye):
                return False
        return True

    grow = [o.id for o in p.reconstruction.objects if o.kind == "set" and o.asset is None and boundary(o)]
    skipped = [o.id for o in p.reconstruction.objects if o.kind == "set" and o.asset is None and o.id not in grow]
    # what the source camera saw, as depth: extensions may not stand in front of it (run/complete.protect_depth)
    import tempfile

    from vrecon.scene.builder import build_scene
    from vrecon.scene.passes import render_passes

    build_scene(p, root=root)
    with tempfile.TemporaryDirectory() as tmp:
        render_passes(tmp, shot_frames=src, shaded=False)
        dep = np.load(Path(tmp) / "depth.npz")["depth"]
    pdepth = {f: dep[i] for i, f in enumerate(src)}
    done, ops = p, [{"op": "boundary_check", "grow": grow, "not_boundary": skipped}]
    for name, v in views.items():
        for o in [o for o in done.reconstruction.objects if o.id in grow]:
            done, r = extend_set_to_view(done, o.id, v, frames=src, protect=cam, protect_frames=src, protect_depth=pdepth)
            ops.append({"view": name, **r})
    save_project(done, out / "project.json")

    res = {"ops": ops, "views": {}}
    for name, v in views.items():
        row = {}
        for tag, proj in (("before", p), ("after", done)):
            paths, _ = render_coverage(proj, out / tag / name, src, view=v, view_frames=frames, root=root)
            row[tag] = shares(paths)
        res["views"][name] = row
    (out / "completion.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res["views"], indent=1))
    print(json.dumps(ops[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
