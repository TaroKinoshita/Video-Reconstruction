"""Colour-pass renders of several projects from camera moves (scripts/camera_moves.py), in one grid.

    python scripts/clay_views.py OUT.png TAG=PROJECT.json [TAG=PROJECT.json ...] [--moves source,orbit30] [--frame N]
        [--kind clay|look|id]
        [--drop obj_background_rest]

Mesh files resolve against each project's folder, or the nearest parent holding skeleton.npz. --drop leaves objects out (e.g. the reference mesh that
should not go into control videos for new cameras).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("projects", nargs="+")
    ap.add_argument("--moves", default="source,orbit30,orbit-30", help="scripts/camera_moves.py moves")
    ap.add_argument("--frame", type=int, default=-1, help="frame to draw (default: the last, where a move ends)")
    ap.add_argument("--kind", default="clay", help="colour pass: clay, look, id")
    ap.add_argument("--drop", default="")
    a = ap.parse_args()
    import cv2
    from camera_moves import move_camera, pivot_of

    from vrecon.control.core import load_frames
    from vrecon.scene.builder import build_scene
    from vrecon.scene.color_passes import render_color_pass
    from vrecon.schema import load_project

    drop = set(filter(None, a.drop.split(",")))
    grid = []
    for spec in a.projects:
        tag, path = spec.split("=", 1)
        path = Path(path).resolve()
        p = load_project(path)
        # mesh files: the project's folder, or the nearest parent that has them (completion outputs sit in a subfolder)
        root = next((d for d in [path.parent, *path.parents] if (d / "skeleton.npz").exists()), path.parent)
        cam = p.reconstruction.cameras[0]
        pv = pivot_of(p)
        from vrecon.analysis.units import units_per_metre

        try:
            upm = units_per_metre(path.parent)
        except FileNotFoundError:
            upm = 1.0
        n = p.source.frame_count
        fps = p.source.fps.num / p.source.fps.den
        frame = a.frame if a.frame >= 0 else n - 1
        row = []
        for mv in a.moves.split(","):
            q = p.model_copy(deep=True)
            q.reconstruction.cameras = [move_camera(cam, mv, pv, upm, n, fps, fps)]
            q.reconstruction.objects = [o for o in q.reconstruction.objects if o.id not in drop]
            build_scene(q, root=root)
            with tempfile.TemporaryDirectory() as tmp:
                from vrecon.scene.color_passes import identity_colors

                x = load_frames(render_color_pass(a.kind, tmp, [frame], identity_colors(q) if a.kind == "id" else {}))[0]
            x = cv2.resize(x, (480, int(480 * x.shape[0] / x.shape[1])))
            cv2.putText(x, f"{tag} {mv}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            row.append(x)
        grid.append(np.concatenate(row, 1))
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.concatenate(grid, 0), cv2.COLOR_RGB2BGR))
    buf.tofile(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
