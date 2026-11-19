"""Checks on a scene and its control video before paying for generation (after CWM 2.0's proxy quality control).

    python scripts/proxy_qc.py IN.blend CATEGORIES.json CONTROL.mp4 OUT.json [--masks MASKS.npz] [--frames 5]

1. Source view vs observed masks (when given): the scene's people rendered from the .blend's source camera against
   the observed person masks (all obs_* labels; else the label overlapping them most); IoU per sampled frame. Below MIN_IOU: fail.
2. Control video: every frame has pixels (not empty), no frame is a copy of the previous one for the whole clip
   (frozen), and the frame count / size meet H3 (17n+5, multiples of 32).
Writes OUT.json {"pass": bool, "checks": {...}}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MIN_IOU = 0.3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("blend", type=Path)
    ap.add_argument("categories", type=Path)
    ap.add_argument("control", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--masks", type=Path)
    ap.add_argument("--frames", type=int, default=5)
    a = ap.parse_args()
    import bpy

    from vrecon.control.core import read_video_rgb
    from vrecon.scene.novel_blend import open_scene, render_view

    checks, ok = {}, True
    if a.masks:
        cats = json.loads(a.categories.read_text(encoding="utf-8"))["categories"]
        rec = open_scene(a.blend)
        cam = rec.reconstruction.cameras[0]
        z = np.load(a.masks)
        m, idmap = z["masks"], {int(k): v for k, v in json.loads(str(z["id_map"])).items()}

        def top(o):
            while o.parent is not None:
                o = o.parent
            return o

        people = {o.get("vrecon_id") for o in bpy.data.objects if o.type == "MESH" and not o.hide_render and (
            o.get("vrecon_id") == "obj_person_observed" or cats.get(top(o).name.split(".")[0]) == "person")}
        n = min(len(m), rec.source.frame_count)
        frames = [int(x) for x in np.linspace(0, n - 1, a.frames)]
        ious = []
        for f in frames:
            idx, imap = render_view(cam, [f])
            r = np.isin(idx[0], [k for k, v in imap.items() if v in people])
            mm = m[f]
            if mm.shape != r.shape:
                import cv2

                mm = cv2.resize(mm.astype(np.uint8), (r.shape[1], r.shape[0]), interpolation=cv2.INTER_NEAREST)
            obs = [k for k, v in idmap.items() if v.startswith("obs")]  # observed people labels (run_sam3.py)
            if obs:
                g = np.isin(mm, obs)
                best = (r & g).sum() / max((r | g).sum(), 1)
            else:
                labs = [k for k in idmap if (mm == k).any()]
                best = max(((r & (mm == k)).sum() / max((r | (mm == k)).sum(), 1) for k in labs), default=0.0)
            ious.append(float(best))
        checks["person_iou_source_view"] = {"frames": frames, "iou": ious, "min": min(ious) if ious else None}
        ok &= bool(ious) and min(ious) >= MIN_IOU
    v = read_video_rgb(a.control)
    empty = [i for i in range(len(v)) if v[i].max() == v[i].min()]
    frozen = len(v) > 1 and all(np.array_equal(v[i], v[i - 1]) for i in range(1, len(v)))
    shape_ok = (len(v) - 5) % 17 == 0 and v.shape[1] % 32 == 0 and v.shape[2] % 32 == 0
    checks["control"] = {"frames": len(v), "size": [int(v.shape[2]), int(v.shape[1])], "empty_frames": empty[:10],
                         "frozen": bool(frozen), "h3_format": bool(shape_ok)}
    ok &= not empty and not frozen and shape_ok
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"pass": bool(ok), "checks": checks}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pass": bool(ok), **{k: (v.get("min") if isinstance(v, dict) and "min" in v else None) for k, v in checks.items()}}))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
