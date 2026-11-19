"""Checks on a scene and its control video before paying for generation (after CWM 2.0's proxy quality control).

    python scripts/proxy_qc.py IN.blend CATEGORIES.json CONTROL.mp4 OUT.json --start S --frames N
        [--check "cat:person=MASKS.npz"] [--check "cat:held=PROPS.npz:obs_0"] [--check "name:Ball=BALL.npz"]
        [--samples 6] [--needs "hand-string-duck connection"]

Only the window that is generated (scene time start .. start + frames/24) is sampled. Each --check compares the
scene's things picked by a selector (cat:<category> from CATEGORIES.json, or name:<prefix> of the top-level name;
the observed body counts as cat:person) rendered from the .blend's source camera with observed masks (the labels after
":" or, without them, every obs_* label, else every label) and reports the IoU per sampled frame where either side
has pixels. Frames where neither shows: skipped; a check with no frame left is "needs review" (no observation).
--needs records a relation nobody has measured yet (status "needs review").
Control video: not empty, not frozen, H3 format (17n+5 frames, sides multiples of 32).

Exit 2 when the control's format is wrong (stop); exit 0 otherwise, with each check's status in OUT.json:
"ok" (min IoU >= MIN_IOU), "low" (below), "needs review".
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
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=124)
    ap.add_argument("--check", action="append", default=[])
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--needs", action="append", default=[])
    a = ap.parse_args()
    import bpy

    from vrecon.control.core import read_video_rgb
    from vrecon.scene.novel_blend import open_scene, render_view

    out = {"checks": {}, "window": {"start": a.start, "frames": a.frames}}
    v = read_video_rgb(a.control)
    empty = [i for i in range(len(v)) if v[i].max() == v[i].min()]
    frozen = len(v) > 1 and all(np.array_equal(v[i], v[i - 1]) for i in range(1, len(v)))
    fmt = (len(v) - 5) % 17 == 0 and v.shape[1] % 32 == 0 and v.shape[2] % 32 == 0 and len(v) == a.frames
    out["control"] = {"frames": len(v), "size": [int(v.shape[2]), int(v.shape[1])], "empty_frames": empty[:10],
                      "frozen": bool(frozen), "format_ok": bool(fmt)}
    if a.check:
        cats = json.loads(a.categories.read_text(encoding="utf-8"))["categories"]
        rec = open_scene(a.blend)
        cam = rec.reconstruction.cameras[0]
        fps, n_src = rec.source.fps.value, rec.source.frame_count
        src_frames = sorted({int(min((a.start + t) * fps, n_src - 1))
                             for t in np.linspace(0, (a.frames - 1) / 24.0, a.samples)})

        def top(o):
            while o.parent is not None:
                o = o.parent
            return o

        def picked(sel: str) -> set:
            kind, _, val = sel.partition(":")
            ids = set()
            for o in bpy.data.objects:
                if o.type != "MESH" or o.hide_render:
                    continue
                name = top(o).name.split(".")[0]
                observed = o.get("vrecon_id") == "obj_person_observed" or bool(top(o).get("vrecon_rig_of"))
                if (kind == "cat" and (cats.get(name) == val or (val == "person" and observed))) or \
                        (kind == "name" and (name.startswith(val) or o.name.startswith(val))):
                    ids.add(o.get("vrecon_id"))
            return ids

        renders = {f: render_view(cam, [f]) for f in src_frames}
        for chk in a.check:
            sel, _, spec = chk.partition("=")
            path, _, labs = spec.rpartition(":") if spec.count(":") > (1 if len(spec) > 1 and spec[1] == ":" else 0) else (spec, "", "")
            if not path.endswith(".npz"):
                path, labs = spec, ""
            z = np.load(path)
            m, idmap = z["masks"], {int(k): v for k, v in json.loads(str(z["id_map"])).items()}
            want = [k for k, v in idmap.items() if v in labs.split(",")] if labs else \
                ([k for k, v in idmap.items() if v.startswith("obs")] or list(idmap))
            ids = picked(sel)
            ious = []
            for f in src_frames:
                if f >= len(m):
                    continue
                idx, imap = renders[f]
                r = np.isin(idx[0], [k for k, v in imap.items() if v in ids])
                mm = m[f]
                if mm.shape != r.shape:
                    import cv2

                    mm = cv2.resize(mm.astype(np.uint8), (r.shape[1], r.shape[0]), interpolation=cv2.INTER_NEAREST)
                g = np.isin(mm, want)
                if not r.any() and not g.any():
                    continue
                ious.append(float((r & g).sum() / max((r | g).sum(), 1)))
            status = "needs review" if not ious else ("ok" if min(ious) >= MIN_IOU else "low")
            out["checks"][chk] = {"status": status, "frames": src_frames, "iou": ious,
                                  "min": min(ious) if ious else None, "objects": sorted(i for i in ids if i)[:20]}
    for n in a.needs:
        out["checks"][f"needs: {n}"] = {"status": "needs review", "why": "no observation measures it yet"}
    out["pass"] = bool(fmt and not empty and not frozen)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"format": out["pass"], **{k: (c["status"], c.get("min")) for k, c in out["checks"].items()}}))
    return 0 if out["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
