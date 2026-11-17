"""Reference images for r2v, cut automatically from the source video with context (scene-independent).

    python scripts/make_ref_crops.py VIDEO MASKS.npz PROJECT.json OUT_DIR --start 5.26 --end 10.43
        [--context 0.6] [--max 8] [--map LABEL=OBJECT ...]

For every mask label paired with a scene object (same rule as scripts/fix_floor_contact.py), the frame inside the
window where the object is largest and does not touch the image border is cut out with `context` x its box size
added on every side, so the crop shows what the object stands on and next to (a lone white shape at a frame
edge was drawn as a sneaker; a chair with its surroundings reads as a chair). The window's first full frame is
picture 1. Writes OUT_DIR/crop_<object>.png, OUT_DIR/frame_full.png and OUT_DIR/crops.json:
    [{"file", "object", "class", "frame", "box", "partial"}] in picture order, for scripts/make_prompts.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("masks", type=Path)
    ap.add_argument("project", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--context", type=float, default=0.6)
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--min-area", type=float, default=0.002, help="share of the frame")
    ap.add_argument("--map", nargs="*", default=[])
    a = ap.parse_args()

    import cv2

    from vrecon.control.core import read_video_rgb
    from vrecon.control.inventory import _class
    from vrecon.schema import load_project
    from vrecon.video import frame_rate, probe

    video = read_video_rgb(a.video)
    fps = float(frame_rate(probe(a.video)))
    f0, f1 = int(round(a.start * fps)), min(int(round(a.end * fps)), len(video) - 1)
    z = np.load(a.masks)
    masks, id_map = z["masks"], json.loads(str(z["id_map"]))
    project = load_project(a.project)
    objs = {o.id: o for o in project.reconstruction.objects}
    pairs = dict(x.split("=", 1) for x in a.map)
    for lab in id_map.values():
        pairs.setdefault(lab, "obj_person_box_" + lab.removeprefix("obs_").zfill(2) if lab.startswith("obs_") else lab)
    # the GVHMR person is stored under obj_person_NN: pair by the label the person boxes would have used
    for o in objs.values():
        if o.kind == "character" and o.id.startswith("obj_person_"):
            pass
    a.out.mkdir(parents=True, exist_ok=True)
    h, w = video.shape[1:3]
    cv2.imencode(".png", video[f0][..., ::-1])[1].tofile(str(a.out / "frame_full.png"))
    rows = [{"file": "frame_full.png", "object": None, "class": "frame", "frame": f0, "box": [0, 0, w, h],
             "partial": False}]
    cands = []
    for k, lab in id_map.items():
        oid = pairs.get(lab)
        m = masks[f0:f1 + 1] == int(k)
        area = m.reshape(len(m), -1).mean(1)
        if area.max() < a.min_area:
            continue
        touch = np.array([fr[:, 0].any() or fr[:, -1].any() or fr[0].any() or fr[-1].any() for fr in m])
        score = np.where(touch, area * 0.3, area)  # prefer frames where the object is whole
        i = int(np.argmax(score))
        ys, xs = np.nonzero(m[i])
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cw, ch = (x1 - x0) * a.context, (y1 - y0) * a.context
        box = [int(max(x0 - cw, 0)), int(max(y0 - ch, 0)), int(min(x1 + cw, w - 1)), int(min(y1 + ch, h - 1))]
        obj = objs.get(oid)
        cands.append((float(area[i]), {"label": lab, "object": oid if obj else None,
                                       "class": _class(obj) if obj else "thing", "frame": f0 + i, "box": box,
                                       "partial": bool(touch[i])}))
    for _, r in sorted(cands, key=lambda c: -c[0])[: a.max]:
        x0, y0, x1, y1 = r["box"]
        name = f"crop_{(r['object'] or r['label']).removeprefix('obj_')}.png"
        cv2.imencode(".png", video[r["frame"]][y0:y1 + 1, x0:x1 + 1][..., ::-1])[1].tofile(str(a.out / name))
        rows.append({"file": name, **r})
    (a.out / "crops.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    for i, r in enumerate(rows, 1):
        print(f"Picture {i}: {r['file']} {r['class']} {r.get('label', '')} partial={r['partial']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
