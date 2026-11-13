"""Look check before V2V: one frame of a 3D render (control_look) per camera move, made photo-like by an image model
on fal, the shapes kept. Not a reconstruction step: the scene stays the only source of geometry.

    python scripts/fal_restyle.py CONTROLS_ROOT OUT_DIR [--frame 120] [--moves source,orbit90] [--quality medium]
        [--reference SOURCE.png]

CONTROLS_ROOT holds controls_<move>/control_look.mp4 and control_canny.mp4 (scripts/export_scene_controls.py).
--reference adds the real source frame as a second image for colours and materials. Writes per move
<move>_render.png, <move>_photo.png and restyle.json (prompt, settings, edge agreement between the render's and the
photo's outlines, cost estimate). Same privacy settings as fal_outside.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fal_outside import PRICE, _data_uri, _read, _write  # noqa: E402

PROMPT = (
    "Turn this 3D render into a realistic photograph of a professional outdoor tennis venue on a sunny day. Keep "
    "every shape, position and the camera exactly as in the render: the court and its lines, the net, the stands' "
    "rows and stairs, the walls, and each person's pose and size. Only change materials, lighting and detail: "
    "hard court surface, net mesh, blue stadium seats with a few spectators, painted walls. The figures are "
    "plain grey/beige mannequins standing in for people: show them as fully clothed tennis players, coaches and "
    "staff (shirts, shorts, shoes), same poses. Do not add or remove objects, do not move the camera, no text or logos.")
REF_NOTE = " The second image is a real photo of this venue from another camera: match its colours and materials."


def call(model, prompt, images, quality, size, timeout=600):
    body = {"prompt": prompt, "image_urls": [_data_uri(i) for i in images], "quality": quality,
            "image_size": {"width": size[0], "height": size[1]}, "num_images": 1, "output_format": "png",
            "sync_mode": True}
    req = urllib.request.Request(f"https://fal.run/{model}", data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": "Key " + os.environ["FAL_KEY"], "Content-Type": "application/json", "X-Fal-Store-IO": "0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        url = json.loads(r.read().decode("utf-8"))["images"][0]["url"]
    if url.startswith("data:"):
        raw = base64.b64decode(url.split(",", 1)[1])
    else:
        with urllib.request.urlopen(url, timeout=120) as r:
            raw = r.read()
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def edge_agreement(render, photo):
    """Share of the render's outline pixels with a photo outline within 3 px (shape kept = high)."""
    def edges(im):
        return cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 0), 60, 150) > 0
    a, b = edges(render), edges(cv2.resize(photo, (render.shape[1], render.shape[0])))
    near = cv2.dilate(b.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return round(float((a & near).sum() / max(a.sum(), 1)), 3)


def main() -> int:
    from vrecon.control.core import read_video_rgb

    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--frame", type=int, default=120)
    ap.add_argument("--moves", default="source,dolly-100+crane4,orbit90,crane10,orbit180")
    ap.add_argument("--quality", default="medium", choices=list(PRICE))
    ap.add_argument("--reference", default=None)
    ap.add_argument("--model", default="openai/gpt-image-2.5/sunburst/edit")
    a = ap.parse_args()
    if not os.environ.get("FAL_KEY"):
        raise SystemExit("FAL_KEY is not set")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = _read(a.reference) if a.reference else None
    log = {"prompt": PROMPT + (REF_NOTE if ref is not None else ""), "model": a.model, "quality": a.quality,
           "frame": a.frame, "reference": a.reference, "views": []}
    for m in a.moves.split(","):
        render = np.ascontiguousarray(read_video_rgb(str(Path(a.root) / f"controls_{m}" / "control_look.mp4"))[a.frame][..., ::-1])
        _write(out / f"{m}_render.png", render)
        t0 = time.time()
        try:
            photo = call(a.model, log["prompt"], [render] + ([ref] if ref is not None else []), a.quality,
                         (render.shape[1], render.shape[0]))
        except Exception as e:
            log["views"].append({"move": m, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(m, "failed:", type(e).__name__)
            continue
        _write(out / f"{m}_photo.png", photo)
        rec = {"move": m, "seconds": round(time.time() - t0, 1), "edge_agreement": edge_agreement(render, photo),
               "cost_estimate_usd": PRICE[a.quality] * render.shape[0] * render.shape[1] / (1024 * 768)}
        log["views"].append(rec)
        print(json.dumps(rec))
    log["cost_estimate_total_usd"] = round(sum(v.get("cost_estimate_usd", 0) for v in log["views"]), 3)
    (out / "restyle.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    ims = []
    for v in log["views"]:
        if "error" in v:
            continue
        r, p = _read(out / f"{v['move']}_render.png"), _read(out / f"{v['move']}_photo.png")
        p = cv2.resize(p, (r.shape[1], r.shape[0]))
        col = np.vstack([cv2.resize(r, (400, 352)), cv2.resize(p, (400, 352))])
        cv2.putText(col, v["move"], (6, 20), 0, 0.55, (0, 255, 255), 2)
        ims.append(col)
    if ims:
        _write(out / "sheet.png", np.hstack(ims))
    print("estimated total: $", log["cost_estimate_total_usd"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
