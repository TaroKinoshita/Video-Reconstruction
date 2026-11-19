"""First-frame appearance reference for a new camera (after CWM 2.0's first-image reference): the image model on fal
redraws the control video's first frame (the layout from the new camera) with the look of the source video.

    python scripts/first_frame_ref.py CONTROL.mp4 SOURCE.png OUT.png [--look look.txt] [--quality medium]
        [--model openai/gpt-image-2.5/sunburst/edit]

Image 1 = the control's first frame (colour-coded clay: layout, sizes, outlines), image 2 = a frame of the source
video (look). The answer is resized to the control's size. Needs FAL_KEY (never printed); requests ask fal not to
store inputs/outputs. Writes OUT.png and OUT.json (prompt, model, seconds).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGEND = ("orange = a fully clothed person (a smooth clay mannequin, or a simple figure without limbs, only gives "
          "the pose and place; draw the person dressed as in image 2), cyan = "
          "something a person holds or pulls, green = ground, blue-grey = walls and stands, brown = furniture, "
          "yellow = small objects, light blue = sky")
PROMPT = ("Image 1 is a colour-coded clay 3D render of a scene seen from a new camera; image 2 is a photo of the same "
          "scene from the original camera. Draw image 1 as a realistic photo with the look of image 2: keep the "
          "camera, the position, size and outline of every object and person exactly as in image 1; take colours, "
          "materials, clothing, lighting and surroundings from image 2. Image 1's flat colours are labels, not looks: "
          + LEGEND + ". Where image 1 shows parts image 2 never shows, continue them plausibly. "
          "No text, no added people.")


def _uri(img) -> str:
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("control", type=Path)
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--look", type=Path)
    ap.add_argument("--quality", default="medium")
    ap.add_argument("--model", default="openai/gpt-image-2.5/sunburst/edit")
    a = ap.parse_args()
    if not os.environ.get("FAL_KEY"):
        raise SystemExit("FAL_KEY is not set")
    sys.path.insert(0, str(ROOT / "src"))
    from vrecon.control.core import read_video_rgb

    first = read_video_rgb(a.control)[0][..., ::-1]
    h, w = first.shape[:2]
    src = cv2.imdecode(np.fromfile(str(a.source), np.uint8), cv2.IMREAD_COLOR)
    prompt = PROMPT + (("\nThe look: " + a.look.read_text(encoding="utf-8").strip()) if a.look else "")
    body = {"prompt": prompt, "image_urls": [_uri(first), _uri(src)], "quality": a.quality,
            "image_size": {"width": w, "height": h}, "num_images": 1, "output_format": "png", "sync_mode": True}
    t0 = time.time()
    req = urllib.request.Request(f"https://fal.run/{a.model}", data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": "Key " + os.environ["FAL_KEY"], "Content-Type": "application/json", "X-Fal-Store-IO": "0"})
    out, err = None, ""
    for attempt in range(3):  # the content checker's verdict varies between calls on the same input
        try:
            with urllib.request.urlopen(urllib.request.Request(req.full_url, data=req.data, method="POST",
                                                               headers=dict(req.header_items())), timeout=600) as r:
                out = json.loads(r.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:  # the reason (e.g. a content filter) is in the body
            err = f"fal {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
            if e.code != 422:
                break
    if out is None:
        raise SystemExit(err)
    url = out["images"][0]["url"]
    raw = base64.b64decode(url.split(",", 1)[1]) if url.startswith("data:") else urllib.request.urlopen(url, timeout=120).read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", img)[1].tofile(str(a.out))
    a.out.with_suffix(".json").write_text(json.dumps({"prompt": prompt, "model": a.model, "quality": a.quality,
                                                       "seconds": round(time.time() - t0, 1)}, indent=1) + "\n",
                                          encoding="utf-8")
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
