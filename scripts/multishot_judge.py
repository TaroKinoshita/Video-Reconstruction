"""Second tier of the automatic check: a vision model compares the film with the scene's render, per cut.

    python scripts/multishot_judge.py SOURCE.mp4 CONTROL.mp4 SHOTS.json OUT.json [--per-cut 3]
                                      [--model openai/gpt-6.1-sol] [--max-usd 1.0] [--dry-run]

For each cut, --per-cut frames (spread over the cut) become one side-by-side picture (film left, render right,
1024 px wide in all) and go to fal's openrouter/router/vision with a fixed question; the answer is JSON with
0-5 marks and the worst problem. It asks what the numbers of scripts/multishot_check.py cannot tell (is the gun
there, does the coat read as a coat, is the framing the same), never replaces them. Spend: fal reports each
call's cost; the run stops before going over --max-usd. Needs FAL_KEY. Sends film frames to an outside service.
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
sys.path.insert(0, str(ROOT / "src"))
ENDPOINT = "https://fal.run/openrouter/router/vision"
SYSTEM = ("You check a 3D reconstruction of a film shot. Left: the film frame. Right: the reconstruction rendered from "
          "the same camera as flat coloured proxies (orange = people, cyan = held objects, yellow = loose objects, "
          "green = ground, blue-grey = buildings and walls, pale blue = sky). Judge geometry and layout only, not "
          "texture, lighting or detail. Be strict and specific. Answer with JSON only.")
MARKS = ("framing", "people", "objects", "set")
QUESTION = """Compare left and right in this picture.
Marks use one fixed scale; judge each row of the picture, then give the mark the rows mostly earn:
 5 = matches; a viewer would call it the same shot
 4 = right, small offsets in place or size
 3 = recognisably the same, one clear error (place, size, pose or missing part)
 2 = partly right, several clear errors
 1 = present but mostly wrong
 0 = absent, or nothing alike
Return JSON with these keys:
 "framing": 0-5, same camera view (where the subject sits, how big, angle),
 "people": 0-5, each person present, in the right place, size and pose,
 "objects": 0-5, important objects (weapons, props) present and in the right place; "n/a" when the film shows none,
 "set": 0-5, ground, walls, buildings in the right places,
 "missing": list of things in the film that the reconstruction lacks,
 "wrong": list of things in the reconstruction that are wrong or should not be there,
 "worst_problem": one sentence."""


def data_uri(img_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def ask(model: str, img: np.ndarray) -> dict:
    body = {"model": model, "prompt": QUESTION, "system_prompt": SYSTEM, "image_urls": [data_uri(img)],
            "temperature": 0, "max_tokens": 4000, "reasoning": True}  # this model refuses reasoning off
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": "Key " + os.environ["FAL_KEY"], "Content-Type": "application/json", "X-Fal-Store-IO": "0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def parse(text: str) -> dict | None:
    """the last JSON object in the text (reasoning may come before it)."""
    b = text.rfind("}")
    for a in [i for i, c in enumerate(text) if c == "{"]:
        try:
            return json.loads(text[a:b + 1])
        except json.JSONDecodeError:
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("control", type=Path)
    ap.add_argument("shots", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--per-cut", type=int, default=3)
    ap.add_argument("--model", default="openai/gpt-6.1-sol")
    ap.add_argument("--max-usd", type=float, default=1.0)
    ap.add_argument("--repeat", type=int, default=3, help="asks per cut; marks are the median, spread is max - min")
    ap.add_argument("--dry-run", action="store_true", help="write the pictures only, send nothing")
    a = ap.parse_args()
    from vrecon.control.core import read_video_rgb

    src, ctl = read_video_rgb(a.source), read_video_rgb(a.control)
    shots = json.loads(a.shots.read_text(encoding="utf-8"))["shots"]
    pics = a.out.parent / (a.out.stem + "_pics")
    pics.mkdir(parents=True, exist_ok=True)
    rep, spent = {"model": a.model, "cuts": {}}, 0.0
    for k, (s0, s1) in enumerate(shots):
        fr = [f for f in np.linspace(s0, s1 - 1, a.per_cut + 2, dtype=int)[1:-1] if f < len(ctl)]
        rows = []
        for f in fr:
            l = cv2.resize(src[f], (512, int(512 * src.shape[1] / src.shape[2])))
            r = cv2.resize(ctl[f], (512, l.shape[0]))
            rows.append(np.concatenate([l, r], 1))
        if not rows:
            continue
        img = np.concatenate(rows, 0)
        cv2.imencode(".jpg", img[..., ::-1])[1].tofile(str(pics / f"cut{k:02d}.jpg"))
        if a.dry_run:
            continue
        if spent >= a.max_usd:
            print("budget reached, stopping", flush=True)
            break
        runs = []
        for _ in range(a.repeat):
            if spent >= a.max_usd:
                break
            t0 = time.time()
            try:
                res = ask(a.model, img)
            except urllib.error.HTTPError as e:
                print(k, "HTTP", e.code, e.read().decode("utf-8", "replace")[:500], flush=True)
                return 1
            spent += float(res.get("usage", {}).get("cost") or 0)
            runs.append({"answer": parse(res.get("output", "")), "usage": res.get("usage"),
                         "seconds": round(time.time() - t0, 1)})
        marks, spread = {}, {}
        for key in MARKS:
            v = [r["answer"][key] for r in runs if r["answer"] and isinstance(r["answer"].get(key), (int, float))]
            na = sum(1 for r in runs if r["answer"] and r["answer"].get(key) == "n/a")
            if v and len(v) >= na:
                marks[key], spread[key] = float(np.median(v)), float(max(v) - min(v))
            elif na:
                marks[key] = "n/a"
        rep["cuts"][k] = {"frames": [int(f) for f in fr], "marks": marks, "spread": spread, "runs": runs}
        print(k, marks, "spread", spread, flush=True)
    if not a.dry_run:
        sp = [v for c in rep["cuts"].values() for v in c["spread"].values()]
        rep["mean_spread"] = float(np.mean(sp)) if sp else None
        print("mean spread", rep["mean_spread"])
    rep["usd"] = spent
    a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print("total USD", round(spent, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
