"""3D bodies for chosen frames of one shot with fal's SAM 3D Body, one call per person per frame (mask given, so the
person is the one SAM 3.1 tracked). Saves the raw reply (metadata JSON) and the person's mesh (.ply, camera space).

  python scripts/v11_fal_body.py VIDEO.mp4 PEOPLE.masks.npz OUT_DIR --frames 0 34 68 ...

Costs about 0.02 USD per call (fal price page, 2026-11). Needs FAL_KEY in the environment (never printed).
Frames and masks are sent to fal as data URIs; nothing is stored there (X-Fal-Store-IO: 0).
ASCII paths only (cv2).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

MODEL = "fal-ai/sam-3/3d-body"


def data_uri(img) -> str:
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def call(body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(f"https://fal.run/{MODEL}", data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": "Key " + os.environ["FAL_KEY"], "Content-Type": "application/json", "X-Fal-Store-IO": "0"})
    for k in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", errors="replace")
            if e.code not in (429, 500, 502, 503, 504) or k == 2:
                return {"http_error": e.code, "body": msg[:2000]}
            time.sleep(20 * (k + 1))
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:  # dropped connection: try again
            if k == 2:
                return {"http_error": "connection", "body": str(e)[:500]}
            time.sleep(20 * (k + 1))
    return {"http_error": "no reply"}


def fetch(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("masks")
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--min-area", type=float, default=0.002, help="skip a person covering less than this share")
    ap.add_argument("--workers", type=int, default=1, help="calls at once")
    ap.add_argument("--min-present", type=float, default=0.5,
                    help="only people present (above --min-area) in this share of the shot's frames")
    a = ap.parse_args()
    d = np.load(a.masks)
    masks, id_map = d["masks"], json.loads(str(d["id_map"]))
    a.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    want, imgs, i = set(a.frames), {}, 0
    while want - set(imgs):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            imgs[i] = f
        i += 1
    present = {num: float(((masks == int(num)).mean(axis=(1, 2)) >= a.min_area).mean()) for num in id_map}
    id_map = {num: lab for num, lab in id_map.items() if present[num] >= a.min_present}
    print("people kept", id_map, flush=True)
    jobs = []
    for fr in a.frames:
        for num, label in id_map.items():
            m = masks[fr] == int(num)
            if m.mean() < a.min_area or (a.out / f"f{fr:04d}_{label}.json").exists():
                continue
            jobs.append((fr, label, m))

    def run(job) -> dict:
        fr, label, m = job
        stem = f"f{fr:04d}_{label}"
        body = {"image_url": data_uri(imgs[fr]), "mask_url": data_uri((m * 255).astype(np.uint8)),
                "export_meshes": True, "include_mhr_params": False, "include_3d_keypoints": False}
        t0 = time.time()
        r = call(body)
        entry = {"frame": fr, "person": label, "seconds": round(time.time() - t0, 1), "error": r.get("http_error")}
        if "http_error" not in r:
            try:
                for j, mesh in enumerate(r.get("meshes") or []):
                    (a.out / f"{stem}_{j}.ply").write_bytes(fetch(mesh["url"]))
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                entry["error"] = f"mesh download: {e}"[:300]
                print(json.dumps(entry), flush=True)
                return entry
            r.pop("visualization", None)
            r.pop("model_glb", None)
            entry["people"] = (r.get("metadata") or {}).get("num_people")
            (a.out / f"{stem}.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(entry), flush=True)
        return entry

    print("calls", len(jobs), flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        log = list(pool.map(run, jobs))
    (a.out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
