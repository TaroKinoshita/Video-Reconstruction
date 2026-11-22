"""Objects in each cut of a multi-cut clip, with one fixed open vocabulary (the same list for every video).

    <sam2env>/Scripts/python.exe gpu/observe/detect_objects.py OUT_DIR TAG --ckpt yolov8s-worldv2.pt

Runs YOLO-World on every frame of OUT_DIR/TAG_shotNN.mp4 (read with ffmpeg: non-ASCII paths), links detections of
the same class over frames by box overlap into tracks, keeps tracks seen in at least MIN_FRAMES frames with mean
score at least MIN_SCORE. People and set pieces are not in the vocabulary (other steps handle them).
Writes OUT_DIR/TAG_objects.json: {cut: [{"label", "frames": [t...], "boxes": [[x0,y0,x1,y1]...], "scores": [...]}]}
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

VOCAB = ["gun", "pistol", "rifle", "knife", "sword", "phone", "bag", "backpack", "suitcase", "umbrella", "bottle",
         "cup", "glass", "book", "box", "ball", "racket", "bat", "chair", "table", "bench", "sofa", "bed", "lamp",
         "car", "bicycle", "motorcycle", "bus", "truck", "helicopter", "dog", "cat", "horse", "bird", "laptop",
         "television", "clock", "sign", "flag", "bucket", "barrel", "crate", "trash can", "cone", "hat", "helmet",
         "sunglasses", "flower pot", "plant", "guitar", "microphone", "camera", "shell casing", "debris"]
MIN_FRAMES, MIN_SCORE, CONF, LINK_IOU = 3, 0.25, 0.12, 0.3


def frames_of(video: Path) -> np.ndarray:
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
                         "-of", "csv=p=0", str(video)], capture_output=True, text=True, encoding="utf-8").stdout
    w, h = (int(x) for x in pr.strip().split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / max(u, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("tag")
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    from ultralytics import YOLOWorld

    model = YOLOWorld(a.ckpt)
    model.set_classes(VOCAB)
    result = {}
    for v in sorted(a.out.glob(f"{a.tag}_shot[0-9][0-9].mp4")):
        k = int(v.stem[-2:])
        fr = frames_of(v)
        tracks = []
        for t, img in enumerate(fr):
            r = model.predict(img, conf=CONF, verbose=False)[0]
            for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                lab = VOCAB[int(c)]
                best = None
                for tr in tracks:
                    if tr["label"] == lab and tr["frames"][-1] in (t - 1, t - 2) and iou(tr["boxes"][-1], b) > LINK_IOU:
                        if best is None or iou(tr["boxes"][-1], b) > iou(best["boxes"][-1], b):
                            best = tr
                if best is not None and best["frames"][-1] != t:
                    best["frames"].append(t)
                    best["boxes"].append(b.tolist())
                    best["scores"].append(float(s))
                else:
                    tracks.append({"label": lab, "frames": [t], "boxes": [b.tolist()], "scores": [float(s)]})
        keep = [tr for tr in tracks if len(tr["frames"]) >= min(MIN_FRAMES, len(fr)) and np.mean(tr["scores"]) >= MIN_SCORE]
        result[k] = keep
        print(k, len(fr), [(tr["label"], len(tr["frames"]), round(float(np.mean(tr["scores"])), 2)) for tr in keep], flush=True)
    (a.out / f"{a.tag}_objects.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
