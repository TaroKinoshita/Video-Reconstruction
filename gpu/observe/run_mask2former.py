"""Semantic labels per frame with Mask2Former (ADE20K, as MAVE labels its static points), in run_sam3.py's file
format, grouped into the classes the evidence uses: support (floor-like), wall, ceiling, trunk (tree, pole,
column), and people (obs_0, so the evidence leaves them out). Runs in env %LOCALAPPDATA%/vrecon/envs/sem (the
observe env's torch plus transformers). ASCII paths only.

    python gpu/observe/run_mask2former.py VIDEO OUT.masks.npz [--model facebook/mask2former-swin-large-ade-semantic]
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import torch

GROUPS = {
    "support": ["floor", "road", "grass", "sidewalk", "earth", "field", "sand", "path", "rug", "stairs", "stairway",
                "step", "runway", "dirt track", "land", "floor, flooring", "ground"],
    "wall": ["wall", "building", "fence", "railing", "house", "skyscraper", "door", "screen door", "bannister",
             "rock", "cliff"],
    "ceiling": ["ceiling"],
    "trunk": ["tree", "pole", "column", "pillar", "palm", "post"],
    "obs_0": ["person"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--model", default="facebook/mask2former-swin-large-ade-semantic")
    a = ap.parse_args()
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    proc = AutoImageProcessor.from_pretrained(a.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(a.model).cuda().eval().half()
    names = {i: n for i, n in model.config.id2label.items()}
    lut = np.zeros(len(names) + 1, np.uint16)
    order = list(GROUPS)
    for i, n in names.items():
        parts = [p.strip() for p in n.split(",")]
        for g, keys in GROUPS.items():
            if n in keys or any(p in keys for p in parts):
                lut[i] = order.index(g) + 1
    cap = cv2.VideoCapture(a.video)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[:, :, ::-1])
    T, H, W = len(frames), *frames[0].shape[:2]
    masks = np.zeros((T, H, W), np.uint16)
    t0 = time.time()
    for t, f in enumerate(frames):
        inp = proc(images=f, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].half()
        with torch.no_grad():
            out = model(**inp)
        seg = proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0].cpu().numpy()
        masks[t] = lut[seg]
    id_map = {str(k + 1): g for k, g in enumerate(order)}
    meta = {"tool": "mask2former", "model": a.model, "groups": GROUPS, "seconds": round(time.time() - t0, 1)}
    np.savez_compressed(a.out, masks=masks, id_map=json.dumps(id_map), meta=json.dumps(meta))
    print(json.dumps({**meta, "share": {g: round(float((masks == k + 1).mean()), 3) for k, g in enumerate(order)}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
