"""Cameras, depth and point maps with Depth Anything 3 (MAVE's choice), written in 4RC's file layout so the rest of
the pipeline reads either: pred_<i>_pts (1, h, w, 3) world points, pred_<i>_conf, pred_<i>_extrinsic
(camera-to-world, OpenCV axes), pred_<i>_intrinsic at the grid. The grid is the picture with its long side at 512 and
no crop (4RC crops; with a full grid the crop offsets the readers compute are zero). Runs in env
%LOCALAPPDATA%/vrecon/envs/da3 (the observe env's torch plus the DA3 package). ASCII paths only.

    python gpu/observe/run_da3.py FRAMES_DIR OUT.npz [--model depth-anything/DA3-LARGE] [--res 504]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path)
    ap.add_argument("out")
    ap.add_argument("--model", default="depth-anything/DA3-LARGE")
    ap.add_argument("--res", type=int, default=504)
    a = ap.parse_args()
    from depth_anything_3.api import DepthAnything3

    files = sorted(a.frames.glob("*.png"))
    imgs = [cv2.imread(str(f))[:, :, ::-1].copy() for f in files]
    H, W = imgs[0].shape[:2]
    s = 512 / max(W, H)
    gw, gh = round(W * s), round(H * s)
    t0 = time.time()
    model = DepthAnything3.from_pretrained(a.model).to("cuda").eval()
    with torch.no_grad():
        pred = model.inference(imgs, process_res=a.res)
    d, conf = pred.depth, pred.conf
    ph, pw = d.shape[1:]
    out = {}
    for i in range(len(files)):
        w2c = np.eye(4)
        w2c[:3, :4] = pred.extrinsics[i][:3, :4]
        c2w = np.linalg.inv(w2c)
        K = pred.intrinsics[i].astype(np.float64).copy()
        K[0] *= gw / pw
        K[1] *= gh / ph
        z = cv2.resize(d[i].astype(np.float32), (gw, gh), interpolation=cv2.INTER_LINEAR)
        c = cv2.resize(conf[i].astype(np.float32), (gw, gh), interpolation=cv2.INTER_LINEAR) if conf is not None \
            else np.ones((gh, gw), np.float32)
        u, v = np.meshgrid(np.arange(gw) + 0.5, np.arange(gh) + 0.5)
        Xc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
        Xw = Xc.reshape(-1, 3) @ c2w[:3, :3].T + c2w[:3, 3]
        out[f"pred_{i}_pts"] = Xw.reshape(1, gh, gw, 3).astype(np.float32)
        out[f"pred_{i}_conf"] = c[None].astype(np.float32)
        out[f"pred_{i}_extrinsic"] = c2w.astype(np.float32)
        out[f"pred_{i}_intrinsic"] = K.astype(np.float32)
    out["meta"] = np.array(json.dumps({"tool": "da3", "model": a.model, "process_res": a.res, "grid": [gw, gh],
                                       "is_metric": str(pred.is_metric)[:40], "frames": [f.name for f in files],
                                       "seconds": round(time.time() - t0, 1)}))
    np.savez_compressed(a.out, **out)
    print(str(out["meta"]), "peak_gb", round(torch.cuda.max_memory_allocated() / 1e9, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
