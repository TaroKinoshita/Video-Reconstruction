"""MegaSaM output (<scene>_droid.npz: depths, intrinsic, cam_c2w for every frame) -> 4RC-style npz.

Picks --frames frames evenly (np.linspace, as 4RC does) and writes n_frames, pred_<i>_extrinsic (c2w, OpenCV),
pred_<i>_intrinsic, pred_<i>_pts ([1, h, w, 3] world points from depth), pred_<i>_conf (ones) and
model_to_source. MegaSaM resizes each frame to about 384*512 pixels and crops the bottom/right edge to a
multiple of 8 (camera_tracking_scripts/test_demo.py image_stream), so the crop offset is 0.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("droid_npz")
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=0)
    a = ap.parse_args()
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", a.video], capture_output=True, encoding="utf-8", check=True)
    w0, h0 = (int(v) for v in probe.stdout.strip().split(",")[:2])
    f = np.sqrt((384 * 512) / (h0 * w0))
    w1, h1 = int(w0 * f), int(h0 * f)
    z = np.load(a.droid_npz)
    depths, K, c2w = z["depths"], z["intrinsic"].astype(np.float64), z["cam_c2w"].astype(np.float64)
    t, h, w = depths.shape
    frames = [int(i) for i in np.linspace(0, t - 1, a.frames, dtype=int)]
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    out = {"n_frames": np.array(len(frames)), "model_to_source": np.array([w1 / w0, h1 / h0, 0.0, 0.0])}
    for i, fr in enumerate(frames):
        pc = rays * depths[fr][..., None]
        pw = pc @ c2w[fr, :3, :3].T + c2w[fr, :3, 3]
        out[f"pred_{i}_extrinsic"] = c2w[fr].astype(np.float32)
        out[f"pred_{i}_intrinsic"] = K.astype(np.float32)
        out[f"pred_{i}_pts"] = pw[None].astype(np.float32)
        out[f"pred_{i}_conf"] = np.ones((1, h, w), np.float32)
    meta = {"tool": "megasam", "frames": frames, "seconds": a.seconds, "model_size": [w, h], "all_frames": t}
    out["meta"] = np.array(json.dumps(meta))
    with open(a.out, "wb") as fh:
        np.savez_compressed(fh, **out)
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
