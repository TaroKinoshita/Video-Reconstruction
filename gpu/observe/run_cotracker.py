"""Point tracks with CoTracker3 Offline. Runs in the separate observe env, not in vrecon's .venv.

    python gpu/observe/run_cotracker.py VIDEO OUT.npz [--max-tracks 400] [--reseed 12]

Reads only the video. Writes xy (float32 [N, T, 2], NaN before the seed frame and while not visible),
size (width, height) and meta (JSON); vrecon reads this with optimize.tracks.load_tracks.

Points start on Shi-Tomasi corners, like the OpenCV tracker: a point on a flat wall has nothing to follow.
They are seeded every `reseed` frames because track_metrics compares each track only for a short
horizon after its first frame: seeding on frame 0 alone would ignore the rest of the clip.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from cotracker.predictor import CoTrackerPredictor
from huggingface_hub import hf_hub_download

WEIGHTS = "scaled_offline.pth"


def read_video(path: str) -> np.ndarray:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", path], capture_output=True, encoding="utf-8", errors="replace", check=True)
    w, h = (int(v) for v in probe.stdout.strip().split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--max-tracks", type=int, default=400, help="shared evenly between the seed frames")
    ap.add_argument("--reseed", type=int, default=12)
    args = ap.parse_args()

    frames = read_video(args.video)
    t, h, w = frames.shape[:3]
    seeds = list(range(0, t - 1, args.reseed))
    per_seed = max(1, args.max_tracks // len(seeds))
    queries = []
    for f in seeds:
        gray = cv2.cvtColor(frames[f], cv2.COLOR_RGB2GRAY)
        pts = cv2.goodFeaturesToTrack(gray, per_seed, 0.01, 8)
        if pts is not None:
            pts = pts.reshape(-1, 2).astype(np.float32)
            queries.append(np.column_stack([np.full(len(pts), f, np.float32), pts]))
    queries = np.concatenate(queries)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CoTrackerPredictor(checkpoint=hf_hub_download("facebook/cotracker3", WEIGHTS), offline=True,
                               window_len=60).to(device)
    video = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)[None].float().to(device)
    start = time.time()
    with torch.no_grad():
        tracks, vis = model(video, queries=torch.from_numpy(queries)[None].to(device))
    seconds = time.time() - start

    xy = tracks[0].cpu().numpy().astype(np.float32)  # [T, N, 2]
    seen = vis[0].cpu().numpy().astype(bool)
    xy = np.transpose(xy, (1, 0, 2))
    seen = np.transpose(seen, (1, 0))
    for i, s in enumerate(queries[:, 0].astype(int)):
        seen[i, :s] = False
    inside = (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    seen &= inside
    xy[~seen] = np.nan
    keep = seen.sum(1) >= 4
    xy = xy[keep]

    meta = {"tool": "cotracker3_offline", "weights": WEIGHTS, "seeding": "shi_tomasi_corners", "max_tracks": args.max_tracks, "reseed": args.reseed,
            "seconds": round(seconds, 2), "device": device,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:  # a file object: numpy would add .npz to a bare path only sometimes
        np.savez_compressed(f, xy=xy, size=np.array([w, h]), meta=json.dumps(meta))
    print(json.dumps({**meta, "tracks": int(len(xy)), "frames": int(t)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
