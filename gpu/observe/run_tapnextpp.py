"""Point tracks with TAPNext++ (google-deepmind/tapnet). Runs in the separate observe env, not in vrecon's .venv.

    python gpu/observe/run_tapnextpp.py VIDEO OUT.npz --ckpt tapnextpp_512.ckpt [--resolution 512]
        [--max-tracks 400] [--reseed 12] [--span 0]

Needs the tapnet repo on PYTHONPATH. Same output as run_cotracker.py (xy [N, T, 2] with NaN, size, meta);
vrecon reads it with optimize.tracks.load_tracks.

TAPNext++ is an online tracker that takes its queries on the first frame it sees, so each seed frame gets
its own pass starting at that frame. Points start on Shi-Tomasi corners, as in the other trackers.
span > 0 stops each track that many frames after its seed (the OpenCV tracker uses 24).
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
from tapnet.tapnextpp.votsp2026.model import TAPNextPP


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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--resolution", type=int, default=512, help="512 for tapnextpp_512.ckpt, 256 for tapnextpp_ckpt.pt")
    ap.add_argument("--max-tracks", type=int, default=400, help="shared evenly between the seed frames")
    ap.add_argument("--reseed", type=int, default=12)
    ap.add_argument("--span", type=int, default=0)
    args = ap.parse_args()

    frames = read_video(args.video)
    t, h, w = frames.shape[:3]
    seeds = list(range(0, t - 1, args.reseed))
    per_seed = max(1, args.max_tracks // len(seeds))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TAPNextPP.from_checkpoint(args.ckpt, device=device, input_resolution=args.resolution)

    out = []
    start = time.time()
    for s in seeds:
        gray = cv2.cvtColor(frames[s], cv2.COLOR_RGB2GRAY)
        pts = cv2.goodFeaturesToTrack(gray, per_seed, 0.01, 8)
        if pts is None:
            continue
        q = pts.reshape(-1, 2).astype(np.float32)
        xy = np.full((len(q), t, 2), np.nan, np.float32)
        end = min(t, s + args.span) if args.span > 0 else t
        state = None
        for f in range(s, end):
            bgr = np.ascontiguousarray(frames[f][..., ::-1])
            pos, vis, state = model.track_frame(bgr, q if state is None else None, state=state)
            if f == s:
                pos, vis = q, np.ones(len(q), bool)  # the seed itself is exact
            inside = (pos[:, 0] >= 0) & (pos[:, 0] < w) & (pos[:, 1] >= 0) & (pos[:, 1] < h)
            ok = vis & inside
            xy[ok, f] = pos[ok]
        out.append(xy)
    seconds = time.time() - start
    xy = np.concatenate(out)
    xy = xy[(~np.isnan(xy[:, :, 0])).sum(1) >= 4]

    meta = {"tool": "tapnextpp", "weights": Path(args.ckpt).name, "resolution": args.resolution,
            "seeding": "shi_tomasi_corners", "max_tracks": args.max_tracks, "reseed": args.reseed, "span": args.span,
            "seconds": round(seconds, 2), "device": device,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:  # a file object: numpy would add .npz to a bare path only sometimes
        np.savez_compressed(fh, xy=xy, size=np.array([w, h]), meta=json.dumps(meta))
    print(json.dumps({**meta, "tracks": int(len(xy)), "frames": int(t)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
