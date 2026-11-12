"""First look at a video: cut it into shots and say, per shot, which reconstruction path fits.

Cheap measurements only (CPU, seconds). The one that decides is the share of pixels that do not change
between frames ("frozen"): on the 6 synthetic 3D clips it is .98 for the static camera and .38-.79 for pan, dolly,
truck, orbit, handheld; the 4 2D clips are .98-.99. Whether a still shot is screen graphics or filmed objects is
NOT decided here: a still camera on a still set is as frozen as a graphics background (syn_01 .98, stock 6077272
.996), and flat-colour share and point tracks did not separate them either (2026-11-12). Those shots are
"undecided" until someone looks at the content (an LLM or a person). People and text stay "unknown" (no detector
or OCR is run here).

    route "scene3d"    the camera moves: camera + objects in 3D (4RC, masks, repair loop)
    route "undecided"  the camera stands still: 2D layers (vrecon.layers2d), 3D objects, or both
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

STILL_PX = 0.5  # a tracked point that moves less than this between frames counts as still
FLAT_GRAD = 8.0  # gray-level gradient below this counts as flat (information only)
FROZEN_DIFF = 1  # gray levels; a pixel that changes by at most this between frames is frozen
FROZEN_STILL = 0.85  # camera counts as still when at least this share of pixels is frozen


def measure_shot(frames: np.ndarray, pairs: int = 12) -> dict:
    """frames: uint8 RGB [T,H,W,3] of one shot."""
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])
    n = len(gray)
    idx = np.unique(np.linspace(0, n - 2, min(pairs, max(n - 1, 1))).astype(int)) if n > 1 else np.array([], int)
    still, moved = [], []
    for t in idx:
        p = cv2.goodFeaturesToTrack(gray[t], 400, 0.01, 6)
        if p is None or len(p) < 10:
            continue
        q, st, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t + 1], p.astype(np.float32), None)
        ok = st.ravel() == 1
        d = np.linalg.norm((q - p).reshape(-1, 2)[ok], axis=1)
        if len(d):
            still.append(float((d < STILL_PX).mean()))
            moved.append(float(np.median(d)))
    frozen = [float((np.abs(gray[t + 1].astype(np.int16) - gray[t]) <= FROZEN_DIFF).mean()) for t in idx]
    flat = []
    for t in np.linspace(0, n - 1, min(6, n)).astype(int):
        g = gray[t].astype(np.float32)
        mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 4
        flat.append(float((mag < FLAT_GRAD).mean()))
    return {"frozen_pixels": float(np.median(frozen)) if frozen else None,
            "still_points": float(np.median(still)) if still else None,
            "median_motion_px": float(np.median(moved)) if moved else None,
            "flat_pixels": float(np.mean(flat)) if flat else None}


def route(m: dict) -> str:
    if m["frozen_pixels"] is None:
        return "unknown"  # too short to tell
    return "undecided" if m["frozen_pixels"] >= FROZEN_STILL else "scene3d"


def triage(video: str | Path) -> dict:
    from vrecon.control.core import read_video_rgb
    from vrecon.video import detect_shots

    frames = read_video_rgb(video)
    shots, info = detect_shots(video, len(frames))
    out = []
    for i, (s, e) in enumerate(shots):
        m = measure_shot(frames[s:e])
        r = route(m)
        out.append({"shot": i + 1, "start": s, "end": e, **m, "camera": {"scene3d": "moving", "undecided": "still"}.get(r, "unknown"),
                    "route": r, "unknown": ["people", "text"] + (["2d_or_3d"] if r == "undecided" else [])})  # no person detector / OCR in this step
    return {"video": str(video), "frames": int(len(frames)), "shots": out, "shot_detection": info,
            "rule": {"FROZEN_DIFF": FROZEN_DIFF, "FROZEN_STILL": FROZEN_STILL}}


__all__ = ["measure_shot", "route", "triage"]
