"""Control-video generation from geometry passes. Pure numpy / OpenCV / ffmpeg."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

FPS = 24
H3_MAX_FRAMES = 243


@dataclass
class ControlSpec:
    """Settings recorded in the manifest so a control video can be reproduced."""

    kind: Literal["depth", "canny"]
    params: dict = field(default_factory=dict)


def snap_h3_frames(n: int, max_frames: int = H3_MAX_FRAMES) -> int:
    """Largest 17k + 5 <= min(n, max_frames) (what the H3 pipeline will actually generate)."""
    n = min(n, max_frames)
    return max((n - 5) // 17 * 17 + 5, 5)


def check_h3_constraints(width: int, height: int, frames: int, fps: float) -> list[str]:
    """Problems that would make H3 crop, resample or drop frames. Empty list = fine."""
    problems = []
    if width % 32 or height % 32:
        problems.append(f"size {width}x{height}: width and height must be multiples of 32")
    if abs(fps - FPS) > 1e-6:
        problems.append(f"fps {fps}: H3 generates at a fixed {FPS} fps")
    snapped = snap_h3_frames(frames)
    if snapped != frames:
        problems.append(f"{frames} frames: H3 will use only the first {snapped} (frame count must be 17n+5, max 243)")
    return problems


def depth_control(
    depth: np.ndarray,
    mode: Literal["clip", "frame"] = "clip",
    lo_pct: float = 2.0,
    hi_pct: float = 85.0,
) -> tuple[np.ndarray, dict]:
    """Depth [T, H, W] (0 = background) -> uint8 RGB frames [T, H, W, 3], near = white.

    mode "clip": one normalisation range for the whole clip, so brightness keeps its meaning over
             time (a camera moving closer makes things brighter). Default for 3D-sourced depth.
    mode "frame": per-frame range, as VideoX-Fun's annotator does on estimated depth.
    Background (no geometry) is treated as the farthest value (black).
    """
    d = depth.astype(np.float32)
    covered = d > 0
    if not covered.any():
        return np.zeros((*d.shape, 3), np.uint8), {"mode": mode, "empty": True}
    far = float(d[covered].max())
    d = np.where(covered, d, far)

    def norm(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        x = (x - lo) / max(hi - lo, 1e-6)
        return ((1.0 - x).clip(0, 1) * 255.0).round().astype(np.uint8)

    if mode == "clip":
        lo, hi = np.percentile(d, lo_pct), np.percentile(d, hi_pct)
        out = norm(d, lo, hi)
        info = {"mode": mode, "lo": float(lo), "hi": float(hi), "lo_pct": lo_pct, "hi_pct": hi_pct}
    elif mode == "frame":
        out = np.empty(d.shape, np.uint8)
        for i in range(d.shape[0]):
            out[i] = norm(d[i], np.percentile(d[i], lo_pct), np.percentile(d[i], hi_pct))
        info = {"mode": mode, "lo_pct": lo_pct, "hi_pct": hi_pct}
    else:
        raise ValueError(f"unknown depth mode {mode!r}")
    return np.repeat(out[..., None], 3, axis=-1), info


def canny_control(frames_rgb: np.ndarray, low: int = 100, high: int = 200) -> tuple[np.ndarray, dict]:
    """uint8 RGB [T, H, W, 3] -> canny edges as uint8 RGB, white on black."""
    import cv2

    out = np.empty_like(frames_rgb)
    for i, frame in enumerate(frames_rgb):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)
        out[i] = np.repeat(edges[..., None], 3, axis=-1)
    return out, {"low": low, "high": high}


def load_frames(paths: list[Path]) -> np.ndarray:
    import cv2

    # imdecode, not imread: OpenCV's file functions cannot open non-ASCII paths on Windows.
    frames = [cv2.cvtColor(cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for p in paths]
    return np.stack(frames)


def read_video_rgb(path: str | Path) -> np.ndarray:
    """Decode a whole video to uint8 RGB [T, H, W, 3] with the system ffmpeg.

    Not cv2.VideoCapture: it cannot open non-ASCII paths on Windows.
    """
    size = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    if size.returncode != 0 or not size.stdout.strip():
        raise ValueError(f"not a readable video: {path}: {size.stderr.strip()[-300:]}")
    w, h = (int(x) for x in size.stdout.strip().split(",")[:2])
    proc = subprocess.run(
        # passthrough: one output frame per stored frame (as OpenCV and the GVHMR / SAM wrappers read them); the
        # default would duplicate or drop frames to make a variable-frame-rate file constant-rate
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[-2000:]}")
    return np.frombuffer(proc.stdout, np.uint8).reshape(-1, h, w, 3)


def encode_mp4(frames_rgb: np.ndarray, path: str | Path, fps: int = FPS, crf: int = 10) -> Path:
    """Encode uint8 RGB frames to H.264 mp4 (yuv420p, high quality) with the system ffmpeg."""
    t, h, w, c = frames_rgb.shape
    if c != 3 or frames_rgb.dtype != np.uint8:
        raise ValueError("frames must be uint8 RGB [T, H, W, 3]")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(frames_rgb).tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[-2000:]}")
    return path


def probe_frames(path: str | Path) -> int:
    """Number of video frames in a file (ffprobe)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, encoding="utf-8", errors="replace", check=True,
    )
    return int(out.stdout.strip())
