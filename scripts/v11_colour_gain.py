"""Colour gain for people: compare a render of the scene with the film over the people masks and return the linear
RGB factor that makes the rendered people as bright as the film (the scene's lighting darkens film colours that
already hold the light). Pure numpy/cv2.

  python scripts/v11_colour_gain.py FILM.mp4 PEOPLE.masks.npz RENDER_DIR [--frames 0-305:10]
Prints "R G B" for mave_blend.py --people-gain (multiply with a gain already used for RENDER_DIR).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vrecon.control.core import read_video_rgb  # noqa: E402


def to_lin(x: np.ndarray) -> np.ndarray:
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("film", type=Path)
    ap.add_argument("masks", type=Path)
    ap.add_argument("render", type=Path)
    ap.add_argument("--frames", default="0-305:10")
    a = ap.parse_args()
    v = read_video_rgb(a.film)
    pm = np.load(a.masks)["masks"]
    rng, step = a.frames.split(":")
    lo, hi = (int(x) for x in rng.split("-"))
    film, rend = [], []
    for f in range(lo, min(hi, len(v) - 1) + 1, int(step)):
        p = a.render / f"r{f:04d}.png"
        if not p.exists():
            continue
        r = cv2.imdecode(np.fromfile(p, np.uint8), 1)[:, :, ::-1]
        r = cv2.resize(r, (v.shape[2], v.shape[1]), interpolation=cv2.INTER_AREA)
        m = cv2.erode((pm[f] > 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        film.append(to_lin(v[f][m].astype(float)))
        rend.append(to_lin(r[m].astype(float)))
    film, rend = np.concatenate(film), np.concatenate(rend)
    g = np.median(film, 0) / np.maximum(np.median(rend, 0), 1e-4)
    print(" ".join(f"{x:.3f}" for x in np.clip(g, 0.5, 2.5)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
