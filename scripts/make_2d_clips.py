"""Synthetic 2D-graphics clips with known ground truth (text, shapes, logo over a static background).

    python scripts/make_2d_clips.py [--out evaluation/synthetic2d] [--only s2d_01 ...]

For each clip, writes to <out>/<clip>/:
    video.mp4     the reference video (640x360, 24 fps, 72 frames)
    gt.json       ground-truth project: reconstruction.layers with keys and easing
    meta.json     what happens (per layer: motion segments and their easing)
    gt_passes/    index.npz (front-most layer per pixel), id_map.json, sprites/<id>.png (git-ignored)

Evaluation data only: reconstruction code must never read gt.json or gt_passes/ (the masks are used as a
labelled ground-truth stand-in, as for the 3D clips). Composited with numpy, not Blender: exact and fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from vrecon.control.core import encode_mp4
from vrecon.layers2d.model import key, render
from vrecon.schema import FileRef, Layer2D, Project, Rational, Shot, ShotMapping, SourceVideo, save_project

W, H, FPS, FRAMES = 640, 360, 24, 72


def _rgba(color_img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.dstack([color_img.astype(np.float32) / 255.0, alpha.astype(np.float32)])


def card(w: int, h: int, c1, c2) -> np.ndarray:
    """Rounded rectangle with diagonal stripes (texture for point tracking)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = c1
    for k in range(-h, w, 18):
        cv2.line(img, (k, 0), (k + h, h), c2, 5, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (245, 245, 245), 3)
    a = np.zeros((h, w), np.float32)
    r = min(w, h) // 6
    cv2.rectangle(a, (r, 0), (w - 1 - r, h - 1), 1.0, -1)
    cv2.rectangle(a, (0, r), (w - 1, h - 1 - r), 1.0, -1)
    for cx, cy in [(r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)]:
        cv2.circle(a, (cx, cy), r, 1.0, -1, cv2.LINE_AA)
    return _rgba(img, a)


def text(t: str, size: float, color, thick: int) -> np.ndarray:
    (tw, th), base = cv2.getTextSize(t, cv2.FONT_HERSHEY_DUPLEX, size, thick)
    w, h = tw + 16, th + base + 16
    a = np.zeros((h, w), np.uint8)
    cv2.putText(a, t, (8, 8 + th), cv2.FONT_HERSHEY_DUPLEX, size, 255, thick, cv2.LINE_AA)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = color
    img[:, :, :] = (img * np.linspace(0.75, 1.0, h)[:, None, None]).astype(np.uint8)  # vertical shade
    return _rgba(img, a / 255.0)


def logo(d: int) -> np.ndarray:
    img = np.zeros((d, d, 3), np.uint8)
    a = np.zeros((d, d), np.float32)
    c = (d // 2, d // 2)
    cv2.circle(img, c, d // 2 - 2, (40, 170, 220), -1, cv2.LINE_AA)
    cv2.circle(a, c, d // 2 - 2, 1.0, -1, cv2.LINE_AA)
    pts = []
    for i in range(10):
        r = (d * 0.42) if i % 2 == 0 else (d * 0.18)
        ang = np.pi / 2 + i * np.pi / 5
        pts.append((c[0] + r * np.cos(ang), c[1] - r * np.sin(ang)))
    cv2.fillPoly(img, [np.round(np.array(pts)).astype(np.int32)], (250, 220, 60), cv2.LINE_AA)
    cv2.ellipse(img, c, (d // 2 - 8, d // 2 - 8), 0, 20, 110, (230, 60, 80), 6, cv2.LINE_AA)
    return _rgba(img, a)


def badge(d: int) -> np.ndarray:
    img = np.zeros((d, d, 3), np.uint8)
    a = np.zeros((d, d), np.float32)
    c = (d // 2, d // 2)
    cv2.circle(img, c, d // 2 - 2, (220, 70, 70), -1, cv2.LINE_AA)
    cv2.circle(a, c, d // 2 - 2, 1.0, -1, cv2.LINE_AA)
    cv2.putText(img, "NEW", (d // 2 - 26, d // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return _rgba(img, a)


def f(n: int) -> float:
    return n / FPS


def specs() -> dict:
    """id -> list of (layer kwargs, sprite). Key times in frames via f()."""
    return {
        "s2d_01": [  # card slides in (ease out), holds, turns 90 deg (ease in-out)
            (dict(id="lay_card", kind="shape", z=0, keys=[
                key(f(0), 90, 180, easing="ease_out"), key(f(20), 420, 180), key(f(40), 420, 180, easing="ease_in_out"),
                key(f(64), 420, 180, rotation=90)]), card(160, 90, (230, 130, 40), (250, 180, 90))),
        ],
        "s2d_02": [  # title fades in while growing, holds, fades out
            (dict(id="lay_title", kind="text", text="SALE 50%", z=0, keys=[
                key(f(6), 320, 170, scale=0.7, opacity=0.0, easing="ease_in_out"), key(f(30), 320, 170),
                key(f(50), 320, 170, easing="ease_in"), key(f(66), 320, 170, opacity=0.0)]),
             text("SALE 50%", 2.0, (255, 235, 120), 4)),
        ],
        "s2d_03": [  # logo arcs over in two moves while spinning
            (dict(id="lay_logo", kind="image", z=0, keys=[
                key(f(4), 100, 280, easing="ease_in_out"), key(f(32), 320, 90, rotation=180, easing="ease_in"),
                key(f(60), 540, 280, rotation=360)]), logo(110)),
        ],
        "s2d_04": [  # lower third: bar slides, name fades in, badge pops and pulses, bar fades out
            (dict(id="lay_bar", kind="shape", z=0, keys=[
                key(f(4), 140, 300, easing="ease_out"), key(f(22), 330, 300), key(f(56), 330, 300, easing="ease_in"),
                key(f(70), 330, 300, opacity=0.0)]), card(400, 64, (50, 90, 200), (80, 130, 230))),
            (dict(id="lay_name", kind="text", text="Jane Doe", z=1, keys=[
                key(f(18), 280, 300, opacity=0.0, easing="ease_out"), key(f(32), 300, 300),
                key(f(56), 300, 300, easing="ease_in"), key(f(70), 300, 300, opacity=0.0)]),
             text("Jane Doe", 1.2, (255, 255, 255), 2)),
            (dict(id="lay_badge", kind="shape", z=2, keys=[
                key(f(35), 500, 250, opacity=0.0, easing="hold"), key(f(36), 500, 250, easing="ease_out"),
                key(f(46), 500, 250, scale=1.3, easing="ease_in"), key(f(56), 500, 250)]), badge(70)),
        ],
    }


def background(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    top, bot = np.array([30, 40, 70], float), np.array([90, 60, 120], float)
    g = top + (bot - top) * np.linspace(0, 1, H)[:, None, None]
    img = np.repeat(g, W, axis=1)
    noise = cv2.GaussianBlur(rng.normal(0, 10, (H, W)).astype(np.float32), (0, 0), 1.5)
    return np.clip(img + noise[..., None], 0, 255)


def segments(layer: Layer2D) -> list[dict]:
    return [{"start_frame": round(a.time * FPS), "end_frame": round(b.time * FPS), "easing": a.easing}
            for a, b in zip(layer.keys, layer.keys[1:])
            if any(getattr(a, c) != getattr(b, c) for c in ("x", "y", "rotation", "scale", "opacity"))]


def make(cid: str, layers_spec, out: Path, seed: int) -> None:
    d = out / cid
    (d / "gt_passes" / "sprites").mkdir(parents=True, exist_ok=True)
    layers, sprites = [], {}
    for kw, spr in layers_spec:
        h, w = spr.shape[:2]
        layers.append(Layer2D(size=(float(w), float(h)), status="human_confirmed",
                              sprite=FileRef(id=f"file_{kw['id']}", path=f"gt_passes/sprites/{kw['id']}.png"), **kw))
        sprites[kw["id"]] = spr
        png = (np.clip(spr, 0, 1) * 255 + 0.5).astype(np.uint8)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(png, cv2.COLOR_RGBA2BGRA))
        buf.tofile(d / "gt_passes" / "sprites" / f"{kw['id']}.png")  # not imwrite: non-ASCII paths
    rgb, index = render(layers, sprites, FRAMES, FPS, (W, H), background(seed))
    encode_mp4(rgb, d / "video.mp4", fps=FPS)
    np.savez_compressed(d / "gt_passes" / "index.npz", index=index)
    (d / "gt_passes" / "id_map.json").write_text(
        json.dumps({str(i + 1): lay.id for i, lay in enumerate(layers)}, indent=1) + "\n", encoding="utf-8")
    project = Project(
        id=f"proj_{cid}",
        source=SourceVideo(file=FileRef(id="file_video", path="video.mp4"), fps=Rational(num=FPS, den=1),
                           frame_count=FRAMES, width=W, height=H),
        shots=[Shot(id="shot_001", start=0, end=FRAMES, transition_from_previous="first")],
        reconstruction={"layers": layers, "shot_mapping": [ShotMapping(shot_id="shot_001", scene_time_start=0.0)]},
    )
    save_project(project, d / "gt.json")
    meta = {"id": cid, "kind": "2d_graphics", "frames": FRAMES, "fps": FPS, "size": [W, H],
            "layers": [{"id": lay.id, "kind": lay.kind, "text": lay.text, "z": lay.z, "segments": segments(lay)}
                       for lay in layers]}
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(cid, "->", d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/synthetic2d")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for n, (cid, spec) in enumerate(specs().items()):
        if a.only and cid not in a.only:
            continue
        make(cid, spec, Path(a.out), seed=100 + n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
