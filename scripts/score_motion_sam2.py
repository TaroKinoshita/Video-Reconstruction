"""Motion of the main person in generated videos, with SAM 2 run locally (PLAN v2 8.2 step 2; no pod needed).

Runs in a Python env with torch + sam2 (not the vrecon venv); reads videos through ffmpeg (no OpenCV paths).

    python scripts/score_motion_sam2.py --ckpt sam2.1_hiera_small.pt --source-masks MASKS.npz --source-label obs_0
        --source-video SRC.mp4 --start 5.26 --out OUT.json  CONTROL.mp4:GENERATED.mp4 [...]

Per pair: the main person is prompted with a box of the largest orange (person label) blob of the control's first
frame and tracked by SAM 2 in both the control and the generated video.
- follows_control: mask IoU generated vs its own control per frame (does the generator keep the scene's motion).
- vs_source: the person's mask height and width over time (each divided by its median, so the view and size do not
  count), compared with the same series of the source video's person mask over the same time window: Pearson
  correlation and mean absolute difference. This compares with the real footage, not with any scene.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def read_video(path: Path) -> np.ndarray:
    info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
                           "-of", "csv=p=0", str(path)], capture_output=True, text=True, encoding="utf-8").stdout
    w, h = (int(x) for x in info.strip().split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def person_box(frame: np.ndarray) -> list[float]:
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # the person label is orange, shaded: by hue, not by value
    m = ((hsv[..., 0] >= 5) & (hsv[..., 0] <= 22) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = st[k, :4]
    return [float(x), float(y), float(x + w), float(y + h)]


def track(pred, frames: np.ndarray, box: list[float]) -> np.ndarray:
    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(f"{d}/{i:05d}.jpg", quality=95)
        st = pred.init_state(video_path=d, offload_video_to_cpu=True)
        pred.add_new_points_or_box(st, frame_idx=0, obj_id=1, box=np.array(box))
        out = np.zeros(frames.shape[:3], bool)
        for fi, _, logits in pred.propagate_in_video(st):
            out[fi] = (logits[0, 0] > 0).cpu().numpy()
    return out


def hw(masks: np.ndarray) -> np.ndarray:
    s = np.full((len(masks), 2), np.nan)
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(ys) > 20:
            s[i] = [np.quantile(ys, .98) - np.quantile(ys, .02), np.quantile(xs, .98) - np.quantile(xs, .02)]
    return s / np.nanmedian(s, 0)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    out = {}
    for j, name in enumerate(("height", "width")):
        ok = np.isfinite(a[:, j]) & np.isfinite(b[:, j])
        out[name] = {"corr": round(float(np.corrcoef(a[ok, j], b[ok, j])[0, 1]), 3),
                     "mad": round(float(np.abs(a[ok, j] - b[ok, j]).mean()), 3)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--source-masks", type=Path, required=True)
    ap.add_argument("--source-label", default="obs_0")
    ap.add_argument("--source-fps", type=float, default=29.83)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("pairs", nargs="+")
    a = ap.parse_args()
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    pred = build_sam2_video_predictor(a.cfg, a.ckpt, device="cuda")
    z = np.load(a.source_masks)
    lab = {v: int(k) for k, v in json.loads(str(z["id_map"])).items()}[a.source_label]
    sm = z["masks"]
    rep = {"pairs": {}}
    for pair in a.pairs:
        cpath, gpath = (Path(p) for p in pair.split(":"))
        ctrl, gen = read_video(cpath), read_video(gpath)
        t = min(len(ctrl), len(gen))
        src_idx = [min(round((a.start + i / 24) * a.source_fps), len(sm) - 1) for i in range(t)]
        src = hw(np.stack([sm[i] == lab for i in src_idx]))
        box = person_box(ctrl[0])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            mc = track(pred, ctrl[:t], box)
            mg = track(pred, gen[:t], box)
        inter = (mc & mg).sum((1, 2))
        union = (mc | mg).sum((1, 2))
        iou = inter / np.maximum(union, 1)
        name = gpath.parent.name
        rep["pairs"][name] = {
            "box": box,
            "follows_control_iou": round(float(np.median(iou)), 3),
            "control_vs_source": compare(hw(mc), src),
            "generated_vs_source": compare(hw(mg), src),
        }
        np.savez_compressed(gpath.parent / "sam2_person.npz", generated=mg, control=mc)
        print(name, json.dumps(rep["pairs"][name]), flush=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
