"""Hand-anchored windows for held props: crop the full-resolution video around each person's hands, so the segmenter
sees a held object large (a blurred staff mid-swing is a few pixels wide in the whole frame and a text prompt misses
it there); then paste the crop masks back into whole frames. Pure numpy/cv2; works for any clip with moving bodies.

  python scripts/v11_hand_windows.py prepare PREP.npz PEOPLE_ANIM.npz BODY_DIR VIDEO OUT_DIR [--frames 110-240]
      [--reach 1.4] [--chunk 64] [--size 512]
  (observe env) python gpu/observe/run_sam3.py OUT_DIR/<win>.mp4 OUT_DIR/<win>.masks.npz --movers "wooden stick" ...
  python scripts/v11_hand_windows.py merge OUT_DIR BASE.masks.npz MERGED.masks.npz

prepare: per person and per run of frames where the body exists (cut into --chunk frames), a square box centred on
the mean of the two hands (projected, smoothed in time), half side = reach metres at the hands' depth (one size per
window, the median); the crop is cut from VIDEO (any resolution; box scaled from PREP pixels) and resized to --size.
windows.json records every frame's box in PREP pixels. merge: masks of every window pasted back (nearest), only on
pixels still empty in BASE, labels kept apart per window as "<win>_<label>".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v11_staffs import body_at, hand_vertices  # noqa: E402
from vrecon.control.core import encode_mp4, read_video_rgb  # noqa: E402


def _smooth(x: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(x.shape[1])], 1)


def prepare(a) -> int:
    prep = np.load(a.prep)
    info = json.loads(str(prep["info"]))
    W, H = info["size"]
    f_img = info["f_img"]
    cam_R, cam_T = prep["cam_R"], prep["cam_T"]
    anim = dict(np.load(a.anim))
    people = json.loads(str(anim["labels"]))
    hv = hand_vertices(a.body)
    ids = np.concatenate(list(hv.values()))
    video = read_video_rgb(str(a.video))
    sx, sy = video.shape[2] / W, video.shape[1] / H
    lo, hi = (int(x) for x in a.frames.split("-")) if a.frames else (0, len(video) - 1)
    a.out.mkdir(parents=True, exist_ok=True)
    wins = {}
    for p in people:
        track = {}
        for f in range(lo, hi + 1):
            X = body_at(anim, p, f)
            if X is None:
                continue
            Xc = (X[ids].mean(0) - cam_T[f]) @ cam_R[f]
            if Xc[2] <= 0.1:
                continue
            track[f] = (f_img * Xc[0] / Xc[2] + W / 2, f_img * Xc[1] / Xc[2] + H / 2, a.reach * f_img / Xc[2])
        fs = sorted(track)
        runs, cur = [], []
        for f in fs:
            if cur and (f != cur[-1] + 1 or len(cur) >= a.chunk):
                runs.append(cur)
                cur = []
            cur.append(f)
        if cur:
            runs.append(cur)
        for run in runs:
            if len(run) < 12:
                continue
            uvr = _smooth(np.array([track[f] for f in run]))
            half = float(np.clip(np.median(uvr[:, 2]), 40, H / 2))
            name = f"h{p}_{run[0]:04d}"
            frames, boxes = [], []
            for f, (u, v, _) in zip(run, uvr):
                x0, y0 = u - half, v - half
                boxes.append([float(x0), float(y0), 2 * half])
                # crop from the full-resolution video, black outside the picture
                X0, Y0, S = int(round(x0 * sx)), int(round(y0 * sy)), int(round(2 * half * sx))
                Sy = int(round(2 * half * sy))
                crop = np.zeros((Sy, S, 3), np.uint8)
                a0, b0 = max(X0, 0), max(Y0, 0)
                a1, b1 = min(X0 + S, video.shape[2]), min(Y0 + Sy, video.shape[1])
                if a1 > a0 and b1 > b0:
                    crop[b0 - Y0:b1 - Y0, a0 - X0:a1 - X0] = video[f, b0:b1, a0:a1]
                frames.append(cv2.resize(crop, (a.size, a.size), interpolation=cv2.INTER_AREA))
            encode_mp4(np.array(frames), a.out / f"{name}.mp4", fps=24)
            wins[name] = {"person": p, "frames": run, "boxes": boxes}
            print(name, run[0], run[-1], f"half {half:.0f}px")
    (a.out / "windows.json").write_text(json.dumps({"size": [W, H], "crop": a.size, "windows": wins}), encoding="utf-8")
    return 0


def merge(a) -> int:
    spec = json.loads((a.dir / "windows.json").read_text(encoding="utf-8"))
    W, H = spec["size"]
    base = np.load(a.base)
    M = base["masks"].copy()
    id_map = json.loads(str(base["id_map"]))
    nid = max(map(int, id_map)) + 1 if id_map else 1
    for name, w in spec["windows"].items():
        p = a.dir / f"{name}.masks.npz"
        if not p.exists():
            print(name, "no masks")
            continue
        d = np.load(p)
        m, mp = d["masks"], json.loads(str(d["id_map"]))
        for n, lab in mp.items():
            k, nid = nid, nid + 1
            id_map[str(k)] = f"{name}_{lab}"
            px = 0
            for t, (f, (x0, y0, s)) in enumerate(zip(w["frames"], w["boxes"])):
                if t >= len(m):
                    break
                sel = (m[t] == int(n)).astype(np.uint8)
                if not sel.any():
                    continue
                S = max(int(round(s)), 1)
                big = cv2.resize(sel, (S, S), interpolation=cv2.INTER_NEAREST).astype(bool)
                X0, Y0 = int(round(x0)), int(round(y0))
                a0, b0 = max(X0, 0), max(Y0, 0)
                a1, b1 = min(X0 + S, W), min(Y0 + S, H)
                if a1 <= a0 or b1 <= b0:
                    continue
                sub = big[b0 - Y0:b1 - Y0, a0 - X0:a1 - X0] & (M[f, b0:b1, a0:a1] == 0)
                M[f, b0:b1, a0:a1][sub] = k
                px += int(sub.sum())
            print(name, lab, "->", k, "pixels", px)
    np.savez_compressed(a.out, masks=M, id_map=np.array(json.dumps(id_map)),
                        meta=base["meta"] if "meta" in base else np.array("{}"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    for n in ("prep", "anim", "body", "video", "out"):
        p.add_argument(n, type=Path)
    p.add_argument("--frames", default=None, help="lo-hi (default: all)")
    p.add_argument("--reach", type=float, default=1.4, help="metres from the hands to the crop edge")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--size", type=int, default=512)
    m = sub.add_parser("merge")
    for n in ("dir", "base", "out"):
        m.add_argument(n, type=Path)
    a = ap.parse_args()
    return prepare(a) if a.cmd == "prepare" else merge(a)


if __name__ == "__main__":
    raise SystemExit(main())
