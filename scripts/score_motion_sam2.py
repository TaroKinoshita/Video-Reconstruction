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
- shape_vs_source: height / width of the mask (divided by its median). Unlike height and width alone it does not
  change when the camera moves closer (dolly), so it holds for any camera path.
- pose_vs_source (with --pose YOLO_POSE.pt): 2D joint angles of the same person (both elbows, both knees, both
  arms against the body) from a pose model, per angle correlation with the source over time (median over angles),
  and the time shift (frames, +: later than the source) that lines up the whole set best. Angles do not depend on
  the person's size in the picture; a turned camera changes them somewhat, the same for both conditions. Its time
  shift is noisy (few source frames with a confident pose); timing_lag_frames from the mask height is the timing.
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


ANGLES = [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16), (11, 5, 7), (12, 6, 8)]  # COCO: elbow, knee, arm


def pose_angles(model, frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Per frame the angles of the detected person that overlaps the mask most (NaN when none or low confidence)."""
    out = np.full((len(frames), len(ANGLES)), np.nan)
    for i in range(0, len(frames), 16):
        res = model.predict(list(frames[i:i + 16]), verbose=False, conf=0.25)
        for j, r in enumerate(res):
            f = i + j
            if r.keypoints is None or len(r.boxes) == 0 or masks[f].sum() < 20:
                continue
            m = masks[f]
            best, bi = -1.0, None
            for k, b in enumerate(r.boxes.xyxy.cpu().numpy().astype(int)):
                x0, y0, x1, y1 = np.clip(b, 0, [m.shape[1] - 1, m.shape[0] - 1] * 2)
                ov = m[y0:y1 + 1, x0:x1 + 1].sum() / max(m.sum(), 1)
                if ov > best:
                    best, bi = ov, k
            if best < 0.3:
                continue
            kp = r.keypoints.xy.cpu().numpy()[bi]
            cf = r.keypoints.conf.cpu().numpy()[bi]
            for n, (a, b, c) in enumerate(ANGLES):
                if min(cf[a], cf[b], cf[c]) < 0.4:
                    continue
                u, v = kp[a] - kp[b], kp[c] - kp[b]
                out[f, n] = np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1)))
    return out


def pose_compare(a: np.ndarray, b: np.ndarray, max_lag: int = 24) -> dict:
    cs = []
    for n in range(a.shape[1]):
        ok = np.isfinite(a[:, n]) & np.isfinite(b[:, n])
        if ok.sum() > 10 and np.std(a[ok, n]) > 1e-6 and np.std(b[ok, n]) > 1e-6:
            cs.append(float(np.corrcoef(a[ok, n], b[ok, n])[0, 1]))
    za = (a - np.nanmean(a, 0)) / (np.nanstd(a, 0) + 1e-9)
    zb = (b - np.nanmean(b, 0)) / (np.nanstd(b, 0) + 1e-9)
    best = (None, -9.0)
    for lag in range(-max_lag, max_lag + 1):  # a later by lag frames: a[t + lag] ~ b[t]
        x, y = (za[lag:], zb[:len(zb) - lag]) if lag >= 0 else (za[:lag], zb[-lag:])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 30:
            c = float(np.mean(x[ok] * y[ok]))
            if c > best[1]:
                best = (lag, c)
    return {"angle_corr_median": round(float(np.median(cs)), 3) if cs else None, "angles_used": len(cs),
            "frames_with_pose": int(np.isfinite(a).any(1).sum()), "best_lag_frames": best[0]}


def shape(masks: np.ndarray) -> np.ndarray:
    s = hw(masks)
    r = s[:, :1] / s[:, 1:]
    return r / np.nanmedian(r)


def lag(a: np.ndarray, b: np.ndarray, max_lag: int = 24) -> int | None:
    """Frames by which series a is later than b (best correlation over shifts)."""
    best = (None, -9.0)
    for d in range(-max_lag, max_lag + 1):
        x, y = (a[d:], b[:len(b) - d]) if d >= 0 else (a[:d], b[-d:])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 30:
            c = float(np.corrcoef(x[ok], y[ok])[0, 1])
            if c > best[1]:
                best = (d, c)
    return best[0]


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    out = {"timing_lag_frames": lag(a[:, 0], b[:, 0])}
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
    ap.add_argument("--source-label", default="auto", help="mask label of the person; auto = largest obs_*")
    ap.add_argument("--source-fps", type=float, help="default: read from --source-video")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--retrack", action="store_true", help="run SAM 2 again even when sam2_person.npz exists")
    ap.add_argument("--pose", help="YOLO pose weights (ultralytics); adds joint-angle comparison")
    ap.add_argument("--source-video", type=Path, help="the source video (needed with --pose)")
    ap.add_argument("pairs", nargs="+")
    a = ap.parse_args()
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    pred = build_sam2_video_predictor(a.cfg, a.ckpt, device="cuda")
    z = np.load(a.source_masks)
    if a.source_fps is None:
        if a.source_video is None:
            raise SystemExit("give --source-fps or --source-video (the masks' frame rate)")
        num = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=avg_frame_rate",
                              "-of", "csv=p=0", str(a.source_video)], capture_output=True, text=True,
                             encoding="utf-8").stdout.strip().split("/")
        a.source_fps = int(num[0]) / int(num[1])
    sm = z["masks"]
    ids = {v: int(k) for k, v in json.loads(str(z["id_map"])).items()}
    if a.source_label == "auto":  # the main person: the observed person label with the most pixels
        a.source_label = max((v for v in ids if v.startswith("obs")), key=lambda v: int((sm == ids[v]).sum()))
    lab = ids[a.source_label]
    rep = {"pairs": {}}
    pose = None
    if a.pose:
        from ultralytics import YOLO

        pose = YOLO(a.pose)
        sv = read_video(a.source_video)
        sv_fps = a.source_fps
    src_pose = None
    for pair in a.pairs:
        cpath, gpath = (Path(p) for p in pair.split(":"))
        ctrl, gen = read_video(cpath), read_video(gpath)
        t = min(len(ctrl), len(gen))
        # source frame of each output frame (24 fps); past the source's end: no frame (not the last one repeated)
        raw_idx = [round((a.start + i / 24) * a.source_fps) for i in range(t)]
        src_idx = [min(i, len(sm) - 1) for i in raw_idx]
        outside = np.array([i >= len(sm) for i in raw_idx])
        src_m = np.stack([sm[i] == lab for i in src_idx])
        src_m[outside] = False
        src, shape_src = hw(src_m), shape(src_m)
        box = person_box(ctrl[0])
        cache = gpath.parent / "sam2_person.npz"
        if cache.is_file() and not a.retrack:  # SAM 2 masks of this pair from an earlier run
            zc = np.load(cache)
            mc, mg = zc["control"][:t], zc["generated"][:t]
        else:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                mc = track(pred, ctrl[:t], box)
                mg = track(pred, gen[:t], box)
        inter = (mc & mg).sum((1, 2))
        union = (mc | mg).sum((1, 2))
        iou = inter / np.maximum(union, 1)
        name = gpath.parent.name
        def share(m):
            return round(float((m.sum((1, 2)) > 20).mean()), 3)

        rep["pairs"][name] = {
            "box": box,
            "measured": {"person": f"source mask label {a.source_label}; generated and control: SAM 2 from the box of the "
                                   "largest person-label blob of the control's first frame",
                         "output_frames": t, "output_fps": 24, "source_fps": a.source_fps, "start_s": a.start,
                         "source_frames": [int(src_idx[0]), int(src_idx[-1])], "frames_past_source_end": int(outside.sum()),
                         "tracked_share": {"source": share(src_m), "control": share(mc), "generated": share(mg)}},
            "follows_control_iou": round(float(np.median(iou)), 3),
            "control_vs_source": compare(hw(mc), src),
            "generated_vs_source": compare(hw(mg), src),
        }
        rep["pairs"][name]["shape_vs_source"] = {"control": {k: v for k, v in compare(np.repeat(shape(mc), 2, 1), np.repeat(shape_src, 2, 1))["height"].items()},
                                                 "generated": compare(np.repeat(shape(mg), 2, 1), np.repeat(shape_src, 2, 1))["height"]}
        if pose is not None:
            if src_pose is None:
                import cv2

                sf = np.stack([sv[min(round((a.start + i / 24) * sv_fps), len(sv) - 1)] for i in range(t)])
                smk = np.stack([cv2.resize((sm[i] == lab).astype(np.uint8), (sf.shape[2], sf.shape[1]),
                                           interpolation=cv2.INTER_NEAREST) > 0 for i in src_idx])
                src_pose = pose_angles(pose, sf, smk)
                rep["source_frames_with_pose"] = int(np.isfinite(src_pose).any(1).sum())
            rep["pairs"][name]["pose_vs_source"] = {"control": pose_compare(pose_angles(pose, ctrl[:t], mc), src_pose),
                                                    "generated": pose_compare(pose_angles(pose, gen[:t], mg), src_pose)}
        np.savez_compressed(gpath.parent / "sam2_person.npz", generated=mg, control=mc)
        print(name, json.dumps(rep["pairs"][name]), flush=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
