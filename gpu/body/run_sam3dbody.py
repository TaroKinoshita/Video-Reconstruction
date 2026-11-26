"""3D bodies with SAM 3D Body run locally (env %LOCALAPPDATA%/vrecon/envs/body, which borrows the observe env's torch;
code in %LOCALAPPDATA%/vrecon/src/sam-3d-body, weights in %LOCALAPPDATA%/vrecon/weights/sam-3d-body-dinov3).
Not part of vrecon's .venv.

  python gpu/body/run_sam3dbody.py VIDEO.mp4 PEOPLE.masks.npz OUT_DIR --focal F [--frames 0 2 4 ...] [--labels obs_0 ...]

One person per call, conditioned on that person's SAM 3.1 mask and box, with our camera's focal length (pixels at the
video's size, principal point at the centre), so the body comes out in our camera. Writes the same files as
scripts/v11_fal_body.py: f<frame>_<label>_0.ply (OpenGL camera axes, translation applied) and f<frame>_<label>.json
({"metadata": {"people": [{focal_length, pred_cam_t, keypoints_3d, keypoints_2d, bbox}]}}), so the rest of the
pipeline reads either. Prints peak GPU memory. ASCII paths only (cv2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(os.environ["LOCALAPPDATA"]) / "vrecon"
sys.path.insert(0, str(ROOT / "src" / "sam-3d-body"))
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402


def write_ply(path: Path, V: np.ndarray, F: np.ndarray) -> None:
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\n"
            f"property float z\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode()
    faces = np.zeros(len(F), np.dtype([("n", "u1"), ("i", "<i4", 3)]))
    faces["n"], faces["i"] = 3, F
    verts = np.zeros(len(V), np.dtype([("p", "<f4", 3), ("c", "u1", 4)]))  # same layout as fal's files
    verts["p"], verts["c"] = V, 200
    path.write_bytes(head + verts.tobytes() + faces.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("masks")
    ap.add_argument("out", type=Path)
    ap.add_argument("--focal", type=float, required=True, help="pixels at the video's size")
    ap.add_argument("--frames", type=int, nargs="*", help="default: every frame")
    ap.add_argument("--labels", nargs="*", help="default: people present (>= --min-area) in half the frames")
    ap.add_argument("--min-area", type=float, default=0.002)
    ap.add_argument("--hands", action="store_true", help="also refine hands (slower)")
    a = ap.parse_args()
    d = np.load(a.masks)
    masks, id_map = d["masks"], json.loads(str(d["id_map"]))
    if a.labels:
        id_map = {n: l for n, l in id_map.items() if l in a.labels}
    else:
        id_map = {n: l for n, l in id_map.items() if ((masks == int(n)).mean(axis=(1, 2)) >= a.min_area).mean() >= 0.5}
    frames = a.frames if a.frames else list(range(len(masks)))
    a.out.mkdir(parents=True, exist_ok=True)
    w = ROOT / "weights" / "sam-3d-body-dinov3"
    model, cfg = load_sam_3d_body(str(w / "model.ckpt"), device="cuda", mhr_path=str(w / "assets" / "mhr_model.pt"))
    est = SAM3DBodyEstimator(model, cfg)
    faces = est.faces.astype(np.int32)
    H, W = masks.shape[1:]
    K = np.array([[[a.focal, 0, W / 2], [0, a.focal, H / 2], [0, 0, 1]]], np.float32)
    cap = cv2.VideoCapture(a.video)
    want, i, done, t0 = set(frames), 0, 0, time.time()
    torch.cuda.reset_peak_memory_stats()
    while want:
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            want.discard(i)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for num, label in id_map.items():
                stem = f"f{i:04d}_{label}"
                m = masks[i] == int(num)
                if m.mean() < a.min_area or (a.out / f"{stem}.json").exists():
                    continue
                ys, xs = np.nonzero(m)
                box = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], np.float32)
                res = est.process_one_image(rgb, bboxes=box, masks=m[None].astype(np.uint8),
                                            cam_int=torch.from_numpy(K), inference_type="full" if a.hands else "body")
                r = res[0]
                V = (r["pred_vertices"] + r["pred_cam_t"]).astype(np.float64) * [1, -1, -1]  # OpenCV -> OpenGL
                write_ply(a.out / f"{stem}_0.ply", V, faces)
                meta = {"focal_length": float(np.asarray(r["focal_length"]).ravel()[0]),
                        "pred_cam_t": np.asarray(r["pred_cam_t"]).tolist(),
                        "keypoints_3d": (r["pred_keypoints_3d"] + r["pred_cam_t"]).tolist(),
                        "keypoints_2d": np.asarray(r["pred_keypoints_2d"]).tolist(),
                        "bbox": np.asarray(r["bbox"]).tolist()}
                (a.out / f"{stem}.json").write_text(json.dumps({"metadata": {"num_people": 1, "people": [meta]},
                                                                 "source": "local"}), encoding="utf-8")
                done += 1
                if done % 20 == 1:
                    print(json.dumps({"frame": i, "person": label, "done": done, "s_per_call": round((time.time() - t0) / done, 2),
                                      "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}), flush=True)
        i += 1
    print(json.dumps({"done": done, "seconds": round(time.time() - t0), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
