"""WP-7 step 1 (review F-04): is a lost camera translation 4RC's error or the conversion's?

    python scripts/diagnose_camera_path.py CLIP_DIR FOUR_RC.npz START_PROJECT.json [ADOPTED_PROJECT.json]

Compares camera centres over the clip, each against the ground truth after the best similarity alignment
(scale, rotation, translation; Umeyama), as a share of the ground-truth path length:
    raw       4RC's own extrinsics (the frames 4RC saw: linspace over the clip, as run/service.py feeds it)
    start     the start scene built from it (analysis/from_4d -> project camera)
    adopted   optionally the scene after repair
Also the travelled path length of each, relative to the ground truth (1.0 = the full truck reproduced).
Ground truth is read here only to score (evaluation use).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray):
    """s, R, t minimising |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    sgn = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sgn[2, 2] = -1
    r = u @ sgn @ vt
    var = (a ** 2).sum() / len(src)
    s = float(np.trace(np.diag(d) @ sgn) / var) if var > 1e-12 else 1.0
    return s, r, mu_d - s * r @ mu_s


def compare(est: np.ndarray, gt: np.ndarray) -> dict:
    path = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    if np.allclose(est, est[0]):
        return {"error_rel_path": float(np.linalg.norm(gt - gt.mean(0), axis=1).mean() / max(path, 1e-9)),
                "travel_rel": 0.0}
    s, r, t = umeyama(est, gt)
    al = (s * (r @ est.T)).T + t
    travel = float(np.linalg.norm(np.diff(al, axis=0), axis=1).sum())
    return {"error_rel_path": float(np.linalg.norm(al - gt, axis=1).mean() / max(path, 1e-9)),
            "travel_rel": travel / max(path, 1e-9), "scale": s}


def main() -> int:
    clip, rc, start = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    adopted = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    from vrecon.metrics.poses import camera_pose
    from vrecon.schema import load_project

    gt = load_project(clip / "gt.json")
    n_clip = gt.shots[0].end - gt.shots[0].start
    z = np.load(rc)
    n = int(z["n_frames"])
    frames = [int(f) for f in np.linspace(0, n_clip - 1, n, dtype=int)]
    gt_c = np.array([camera_pose(gt.reconstruction.cameras[0], f)[1] for f in frames])
    # pred_<i>_extrinsic is OpenCV camera-to-world (analysis/from_4d.Model4D.c2w): the centre is its translation
    raw = np.array([np.asarray(z[f"pred_{i}_extrinsic"], float)[:3, 3] for i in range(n)])
    rep = {"clip": clip.name, "frames": len(frames),
           "gt_path_length": float(np.linalg.norm(np.diff(gt_c, axis=0), axis=1).sum()),
           "raw_4rc": compare(raw, gt_c)}
    for name, path in (("start_scene", start), ("adopted", adopted)):
        if path is None:
            continue
        p = load_project(path)
        est = np.array([camera_pose(p.reconstruction.cameras[0], f)[1] for f in frames])
        rep[name] = compare(est, gt_c)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
