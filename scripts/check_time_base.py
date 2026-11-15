"""WP-2 step 4: what removing padded frames changes for tennis, without re-running any observer.

    python scripts/check_time_base.py [--video runs/human/tennis/tennis.mp4]
        [--skeleton runs/human/tennis/gvhmr_skeleton.npz] [--out runs/time_base/tennis.json]

Reads the stored video and the stored GVHMR skeleton. Before: the body keyed at original frame / container fps
(how from_gvhmr keyed it), sampled at 24 fps as the H3 export does. After: the padded frames removed with the
frame map of vrecon.video.detect_cadence, the rest keyed at the project's new rate, sampled at 24 fps the same
way. Reports the detected cadence and, for the body's root (in camera space, the part placement uses), the share
of 24 fps control frames in which it does not move (step under 10% of the median step).
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sample(times: np.ndarray, pos: np.ndarray, out_times: np.ndarray) -> np.ndarray:
    """Hold-free linear interpolation of positions [N,3] at out_times (as Blender's linear keys do)."""
    return np.stack([np.interp(out_times, times, pos[:, k]) for k in range(pos.shape[1])], axis=1)


def still_share(pos: np.ndarray) -> tuple[float, float, float]:
    """share of steps under 10% of the median step, the median step, and the unevenness of the steps: the
    median over frames of |step - mean of its two neighbours| / median step (0 for a smooth path)."""
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    med = float(np.median(step))
    jag = np.abs(step[1:-1] - 0.5 * (step[:-2] + step[2:])) / med
    return float(np.mean(step < 0.1 * med)), med, float(np.median(jag))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "runs/human/tennis/tennis.mp4")
    ap.add_argument("--skeleton", type=Path, default=ROOT / "runs/human/tennis/gvhmr_skeleton.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/time_base/tennis.json")
    a = ap.parse_args()

    from vrecon.control.core import FPS as H3_FPS
    from vrecon.control.core import read_video_rgb
    from vrecon.video import STANDARD_RATES, detect_cadence, frame_rate, probe

    info = probe(a.video)
    frames = read_video_rgb(a.video)
    cad = detect_cadence(frames.mean(axis=3).astype(np.uint8))
    d = np.load(a.skeleton)
    n, fps0 = int(d["frames"]), float(d["fps"])
    root = d["joints_cam"][:, 0]  # pelvis in camera space, per original frame
    rep = {"video": a.video.relative_to(ROOT).as_posix(), "frames": int(len(frames)), "container_fps": str(info.fps), "frame_rate": str(frame_rate(info)),
           "skeleton_frames": n, "skeleton_fps": fps0}
    if cad is None:
        rep["cadence"] = None
        print(json.dumps(rep, indent=1))
        return 0
    kept = [i for i in range(n) if i not in set(cad.phase_frames)]
    new = (frame_rate(info) * Fraction(len(kept), n)).limit_denominator(1001)
    near = min(STANDARD_RATES, key=lambda r: abs(float(r - new)))
    new_fps = near if abs(float(near - new)) < 0.01 * float(near) else new
    rep["cadence"] = {"period": cad.period, "phase": cad.phase, "removed": len(cad.phase_frames), "found_copies": len(cad.repeats), "share": cad.share,
                      "kept": len(kept), "new_fps": f"{new_fps.numerator}/{new_fps.denominator}"}
    # repeated video frames: does GVHMR give (nearly) the same root there?
    step_all = np.linalg.norm(np.diff(root, axis=0), axis=1)
    rep_steps = step_all[[i - 1 for i in cad.repeats if i >= 1]]
    rep["gvhmr_step_at_repeats_over_median"] = float(np.median(rep_steps) / np.median(step_all))

    dur_before = n / fps0
    t_before = np.arange(n) / fps0
    t_after = np.arange(len(kept)) / float(new_fps)
    out_before = np.arange(int(dur_before * H3_FPS)) / H3_FPS
    out_after = np.arange(int(t_after[-1] * H3_FPS) + 1) / H3_FPS
    s_b, med_b, j_b = still_share(sample(t_before, root, out_before))
    s_a, med_a, j_a = still_share(sample(t_after, root[kept], out_after))
    rep["before"] = {"keys": n, "control_frames": len(out_before), "still_share": s_b, "median_step_m": med_b, "unevenness": j_b}
    rep["after"] = {"keys": len(kept), "control_frames": len(out_after), "still_share": s_a, "median_step_m": med_a, "unevenness": j_a,
                    "resampling_needed": float(new_fps) != H3_FPS}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
