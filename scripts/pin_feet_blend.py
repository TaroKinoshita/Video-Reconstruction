"""Stop a placed body's planted feet from sliding (PLAN v2 8, foot contact), without changing how it looks from the
source camera.

    python scripts/pin_feet_blend.py IN.blend OUT.blend [--report OUT.json] [--height 0.04] [--sigma 2]

The observed body is placed on the source camera's rays at GVHMR's per-frame depth (analysis/place_person); depth
noise along the ray shows as feet sliding on the floor. Per frame, the feet joints within `height` m of their lowest
level are planted; each planted run keeps the foot where the run began. The correction that would do that (mean
over planted feet) is kept only along the camera's horizontal viewing direction (the depth direction: the picture
from the source camera barely changes), smoothed over time (Gaussian, sigma frames), and added to the rig's root
location keys. Reports the median sliding speed of planted feet before and after.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def slide(P: dict, fps: float, height: float) -> dict:
    out = {}
    for f, p in P.items():
        zmin = np.quantile(p[:, 2], 0.05)
        c = p[:, 2] < zmin + height
        v = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) * fps
        cc = c[1:] & c[:-1]
        out[f] = round(float(np.median(v[cc])) * 100, 1) if cc.any() else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("blend", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--height", type=float, default=0.04)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--all-directions", action="store_true", help="correct along the floor in every direction (moves "
                    "the body a little in the source picture too); default: the depth direction only")
    a = ap.parse_args()
    import bpy

    from vrecon.metrics.poses import camera_pose
    from vrecon.scene.novel_blend import open_scene

    rec = open_scene(a.blend)
    cam = rec.reconstruction.cameras[0]
    sc = bpy.context.scene
    fps = sc.render.fps / sc.render.fps_base
    arm = next(o for o in bpy.data.objects if o.get("vrecon_rig_of"))
    names = [b.name for b in arm.pose.bones]
    feet = [x for x in ("left_foot", "right_foot", "left_ankle", "right_ankle") if x in names]
    n = sc.frame_end - sc.frame_start + 1

    def sample():
        P = {f: [] for f in feet}
        for i in range(n):
            sc.frame_set(sc.frame_start + i)
            for f in feet:
                P[f].append(list(arm.matrix_world @ arm.pose.bones[f].head))
        return {f: np.array(v) for f, v in P.items()}

    P = sample()
    before = slide(P, fps, a.height)
    corr = np.zeros((n, 3))
    cnt = np.zeros(n)
    for f, p in P.items():
        zmin = np.quantile(p[:, 2], 0.05)
        planted = p[:, 2] < zmin + a.height
        anchor = None
        for i in range(n):
            if planted[i]:
                if anchor is None:
                    anchor = p[i].copy()
                corr[i, :2] += (anchor - p[i])[:2]
                cnt[i] += 1
            else:
                anchor = None
    corr[cnt > 0] /= cnt[cnt > 0, None]
    # only along the camera's horizontal viewing direction (unless --all-directions)
    for i in range(n if not a.all_directions else 0):
        r, _, _ = camera_pose(cam, i)
        fwd = np.asarray(r)[:, 2].copy()
        fwd[2] = 0.0
        fwd /= np.linalg.norm(fwd) + 1e-9
        corr[i] = fwd * float(corr[i] @ fwd)
    g = np.exp(-0.5 * (np.arange(-6, 7) / a.sigma) ** 2)
    g /= g.sum()
    pad = np.pad(corr, ((6, 6), (0, 0)), mode="edge")
    corr = np.stack([np.convolve(pad[:, k], g, mode="valid") for k in range(3)], 1)
    base = np.array(list(arm.location))  # read once: keys written below would otherwise be read back and add up
    for i in range(n):
        arm.location = (base + corr[i]).tolist()
        arm.keyframe_insert("location", frame=sc.frame_start + i)
    after = slide(sample(), fps, a.height)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.out.resolve()))
    rep = {"slide_cm_s_before": before, "slide_cm_s_after": after,
           "max_correction_m": round(float(np.abs(corr).max()), 3)}
    if a.report:
        a.report.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
