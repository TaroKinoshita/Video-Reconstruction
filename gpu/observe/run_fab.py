"""FAB (Cao 2026, github.com/noahcao/FAB) body per frame for a cut, for close-ups where a whole-body video tracker
(GVHMR) fails: FAB estimates the body from each image, part of the body may be out of the picture. Runs in the fab env.

    <fab env>/python gpu/observe/run_fab.py FRAMES_DIR MASKS.npz FOCAL_PX OUT.npz

The person is the largest SAM mask's box per frame (no detector: in a close-up it often finds nothing or a face).
Output: joints [T, 22, 3] SMPL joints in the Blender camera frame (x right, y up, -z forward), metres; ok [T] bool;
contact [T, 16] (FAB's contact probabilities); visible [T, 22] (FAB's joint visibility where available).
The Unitree G1 head (robot retargeting) is switched off: it is not needed and needs robot files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> int:
    frames_dir, masks_p, focal, out = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
    import fab.paths
    import fab.retarget.g1 as g1

    class _NoKin:
        def to(self, *_):
            return None

    g1.G1Kinematics = lambda *a, **k: _NoKin()
    _req = fab.paths.require
    fab.paths.require = lambda p, *a, **k: p if "G1" in str(a[:1]) else _req(p, *a, **k)
    import cv2
    import torch
    from fab.predictor import FABPredictor
    from bench.body_models import SMPL

    pred = FABPredictor("fab-l", fast=False, detector="")
    smpl = SMPL(fab.paths.body_models_dir() / "smpl" / "SMPL_NEUTRAL.pkl")
    Jr = smpl.J_regressor.numpy()[:22]
    masks = np.load(masks_p)["masks"]
    files = sorted(frames_dir.glob("*.png"))
    T = min(len(files), len(masks))
    J = np.zeros((T, 22, 3))
    ok = np.zeros(T, bool)
    con = np.zeros((T, 16))
    for t in range(T):
        img = cv2.imdecode(np.fromfile(str(files[t]), np.uint8), cv2.IMREAD_COLOR)[..., ::-1].copy()
        H, W = img.shape[:2]
        m = masks[t]
        ids = [i for i in np.unique(m) if i]
        if not ids:
            continue
        big = max(ids, key=lambda i: (m == i).sum())
        ys, xs = np.nonzero(m == big)
        sx, sy = W / m.shape[1], H / m.shape[0]
        box = np.array([[xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy]], float)
        K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])
        with torch.no_grad():
            people = pred.predict(img, K=K, boxes=box)
        if not people:
            continue
        p = people[0]
        mesh = np.asarray(p["mesh_smpl_fit"], float)  # already in the camera frame (checked: its head projects on
        jc = Jr @ mesh  # the 2D head keypoints; dense_surface_cam does not, in close-ups)  OpenCV camera frame
        J[t] = jc * [1, -1, -1]
        ok[t] = True
        if "contact" in p:
            con[t] = np.asarray(p["contact"], float)
        print(t, "ok", np.round(jc[0], 2), flush=True)
    np.savez_compressed(out, joints=J, ok=ok, contact=con)
    print({"frames": T, "ok": int(ok.sum())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
