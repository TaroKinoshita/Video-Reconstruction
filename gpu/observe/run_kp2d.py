"""2D body keypoints (COCO-17, ViTPose-H as in GVHMR) on one video, inside given person boxes. Runs in WSL, gvhmr env:

    cd ~/vrecon/GVHMR && ~/vrecon/bin/micromamba run -r ~/vrecon/mamba -n gvhmr \
        python /mnt/c/.../gpu/observe/run_kp2d.py VIDEO BOXES.npz OUT.npz

BOXES.npz: xyxy [T, 4] in video pixels, NaN where the person is not seen (from the SAM person masks).
OUT.npz: kp2d [T, 17, 3] (x, y in video pixels, confidence 0..1), ok [T] (a box was given for that frame).
These are observations of the picture (independent of the 3D body estimators); the fits and the checks of
docs/PLAN_v4.md compare the scene's people with them.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy  # noqa: E402
from hmr4d.utils.preproc import VitPoseExtractor  # noqa: E402


def main() -> int:
    video, boxes, out = sys.argv[1], sys.argv[2], sys.argv[3]
    xyxy = np.load(boxes)["xyxy"].astype(np.float32)
    ok = np.isfinite(xyxy).all(1)
    if not ok.any():
        np.savez_compressed(out, kp2d=np.zeros((len(xyxy), 17, 3), np.float32), ok=ok)
        return 0
    idx = np.flatnonzero(ok)
    near = idx[np.abs(np.arange(len(xyxy))[:, None] - idx[None]).argmin(1)]
    filled = torch.from_numpy(xyxy[near])
    bbx_xys = get_bbx_xys_from_xyxy(filled, base_enlarge=1.2).float()
    kp = VitPoseExtractor(tqdm_leave=False).extract(video, bbx_xys).numpy().astype(np.float32)
    kp[~ok, :, 2] = 0
    np.savez_compressed(out, kp2d=kp, ok=ok)
    print({"frames": len(kp), "with_person": int(ok.sum())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
