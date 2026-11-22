"""2D keypoints of every cut's main person (docs/PLAN_v4.md phase 0): boxes from the SAM person masks, ViTPose-H
in WSL (gpu/observe/run_kp2d.py). Writes DIAG/kp2d_TAGNN.npz per cut (kp2d [T,17,3] in the cut video's pixels).

    python scripts/multishot_kp2d.py DIAG_DIR TAG
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from multishot_pipeline import ROOT, wsl_path  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))


def boxes(masks: np.ndarray, w: int, h: int) -> np.ndarray:
    out = np.full((len(masks), 4), np.nan, np.float32)
    for t, m in enumerate(masks):
        m = m > 0
        if m.sum() < 20:
            continue
        ys, xs = np.nonzero(m)
        sy, sx = h / m.shape[0], w / m.shape[1]
        out[t] = [xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy]
    return out


def main() -> int:
    from vrecon.control.core import read_video_rgb

    diag, tag = Path(sys.argv[1]).resolve(), sys.argv[2]
    for mp in sorted(diag.glob(f"{tag}_shot[0-9][0-9].masks.npz")):
        k = int(mp.name[len(tag) + 5:len(tag) + 7])
        vid = diag / f"{tag}_shot{k:02d}.mp4"
        out = diag / f"kp2d_{tag}{k:02d}.npz"
        if out.exists() or not vid.exists():
            continue
        v = read_video_rgb(vid)
        bp = diag / f"kp2d_{tag}{k:02d}.boxes.npz"
        np.savez(bp, xyxy=boxes(np.load(mp)["masks"], v.shape[2], v.shape[1]))
        cmd = (f"cd ~/vrecon/GVHMR && ~/vrecon/bin/micromamba run -r ~/vrecon/mamba -n gvhmr python "
               f"'{wsl_path(ROOT)}/gpu/observe/run_kp2d.py' '{wsl_path(vid)}' '{wsl_path(bp)}' '{wsl_path(out)}'")
        r = subprocess.run(["wsl", "-e", "bash", "-lc", cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(k, r.stdout.strip().splitlines()[-1:] if r.returncode == 0 else r.stderr[-800:], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
