"""Side-by-side comparison video of one cut: source (left) and a reconstruction render (right).

Same frames, same size, same speed (source fps). Each frame carries its index inside the cut and the index in
the full video, so a person can find a frame a model points at. Reconstruction frames are PNGs rendered from
the .blend over the cut's frame range, in order (frame i of the cut = i-th PNG).

  python scripts/v11_compare_cut.py SOURCE_CUT.mp4 RENDER_DIR OUT.mp4 --offset 83 --width 960
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vrecon.control.core import encode_mp4, load_frames, read_video_rgb  # noqa: E402


def _label(img: np.ndarray, text: str, org: tuple[int, int]) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (org[0] - 4, org[1] - th - 4), (org[0] + tw + 4, org[1] + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("render_dir")
    ap.add_argument("out")
    ap.add_argument("--offset", type=int, default=0, help="index of the cut's first frame in the full video")
    ap.add_argument("--width", type=int, default=960, help="width of each half")
    ap.add_argument("--fps", default="24000/1001")
    ap.add_argument("--right-label", default="RECON")
    a = ap.parse_args()

    src = read_video_rgb(a.source)
    pngs = sorted(Path(a.render_dir).glob("*.png"))
    if len(pngs) != len(src):
        raise SystemExit(f"frame count differs: source {len(src)}, render {len(pngs)}")
    ren = load_frames(pngs)
    h = int(round(src.shape[1] * a.width / src.shape[2] / 2) * 2)
    out = []
    for i, (s, r) in enumerate(zip(src, ren)):
        s = cv2.resize(s, (a.width, h), interpolation=cv2.INTER_AREA)
        r = cv2.resize(r[..., :3], (a.width, h), interpolation=cv2.INTER_AREA)
        row = np.concatenate([s, r], axis=1)
        _label(row, f"SOURCE  cut frame {i:03d} / video frame {a.offset + i:03d}", (8, 22))
        _label(row, a.right_label, (a.width + 8, 22))
        out.append(row)
    encode_mp4(np.stack(out), a.out, fps=a.fps, crf=16)
    print(a.out, len(out), "frames", f"{2 * a.width}x{h}")


if __name__ == "__main__":
    main()
