"""Side-by-side sheet: source frames vs rendered frames (a check picture, local only).

    python scripts/sheet_compare.py SOURCE.mp4 RENDER.mp4|RENDER_DIR OUT.png FRAME [FRAME ...]
RENDER_DIR holds NNNN.png named by timeline frame (= source frame + 1).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrecon.control.core import read_video_rgb  # noqa: E402

src = read_video_rgb(sys.argv[1])
r = Path(sys.argv[2])
vid = read_video_rgb(r) if r.suffix == ".mp4" else None
rows = []
for f in map(int, sys.argv[4:]):
    b = vid[f] if vid is not None else cv2.imdecode(np.fromfile(r / f"{f + 1:04d}.png", np.uint8), 1)[..., ::-1]
    t = np.hstack([cv2.resize(src[f], (480, 204)), cv2.resize(b, (480, 204))]).copy()
    cv2.putText(t, f"f{f}", (8, 28), 0, 0.9, (255, 255, 0), 2)
    rows.append(t)
if len(rows) % 2:
    rows.append(np.zeros_like(rows[0]))
sheet = np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows), 2)])
Path(sys.argv[3]).write_bytes(cv2.imencode(".png", sheet[..., ::-1])[1].tobytes())
