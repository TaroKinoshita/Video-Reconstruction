"""Turn geometry passes into control videos for structure-conditioned V2V (no bpy).

Target: MiniMax-H3 + Fun ControlNet Union 2.0 (PLAN.md 9.1). Its constraints, from VideoX-Fun
examples/minimax_h3_fun/predict_v2v_control.py (commit e8c314f):
  - fixed 24 fps
  - width and height must be multiples of 32
  - frame count is snapped down to 17n + 5 (max 243, under 15 s)
Conventions follow VideoX-Fun's own annotators (comfyui/annotator/nodes.py):
  - depth: near = white, far = black, normalised between the 2nd and 85th percentile
  - canny: cv2.Canny(gray, 100, 200), white edges on black
"""

from .core import (
    FPS,
    ControlSpec,
    canny_control,
    check_h3_constraints,
    depth_control,
    encode_mp4,
    snap_h3_frames,
)

__all__ = [
    "FPS", "ControlSpec", "canny_control", "check_h3_constraints", "depth_control", "encode_mp4",
    "snap_h3_frames",
]
