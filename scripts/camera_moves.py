"""Camera moves made from the source camera, shared by completion (scripts/complete_scene.py: what the new
camera will see gets built) and control export (scripts/export_scene_controls.py: what V2V gets).

A move name:
    source        the source camera
    orbitN        turn N degrees about the vertical axis through the pivot (the main character) over the clip
    dollyN        move N percent of the way towards the pivot (horizontally), same view direction
    craneN        rise N metres and keep looking at the pivot (at chest height)
    orbitN+craneM moves can be combined with "+", applied in order
The move runs linearly from the source pose at the first frame to the full move at the last.
"""

from __future__ import annotations

import numpy as np


def _rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def apply_move(r: np.ndarray, c: np.ndarray, move: str, t: float, pivot: np.ndarray, units_per_m: float):
    """(R cam-to-world OpenCV, centre) after `move` at progress t in [0, 1]."""
    from vrecon.metrics.poses import _rot
    from vrecon.scene.convert import look_at_opencv, matrix_to_quat

    for part in filter(None, move.split("+")):
        if part == "source":
            continue
        if part.startswith("orbit"):
            rz = _rz(float(part[5:]) * t)
            c = rz @ (c - pivot) + pivot
            r = rz @ r
        elif part.startswith("dolly"):
            target = np.array([pivot[0], pivot[1], c[2]])
            c = c + (target - c) * float(part[5:]) / 100 * t
        elif part.startswith("crane"):
            c = c + np.array([0, 0, float(part[5:]) * units_per_m * t])
            look = pivot + np.array([0, 0, 1.2 * units_per_m])
            # turn from the source's view towards looking at the pivot as it rises (slerp), so t = 0 is unchanged
            q0 = np.array(matrix_to_quat(r.tolist()))
            q1 = np.array(look_at_opencv(tuple(c), tuple(look)))
            if q0 @ q1 < 0:
                q1 = -q1
            om = np.arccos(np.clip(q0 @ q1, -1, 1))
            q = q0 if om < 1e-6 else (np.sin((1 - t) * om) * q0 + np.sin(t * om) * q1) / np.sin(om)
            r = np.asarray(_rot(tuple(q / np.linalg.norm(q))))
        else:
            raise ValueError(f"unknown move {part!r}")
    return r, c


def move_camera(src, move: str, pivot: np.ndarray, units_per_m: float, n_frames: int, fps_out: float,
                fps_src: float, step: int = 6, intrinsics=None):
    """A Camera (keys every `step` output frames and the last) for an output clip of n_frames at fps_out, following
    the source camera's own motion (sampled at the same time) plus the move."""
    from vrecon.metrics.poses import camera_pose
    from vrecon.scene.convert import matrix_to_quat
    from vrecon.schema import CameraKey, Pose

    keys = []
    for fr in sorted(set(range(0, n_frames, step)) | {n_frames - 1}):
        r, c, _ = camera_pose(src, fr / fps_out * fps_src)
        r, c = apply_move(np.asarray(r, float), np.asarray(c, float), move, fr / max(n_frames - 1, 1), pivot, units_per_m)
        keys.append(CameraKey(shot_frame=fr, pose=Pose(translation=tuple(map(float, c)), rotation=matrix_to_quat(r.tolist()))))
    upd = {"keys": keys}
    if intrinsics is not None:
        upd["intrinsics"] = intrinsics
    return src.model_copy(update=upd)


def pivot_of(project) -> np.ndarray:
    ch = next((o for o in project.reconstruction.objects if o.kind == "character"), None)
    if ch is None:
        raise ValueError("no character to move about")
    p = np.array((ch.keys[0] if ch.keys else ch.rest).pose.translation, float)
    p[2] = 0.0
    return p
