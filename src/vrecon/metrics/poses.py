"""Evaluate stored camera / object animation at a given frame, the way the builder plays it back.

The builder keys location, quaternion and lens/shift with LINEAR interpolation (component-wise),
and Blender normalises the quaternion when evaluating. These helpers reproduce that so metrics
measure exactly what the .blend shows.
"""

from __future__ import annotations

import numpy as np

from vrecon.schema import Camera, Intrinsics, Project, SceneObject
from vrecon.scene.convert import continuous_quats, quat_to_matrix, scene_time_to_shot_frame


def _lerp_keys(times: list[float], values: list[np.ndarray], t: float) -> np.ndarray:
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    i = int(np.searchsorted(times, t, side="right")) - 1
    a = (t - times[i]) / (times[i + 1] - times[i])
    return values[i] * (1 - a) + values[i + 1] * a


def _rot(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.asarray(quat_to_matrix(tuple(q / np.linalg.norm(q))))


def object_pose(project: Project, obj: SceneObject, shot_id: str, shot_frame: float):
    """(R 3x3, t 3, scale 3) of an object at a shot frame."""
    if not obj.keys:
        p = obj.rest
        return _rot(p.pose.rotation), np.array(p.pose.translation, float), np.array(p.scale, float)
    mapping = next((m for m in project.reconstruction.shot_mapping if m.shot_id == shot_id), None)
    start = mapping.scene_time_start if mapping else 0.0
    frames = [scene_time_to_shot_frame(k.time, start, project.source.fps) for k in obj.keys]
    quats = continuous_quats([k.pose.rotation for k in obj.keys])
    q = _lerp_keys(frames, [np.array(x, float) for x in quats], shot_frame)
    t = _lerp_keys(frames, [np.array(k.pose.translation, float) for k in obj.keys], shot_frame)
    s = _lerp_keys(frames, [np.array(k.scale, float) for k in obj.keys], shot_frame)
    return _rot(q), t, s


def camera_pose(cam: Camera, shot_frame: float):
    """(R cam-to-world with OpenCV axes, center 3, K 3x3) at a shot frame."""
    frames = [k.shot_frame for k in cam.keys]
    quats = continuous_quats([k.pose.rotation for k in cam.keys])
    q = _lerp_keys(frames, [np.array(x, float) for x in quats], shot_frame)
    c = _lerp_keys(frames, [np.array(k.pose.translation, float) for k in cam.keys], shot_frame)
    intr = [k.intrinsics or cam.intrinsics for k in cam.keys]
    fx = float(_lerp_keys(frames, [np.array([i.fx]) for i in intr], shot_frame)[0])
    cx = float(_lerp_keys(frames, [np.array([i.cx]) for i in intr], shot_frame)[0])
    cy = float(_lerp_keys(frames, [np.array([i.cy]) for i in intr], shot_frame)[0])
    k = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])
    return _rot(q), c, k


def project_points(points_world: np.ndarray, r_c2w: np.ndarray, center: np.ndarray, k: np.ndarray):
    """World points [N,3] -> pixels [N,2] and depth [N] (OpenCV camera)."""
    pc = (points_world - center) @ r_c2w  # = R^T (p - c) for each row
    z = pc[:, 2]
    uv = (pc[:, :2] / z[:, None]) * np.array([k[0, 0], k[1, 1]]) + np.array([k[0, 2], k[1, 2]])
    return uv, z


def backproject(us: np.ndarray, vs: np.ndarray, depth: np.ndarray, r_c2w, center, k) -> np.ndarray:
    """Pixels with view-Z depth -> world points [N,3]."""
    x = (us - k[0, 2]) / k[0, 0] * depth
    y = (vs - k[1, 2]) / k[1, 1] * depth
    pc = np.stack([x, y, depth], axis=1)
    return pc @ r_c2w.T + center


def intrinsics_matrix(intr: Intrinsics) -> np.ndarray:
    return np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1.0]])
