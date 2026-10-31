"""Pure-math conversions from vrecon conventions to Blender (no bpy).

See docs/CONVENTIONS.md sections 2, 3 and 5.3.
"""

from __future__ import annotations

from dataclasses import dataclass

from vrecon.schema import Intrinsics, Rational

Quat = tuple[float, float, float, float]

# 180 degrees about the camera x axis: OpenCV (y down, z forward) -> Blender camera (y up, -z forward)
_Q_FLIP_X: Quat = (0.0, 1.0, 0.0, 0.0)

DEFAULT_SENSOR_WIDTH_MM = 36.0
FRAME_OFFSET = 1  # blender_frame = shot_frame + FRAME_OFFSET


def quat_mul(a: Quat, b: Quat) -> Quat:
    """Hamilton product a*b, (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_to_matrix(q: Quat) -> list[list[float]]:
    """3x3 rotation matrix (rows) from a unit quaternion (w, x, y, z)."""
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def opencv_camera_rotation_to_blender(q_cv: Quat) -> Quat:
    """Camera-to-world rotation with OpenCV camera axes -> Blender camera object rotation."""
    return quat_mul(q_cv, _Q_FLIP_X)


def continuous_quats(quats: list[Quat]) -> list[Quat]:
    """Flip signs so neighbours have a non-negative dot product (q and -q are the same rotation,
    but interpolation between keys would take the long way round)."""
    out: list[Quat] = []
    for q in quats:
        if out and sum(a * b for a, b in zip(out[-1], q)) < 0:
            q = (-q[0], -q[1], -q[2], -q[3])
        out.append(q)
    return out


@dataclass(frozen=True)
class BlenderLens:
    lens_mm: float
    sensor_width_mm: float
    shift_x: float
    shift_y: float


SQUARE_PIXEL_TOLERANCE = 1e-3  # relative |fx - fy| / max(fx, fy)


def intrinsics_to_blender(intr: Intrinsics, sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM) -> BlenderLens:
    """Pixel intrinsics (integer pixel centers) -> Blender lens/shift with sensor_fit AUTO.

    Only square pixels (fx ~= fy) and no distortion are supported for now; anything else raises
    instead of silently producing a wrong camera.
    """
    if intr.distortion:
        raise NotImplementedError("lens distortion is not supported by the builder yet")
    if abs(intr.fx - intr.fy) / max(intr.fx, intr.fy) > SQUARE_PIXEL_TOLERANCE:
        raise NotImplementedError(f"non-square pixels (fx={intr.fx}, fy={intr.fy}) are not supported yet")
    s = max(intr.width, intr.height)
    f = (intr.fx + intr.fy) / 2
    return BlenderLens(
        lens_mm=f * sensor_width_mm / s,
        sensor_width_mm=sensor_width_mm,
        shift_x=((intr.width - 1) / 2 - intr.cx) / s,
        shift_y=(intr.cy - (intr.height - 1) / 2) / s,
    )


def scene_time_to_shot_frame(time: float, scene_time_start: float, fps: Rational) -> float:
    """Event time (seconds) -> shot frame (may be fractional)."""
    return (time - scene_time_start) * fps.num / fps.den


def shot_frame_to_blender(shot_frame: float) -> float:
    return shot_frame + FRAME_OFFSET
