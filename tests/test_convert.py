import math

import pytest

from vrecon.schema import Intrinsics, Rational
from vrecon.scene.convert import (
    continuous_quats,
    intrinsics_to_blender,
    opencv_camera_rotation_to_blender,
    quat_mul,
    quat_to_matrix,
    scene_time_to_shot_frame,
    shot_frame_to_blender,
)


def test_quat_mul_identity():
    q = (0.5, 0.5, 0.5, 0.5)
    assert quat_mul((1, 0, 0, 0), q) == q
    assert quat_mul(q, (1, 0, 0, 0)) == q


def test_opencv_to_blender_identity_camera():
    # OpenCV camera looking along world +Z with y down == Blender camera rotated 180 deg about x.
    q = opencv_camera_rotation_to_blender((1, 0, 0, 0))
    m = quat_to_matrix(q)
    forward_blender = [-m[0][2], -m[1][2], -m[2][2]]  # Blender camera looks along local -Z
    up_blender = [m[0][1], m[1][1], m[2][1]]
    assert forward_blender == pytest.approx([0, 0, 1])
    assert up_blender == pytest.approx([0, -1, 0])  # OpenCV y is down, so "up" is world -y


def test_continuous_quats_flips_sign():
    a = (1.0, 0.0, 0.0, 0.0)
    b = (-0.99, -0.141, 0.0, 0.0)
    out = continuous_quats([a, b])
    assert out[1][0] > 0


def test_intrinsics_centered():
    lens = intrinsics_to_blender(Intrinsics(width=640, height=360, fx=500, fy=500, cx=319.5, cy=179.5))
    assert lens.lens_mm == pytest.approx(500 * 36 / 640)
    assert lens.shift_x == pytest.approx(0)
    assert lens.shift_y == pytest.approx(0)


def test_intrinsics_rejects_unsupported():
    with pytest.raises(NotImplementedError):
        intrinsics_to_blender(Intrinsics(width=640, height=360, fx=500, fy=550, cx=320, cy=180))
    with pytest.raises(NotImplementedError):
        intrinsics_to_blender(
            Intrinsics(width=640, height=360, fx=500, fy=500, cx=320, cy=180, distortion=[0.1, 0, 0, 0])
        )


def test_time_mapping():
    fps = Rational(num=24000, den=1001)
    assert scene_time_to_shot_frame(1001 / 24000 * 10, 0.0, fps) == pytest.approx(10)
    assert scene_time_to_shot_frame(2.0, 1.0, Rational(num=24, den=1)) == pytest.approx(24)
    assert shot_frame_to_blender(0) == 1
    assert math.isclose(shot_frame_to_blender(2.5), 3.5)
