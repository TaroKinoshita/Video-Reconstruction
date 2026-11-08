import numpy as np

from vrecon.analysis.from_4d import fit_plane, model_to_source, upright_box


def test_model_to_source_matches_4rc_preprocessing():
    # 640x360 -> long side 512 (512x288) -> centre crop to 504x280
    s, ox, oy = model_to_source(280, 504, 640, 360)
    assert s == 0.8 and (ox, oy) == (4, 4)


def test_fit_plane_ignores_outliers():
    rng = np.random.default_rng(1)
    floor = np.c_[rng.uniform(-1, 1, (2000, 2)), 0.5 + rng.normal(0, 1e-4, 2000)]
    junk = rng.uniform(-1, 1, (300, 3))
    n, d, inl = fit_plane(np.r_[floor, junk], tol=0.005)
    assert abs(abs(n[2]) - 1) < 1e-3 and abs(abs(d) - 0.5) < 1e-3
    assert inl[:2000].mean() > 0.99


def test_upright_box_recovers_rotated_box():
    rng = np.random.default_rng(2)
    local = rng.uniform(-0.5, 0.5, (5000, 3)) * [0.4, 0.1, 0.3] + [0, 0, 0.15]
    yaw = 0.6
    c, s = np.cos(yaw), np.sin(yaw)
    p = local @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + [1.0, 2.0, 0]
    centre, got_yaw, size = upright_box(p, lo=0.0, hi=1.0)
    assert np.allclose(centre, [1.0, 2.0, 0.15], atol=0.01)
    assert abs(np.sin(got_yaw - yaw) ) < 0.02  # box axes are only defined up to 180 degrees
    assert np.allclose(size, [0.4, 0.1, 0.3], atol=0.01)


def test_snap_still_ends_removes_wobble_before_and_after_the_move():
    from vrecon.analysis.from_4d import _snap_still_ends

    rng = np.random.default_rng(0)
    frames = list(range(0, 60, 2))
    x = np.clip((np.array(frames) - 20) / 20, 0, 1) * 2.0  # still until 20, moves 2 m until 40, then still
    xy = np.c_[x, np.zeros_like(x)] + rng.normal(0, 0.01, (len(frames), 2))
    f, p, y = _snap_still_ends(frames, xy, np.zeros(len(frames)))
    assert 20 <= f[0] <= 22 and 38 <= f[-1] <= 40  # within one sample of the true onset and stop
    assert np.allclose(p[0], [0, 0], atol=0.02) and np.allclose(p[-1], [2, 0], atol=0.02)


def test_snap_still_ends_leaves_a_return_trip():
    from vrecon.analysis.from_4d import _snap_still_ends

    frames = list(range(30))
    xy = np.c_[np.sin(np.linspace(0, np.pi, 30)), np.zeros(30)]
    f, p, _ = _snap_still_ends(frames, xy, np.zeros(30))
    assert f == frames and np.array_equal(p, xy)


def test_deepen_extends_away_from_the_camera_keeping_the_front_face():
    from vrecon.analysis.from_4d import _deepen

    xy, size = _deepen(np.array([0.0, 1.0]), 0.0, (1.0, 0.1, 0.5))  # wide box facing the camera (+Y view)
    assert np.isclose(size[1], 0.5) and np.isclose(xy[1] - size[1] / 2, 1.0 - 0.05)  # front face stays
    xy2, size2 = _deepen(np.array([0.0, 1.0]), 0.0, (1.0, 0.8, 0.5))  # already deep: unchanged
    assert size2 == (1.0, 0.8, 0.5) and np.allclose(xy2, [0.0, 1.0])
