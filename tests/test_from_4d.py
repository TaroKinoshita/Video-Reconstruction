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
