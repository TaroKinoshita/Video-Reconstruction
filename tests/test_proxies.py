import numpy as np

from vrecon.analysis.proxies import board_proxy, near_band, plane_proxy


def test_plane_proxy_recovers_tilted_stands():
    rng = np.random.default_rng(0)
    # stands: 10 wide, rising 3 over 4 deep (37 deg back), seen from a camera at y = -10
    s, h = rng.uniform(-5, 5, 20000), rng.uniform(0, 1, 20000)
    p = np.c_[s, 5 + 4 * h, 3 * h] + rng.normal(0, 0.02, (20000, 3))
    c, q, size, r = plane_proxy(p, eye=np.array([0.0, -10.0, 1.5]))
    assert abs(r["tilt_deg"] - np.degrees(np.arctan2(3, 4))) < 2
    assert abs(size[0] - 10) < 0.6 and abs(size[1] - 5) < 0.4


def test_board_and_band():
    rng = np.random.default_rng(1)
    p = np.c_[rng.uniform(-3, 3, 5000), rng.normal(0, 0.01, 5000), rng.uniform(0, 1, 5000)]
    far = np.c_[rng.uniform(-3, 3, 500), np.full(500, 20.0), rng.uniform(0, 1, 500)]
    kept = near_band(np.r_[p, far], np.array([0.0, -10.0, 1.0]))
    assert (kept[:, 1] < 1).all()
    _, _, size, _ = board_proxy(kept)
    assert abs(size[0] - 6) < 0.4 and size[1] < 0.1 and abs(size[2] - 1) < 0.1
