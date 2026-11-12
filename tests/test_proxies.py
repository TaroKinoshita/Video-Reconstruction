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


def test_ground_box_from_mask():
    from vrecon.analysis.proxies import ground_box

    # camera 1.5 up at y = -10 looking along +y (OpenCV: x right, y down, z forward)
    r = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
    c, k = np.array([0.0, -10.0, 1.5]), np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    # a person 0.5 wide, 1.8 tall standing at (1, 0): its front face projected
    def px(x, y, z):
        q = r.T @ (np.array([x, y, z]) - c)
        return 320 + 500 * q[0] / q[2], 240 + 500 * q[1] / q[2]
    (u0, v1), (u1, v0) = px(0.75, 0, 0), px(1.25, 0, 1.8)
    mask = np.zeros((480, 640), bool)
    mask[int(v0):int(v1), int(u0):int(u1)] = True
    centre, yaw, size = ground_box(mask, r, c, k, depth_ratio=0.0)
    assert abs(centre[0] - 1) < 0.05 and abs(centre[1]) < 0.1
    assert abs(size[0] - 0.5) < 0.05 and abs(size[2] - 1.8) < 0.08
