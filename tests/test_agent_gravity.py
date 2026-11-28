import numpy as np

from vrecon.agent import gravity


def test_recovers_tilted_up_with_outliers():
    rng = np.random.default_rng(1)
    t = np.radians(7)
    up = np.array([np.sin(t), 0, np.cos(t)])

    def noisy(v, n, s):
        x = v + rng.normal(0, s, (n, 3))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    ground = np.concatenate([noisy(up, 500, 0.05), noisy(np.array([0, 1.0, 0]), 100, 0.3)])  # 1/6 outliers
    side = np.cross(up, [0, 1.0, 0])
    walls = np.concatenate([noisy(side / np.linalg.norm(side), 200, 0.03), noisy(np.array([0, 1.0, 0]), 200, 0.03)])
    cam = noisy(np.array([0, 0.3, 1.0]) / np.linalg.norm([0, 0.3, 1.0]), 50, 0.01)  # tilted camera: weak cue
    body = noisy(up, 40, 0.1)
    r = gravity.estimate(ground, np.zeros((0, 3)), walls, cam, body)
    assert np.degrees(np.arccos(r["up"] @ up)) < 1.5
