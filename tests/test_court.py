import numpy as np

from vrecon.analysis.court import (COURT_LINES, camera_fixed_centre, camera_from_points, detect_lines, match_court,
                                   project)
from vrecon.scene.convert import look_at_opencv
from vrecon.metrics.poses import _rot


def _camera(w=800, h=450, f=1500.0, centre=(2.0, -19.0, 1.9), look=(-4.0, -2.0, 0.0)):
    # turned off the court's axis like the tennis clip (14 deg): looking straight down the court makes the
    # across lines parallel in the picture and the focal length unobservable from the court alone
    Rc2w = np.asarray(_rot(look_at_opencv(centre, look)))
    R = Rc2w.T
    t = -R @ np.array(centre)
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    return K, R, t


def _draw(K, R, t, w=800, h=450):
    import cv2

    img = np.zeros((h, w, 3), np.uint8) + 40
    for p0, p1 in COURT_LINES.values():
        X = np.c_[np.linspace(p0[0], p1[0], 800), np.linspace(p0[1], p1[1], 800), np.zeros(800)]
        X = X[(X @ R.T + t)[:, 2] > 0.1]
        uv = project(K, R, t, X)
        for a, b in zip(uv[:-1], uv[1:]):
            cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (255, 255, 255), 2)
    return img


def test_court_match_recovers_camera():
    from vrecon.analysis.court import white_lines

    K, R, t = _camera()
    img = _draw(K, R, t)
    white = white_lines(img, top=0)
    best, cands = match_court(detect_lines(white), white, 800, 450)
    top = [c for c in cands if c["score"] >= best["score"] - 1e-6]
    best = min(top, key=lambda c: c["centre_from_h"][1])
    K2, R2, t2, f2, err = camera_from_points(best["img"], best["court"], 800, 450)
    assert abs(f2 - 1500) / 1500 < 0.03 and err < 1.5
    assert np.linalg.norm(-R2.T @ t2 - np.array([2.0, -19.0, 1.9])) < 0.6


def test_fixed_centre_camera():
    K, R, t = _camera(f=2000.0)
    X = np.array([[-5.485, -11.885, 0], [5.485, -11.885, 0], [-4.115, -6.4, 0], [4.115, -6.4, 0], [0, -6.4, 0]], float)
    uv = project(K, R, t, X)
    K2, R2, t2, f2, err = camera_fixed_centre(uv, X, -R.T @ t, 800, 450)
    assert abs(f2 - 2000) < 5 and err < 0.1 and np.allclose(R2, R, atol=1e-3)
