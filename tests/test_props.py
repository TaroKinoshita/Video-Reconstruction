import json

import numpy as np
import pytest

from vrecon.analysis import props as P
from vrecon.analysis.prop_fit import tracklets
from vrecon.analysis.prop_table import load_table, lookup


def test_table_lookup_and_override(tmp_path):
    assert lookup("Tennis Ball")["size"] == (0.067,)
    with pytest.raises(KeyError):
        lookup("frisbee")
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({"frisbee": {"shape": "stick", "size": [0.27, 0.27, 0.03], "contacts": ["right_wrist"],
                                             "reach": 0.05, "colour": [1, 1, 1]}}), encoding="utf-8")
    e = lookup("frisbee", load_table(extra))
    assert e["source"] == str(extra) and e["shape"] == "stick"


def test_tracklets_keep_two_balls_apart():
    m = np.zeros((10, 60, 120), bool)
    for f in range(10):
        m[f, 10:14, 10 + 2 * f:14 + 2 * f] = True   # a ball moving right
        m[f, 40:44, 100:104] = True                 # another lying still
    tr = tracklets(m, link=20, min_track=5)
    assert len(tr) == 2 and all(len(t) == 10 for t in tr)


def test_frame_times_repeats_share_a_time():
    rng = np.random.default_rng(0)
    v = np.zeros((40, 8, 8, 3), np.uint8)
    for i in range(40):
        v[i] = rng.integers(0, 255, (8, 8, 3))
    for i in range(5, 40, 5):
        v[i] = v[i - 1]  # every 5th frame a copy (24 -> 30 fps)
    t = P.frame_times(v, fps=30.0)
    assert t[5] == t[4] and t[6] > t[5] and np.all(np.diff(t) >= 0)
    still = np.zeros((40, 8, 8, 3), np.uint8)  # a still shot: no cadence, no repeats
    still[20:, 0, 0] = 9
    assert np.all(np.diff(P.frame_times(still, fps=30.0)) > 0)


def test_ballistic_fit_recovers_a_toss():
    # one still camera at the origin looking along +y; a ball tossed 6 m in front
    R = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # camera x right, y down, z forward -> world x, -z, y
    K = np.array([[800.0, 0, 320], [0, 800, 240], [0, 0, 1]])
    c = np.zeros(3)
    p0, v = np.array([0.2, 6.0, 0.0]), np.array([0.3, 0.0, 5.0])
    t_n, fps = 20, 30.0
    cen = np.zeros((t_n, 2))
    for f in range(t_n):
        t = f / fps
        p = p0 + v * t - np.array([0, 0, 9.81 * t * t / 2])
        pc = (p - c) @ R
        cen[f] = pc[:2] / pc[2] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    pos, how, info = P.ball_fit(cen, [(R, c, K)] * t_n, fps, splits=[], floor_z=None, models=("still", "ballistic"))
    assert info[0]["model"] == "ballistic"
    assert np.allclose(pos[0], p0, atol=0.05)
