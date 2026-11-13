import numpy as np

from vrecon.venue.fit import fit_layout
from vrecon.venue.generate import rows_along, wall_along
from vrecon.venue.layout import check


def test_layout_check_and_defaults():
    lay = {"parts": [{"id": "n", "type": "stand", "front": [[-5, 20], [5, 20]], "rows": 3, "status": "observed"},
                     {"id": "x", "type": "spaceship", "status": "observed"}]}
    probs = check(lay)
    assert any("unknown type" in p for p in probs)
    assert lay["parts"][0]["rise"] == 0.45 and lay["parts"][0]["row_depth"] == 0.85


def test_stand_rows_climb_away_from_the_court():
    v, f = rows_along([[-5, 20], [5, 20]], rows=4, row_depth=1.0, rise=0.5, base_height=1.0)
    assert len(v) == 4 * 8 and f.max() < len(v)
    back = v[v[:, 1] > 23.5]
    assert np.isclose(back[:, 2].max(), 1.0 + 4 * 0.5) and v[:, 1].min() >= 20 - 1e-9
    v2, _ = rows_along([[5, -20], [-5, -20]], rows=2, row_depth=1.0, rise=0.5, base_height=0.0)
    assert v2[:, 1].max() <= -20 + 1e-9  # the south stand climbs towards -Y
    vw, _ = wall_along([[-5, 18], [5, 18]], height=2.0, thickness=0.3)
    assert np.isclose(vw[:, 2].max(), 2.0) and vw[:, 1].min() >= 18 - 1e-9


def test_fit_moves_stand_onto_observation_and_clips():
    lay = {"parts": [{"id": "n", "type": "stand", "front": [[-5, 20], [5, 20]], "rows": 4, "row_depth": 1.0, "rise": 0.5,
                      "base_height": 0.0, "status": "observed", "fit": "obs"}]}
    check(lay)
    obs = {"things": {"obs": {"centroid": [0, 22.5, 1.0], "z_range": [0, 2.0]}}}
    out, log = fit_layout(lay, obs)
    # centroid height 1.0 -> row 1.5 -> 1.5 m behind the front -> front at 21.0
    assert np.isclose(out["parts"][0]["front"][0][1], 21.0)
    far = {"things": {"obs": {"centroid": [0, 30.0, 1.0], "z_range": [0, 2.0]}}}
    out, log = fit_layout(lay, far)
    assert np.isclose(out["parts"][0]["front"][0][1], 22.0) and log[-1]["clipped"]


def test_corners_join_and_aisles_are_stairs():
    from vrecon.venue.generate import join_corners, rows_along

    north = {"id": "n", "front": [[-20.0, 20.0], [20.0, 20.0]], "rows": 3, "row_depth": 1.0, "rise": 0.5, "base_height": 0.0,
             "status": "observed"}
    west = {"id": "w", "front": [[-14.0, -20.0], [-14.0, 15.0]], "rows": 3, "row_depth": 1.0, "rise": 0.5, "base_height": 0.0,
            "status": "hypothesis"}
    corners = join_corners([north, west])
    assert len(corners) == 1 and np.allclose(corners[0]["corner"], [-14.0, 20.0])
    assert north["front"][0] == [-14.0, 20.0] and west["front"][-1] == [-14.0, 20.0]  # trimmed and extended
    assert corners[0]["status"] == "hypothesis"
    v, f, cols = rows_along([[-10, 20], [10, 20]], 2, 1.0, 0.5, 0.0, aisles={"every": 10.0, "width": 1.4},
                            colour=(0.2, 0.3, 0.6))
    from vrecon.venue.generate import STAIR

    grey = np.all(np.isclose(cols, STAIR, atol=0.01), axis=1)
    assert grey.any()  # stair steps in the aisle


def test_crowd_sits_on_seat_rows_not_aisles():
    from vrecon.venue.generate import crowd_along

    v, f, c = crowd_along([[-20, 15], [20, 15]], 10, 0.85, 0.45, 1.0, {"every": 10.0, "width": 1.2}, density=1.0)
    assert len(v) and len(c) == len(v)
    assert v[:, 1].min() > 15 and v[:, 2].min() > 1.4  # behind the front, above the first row's top
    aisle_x = [-10.0, 0.0, 10.0]
    assert all(np.abs(v[:, 0] - x).min() > 0.3 for x in aisle_x)
