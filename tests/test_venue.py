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
