"""analysis/floor_contact.py: an object stands on the floor only when the floor mask lies under its mask."""

import numpy as np

from vrecon.analysis.floor_contact import check, foot_pixel, is_floor_piece, stands_on_floor, surface_depth
from vrecon.schema import SceneObject


def _frame(bottom_on_floor: bool):
    m = np.zeros((40, 40), bool)
    m[10:20, 15:20] = True  # object, lowest row 19
    floor = np.zeros((40, 40), bool)
    floor[30:, :] = True  # floor starts at row 30
    if bottom_on_floor:
        floor[20:, :] = True
    return m, floor


def test_standing_and_hidden_feet():
    m, f = _frame(True)
    assert foot_pixel(m) == (17.0, 19.0) and stands_on_floor(m, f) is True
    m, f = _frame(False)
    assert stands_on_floor(m, f) is False
    edge = np.zeros((40, 40), bool)
    edge[30:40, 5:10] = True  # feet cut by the image's bottom edge: no verdict
    assert stands_on_floor(edge, f) is None
    res = check({i: _frame(False)[0] for i in range(5)}, {i: _frame(False)[1] for i in range(5)})
    assert res["on_floor"] is False and res["frames"] == 5
    res = check({i: _frame(True)[0] for i in range(5)}, {i: _frame(True)[1] for i in range(5)})
    assert res["on_floor"] is True


def test_surface_depth_and_floor_pieces():
    m, _ = _frame(False)
    d = np.zeros((40, 40), np.float32)
    d[19, 17] = 12.5
    assert surface_depth({0: m, 1: m}, {0: d, 1: d}) == 12.5
    assert is_floor_piece(SceneObject(id="obj_court_lines", kind="set"))
    assert not is_floor_piece(SceneObject(id="obj_wall_far", kind="set"))
