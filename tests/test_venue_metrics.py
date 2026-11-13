import numpy as np

from vrecon.metrics.venue import category, observed_kept, score_view


def test_categories_from_both_naming_schemes():
    assert category("obj_stand_north") == category("obj_stands_encl090") == "stands"
    assert category("obj_barrier_west") == category("obj_wall_03") == "wall"
    assert category("obj_person_box_02") == category("obj_player_a") == "person"


def test_score_view_counts_by_category_not_id():
    gi = np.zeros((4, 4), int)
    gi[:2] = 1  # stands (ground truth id 1)
    gi[2:, :2] = 2  # court
    gm = {1: "obj_stand_north", 2: "obj_court"}
    ri = np.zeros((4, 4), int)
    ri[:2] = 7  # stands under another id
    ri[3, 3] = 8  # something where the ground truth has nothing
    rm = {7: "obj_stands_encl180", 8: "obj_wall_01"}
    d = np.ones((4, 4))
    s = score_view(gi, gm, d, ri, rm, d * 1.1)
    assert s["category_iou"]["stands"] == 1.0 and s["category_iou"]["court"] == 0.0
    assert abs(s["covered"] - 8 / 12) < 1e-9 and abs(s["extra"] - 1 / 9) < 1e-9
    assert abs(s["depth_rel"] - 0.1) < 1e-9
    assert abs(observed_kept(gi, gm, ri, rm, {"stands", "court"}) - 8 / 12) < 1e-9
