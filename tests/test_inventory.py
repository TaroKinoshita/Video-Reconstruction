"""control/inventory.py: visible objects and the prompt sentence."""

import numpy as np

import vrecon.control.inventory as inv
from vrecon.control.inventory import inventory_sentence, visible_objects
from vrecon.schema import FileRef, Project, Rational, SceneObject, Shot, SourceVideo


def test_inventory_counts_people_and_things():
    p = Project(id="proj_i", source=SourceVideo(file=FileRef(id="file_v", path="v.mp4"), fps=Rational(num=24, den=1),
                                                frame_count=10, width=30, height=20),
                shots=[Shot(id="shot_001", start=0, end=10, transition_from_previous="first")],
                reconstruction={"objects": [SceneObject(id="obj_player", kind="character"),
                                            SceneObject(id="obj_person_box_02", kind="rigid"),
                                            SceneObject(id="obj_chair", kind="rigid"),
                                            SceneObject(id="obj_crowd_far", kind="set"),
                                            SceneObject(id="obj_tiny", kind="rigid")]})
    idx = np.zeros((10, 20, 30), np.int32)
    idx[:, 5:15, 12:18] = 1  # player, middle
    idx[:, 5:12, 0:4] = 2  # another person, left
    idx[:, 10:14, 25:29] = 3  # chair, right
    idx[:, 0:3, :] = 4  # crowd
    idx[0, 19, 29] = 5  # tiny: one pixel (1/600 of the frame) in 1 of 10 frames
    idx[0, 19, 28] = 0
    assert inv.MIN_FRAMES == 0.1  # the tiny object sits on that line: seen in 1 of 10 frames is not enough below
    idx[0, 19, 29] = 0
    idx[0:1, 19, 29] = 5
    vis = visible_objects(idx, {1: "obj_player", 2: "obj_person_box_02", 3: "obj_chair", 4: "obj_crowd_far", 5: "obj_tiny"}, p)
    assert {o["id"] for o in vis} == {"obj_player", "obj_person_box_02", "obj_chair", "obj_crowd_far"}
    s = inventory_sentence(vis, "obj_player", "the player")
    assert "Exactly 2 people" in s and "the player" in s and "on the left" in s
    assert "a chair on the right" in s and "Spectators only sit in the stands" in s
