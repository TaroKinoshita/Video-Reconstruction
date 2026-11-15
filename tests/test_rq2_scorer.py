"""RQ2 scorer (compare/generated.py) and the status pass (control/status.py) on small numpy scenes."""

from __future__ import annotations

import numpy as np

from vrecon.compare.generated import masks_from_index, onset_2d, pair_masks, score_generated
from vrecon.control.status import HYPOTHESIS, OBSERVED, STANDARD, apply_mode, shares, status_pass
from vrecon.schema import (
    Camera, CameraKey, FileRef, Intrinsics, Pose, Project, Rational, SceneObject, Shot, SourceVideo, StatusSpan,
    TransformKey,
)


def _project(n=10):
    intr = Intrinsics(width=32, height=24, fx=30, fy=30, cx=15.5, cy=11.5)
    return Project(
        id="proj_t", source=SourceVideo(file=FileRef(id="file_v", path="v.mp4"), fps=Rational(num=10, den=1),
                                        frame_count=n, width=32, height=24),
        shots=[Shot(id="shot_001", start=0, end=n, transition_from_previous="first")],
        reconstruction={"cameras": [Camera(id="cam_001", shot_id="shot_001", intrinsics=intr,
                                           keys=[CameraKey(shot_frame=0, pose=Pose())])],
                        "objects": [
                            SceneObject(id="obj_ball", kind="rigid", status="observed",
                                        spans=[StatusSpan(start=0.5, end=0.9, status="hypothesis")]),
                            SceneObject(id="obj_court", kind="set", status="standard"),
                            SceneObject(id="obj_stand", kind="set", status="hypothesis",
                                        rest=TransformKey(time=0, pose=Pose()))]})


def _index(n=10, x0=2, dx=2):
    idx = np.zeros((n, 24, 32), np.int32)
    idx[:, 16:, :] = 2  # court
    idx[:, :4, :] = 3  # stand
    for f in range(n):
        x = x0 + (dx * max(f - 3, 0))
        idx[f, 6:12, x:x + 6] = 1  # ball, moves from frame 4
    return idx


ID_MAP = {1: "obj_ball", 2: "obj_court", 3: "obj_stand"}


def test_status_pass_follows_spans():
    st = status_pass(_index(), ID_MAP, _project())
    assert st[0, 8, 4] == OBSERVED and st[6, 8, 10] == HYPOTHESIS  # ball: hypothesis from 0.5 s (frame 5)
    assert st[0, 20, 0] == STANDARD and st[0, 0, 0] == HYPOTHESIS and st[0, 13, 30] == 0
    s = shares(st)["clip"]
    assert abs(sum(s.values()) - 1) < 1e-9


def test_modes():
    idx = _index()
    st = status_pass(idx, ID_MAP, _project())
    frames = np.random.default_rng(0).integers(0, 255, (10, 24, 32, 3), dtype=np.uint8)
    assert apply_mode("depth", frames, st, "full") is frames
    drop = apply_mode("depth", frames, st, "drop")
    assert (drop[st == HYPOTHESIS] == 0).all() and (drop[st != HYPOTHESIS] == frames[st != HYPOTHESIS]).all()
    soft = apply_mode("canny", frames, st, "soft")
    assert (soft[st == HYPOTHESIS] == 0).all()
    flat = apply_mode("look", frames, st, "soft")
    hyp = flat[0][st[0] == HYPOTHESIS]
    assert (hyp == hyp[0]).all()


def test_perfect_and_shifted_generation():
    idx = _index()
    st = status_pass(idx, ID_MAP, _project())
    ctrl = {"index": idx, "id_map": ID_MAP, "status": st, "canny": None}
    video = np.zeros((10, 24, 32, 3), np.uint8)
    rep = score_generated(video, ctrl, masks_from_index(idx, ID_MAP), movers=["obj_ball"])
    assert all(o["iou"] == 1.0 for o in rep["objects"].values())
    assert rep["objects"]["obj_ball"]["onset"]["error_frames"] == 0
    # the ball starts two frames late in the "generation"
    late = _index()
    for f in range(10):
        late[f][late[f] == 1] = 0
        x = 2 + 2 * max(f - 5, 0)
        late[f, 6:12, x:x + 6] = 1
    rep = score_generated(video, ctrl, {"a": late == 1, "b": late == 2, "c": late == 3}, movers=["obj_ball"])
    assert rep["objects"]["obj_ball"]["label"] == "a"
    assert rep["objects"]["obj_ball"]["onset"]["error_frames"] == 2
    assert rep["objects"]["obj_ball"]["iou"] < 1.0 and rep["objects"]["obj_court"]["iou"] == 1.0
    # a lost object scores 0
    rep = score_generated(video, ctrl, {"b": idx == 2}, movers=[])
    assert rep["objects"]["obj_ball"] == {**rep["objects"]["obj_ball"], "matched": False, "iou": 0.0}


def test_unchanged_region_and_helpers():
    idx = _index()
    ctrl = {"index": idx, "id_map": ID_MAP, "status": np.ones_like(idx, np.uint8), "canny": None}
    base = np.full((10, 24, 32, 3), 100, np.uint8)
    other = base.copy()
    other[:, 16:] = 110  # the court changed by 10 grey levels
    rep = score_generated(other, ctrl, base_video=base, unchanged=["obj_court"])
    assert abs(rep["unchanged"]["mad"] - 10) < 1e-6
    assert onset_2d(np.array([[0, 0]] * 5 + [[float(i), 0] for i in range(1, 20)])) == 4
    assert pair_masks({"x": idx == 2}, {"obj_court": idx == 2, "obj_ball": idx == 1}) == {"obj_court": "x"}
