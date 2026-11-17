"""optimize 'cam_path': camera drift and path scale handed over to the moving objects."""

import numpy as np

from vrecon.optimize import Param, apply_params
from vrecon.schema import (Camera, CameraKey, FileRef, Intrinsics, Pose, Project, Rational, SceneObject, Shot,
                           SourceVideo, TransformKey)


def _proj():
    intr = Intrinsics(width=64, height=48, fx=60, fy=60, cx=31.5, cy=23.5)
    cam = Camera(id="cam_001", shot_id="shot_001", intrinsics=intr,
                 keys=[CameraKey(shot_frame=f, pose=Pose(translation=(0.1 * f, -5.0, 1.5))) for f in (0, 10, 20)])
    return Project(id="proj_c", source=SourceVideo(file=FileRef(id="file_v", path="v.mp4"), fps=Rational(num=10, den=1),
                                                   frame_count=21, width=64, height=48),
                   shots=[Shot(id="shot_001", start=0, end=21, transition_from_previous="first")],
                   reconstruction={"cameras": [cam], "objects": [
                       SceneObject(id="obj_floor", kind="set", size=(10, 10, 0.1)),
                       SceneObject(id="obj_box", kind="rigid", size=(1, 1, 1),
                                   keys=[TransformKey(time=t, pose=Pose(translation=(0.0, 0.0, 0.5))) for t in (0.0, 1.0, 2.0)])]})


def test_drift_moves_camera_and_objects_together():
    p = _proj()
    q, ch = apply_params(p, [Param("cam_path", "cam_001", 0, 0.1)], np.array([2.0]))
    cam = [k.pose.translation[0] for k in q.reconstruction.cameras[0].keys]
    box = [k.pose.translation[0] for k in q.reconstruction.objects[1].keys]
    assert np.allclose(np.array(cam) - [0.0, 1.0, 2.0], [-1.0, 0.0, 1.0])  # +2 over the shot, zero at mid-shot
    assert np.allclose(box, [-1.0, 0.0, 1.0])  # the box gets the same shift at its key times
    assert q.reconstruction.objects[0].rest == p.reconstruction.objects[0].rest  # set pieces stay
    assert {c.target_id for c in ch} == {"cam_001", "obj_box"}


def test_scale_about_the_mean():
    q, _ = apply_params(_proj(), [Param("cam_path", "cam_001", 3, 0.5)], np.array([1.0]))
    xs = [k.pose.translation[0] for k in q.reconstruction.cameras[0].keys]
    assert np.allclose(xs, [-1.0, 1.0, 3.0])  # path 0..2 around its mean 1, doubled
