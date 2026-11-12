import json

import numpy as np
import pytest

from vrecon.analysis.from_gvhmr import character_from_gvhmr
from vrecon.schema import (Camera, CameraKey, FileRef, Intrinsics, Pose, Project, Rational, Shot, ShotMapping,
                           SourceVideo)

NAMES = ["pelvis", "spine", "head"]
PARENTS = [-1, 0, 1]


def _quat_z(a):
    return [np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)]


def _skeleton(path, n=5, fps=25.0):
    rest = np.array([[0, 0, 0], [0, 0, 0.3], [0, 0, 0.6]], float)
    local = np.zeros((n, 3, 4))
    local[..., 0] = 1.0
    for t in range(n):
        local[t, 1] = _quat_z(0.2 * t)  # spine turns about z
    root = np.array([[0.1 * t, 0, 1.0] for t in range(n)])
    jw = np.stack([root + rest[j] for j in range(3)], 1)
    jc = jw - np.array([0, -3.0, 0])  # camera frame = world shifted (camera at y=-3, identity rotation)
    np.savez(path, fps=fps, frames=n, width=64, height=48, joint_names=np.array(NAMES), parents=np.array(PARENTS),
             rest_joints=rest, local_rot=local, root_pos=root, joints_world=jw, joints_cam=jc,
             root_rot_cam=local[:, 0], K=np.tile(np.eye(3), (n, 1, 1)), floor_z_before=0.0,
             meta=json.dumps({"commit": "abc1234", "assumptions": ["test"]}))
    return root


def test_keys_follow_video_time(tmp_path):
    root = _skeleton(tmp_path / "s.npz", n=5, fps=25.0)
    obj = character_from_gvhmr(tmp_path / "s.npz", "s.npz", start_time=2.0)
    assert obj.kind == "character" and len(obj.keys) == 5
    assert obj.keys[3].time == pytest.approx(2.0 + 3 / 25.0)
    assert obj.keys[4].pose.translation == pytest.approx(tuple(root[4]))


def test_camera_placement(tmp_path):
    _skeleton(tmp_path / "s.npz")
    cam = Camera(id="cam_001", shot_id="shot_001",
                 intrinsics=Intrinsics(width=64, height=48, fx=50, fy=50, cx=32, cy=24),
                 keys=[CameraKey(shot_frame=0, pose=Pose(translation=(0.0, -3.0, 0.0), rotation=(1, 0, 0, 0)))])
    obj = character_from_gvhmr(tmp_path / "s.npz", "s.npz", camera=cam)
    # camera at (0,-3,0) looking with identity axes: world = cam + (0,-3,0) -> back to the skeleton's world
    assert obj.keys[2].pose.translation == pytest.approx((0.2, 0.0, 1.0))


def test_builder_poses_the_rig(tmp_path):
    pytest.importorskip("bpy")
    from vrecon.scene.builder import build_scene
    import bpy

    _skeleton(tmp_path / "s.npz", n=5, fps=25.0)
    obj = character_from_gvhmr(tmp_path / "s.npz", "s.npz")
    cam = Camera(id="cam_001", shot_id="shot_001",
                 intrinsics=Intrinsics(width=64, height=48, fx=50, fy=50, cx=32, cy=24),
                 keys=[CameraKey(shot_frame=0, pose=Pose(translation=(0.0, -5.0, 1.0), rotation=(1, 0, 0, 0)))])
    project = Project(id="proj_t", source=SourceVideo(file=FileRef(id="file_v", path="v.mp4"),
                                                       fps=Rational(num=25, den=1), frame_count=5, width=64, height=48),
                      shots=[Shot(id="shot_001", start=0, end=5)],
                      reconstruction={"cameras": [cam], "objects": [obj],
                                      "shot_mapping": [ShotMapping(shot_id="shot_001", scene_time_start=0.0)]})
    build_scene(project, root=tmp_path)
    arm = next(o for o in bpy.data.objects if o.get("vrecon_rig_of"))
    body = next(o for o in bpy.data.objects if o.get("vrecon_id") == obj.id)
    assert body.type == "MESH" and arm.type == "ARMATURE"
    bpy.context.scene.frame_set(1 + 4)
    pelvis = np.array(arm.matrix_world @ arm.pose.bones["pelvis"].head)
    head = np.array(arm.matrix_world @ arm.pose.bones["head"].head)
    assert pelvis == pytest.approx([0.4, 0, 1.0], abs=1e-4)
    assert head == pytest.approx([0.4, 0, 1.6], abs=1e-4)  # spine turn about z leaves the head above the pelvis


def test_place_in_scene_recovers_scale_and_floor(tmp_path):
    from vrecon.analysis.place_person import place_in_scene
    from vrecon.scene.convert import matrix_to_quat

    names = ["pelvis", "left_ankle", "right_ankle", "head"]
    rest = np.array([[0, 0, 0], [-0.1, 0, -0.85], [0.1, 0, -0.85], [0, 0, 0.6]])
    n, s_true = 4, 0.1
    root = np.array([[0.2 * t, 0, 0.93] for t in range(n)])  # ankles at 0.08 m: standing
    jw = np.stack([root + rest[j] for j in range(4)], 1)
    rc = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # cam->world: looks along +y, image down = -z
    c = np.array([0.0, -4.0, 1.2])
    jc = (jw - c) @ rc  # world -> camera (rows)
    local = np.zeros((n, 4, 4))
    local[..., 0] = 1.0
    k = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    np.savez(tmp_path / "s.npz", fps=25.0, frames=n, width=640, height=480, joint_names=np.array(names),
             parents=np.array([-1, 0, 0, 0]), rest_joints=rest, local_rot=local, root_pos=root, joints_world=jw,
             joints_cam=jc, root_rot_cam=np.tile([1.0, 0, 0, 0], (n, 1)), K=np.tile(k, (n, 1, 1)),
             floor_z_before=0.0, meta=json.dumps({"commit": "x", "assumptions": []}))
    cam = Camera(id="cam_001", shot_id="shot_001", intrinsics=Intrinsics(width=640, height=480, fx=500, fy=500, cx=320, cy=240),
                 keys=[CameraKey(shot_frame=0, pose=Pose(translation=tuple(c * s_true), rotation=matrix_to_quat(rc.tolist())))])
    obj, rep = place_in_scene(tmp_path / "s.npz", "s.npz", cam)
    assert rep["scale_units_per_metre"] == pytest.approx(s_true, rel=1e-6)
    assert obj.keys[3].pose.translation == pytest.approx(tuple(root[3] * s_true), abs=1e-9)
    assert obj.rest.scale == pytest.approx((s_true,) * 3)
