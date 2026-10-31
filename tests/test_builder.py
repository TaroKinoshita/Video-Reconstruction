"""Builder tests (need bpy). Each test rebuilds the Blender session from scratch."""

import math
import random

import pytest

bpy = pytest.importorskip("bpy")

import mathutils  # noqa: E402  (ships with bpy)
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402

from vrecon.schema import (  # noqa: E402
    Camera,
    CameraKey,
    FileRef,
    Intrinsics,
    Pose,
    Project,
    Rational,
    SceneObject,
    Shot,
    ShotMapping,
    SourceVideo,
    TransformKey,
)
from vrecon.scene.builder import ID_PROP, build_scene, render_frames, save_blend  # noqa: E402
from vrecon.scene.convert import quat_to_matrix  # noqa: E402


def random_quat(rng: random.Random):
    q = [rng.gauss(0, 1) for _ in range(4)]
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def make_project(cameras, objects=(), fps=Rational(num=24, den=1), frames=48, size=(640, 360), mapping=()):
    return Project(
        id="proj_test",
        source=SourceVideo(
            file=FileRef(id="file_src", path="in.mp4"), fps=fps, frame_count=frames, width=size[0], height=size[1]
        ),
        shots=[Shot(id="shot_001", start=0, end=frames, transition_from_previous="first")],
        reconstruction={"cameras": list(cameras), "objects": list(objects), "shot_mapping": list(mapping)},
    )


def opencv_project(intr: Intrinsics, pose: Pose, p_world):
    """Reference projection straight from the stored data (CONVENTIONS.md)."""
    r = quat_to_matrix(pose.rotation)
    d = [p_world[i] - pose.translation[i] for i in range(3)]
    pc = [sum(r[j][i] * d[j] for j in range(3)) for i in range(3)]  # R^T (p - t)
    return intr.fx * pc[0] / pc[2] + intr.cx, intr.fy * pc[1] / pc[2] + intr.cy, pc[2]


def blender_project(p_world, intr: Intrinsics):
    scene = bpy.context.scene
    n = world_to_camera_view(scene, scene.camera, mathutils.Vector(p_world))
    return n.x * intr.width - 0.5, (1 - n.y) * intr.height - 0.5


@pytest.mark.parametrize(
    "intr",
    [
        Intrinsics(width=1920, height=1080, fx=1500, fy=1500, cx=900.0, cy=600.0),
        Intrinsics(width=1080, height=1920, fx=1200, fy=1200, cx=500.0, cy=1000.0),
        Intrinsics(width=640, height=360, fx=500, fy=500, cx=319.5, cy=179.5),
    ],
)
def test_camera_reprojection(intr):
    rng = random.Random(0)
    pose = Pose(translation=(1.0, 2.0, 3.0), rotation=random_quat(rng))
    cam = Camera(id="cam_001", shot_id="shot_001", intrinsics=intr, keys=[CameraKey(shot_frame=0, pose=pose)])
    build_scene(make_project([cam]))
    bpy.context.scene.frame_set(1)
    r = quat_to_matrix(pose.rotation)
    worst = 0.0
    for _ in range(30):
        pc = [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(2, 6)]  # point in camera coords
        pw = [sum(r[i][j] * pc[j] for j in range(3)) + pose.translation[i] for i in range(3)]
        u, v, _ = opencv_project(intr, pose, pw)
        ub, vb = blender_project(pw, intr)
        worst = max(worst, abs(ub - u), abs(vb - v))
    assert worst < 1e-2, f"max reprojection error {worst} px"


def test_camera_animation_and_zoom():
    rng = random.Random(1)
    base = Intrinsics(width=640, height=360, fx=500, fy=500, cx=319.5, cy=179.5)
    zoom = base.model_copy(update={"fx": 800.0, "fy": 800.0})
    k0 = CameraKey(shot_frame=0, pose=Pose(translation=(0, -5, 1), rotation=random_quat(rng)))
    k1 = CameraKey(shot_frame=20, pose=Pose(translation=(2, -4, 1.5), rotation=random_quat(rng)), intrinsics=zoom)
    cam = Camera(id="cam_001", shot_id="shot_001", intrinsics=base, keys=[k0, k1])
    build_scene(make_project([cam]))
    scene = bpy.context.scene
    for key, intr in ((k0, base), (k1, zoom)):
        scene.frame_set(key.shot_frame + 1)
        r = quat_to_matrix(key.pose.rotation)
        pc = [0.3, -0.2, 4.0]
        pw = [sum(r[i][j] * pc[j] for j in range(3)) + key.pose.translation[i] for i in range(3)]
        u, v, _ = opencv_project(intr, key.pose, pw)
        ub, vb = blender_project(pw, intr)
        assert abs(ub - u) < 1e-2 and abs(vb - v) < 1e-2
    # linear interpolation in between
    scene.frame_set(11)
    assert scene.camera.data.lens == pytest.approx((500 + 800) / 2 * 36 / 640, rel=1e-4)
    assert tuple(scene.camera.location) == pytest.approx((1.0, -4.5, 1.25), abs=1e-5)


def test_object_keys_follow_scene_time():
    fps = Rational(num=24000, den=1001)
    cam = Camera(
        id="cam_001",
        shot_id="shot_001",
        intrinsics=Intrinsics(width=320, height=180, fx=250, fy=250, cx=159.5, cy=89.5),
        keys=[CameraKey(shot_frame=0, pose=Pose(translation=(0, 0, -5)))],
    )
    t = lambda frame: 1.0 + frame * 1001 / 24000  # noqa: E731  scene time of a shot frame
    box = SceneObject(
        id="obj_box",
        size=(1, 2, 0.5),
        keys=[
            TransformKey(time=t(0), pose=Pose(translation=(0, 0, 0))),
            TransformKey(time=t(10), pose=Pose(translation=(2, 0, 0))),
        ],
    )
    report = build_scene(
        make_project([cam], [box], fps=fps, mapping=[ShotMapping(shot_id="shot_001", scene_time_start=1.0)])
    )
    assert report.objects == ["obj_box"]
    scene = bpy.context.scene
    obj = next(o for o in bpy.data.objects if o.get(ID_PROP) == "obj_box")
    for shot_frame, x in ((0, 0.0), (5, 1.0), (10, 2.0), (30, 2.0)):
        scene.frame_set(shot_frame + 1)
        assert obj.location.x == pytest.approx(x, abs=1e-4), shot_frame
    assert tuple(obj.dimensions) == pytest.approx((1, 2, 0.5))


def test_render_save_reload(tmp_path):
    cam = Camera(
        id="cam_001",
        shot_id="shot_001",
        intrinsics=Intrinsics(width=320, height=180, fx=250, fy=250, cx=159.5, cy=89.5),
        keys=[CameraKey(shot_frame=0, pose=Pose(translation=(0, -6, 0), rotation=(0.7071068, -0.7071068, 0, 0)))],
    )
    box = SceneObject(id="obj_box", size=(1, 1, 1))
    project = make_project([cam], [box])
    build_scene(project)
    paths = render_frames(tmp_path / "frames", [0], engine="BLENDER_WORKBENCH", resolution_percentage=50)
    assert paths[0].is_file() and paths[0].stat().st_size > 0
    blend = save_blend(tmp_path / "scene.blend")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene = bpy.context.scene
    assert scene["vrecon_project_id"] == "proj_test"
    assert scene["vrecon_shot_id"] == "shot_001"
    assert scene["vrecon_frame_offset"] == 1
    assert scene.frame_start == 1 and scene.frame_end == 48
    ids = {o.get(ID_PROP) for o in bpy.data.objects if o.get(ID_PROP)}
    assert ids == {"cam_001", "obj_box"}
    assert scene.camera.get(ID_PROP) == "cam_001"


def test_shot_without_camera_fails():
    with pytest.raises(ValueError, match="exactly one camera"):
        build_scene(make_project([]))
