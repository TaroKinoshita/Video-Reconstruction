"""Editability test (H5) and the build -> .blend -> import round trip (WP-8)."""

import pytest

bpy = pytest.importorskip("bpy")

from vrecon import edits  # noqa: E402
from vrecon.schema import Camera, CameraKey, Intrinsics, Pose, SceneObject, TransformKey  # noqa: E402
from vrecon.scene.blend_import import import_current  # noqa: E402
from vrecon.scene.builder import build_scene, save_blend  # noqa: E402
from vrecon.scene.edit_blend import run_all, run_edit  # noqa: E402

from test_builder import make_project  # noqa: E402

LOOK = (0.7071068, 0.7071068, 0.0, 0.0)  # OpenCV camera looking along +Y (x right, y down = -Z)


def _project(frames=48):
    intr = Intrinsics(width=320, height=180, fx=250, fy=250, cx=159.5, cy=89.5)
    cam = Camera(id="cam_001", shot_id="shot_001", intrinsics=intr, keys=[
        CameraKey(shot_frame=0, pose=Pose(translation=(0, -8, 1), rotation=LOOK)),
        CameraKey(shot_frame=frames - 1, pose=Pose(translation=(0, -6, 1), rotation=LOOK))])
    mover = SceneObject(id="obj_mover", size=(0.5, 0.5, 0.5), keys=[
        TransformKey(time=0.0, pose=Pose(translation=(-1, 0, 0.25))),
        TransformKey(time=(frames - 1) / 24, pose=Pose(translation=(1, 0, 0.25)))])
    still = SceneObject(id="obj_still", size=(0.4, 0.4, 1.0), rest=TransformKey(time=0, pose=Pose(translation=(0, 2, 0.5))))
    floor = SceneObject(id="obj_floor", kind="set", size=(10, 10, 0.01), rest=TransformKey(time=0, pose=Pose()))
    return make_project([cam], [mover, still, floor], frames=frames)


@pytest.fixture
def built(tmp_path):
    p = _project()
    build_scene(p)
    return p, save_blend(tmp_path / "scene.blend")


def test_round_trip_matches_project(built):
    p, path = built
    bpy.ops.wm.open_mainfile(filepath=str(path))
    got = {o.id: o for o in import_current().reconstruction.objects}
    assert set(got) == {"obj_mover", "obj_still", "obj_floor"}
    assert got["obj_still"].size == pytest.approx((0.4, 0.4, 1.0), rel=1e-4)
    assert got["obj_still"].rest.pose.translation == pytest.approx((0, 2, 0.5), abs=1e-5)
    keys = got["obj_mover"].keys
    assert len(keys) == 48
    assert keys[0].pose.translation == pytest.approx((-1, 0, 0.25), abs=1e-5)
    assert keys[-1].pose.translation == pytest.approx((1, 0, 0.25), abs=1e-5)
    assert keys[24].pose.translation[0] == pytest.approx(-1 + 2 * 24 / 47, abs=1e-4)


def test_target_is_the_mover():
    assert edits.choose_target(_project()) == "obj_mover"


def test_all_edits_land_on_a_built_scene(built):
    _, path = built
    rep = run_all(path)
    assert rep["edit_success_share"] == 1.0, rep
    assert {r["target"] for r in rep["edits"]} == {"obj_mover"}


def test_timing_edit_fails_when_a_parent_carries_the_motion(built, tmp_path):
    _, path = built
    bpy.ops.wm.open_mainfile(filepath=str(path))
    mover = next(o for o in bpy.data.objects if o.get("vrecon_id") == "obj_mover")
    rig = bpy.data.objects.new("rig", None)
    bpy.context.scene.collection.objects.link(rig)
    mover.animation_data.action = None  # motion moves to the parent empty
    mover.location = (0, 0, 0.25)
    mover.parent = rig
    for f, x in ((1, -1.0), (48, 1.0)):
        rig.location = (x, 0, 0)
        rig.keyframe_insert("location", frame=f)
    other = tmp_path / "parented.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(other))
    r = run_edit(other, "action_timing")
    assert not r["success"] and "no keyframes" in r["problems"][0]


def test_size_edit_fails_on_shared_mesh(built, tmp_path):
    _, path = built
    bpy.ops.wm.open_mainfile(filepath=str(path))
    objs = {o.get("vrecon_id"): o for o in bpy.data.objects}
    objs["obj_still"].data = objs["obj_mover"].data
    other = tmp_path / "shared.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(other))
    r = run_edit(other, "object_size")
    assert not r["success"]
