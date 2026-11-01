"""handoff -> (simulated human edit in Blender) -> ingest. Needs bpy."""

import math
import random

import pytest

bpy = pytest.importorskip("bpy")

import mathutils  # noqa: E402

from vrecon.schema import (  # noqa: E402
    Camera,
    CameraKey,
    FileRef,
    Intrinsics,
    Lock,
    Pose,
    Project,
    Rational,
    SceneObject,
    Shot,
    ShotMapping,
    SourceVideo,
    TransformKey,
)
from vrecon.scene.builder import ID_PROP, build_scene  # noqa: E402
from vrecon.scene.handoff import handoff, ingest_blend  # noqa: E402
from vrecon.scene.ingest import IngestConflict, _keys_close, _lens_close, _xform_close  # noqa: E402
from vrecon.scene.readback import read_current  # noqa: E402

INTR = Intrinsics(width=640, height=360, fx=500, fy=500, cx=330.0, cy=170.0)
FPS = Rational(num=24000, den=1001)


def make_project(cam_keys=None, extra_objects=()):
    rng = random.Random(3)
    q = [rng.gauss(0, 1) for _ in range(4)]
    n = math.sqrt(sum(c * c for c in q))
    cam_keys = cam_keys or [CameraKey(shot_frame=0, pose=Pose(translation=(0, -6, 1.5), rotation=tuple(c / n for c in q)))]
    t0 = 2.0
    return Project(
        id="proj_rb",
        source=SourceVideo(file=FileRef(id="file_src", path="in.mp4"), fps=FPS, frame_count=60, width=640, height=360),
        shots=[Shot(id="shot_001", start=0, end=48, transition_from_previous="first")],
        reconstruction={
            "cameras": [Camera(id="cam_001", shot_id="shot_001", intrinsics=INTR, keys=cam_keys)],
            "objects": [
                SceneObject(id="obj_static", size=(1, 1, 1), rest=TransformKey(time=0, pose=Pose(translation=(0, 0, 0.5)))),
                SceneObject(
                    id="obj_moving",
                    size=(0.5, 0.5, 0.5),
                    keys=[
                        TransformKey(time=t0, pose=Pose(translation=(-1, 1, 0.25))),
                        TransformKey(time=t0 + 10 * 1001 / 24000, pose=Pose(translation=(1, 1, 0.25))),
                    ],
                ),
                *extra_objects,
            ],
            "shot_mapping": [ShotMapping(shot_id="shot_001", scene_time_start=t0)],
        },
    )


def obj(vid):
    return next(o for o in bpy.data.objects if o.get(ID_PROP) == vid)


def edit_blend(path, fn):
    """Simulate a human: open the .blend, change it, save."""
    bpy.ops.wm.open_mainfile(filepath=str(path))
    fn()
    bpy.ops.wm.save_mainfile()


@pytest.fixture
def handed(tmp_path):
    project = make_project()
    snap = handoff(project, tmp_path / "handoff")
    return project, snap


def test_no_edit_means_no_changes(handed):
    project, snap = handed
    result = ingest_blend(project, snap)
    assert result.status == "no_changes"
    assert result.warnings == []
    assert result.project.revision == project.revision


def test_move_static_object(handed):
    project, snap = handed

    def move():
        obj("obj_static").location = (0.5, 0.25, 0.5)

    edit_blend(snap.blend_path, move)
    result = ingest_blend(project, snap)
    assert result.status == "applied"
    assert [(e.target_id, e.field) for e in result.edits] == [("obj_static", "rest.pose.translation")]
    edit = result.edits[0]
    assert edit.source == "blender_ingest"
    assert edit.old == [0.0, 0.0, 0.5] and edit.new == pytest.approx([0.5, 0.25, 0.5])
    new_obj = next(o for o in result.project.reconstruction.objects if o.id == "obj_static")
    assert new_obj.rest.pose.translation == pytest.approx((0.5, 0.25, 0.5))
    assert result.project.revision == project.revision + 1
    assert result.project.human.edits == result.edits
    assert project.reconstruction.objects[0].rest.pose.translation == (0, 0, 0.5)  # input untouched


def test_euler_rotation_by_user_is_read_correctly(handed):
    project, snap = handed

    def rotate():
        o = obj("obj_static")
        o.rotation_mode = "XYZ"
        o.rotation_euler = (0, 0, math.radians(30))

    edit_blend(snap.blend_path, rotate)
    result = ingest_blend(project, snap)
    assert [e.field for e in result.edits] == ["rest.pose.rotation"]
    w, x, y, z = next(o for o in result.project.reconstruction.objects if o.id == "obj_static").rest.pose.rotation
    assert (w, x, y, z) == pytest.approx((math.cos(math.radians(15)), 0, 0, math.sin(math.radians(15))), abs=1e-6)


def test_retime_animated_object(handed):
    project, snap = handed

    def retime():
        o = obj("obj_moving")
        from bpy_extras import anim_utils

        cb = anim_utils.action_get_channelbag_for_slot(o.animation_data.action, o.animation_data.action_slot)
        for fc in cb.fcurves:
            for kp in fc.keyframe_points:
                if abs(kp.co[0] - 11) < 1e-6:  # second key: shot frame 10 -> 16
                    kp.co[0] = 17
                    kp.handle_left[0] = kp.handle_right[0] = 17

    edit_blend(snap.blend_path, retime)
    result = ingest_blend(project, snap)
    assert [(e.target_id, e.field) for e in result.edits] == [("obj_moving", "keys")]
    keys = next(o for o in result.project.reconstruction.objects if o.id == "obj_moving").keys
    assert [k.time for k in keys] == pytest.approx([2.0, 2.0 + 16 * 1001 / 24000])
    assert keys[1].pose.translation == pytest.approx((1, 1, 0.25))


def test_camera_move_and_lens_roundtrip(handed, tmp_path):
    project, snap = handed

    def change_camera():
        c = obj("cam_001")
        c.location.x += 0.3
        c.rotation_quaternion = (mathutils.Quaternion((0, 0, 1), 0.1) @ c.rotation_quaternion)
        c.data.lens = 40.0
        c.data.shift_x = 0.05

    edit_blend(snap.blend_path, change_camera)
    edited = read_current()
    result = ingest_blend(project, snap)
    assert sorted(e.field for e in result.edits) == ["intrinsics", "keys"]
    cam = result.project.reconstruction.cameras[0]
    assert cam.intrinsics.fx == pytest.approx(40.0 * 640 / 36)
    assert cam.intrinsics.cx == pytest.approx((640 - 1) / 2 - 0.05 * 640)

    # Rebuilding from the ingested project reproduces the edited Blender camera.
    build_scene(result.project)
    rebuilt = read_current().objects["cam_001"]
    want = edited.objects["cam_001"]
    assert _xform_close(rebuilt.static, want.static)
    assert _lens_close(rebuilt.camera_static, want.camera_static)


def test_camera_zoom_keys(tmp_path):
    zoomed = INTR.model_copy(update={"fx": 700.0, "fy": 700.0})
    keys = [
        CameraKey(shot_frame=0, pose=Pose(translation=(0, -6, 1.5), rotation=(0.7071068, -0.7071068, 0, 0))),
        CameraKey(shot_frame=20, pose=Pose(translation=(1, -6, 1.5), rotation=(0.7071068, -0.7071068, 0, 0)), intrinsics=zoomed),
    ]
    project = make_project(cam_keys=keys)
    snap = handoff(project, tmp_path / "h")

    def more_zoom():
        bpy.context.scene.frame_set(21)
        c = obj("cam_001")
        c.data.lens = 50.0
        c.data.keyframe_insert("lens", frame=21)

    edit_blend(snap.blend_path, more_zoom)
    edited = read_current()
    result = ingest_blend(project, snap)
    assert [e.field for e in result.edits] == ["keys"]
    new_keys = result.project.reconstruction.cameras[0].keys
    assert [k.shot_frame for k in new_keys] == [0, 20]
    assert new_keys[1].intrinsics.fx == pytest.approx(50.0 * 640 / 36)
    build_scene(result.project)
    assert _keys_close(read_current().objects["cam_001"].keys, edited.objects["cam_001"].keys)


def test_out_of_scope_edits_are_warnings(handed):
    project, snap = handed

    def messy():
        bpy.ops.mesh.primitive_uv_sphere_add()  # new object without vrecon_id
        o = obj("obj_static")
        o.data.vertices[0].co.x += 0.2  # shape edit
        o.modifiers.new("bevel", "BEVEL")
        bpy.data.objects.remove(obj("obj_moving"))  # deletion
        bpy.context.scene.render.resolution_x = 800

    edit_blend(snap.blend_path, messy)
    result = ingest_blend(project, snap)
    assert result.status == "no_changes"
    text = "\n".join(result.warnings)
    for needle in ("new object", "shape edited", "modifiers changed", "obj_moving: deleted", "resolution_x"):
        assert needle in text, needle


def test_project_changed_since_handoff_conflicts(handed):
    project, snap = handed
    newer = project.model_copy(update={"revision": project.revision + 1})
    with pytest.raises(IngestConflict, match="changed since handoff"):
        ingest_blend(newer, snap)


def test_blend_from_another_handoff_conflicts(handed, tmp_path):
    project, snap = handed
    other = handoff(make_project().model_copy(update={"id": "proj_other"}), tmp_path / "other")
    with pytest.raises(IngestConflict, match="does not belong"):
        ingest_blend(project, snap, other.blend_path)


def test_human_edits_apply_even_to_locked_fields(handed):
    """Locks restrict the AI, not the human. The lock stays in place."""
    project, snap = handed
    project = project.model_copy(deep=True)
    project.human.locks.append(Lock(id="lock_1", target_id="obj_static", field="rest.pose"))

    def move():
        obj("obj_static").location.z = 1.0

    edit_blend(snap.blend_path, move)
    result = ingest_blend(project, snap)
    assert [e.field for e in result.edits] == ["rest.pose.translation"]
    assert result.project.is_locked("obj_static", "rest.pose.translation")
