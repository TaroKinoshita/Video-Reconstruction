import numpy as np
import pytest

from vrecon.analysis.appearance import body_part, character_look
from vrecon.analysis.background import background_mesh
from vrecon.analysis.from_4d import Model4D
from vrecon.schema import FileRef, Pose, Project, Rational, SceneObject, Shot, SourceVideo, TransformKey


def test_body_parts():
    assert body_part("left_knee") == "lower" and body_part("spine3") == "upper" and body_part("neck") == "head"


def test_character_look_takes_colours_along_bones():
    video = np.zeros((1, 40, 40, 3), np.uint8)
    video[0, :20] = (0, 0, 255)  # upper half blue
    video[0, 20:] = (255, 0, 0)  # lower half red
    names, parents = ["pelvis", "spine", "knee"], [-1, 0, 0]
    uv = np.array([[[20, 20], [20, 2], [20, 38]]], float)  # spine goes up, knee goes down
    look = character_look(video, uv, names, parents)
    # bones are coloured by their PARENT's part: both start at the pelvis -> "lower" mixes; check keys exist
    assert set(look) == {"lower"}
    names, parents = ["pelvis", "spine", "spine2", "hip", "knee"], [-1, 0, 1, 0, 3]
    uv = np.array([[[20, 20], [20, 10], [20, 2], [20, 30], [20, 38]]], float)
    look = character_look(video, uv, names, parents)
    assert look["upper"] == pytest.approx((0, 0, 1)) and look["lower"][0] == pytest.approx(1.0)


def _model(n=3, h=8, w=10):
    # a wall at y = 50 in front of a camera at the origin looking along +y (aligned world = model world); far enough
    # that the grid spacing stays under the depth-jump limit (3% of the distance)
    xs, zs = np.meshgrid(np.linspace(-1, 1, w), np.linspace(2, 0, h))
    pts = np.stack([xs, np.full_like(xs, 50.0), zs], -1)
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, 2, 3] = 1.0
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    return Model4D(np.arange(n), c2w, np.tile(np.eye(3), (n, 1, 1)), np.tile(pts, (n, 1, 1, 1)),
                   np.ones((n, h, w)), np.stack([u, v], -1).astype(float))


def test_background_mesh_grid_and_exclusion():
    m = _model()
    video = np.full((3, 8, 10, 3), 128, np.uint8)
    v, f, c, rep = background_mesh(m, np.eye(4), video, {}, stride=1)
    assert rep["still_camera"] and len(v) == 80 and len(f) == 2 * 7 * 9
    assert c[0] == pytest.approx([128 / 255] * 3)
    # a box covering columns 0-4 in every frame leaves them out
    v2, f2, _, _ = background_mesh(m, np.eye(4), video, {i: [(-1, -1, 4.4, 9)] for i in range(3)}, stride=1)
    assert len(v2) == 40
    # the floor band (z near 0) is dropped with floor_tol
    v3, _, _, _ = background_mesh(m, np.eye(4), video, {}, stride=1, floor_tol=0.1)
    assert (np.abs(v3[:, 2]) >= 0.1).all()


def _project(objs):
    return Project(id="proj_t", source=SourceVideo(file=FileRef(id="file_v", path="v.mp4"), fps=Rational(num=24, den=1),
                                                    frame_count=2, width=64, height=48),
                   shots=[Shot(id="shot_001", start=0, end=2)], reconstruction={"objects": objs})


def test_identity_colours_are_stable_and_separated():
    pytest.importorskip("bpy")
    from vrecon.scene.color_passes import identity_colors

    rest = TransformKey(time=0.0, pose=Pose())
    a = [SceneObject(id="obj_floor", kind="set", rest=rest), SceneObject(id="obj_box", kind="proxy", rest=rest)]
    c1 = identity_colors(_project(a))
    c2 = identity_colors(_project(a + [SceneObject(id="obj_zz", kind="proxy", rest=rest)]))
    assert c1["obj_box"] == c2["obj_box"] and c1["obj_floor"]["role"] == "set"
    assert abs(c2["obj_box"]["luminance"] - c2["obj_zz"]["luminance"]) > 0.1  # neighbours differ in brightness


def test_builder_sets_look_attribute(tmp_path):
    bpy = pytest.importorskip("bpy")
    from vrecon.scene.builder import LOOK_ATTR, build_scene
    from vrecon.schema import Camera, CameraKey, Intrinsics

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    np.savez(tmp_path / "m.npz", verts=verts, faces=np.array([[0, 1, 2]]), colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]))
    cam = Camera(id="cam_001", shot_id="shot_001", intrinsics=Intrinsics(width=64, height=48, fx=50, fy=50, cx=32, cy=24),
                 keys=[CameraKey(shot_frame=0, pose=Pose())])
    p = _project([SceneObject(id="obj_bg", kind="set", asset=FileRef(id="file_m", path="m.npz")),
                  SceneObject(id="obj_box", size=(1, 1, 1), look={"all": (0.2, 0.4, 0.6)})])
    p.reconstruction.cameras = [cam]
    build_scene(p, root=tmp_path)
    bg = bpy.data.objects["obj_bg"].data.color_attributes[LOOK_ATTR]
    box = bpy.data.objects["obj_box"].data.color_attributes[LOOK_ATTR]
    assert tuple(bg.data[1].color)[:3] == pytest.approx((0, 1, 0))
    assert tuple(box.data[0].color)[:3] == pytest.approx((0.2, 0.4, 0.6))


def test_novel_camera_carried_relative_to_source():
    from vrecon.metrics.novel import carry_relative
    from vrecon.metrics.poses import camera_pose
    from vrecon.schema import Camera, CameraKey, Intrinsics

    intr = Intrinsics(width=64, height=48, fx=50, fy=50, cx=32, cy=24)
    look_y = (0.7071068, -0.7071068, 0.0, 0.0)  # OpenCV camera looking along +y (x right, y down = -z)

    def cam(c, q=look_y):
        return Camera(id="cam_001", shot_id="shot_001", intrinsics=intr, keys=[CameraKey(shot_frame=0, pose=Pose(translation=c, rotation=q))])

    gt_src, novel = cam((0.0, -5.0, 2.0)), cam((1.0, -5.0, 2.0))  # novel: 1 m to the right
    rec_src = cam((10.0, 0.0, 0.2))  # another world, 10x smaller (camera height 0.2 vs 2)
    out = carry_relative(novel, gt_src, rec_src)
    _, c, _ = camera_pose(out, 0)
    assert c == pytest.approx([10.1, 0.0, 0.2], abs=1e-6)


def test_coverage_front_seen_back_unseen():
    pytest.importorskip("bpy")
    from vrecon.scene.builder import build_scene
    from vrecon.scene.coverage import compute_coverage
    from vrecon.schema import Camera, CameraKey, Intrinsics
    import bpy

    look_y = (0.7071068, -0.7071068, 0.0, 0.0)  # OpenCV camera looking along +y
    cam = Camera(id="cam_001", shot_id="shot_001", intrinsics=Intrinsics(width=64, height=48, fx=60, fy=60, cx=32, cy=24),
                 keys=[CameraKey(shot_frame=0, pose=Pose(translation=(0.0, -5.0, 0.0), rotation=look_y))])
    p = _project([SceneObject(id="obj_box", size=(1, 1, 1))])
    p.reconstruction.cameras = [cam]
    build_scene(p)
    cov = compute_coverage([0])["obj_box"]
    me = bpy.data.objects["obj_box"].data
    ny = np.array([pl.normal[1] for pl in me.polygons])
    assert (cov[ny < -0.9] == 2).all()  # facing the camera
    assert (cov[ny > 0.9] == 0).all()  # the back


def test_rule_completions():
    from vrecon.run.complete import extend_set_to_view, mirror, repeat
    from vrecon.schema import Camera, CameraKey, Intrinsics

    floor = SceneObject(id="obj_floor", kind="set", size=(2.0, 2.0, 0.002),
                        rest=TransformKey(time=0.0, pose=Pose(translation=(0.0, 0.0, -0.001))))
    box = SceneObject(id="obj_box", size=(1.0, 0.5, 0.5), rest=TransformKey(time=0.0, pose=Pose(translation=(2.0, 1.0, 0.25))))
    p = _project([floor, box])
    # a camera 2 m up looking down 45 deg along +y sees far more floor than 2 x 2 m
    q = (0.3826834, -0.9238795, 0.0, 0.0)  # OpenCV axes turned -135 deg about x: looking along +y and 45 deg down
    cam = Camera(id="cam_001", shot_id="shot_001", intrinsics=Intrinsics(width=64, height=48, fx=40, fy=40, cx=32, cy=24),
                 keys=[CameraKey(shot_frame=0, pose=Pose(translation=(0.0, -3.0, 2.0), rotation=q))])
    p2, r = extend_set_to_view(p, "obj_floor", cam, frames=[0])
    f2 = next(o for o in p2.reconstruction.objects if o.id == "obj_floor")
    assert r["changed"] and f2.size[0] > 2.0 and f2.size[2] == pytest.approx(0.002) and f2.status == "hypothesis"
    assert f2.rest.pose.translation[2] == pytest.approx(-0.001)  # stays in its plane
    p3, r = mirror(p, "obj_box", point=(0, 0, 0), normal=(1, 0, 0))
    m = next(o for o in p3.reconstruction.objects if o.id == r["new"])
    assert m.rest.pose.translation == pytest.approx((-2.0, 1.0, 0.25)) and m.status == "hypothesis"
    p4, r = repeat(p, "obj_box", offset=(0, 1.0, 0), count=2)
    assert [o.rest.pose.translation[1] for o in p4.reconstruction.objects if o.id in r["new"]] == pytest.approx([2.0, 3.0])
