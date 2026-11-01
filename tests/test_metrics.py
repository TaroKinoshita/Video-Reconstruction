"""Scoring tools: perfect score on ground truth, invariance to unobservable world changes, sensitivity to errors."""

import json
import math

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

from vrecon.evaluate import render_reconstruction_passes, score  # noqa: E402
from vrecon.metrics import onset_frame, umeyama  # noqa: E402
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
    SourceVideo,
    TransformKey,
    save_project,
)
from vrecon.scene.convert import look_at_opencv, matrix_to_quat, quat_mul, quat_to_matrix  # noqa: E402

W, H, N = 160, 96, 16


def gt_project() -> Project:
    intr = Intrinsics(width=W, height=H, fx=150, fy=150, cx=(W - 1) / 2, cy=(H - 1) / 2)
    keys = []
    for f in (0, 5, 10, 15):
        eye = (-1.0 + 2.0 * f / 15, -6.0, 1.6)
        keys.append(CameraKey(shot_frame=f, pose=Pose(translation=eye, rotation=look_at_opencv(eye, (0, 1, 0.5)))))
    q = lambda a: (math.cos(a / 2), 0, 0, math.sin(a / 2))  # noqa: E731
    return Project(
        id="proj_gt",
        source=SourceVideo(file=FileRef(id="file_v", path="video.mp4"), fps=Rational(num=24, den=1), frame_count=N,
                           width=W, height=H),
        shots=[Shot(id="shot_001", start=0, end=N)],
        reconstruction={
            "cameras": [Camera(id="cam_001", shot_id="shot_001", intrinsics=intr, keys=keys)],
            "objects": [
                SceneObject(id="obj_floor", kind="set", size=(20, 20, 0.02),
                            rest=TransformKey(time=0, pose=Pose(translation=(0, 0, -0.01)))),
                SceneObject(id="obj_a", size=(0.8, 0.8, 0.8), keys=[
                    TransformKey(time=4 / 24, pose=Pose(translation=(-1.2, 0.5, 0.4))),
                    TransformKey(time=12 / 24, pose=Pose(translation=(0.8, 0.5, 0.4), rotation=q(0.6)))]),
                SceneObject(id="obj_b", size=(0.5, 0.5, 1.2),
                            rest=TransformKey(time=0, pose=Pose(translation=(1.4, 1.8, 0.6)))),
            ],
        },
    )


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    d = tmp_path_factory.mktemp("ref")
    gt = gt_project()
    render_reconstruction_passes(gt, "shot_001", d / "gt_passes")
    save_project(gt, d / "gt.json")
    (d / "meta.json").write_text(json.dumps({"events": [{"object": "obj_a", "onset_frame": onset_frame(gt, "obj_a", "shot_001")}]}))
    return d


def transformed(project: Project, s: float, angle: float, t) -> Project:
    """Apply one similarity transform to the whole world (cameras, objects, sizes)."""
    qz = (math.cos(angle / 2), 0, 0, math.sin(angle / 2))
    r = np.array(quat_to_matrix(qz))
    p = project.model_copy(deep=True)

    def pt(x):
        return tuple((s * r @ np.array(x) + np.array(t)).tolist())

    for cam in p.reconstruction.cameras:
        for k in cam.keys:
            k.pose = Pose(translation=pt(k.pose.translation), rotation=quat_mul(qz, k.pose.rotation))
    for o in p.reconstruction.objects:
        o.size = tuple(s * v for v in o.size)
        for k in [o.rest, *o.keys]:
            k.pose = Pose(translation=pt(k.pose.translation), rotation=quat_mul(qz, k.pose.rotation))
    return p


def test_ground_truth_scores_perfectly(reference):
    rep = score(reference, gt_project())
    s = rep["summary"]
    assert rep["pairs"] == {"obj_floor": "obj_floor", "obj_a": "obj_a", "obj_b": "obj_b"}
    assert s["mask_iou_mean"] == pytest.approx(1.0)
    assert s["centroid_error_mean"] == pytest.approx(0.0, abs=1e-9)
    assert s["reprojection_error_mean"] < 1e-4
    assert s["camera_rotation_deg_mean"] < 1e-3 and s["camera_focal_rel_mean"] < 1e-9
    assert s["camera_center_rel_mean"] < 1e-6
    assert s["timing_abs_error_frames_mean"] == 0


def test_unobservable_world_change_scores_perfectly(reference):
    rec = transformed(gt_project(), s=2.5, angle=0.7, t=(3.0, -1.0, 0.5))
    rec = rec.model_copy(update={"id": "proj_rec"})
    s = score(reference, rec)["summary"]
    assert s["mask_iou_mean"] > 0.995
    assert s["reprojection_error_mean"] < 1e-3
    assert s["camera_rotation_deg_mean"] < 1e-3
    assert s["camera_center_rel_mean"] < 1e-6


def test_moved_object_is_caught(reference):
    rec = gt_project()
    b = next(o for o in rec.reconstruction.objects if o.id == "obj_b")
    b.rest = TransformKey(time=0, pose=Pose(translation=(1.7, 1.8, 0.6)))
    rep = score(reference, rec)
    assert rep["mask_iou"]["obj_b"]["mean"] < 0.8
    assert rep["anchored_reprojection"]["obj_b"]["mean"] > 0.01
    assert rep["mask_iou"]["obj_a"]["mean"] == pytest.approx(1.0)  # untouched object stays perfect
    assert rep["summary"]["camera_rotation_deg_mean"] < 1e-3


def test_wrong_focal_is_caught(reference):
    rec = gt_project()
    cam = rec.reconstruction.cameras[0]
    cam.intrinsics = cam.intrinsics.model_copy(update={"fx": 165.0, "fy": 165.0})
    s = score(reference, rec)["summary"]
    assert s["camera_focal_rel_mean"] == pytest.approx(0.1)
    assert s["reprojection_error_mean"] > 0.005


def test_late_onset_is_caught(reference):
    rec = gt_project()
    a = next(o for o in rec.reconstruction.objects if o.id == "obj_a")
    a.keys = [k.model_copy(update={"time": k.time + 3 / 24}) for k in a.keys]
    rep = score(reference, rec)
    assert rep["timing"]["obj_a"]["error_frames"] == 3


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    r = np.array(quat_to_matrix(matrix_to_quat(np.linalg.qr(rng.normal(size=(3, 3)))[0].tolist())))
    if np.linalg.det(r) < 0:
        r[:, 0] *= -1
    dst = (1.7 * (r @ src.T)).T + np.array([1, 2, 3])
    s, rr, t = umeyama(src, dst)
    assert s == pytest.approx(1.7) and np.allclose(rr, r, atol=1e-8) and np.allclose(t, [1, 2, 3])
    assert umeyama(np.zeros((5, 3)), dst[:5]) is None


def _llm_like_blend(path, scale_res=1, strip_ids=True, extra=True, fit="HORIZONTAL"):
    """A .blend as an LLM might leave it: no vrecon ids, other camera settings, an extra object."""
    from vrecon.scene.builder import build_scene

    build_scene(gt_project())
    scene = bpy.context.scene
    if strip_ids:
        for o in bpy.data.objects:
            if "vrecon_id" in o.keys():
                del o["vrecon_id"]
    cam = scene.camera.data
    if fit == "HORIZONTAL":  # same field of view expressed differently
        cam.sensor_fit = "HORIZONTAL"
        cam.sensor_width = 24.0
        cam.lens = 150 * 24.0 / W
    if extra:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(-2.5, 3.0, 0.3))
        bpy.context.object.name = "Ball"
    scene.render.resolution_x *= scale_res
    scene.render.resolution_y *= scale_res
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    return path


def test_score_blend_matches_by_overlap(reference, tmp_path):
    from vrecon.evaluate import score_blend

    blend = _llm_like_blend(tmp_path / "scene.blend", scale_res=2)
    rep = score_blend(reference, blend)
    s = rep["summary"]
    assert set(rep["pairs"]) == {"obj_floor", "obj_a", "obj_b"}
    assert all(v.startswith("obj_") for v in rep["pairs"].values())
    assert "obj_Ball" not in rep["pairs"].values()
    assert s["mask_iou_mean"] > 0.99
    assert s["camera_rotation_deg_mean"] < 1e-3 and s["camera_focal_rel_mean"] < 1e-6
    assert s["reprojection_error_mean"] < 1e-3
    assert s["timing_abs_error_frames_mean"] == 0
    assert any("changed to" in w for w in rep["import_warnings"])


def test_score_blend_wrong_aspect_is_refused(reference, tmp_path):
    from vrecon.evaluate import score_blend

    blend = _llm_like_blend(tmp_path / "scene.blend", extra=False)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    bpy.context.scene.render.resolution_x = W * 2
    bpy.ops.wm.save_mainfile()
    with pytest.raises(ValueError, match="aspect"):
        score_blend(reference, blend)


def test_onset_definition_matches_keyframe():
    gt = gt_project()  # obj_a's starting key is at frame 4
    assert onset_frame(gt, "obj_a", "shot_001") == 4
    assert onset_frame(gt, "obj_b", "shot_001") is None  # never moves
