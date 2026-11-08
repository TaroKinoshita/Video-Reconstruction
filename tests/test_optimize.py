"""Optimizer + run manager on a synthetic reference: a wrong focal length and a misplaced object get fixed."""

import pytest

pytest.importorskip("bpy")

from test_metrics import gt_project, reference  # noqa: E402,F401  (fixture)
from vrecon.evaluate import score  # noqa: E402
from vrecon.optimize import Param, gt_stand_in_masks, optimize, silhouette_metrics  # noqa: E402
from vrecon.run import AcceptRule, RunManager  # noqa: E402
from vrecon.schema import Pose, TransformKey, save_project  # noqa: E402

PAIRS = {"obj_a": "obj_a", "obj_b": "obj_b"}


def wrong_project():
    p = gt_project()
    cam = p.reconstruction.cameras[0]
    cam.intrinsics = cam.intrinsics.model_copy(update={"fx": 125.0, "fy": 125.0})  # 17% too short
    b = next(o for o in p.reconstruction.objects if o.id == "obj_b")
    b.rest = TransformKey(time=0, pose=Pose(translation=(1.1, 1.4, 0.6)))  # misplaced
    return p


def test_ground_truth_has_low_silhouette_loss(reference):  # noqa: F811
    obs = gt_stand_in_masks(reference)
    assert obs.observation.provenance.tool == "ground_truth_stand_in"
    good = silhouette_metrics(gt_project(), obs, PAIRS, down=1)["silhouette_loss"]
    bad = silhouette_metrics(wrong_project(), obs, PAIRS, down=1)["silhouette_loss"]
    assert good < 0.12  # edge pixels dominate at 160x96 (0.03 on the 640x352 clips)
    assert bad > good + 0.1


def test_search_then_adopt_improves_true_score(reference, tmp_path):  # noqa: F811
    obs = gt_stand_in_masks(reference)
    save_project(wrong_project(), tmp_path / "project.json")

    def evaluate(p):
        return {"silhouette_loss": silhouette_metrics(p, obs, PAIRS)["silhouette_loss"]}

    mgr = RunManager(tmp_path / "project.json", tmp_path / "run", evaluate, AcceptRule(primary="silhouette_loss"))
    before = score(reference, mgr.adopted)["summary"]

    # hypothesis 1 (as an LLM might pick from the diagnostics): the focal length is off
    prop, info = optimize(mgr.adopted, obs, PAIRS, [Param("focal", "cam_001", step=0.1)], "focal too short", "p1",
                          max_evals=40)
    assert mgr.try_proposal(prop).status == "accepted"
    # hypothesis 2: obj_b is misplaced on the floor
    prop, info = optimize(mgr.adopted, obs, PAIRS, [Param("translation", "obj_b", 0, 0.2),
                                                    Param("translation", "obj_b", 1, 0.2)], "obj_b misplaced", "p2",
                          max_evals=80)
    assert mgr.try_proposal(prop).status == "accepted"

    after = score(reference, mgr.adopted)["summary"]  # ground truth only used here, to check
    assert after["camera_focal_rel_mean"] < before["camera_focal_rel_mean"] / 3
    assert after["reprojection_error_mean"] < before["reprojection_error_mean"] / 3
    assert after["mask_iou_mean"] > before["mask_iou_mean"]
    assert mgr.adopted.revision == 2


def test_service_cli_and_mcp_tools(reference, tmp_path):  # noqa: F811
    import asyncio
    import json

    from vrecon.cli import main
    from vrecon.mcp_server import server
    from vrecon.run import service

    save_project(wrong_project(), tmp_path / "project.json")
    rd = str(tmp_path / "run")
    service.init_run(str(tmp_path / "project.json"), rd, str(reference))
    st = service.state(rd)
    assert st["revision"] == 0 and st["pairs"] == {**PAIRS, "obj_floor": "obj_floor"}
    d = service.diagnose(rd)
    assert d["objects"]["obj_b"]["loss"] > d["objects"]["obj_a"]["loss"]  # the misplaced one stands out
    assert main(["run", "optimize", rd, "--param", "focal:cam_001", "--hypothesis", "focal", "--max-evals", "30"]) == 0
    hist = service.history(rd)
    assert hist[-1]["status"] == "accepted" and hist[-1]["source"] == "optimizer"
    bad = service.submit(rd, {"id": "llm1", "hypothesis": "move b far away",
                              "changes": [{"target_id": "obj_b", "field": "rest.pose.translation", "value": [9, 9, 0]}]})
    assert bad["status"] == "rejected"

    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"run_state", "run_diagnose", "run_optimize", "run_submit", "run_history", "run_rollback"} <= names
    res = asyncio.run(server.call_tool("run_state", {"run_dir": rd}))
    assert "revision" in json.dumps(res, default=str)


def test_time_search_fixes_late_onset(reference):  # noqa: F811
    obs = gt_stand_in_masks(reference)
    p = gt_project()
    a = next(o for o in p.reconstruction.objects if o.id == "obj_a")
    a.keys = [k.model_copy(update={"time": k.time + 4 / 24}) for k in a.keys]  # 4 frames late
    from vrecon.run import apply_changes

    prop, info = optimize(p, obs, PAIRS, [Param("time", "obj_a", step=4.0)], "obj_a starts late", "t1", max_evals=40)
    assert info["loss_after"] < info["loss_before"]
    fixed = apply_changes(p, prop.changes)
    assert abs(score(reference, fixed)["timing"]["obj_a"]["error_frames"]) <= 1


def test_track_loss_sees_focal_error(tmp_path):
    import subprocess

    from test_metrics import gt_project as gtp
    from vrecon.optimize.tracks import track_metrics, track_points
    from vrecon.scene.builder import build_scene, render_frames
    from vrecon.scene.look import apply_look
    from vrecon.control.core import encode_mp4, load_frames

    gt = gtp()  # camera trucks sideways
    build_scene(gt)
    apply_look(1)
    frames = render_frames(tmp_path / "f", list(range(16)), engine="BLENDER_WORKBENCH")
    encode_mp4(load_frames(frames), tmp_path / "v.mp4", crf=12)
    tr = track_points(tmp_path / "v.mp4", reseed=4, span=12)
    assert len(tr.xy) > 10 and tr.observation.provenance.tool == "opencv_lk"
    good = track_metrics(gt, tr)["track_loss"]
    wrong = gt.model_copy(deep=True)
    cam = wrong.reconstruction.cameras[0]
    cam.intrinsics = cam.intrinsics.model_copy(update={"fx": 180.0, "fy": 180.0})
    assert track_metrics(wrong, tr)["track_loss"] > 2 * good


def test_run_auto(reference, tmp_path):  # noqa: F811
    from vrecon.cli import main
    from vrecon.run import service

    save_project(wrong_project(), tmp_path / "project.json")
    rd = str(tmp_path / "run")
    service.init_run(str(tmp_path / "project.json"), rd, str(reference))
    out = service.auto(rd, passes=1, max_evals=30)
    assert out["accepted"] >= 1 and out["metrics_after"]["silhouette_loss"] < out["metrics_before"]["silhouette_loss"]
    assert main(["run", "history", rd, "--last", "1"]) == 0


def test_height_change_keeps_bottom_face(reference):  # noqa: F811
    import numpy as np

    from vrecon.optimize import apply_params
    from vrecon.schema import load_project

    p = load_project(reference / "gt.json")
    o = next(o for o in p.reconstruction.objects if o.kind != "set")
    yaw = (float(np.cos(0.3)), 0.0, 0.0, float(np.sin(0.3)))  # upright box turned about Z: up stays +Z
    o.rest = TransformKey(time=0.0, pose=Pose(translation=(1.0, 2.0, o.size[2] / 2), rotation=yaw))
    o.keys = []
    q, _ = apply_params(p, [Param("size", o.id, 2)], np.array([0.4]))
    o2 = next(x for x in q.reconstruction.objects if x.id == o.id)
    assert o2.size[2] == pytest.approx(o.size[2] + 0.4)
    assert o2.rest.pose.translation[2] - o2.size[2] / 2 == pytest.approx(0.0, abs=1e-9)
    assert o2.rest.pose.translation[:2] == pytest.approx((1.0, 2.0))


def test_yaw_and_spin_turn_about_up_axis(reference):  # noqa: F811
    import numpy as np

    from vrecon.optimize import apply_params
    from vrecon.scene.convert import quat_to_matrix
    from vrecon.schema import load_project

    p = load_project(reference / "gt.json")
    o = next(o for o in p.reconstruction.objects if o.kind != "set")
    o.keys = [TransformKey(time=float(i), pose=Pose(translation=(float(i), 0.0, 0.5))) for i in range(3)]

    def yaw_of(k):
        r = np.array(quat_to_matrix(k.pose.rotation))
        assert r[:, 2] == pytest.approx([0, 0, 1], abs=1e-9)  # stays upright
        return float(np.arctan2(r[1, 0], r[0, 0]))

    q, _ = apply_params(p, [Param("yaw", o.id)], np.array([0.4]))
    o2 = next(x for x in q.reconstruction.objects if x.id == o.id)
    assert [yaw_of(k) for k in o2.keys] == pytest.approx([0.4] * 3)
    assert o2.keys[1].pose.translation == o.keys[1].pose.translation
    q, _ = apply_params(p, [Param("spin", o.id)], np.array([0.6]))
    o3 = next(x for x in q.reconstruction.objects if x.id == o.id)
    assert [yaw_of(k) for k in o3.keys] == pytest.approx([0.0, 0.3, 0.6])
