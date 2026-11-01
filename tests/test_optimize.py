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
    good = silhouette_metrics(gt_project(), obs, PAIRS)["silhouette_loss"]
    bad = silhouette_metrics(wrong_project(), obs, PAIRS)["silhouette_loss"]
    assert good < 0.15  # floor from the quick hull raster vs Blender's render at 160x96
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
    assert st["revision"] == 0 and st["pairs"] == PAIRS
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
