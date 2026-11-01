import json

import pytest

from test_schema import make_project
from vrecon.run import AcceptRule, Change, Proposal, RunManager
from vrecon.schema import Lock, load_project, save_project

TARGET = (2.0, 0.0, 0.5)  # where obj_box "should" be at rest


def dist(p):  # toy evaluator: distance of obj_box's rest position to TARGET, plus camera x as a guard
    box = next(o for o in p.reconstruction.objects if o.id == "obj_box")
    d = sum((a - b) ** 2 for a, b in zip(box.rest.pose.translation, TARGET)) ** 0.5
    if d > 100:
        raise RuntimeError("evaluator crashed")
    return {"err": d, "cam_x": abs(p.reconstruction.cameras[0].keys[0].pose.translation[0])}


@pytest.fixture
def mgr(tmp_path):
    save_project(make_project(), tmp_path / "project.json")
    return RunManager(tmp_path / "project.json", tmp_path / "run", dist,
                      AcceptRule(primary="err", guards={"cam_x": 0.01}), max_attempts=6)


def move(pid, xyz):
    return Proposal(id=pid, changes=[Change(target_id="obj_box", field="rest.pose.translation", value=xyz)])


def test_accept_reject_and_history(mgr, tmp_path):
    assert mgr.try_proposal(move("p1", (1.0, 0, 0.5))).status == "accepted"
    assert load_project(tmp_path / "project.json").revision == 1
    assert (tmp_path / "run/snapshots/rev_0000.json").is_file()
    out = mgr.try_proposal(move("p2", (-5.0, 0, 0.5)))  # worse
    assert out.status == "rejected" and "did not improve" in out.reasons[0]
    assert load_project(tmp_path / "project.json").revision == 1  # unchanged on disk
    lines = (tmp_path / "run/history.jsonl").read_text().splitlines()
    assert [json.loads(x)["outcome"]["status"] for x in lines] == ["accepted", "rejected"]


def test_guard_metric_blocks(mgr):
    p = Proposal(id="p", changes=[
        Change(target_id="obj_box", field="rest.pose.translation", value=(1.5, 0, 0.5)),
        Change(target_id="cam_001", field="keys", value=[{"shot_frame": 0, "pose": {"translation": (3, -5, 1)}}]),
    ])
    out = mgr.try_proposal(p)
    assert out.status == "rejected" and any("cam_x got worse" in r for r in out.reasons)


def test_locked_field_rejected(tmp_path):
    p = make_project()
    p.human.locks.append(Lock(id="lock_1", target_id="obj_box", field="rest.pose"))
    save_project(p, tmp_path / "project.json")
    m = RunManager(tmp_path / "project.json", tmp_path / "run", dist, AcceptRule(primary="err"))
    out = m.try_proposal(move("p", (1.0, 0, 0.5)))
    assert out.status == "rejected" and any("locked" in r for r in out.reasons)


def test_code_candidate_with_undeclared_change_rejected(mgr):
    cand = mgr.adopted.model_copy(deep=True)
    cand.reconstruction.objects[0].rest.pose.translation = (1.0, 0, 0.5)  # declared
    cand.reconstruction.cameras[0].intrinsics.fx = 600.0  # sneaky, undeclared
    p = Proposal(id="code1", source="code", declared=[("obj_box", "rest.pose")])
    out = mgr.try_proposal(p, candidate=cand)
    assert out.status == "rejected" and any("cam_001.intrinsics.fx changed but was not declared" in r
                                            for r in out.reasons)


def test_errors_and_budget(mgr):
    assert mgr.try_proposal(move("bad", (500.0, 0, 0))).status == "error"  # evaluator crash
    assert mgr.try_proposal(Proposal(id="x", changes=[Change(target_id="obj_nope", field="size", value=(1, 1, 1))])
                            ).status == "error"
    assert mgr.adopted.revision == 0
    for i in range(4):
        mgr.try_proposal(move(f"w{i}", (-5.0 - i, 0, 0.5)))
    assert mgr.try_proposal(move("late", (1.9, 0, 0.5))).status == "budget_exhausted"


def test_rollback_and_resume(mgr, tmp_path):
    mgr.try_proposal(move("p1", (1.0, 0, 0.5)))
    mgr.try_proposal(move("p2", (1.8, 0, 0.5)))
    back = mgr.rollback(1)
    assert back.revision == 3 and back.reconstruction.objects[0].rest.pose.translation[0] == pytest.approx(1.0)
    again = RunManager(tmp_path / "project.json", tmp_path / "run", dist, AcceptRule(primary="err"), max_attempts=6)
    assert again.attempts == 2 and again.adopted.revision == 3
