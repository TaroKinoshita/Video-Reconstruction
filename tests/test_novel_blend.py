"""Novel views scored from the .blend as it is (evaluate.novel_from_blend)."""

from pathlib import Path

import pytest

bpy = pytest.importorskip("bpy")

from vrecon.evaluate import novel_from_blend  # noqa: E402
from vrecon.scene.builder import build_scene, save_blend  # noqa: E402
from vrecon.schema import load_project  # noqa: E402

CLIP = Path("evaluation/synthetic_v2/syn2_02")


@pytest.fixture
def gt_blend(tmp_path):
    gt = load_project(CLIP / "gt.json")
    build_scene(gt, root=CLIP)
    return save_blend(tmp_path / "gt.blend"), {o.id: o.id for o in gt.reconstruction.objects}


def test_ground_truth_scores_one(gt_blend):
    b, ids = gt_blend
    s = novel_from_blend(CLIP, b, ids)["summary"]
    assert s["novel_iou_10"] == pytest.approx(1.0, abs=1e-3) and s["novel_iou_30"] == pytest.approx(1.0, abs=1e-3)


def test_moving_the_whole_scene_up_changes_nothing(gt_blend, tmp_path):
    b, ids = gt_blend
    bpy.ops.wm.open_mainfile(filepath=str(Path(b).resolve()))
    rig = bpy.data.objects.new("lift", None)
    bpy.context.scene.collection.objects.link(rig)
    for o in list(bpy.data.objects):
        if o.parent is None and o is not rig:
            o.parent = rig
    rig.location.z = 2.0
    moved = tmp_path / "up.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(moved.resolve()))
    r = novel_from_blend(CLIP, moved, ids)
    assert r["floors"]["rec"] == pytest.approx(2.0, abs=0.05)
    assert r["summary"]["novel_iou_10"] == pytest.approx(1.0, abs=1e-3)
