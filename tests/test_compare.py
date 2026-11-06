import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def put(root, cond, clip, att, iou, rep, valid=True, cost=None):
    d = root / cond / clip / att
    d.mkdir(parents=True)
    (d / "score.json").write_text(json.dumps({"summary": {"mask_iou_mean": iou, "reprojection_error_mean": rep}}))
    if cost is not None or not valid:
        (d / "run.json").write_text(json.dumps({"valid": valid, "agent": {"result": {"total_cost_usd": cost,
                                                                                      "duration_ms": 60000}}}))


def test_compare_report(tmp_path):
    r = tmp_path / "res"
    for att, (b_rep, c_rep) in enumerate([(0.02, 0.01), (0.03, 0.012), (0.025, 0.011)]):
        put(r, "B", "syn_01", f"a{att}", 0.6, b_rep, cost=1.0)
        put(r, "C", "syn_01", f"a{att}", 0.8, c_rep)
    put(r, "B", "syn_02", "a0", 0.7, 0.01, cost=2.0)
    put(r, "C", "syn_02", "a0", 0.6, 0.02)
    put(r, "B", "syn_02", "a1", 0.1, 0.9, valid=False)  # invalid run must be ignored
    out = subprocess.run([sys.executable, str(ROOT / "scripts/compare_conditions.py"), str(r)],
                         capture_output=True, encoding="utf-8", errors="replace", check=True).stdout
    assert "C better on 1/2 clips" in out  # primary: reprojection (syn_01 yes, syn_02 no)
    assert "B: 5 (invalid 1)" in out
    assert "| syn_01 | 0.025 | 0.011 | yes |" in out
