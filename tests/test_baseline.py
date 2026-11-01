"""Baseline-B post-processing: run record, leak check, scoring of the agent's scene.blend."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

bpy = pytest.importorskip("bpy")

from test_metrics import _llm_like_blend, reference  # noqa: E402,F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]


def fake_run(tmp_path: Path, log_lines: list[dict]) -> Path:
    run = tmp_path / "run"
    (run / "workspace" / "output").mkdir(parents=True)
    (run / "logs").mkdir()
    (run / "logs" / "agent.jsonl").write_text("\n".join(json.dumps(x) for x in log_lines) + "\n")
    (run / "logs" / "blender_version.txt").write_text("Blender 5.2.3\n")
    _llm_like_blend(run / "workspace" / "output" / "scene.blend")
    return run


def finish(run: Path, clip: Path):
    return subprocess.run(
        [sys.executable, str(ROOT / "baselines/b_llm_mcp/finish_run.py"), "--run", str(run), "--clip", str(clip),
         "--model", "test-model", "--mcp-version", "2.2.6", "--start", "s", "--end", "e", "--exit-code", "0",
         "--minutes", "60", "--turns", "200"],
        capture_output=True, text=True, timeout=600,
    )


LOG = [
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "mcp__blender__get_scene_info"}]}},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "mcp__blender__execute_blender_code"},
                                                  {"type": "tool_use", "name": "mcp__blender__execute_blender_code"}]}},
    {"type": "result", "subtype": "success", "is_error": False, "num_turns": 7, "duration_ms": 1234,
     "total_cost_usd": 0.5, "usage": {"input_tokens": 10}},
]


def test_finish_records_and_scores(tmp_path, reference):  # noqa: F811
    run = fake_run(tmp_path, LOG)
    proc = finish(run, reference)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((run / "run.json").read_text())
    assert record["valid"] and record["ground_truth_leak"] == []
    assert record["agent"]["tool_calls"] == {"mcp__blender__get_scene_info": 1, "mcp__blender__execute_blender_code": 2}
    assert record["agent"]["result"]["num_turns"] == 7
    assert record["deliverables"]["scene.blend"] and not record["deliverables"]["report.md"]
    score = json.loads((run / "score.json").read_text())
    assert score["summary"]["mask_iou_mean"] > 0.99


def test_finish_flags_ground_truth_leak(tmp_path, reference):  # noqa: F811
    leaky = LOG + [{"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Read", "input": {"file_path": "/x/evaluation/synthetic/syn_01/gt.json"}}]}}]
    run = fake_run(tmp_path, leaky)
    proc = finish(run, reference)
    record = json.loads((run / "run.json").read_text())
    assert record["valid"] is False and "gt.json" in record["ground_truth_leak"]
    assert "INVALID RUN" in proc.stderr
