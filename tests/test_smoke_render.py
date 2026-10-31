"""Environment smoke test: Blender builds, renders, saves and reloads a scene."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("bpy")

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_render(tmp_path):
    # Run in a subprocess: bpy keeps global state, so isolate it from other tests.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_render.py"), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("SMOKE_RESULT ")), None)
    assert line is not None, proc.stdout[-2000:] + proc.stderr[-2000:]
    result = json.loads(line.removeprefix("SMOKE_RESULT "))
    assert result["ok"], result
