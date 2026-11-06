"""End-to-end through the vrecon command: handoff -> edit in Blender -> ingest."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("bpy")

from vrecon.schema import load_project, save_project  # noqa: E402

from test_readback import make_project  # noqa: E402


def run(*args, check=True):
    proc = subprocess.run([sys.executable, "-m", "vrecon.cli", *args], capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    if check and proc.returncode != 0:
        raise AssertionError(proc.stdout + proc.stderr)
    return proc


def test_cli_handoff_edit_ingest(tmp_path: Path):
    project_path = tmp_path / "project.json"
    save_project(make_project(), project_path)
    out = tmp_path / "handoff"

    run("handoff", str(project_path), "--out", str(out))
    blend = out / "scene.blend"
    assert blend.is_file() and (out / "handoff.json").is_file()

    # A "human" edits the file in a separate Blender process.
    edit = textwrap.dedent(
        f"""
        import bpy
        bpy.ops.wm.open_mainfile(filepath={str(blend)!r})
        o = next(o for o in bpy.data.objects if o.get("vrecon_id") == "obj_static")
        o.location.x = 2.0
        bpy.ops.wm.save_mainfile()
        """
    )
    subprocess.run([sys.executable, "-c", edit], check=True, capture_output=True, timeout=600)

    dry = run("ingest", str(project_path), "--handoff", str(out / "handoff.json"), "--dry-run")
    assert "obj_static.rest.pose.translation" in dry.stdout
    assert load_project(project_path).revision == 0  # dry run saved nothing

    real = run("ingest", str(project_path), "--handoff", str(out / "handoff.json"))
    assert "revision 1" in real.stdout
    saved = load_project(project_path)
    assert saved.revision == 1
    assert saved.reconstruction.objects[0].rest.pose.translation[0] == pytest.approx(2.0)

    # Ingesting the same handoff again is refused: the project moved on.
    again = run("ingest", str(project_path), "--handoff", str(out / "handoff.json"), check=False)
    assert again.returncode == 2 and "changed since handoff" in again.stderr
