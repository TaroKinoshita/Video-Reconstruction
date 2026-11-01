"""handoff / ingest entry points that touch Blender and files (PLAN.md 7.2.1)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from vrecon.schema import Project

from .builder import build_scene, save_blend
from .ingest import HandoffSnapshot, IngestResult, ingest
from .readback import file_sha256, read_blend


def _write_json_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def handoff(project: Project, out_dir: str | Path, shot_id: str | None = None) -> HandoffSnapshot:
    """Build the shot, save `scene.blend`, and save the baseline readout to `handoff.json`."""
    out = Path(out_dir)
    report = build_scene(project, shot_id)
    blend = save_blend(out / "scene.blend")
    readout = read_blend(blend)  # read the saved file itself, so the baseline matches exactly
    snapshot = HandoffSnapshot(
        project_id=project.id,
        revision=project.revision,
        shot_id=report.shot_id,
        blend_path=str(blend),
        blend_sha256=file_sha256(blend),
        readout=readout,
    )
    _write_json_atomic(out / "handoff.json", snapshot.model_dump_json(indent=2))
    return snapshot


def load_snapshot(path: str | Path) -> HandoffSnapshot:
    return HandoffSnapshot.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def ingest_blend(project: Project, snapshot: HandoffSnapshot, blend_path: str | Path | None = None) -> IngestResult:
    """Read the edited .blend and diff it against the snapshot. Does not save anything."""
    blend = Path(blend_path or snapshot.blend_path)
    return ingest(project, snapshot, read_blend(blend))
