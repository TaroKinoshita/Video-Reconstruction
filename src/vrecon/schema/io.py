"""Load and save project state with version checks and migrations (PLAN.md section 10)."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .project import SCHEMA_VERSION, Project

# Migrations: from_version -> function(raw dict) -> raw dict at the next version.
# Add one entry per schema change, e.g. MIGRATIONS["0.1.0"] = _0_1_0_to_0_2_0.
def _0_1_0_to_0_2_0(raw: dict[str, Any]) -> dict[str, Any]:
    return {**raw, "schema_version": "0.2.0"}  # reconstruction.layers added, defaults to []


def _0_2_0_to_0_3_0(raw: dict[str, Any]) -> dict[str, Any]:
    return {**raw, "schema_version": "0.3.0"}  # SceneObject.look added, defaults to {}


MIGRATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {"0.1.0": _0_1_0_to_0_2_0, "0.2.0": _0_2_0_to_0_3_0}


class SchemaVersionError(ValueError):
    pass


def _parse(version: str) -> tuple[int, int, int]:
    try:
        major, minor, patch = (int(p) for p in version.split("."))
    except ValueError as exc:
        raise SchemaVersionError(f"bad schema_version {version!r}") from exc
    return major, minor, patch


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade raw project data to SCHEMA_VERSION, or raise SchemaVersionError."""
    version = raw.get("schema_version")
    if not isinstance(version, str):
        raise SchemaVersionError("missing schema_version")
    if _parse(version) > _parse(SCHEMA_VERSION):
        raise SchemaVersionError(
            f"data is schema {version}, newer than this code ({SCHEMA_VERSION}); update vrecon"
        )
    while version != SCHEMA_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            raise SchemaVersionError(f"no migration from schema {version} to {SCHEMA_VERSION}")
        raw = step(raw)
        new_version = raw.get("schema_version")
        if not isinstance(new_version, str) or _parse(new_version) <= _parse(version):
            raise SchemaVersionError(f"migration from {version} did not advance the version")
        version = new_version
    return raw


def load_project(path: str | os.PathLike[str]) -> Project:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Project.model_validate(migrate(raw))


def save_project(project: Project, path: str | os.PathLike[str]) -> None:
    """Validate, then write atomically (a crash never leaves a half-written file)."""
    project = Project.model_validate(project.model_dump())  # re-run cross-field checks
    data = project.model_dump_json(indent=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def json_schema() -> dict[str, Any]:
    """JSON Schema of the project file, for other languages and LLM tools."""
    return Project.model_json_schema()
