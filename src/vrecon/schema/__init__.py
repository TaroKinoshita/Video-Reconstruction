"""Project state types (four layers), schema version, load/save.

See docs/PLAN.md section 5 and docs/CONVENTIONS.md.
"""

from .common import Evidence, FileRef, Intrinsics, Pose, Provenance, Rational, Scale
from .io import MIGRATIONS, SchemaVersionError, json_schema, load_project, migrate, save_project
from .project import (
    SCHEMA_VERSION,
    Artifact,
    Camera,
    CameraKey,
    Event,
    HumanEdit,
    HumanLayer,
    Layer2D,
    LayerKey,
    Lock,
    Observation,
    Project,
    Reconstruction,
    SceneObject,
    Shot,
    ShotMapping,
    ShotOverride,
    SourceVideo,
    TransformKey,
)

__all__ = [
    "SCHEMA_VERSION", "MIGRATIONS", "SchemaVersionError",
    "Artifact", "Camera", "CameraKey", "Event", "Evidence", "FileRef", "HumanEdit", "HumanLayer",
    "Intrinsics", "Layer2D", "LayerKey", "Lock", "Observation", "Pose", "Project", "Provenance", "Rational",
    "Reconstruction", "Scale", "SceneObject", "Shot", "ShotMapping", "ShotOverride",
    "SourceVideo", "TransformKey",
    "json_schema", "load_project", "migrate", "save_project",
]
