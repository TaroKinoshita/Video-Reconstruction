"""Project state types (four layers), schema version, load/save.

See docs/PLAN.md section 5 and docs/CONVENTIONS.md.
"""

from .common import Evidence, FileRef, Intrinsics, Pose, Provenance, Rational, Scale
from .io import MIGRATIONS, SchemaVersionError, json_schema, load_project, migrate, save_project
from .project import (
    SCHEMA_VERSION,
    Artifact,
    Cadence,
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
    StatusSpan,
    TransformKey,
    frame_time,
    time_frame,
)

__all__ = [
    "SCHEMA_VERSION", "MIGRATIONS", "SchemaVersionError",
    "Artifact", "Cadence", "Camera", "CameraKey", "Event", "Evidence", "FileRef", "HumanEdit", "HumanLayer",
    "Intrinsics", "Layer2D", "LayerKey", "Lock", "Observation", "Pose", "Project", "Provenance", "Rational",
    "Reconstruction", "Scale", "SceneObject", "Shot", "ShotMapping", "ShotOverride",
    "SourceVideo", "StatusSpan", "TransformKey",
    "frame_time", "json_schema", "load_project", "time_frame", "migrate", "save_project",
]
