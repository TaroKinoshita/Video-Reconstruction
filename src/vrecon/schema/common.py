"""Shared value types. Conventions: docs/CONVENTIONS.md."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

Vec3 = tuple[float, float, float]

ID_PREFIXES = ("obj_", "cam_", "shot_", "evt_", "obs_", "art_", "file_", "edit_", "lock_", "proj_", "ovr_", "lay_")
_ID_RE = re.compile(r"^[a-z]+_[A-Za-z0-9_\-]+$")


def _check_id(value: str) -> str:
    if not _ID_RE.match(value) or not value.startswith(ID_PREFIXES):
        raise ValueError(f"invalid id {value!r}: must start with one of {ID_PREFIXES}")
    return value


Id = Annotated[str, AfterValidator(_check_id)]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Model(BaseModel):
    """Base: unknown fields are errors (catches typos in saved data)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Rational(Model):
    """Frame rate as a fraction, e.g. 24000/1001."""

    num: int = Field(gt=0)
    den: int = Field(gt=0)

    def seconds(self, frames: float) -> float:
        return frames * self.den / self.num


def _check_quat(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(c * c for c in q))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"rotation must be a unit quaternion (w, x, y, z); norm={norm:.6f}")
    return q


Quat = Annotated[tuple[float, float, float, float], AfterValidator(_check_quat)]


class Pose(Model):
    """Rigid transform local->world. For cameras: OpenCV camera axes, camera-to-world."""

    translation: Vec3 = (0.0, 0.0, 0.0)
    rotation: Quat = (1.0, 0.0, 0.0, 0.0)


class Intrinsics(Model):
    """Pinhole intrinsics in pixels; pixel centers are integers (OpenCV)."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    distortion: list[float] | None = None  # OpenCV coefficient order; None = no distortion


class Provenance(Model):
    """Where a result came from (PLAN.md section 10)."""

    tool: str
    tool_version: str | None = None
    model: str | None = None
    weights: str | None = None
    license: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class FileRef(Model):
    """External file referenced by path (relative to the project root) and hash."""

    id: Id
    path: str
    sha256: str | None = None
    media_type: str | None = None

    @model_validator(mode="after")
    def _relative_path(self) -> FileRef:
        if self.path.startswith("/") or ".." in self.path.split("/"):
            raise ValueError(f"file path must be relative to the project root: {self.path!r}")
        return self


class Scale(Model):
    """How the metric scale was decided (CONVENTIONS.md section 1)."""

    source: Literal["assumed", "estimated", "human"] = "assumed"
    note: str = ""
    # scene units per metre, when known (1.0 = the scene is in metres). None = unknown: a scene from a 4D model
    # or an LLM is in its own units, and thresholds in metres must not be applied to it.
    units_per_metre: float | None = Field(default=None, gt=0)
    basis: str = ""  # what fixed it, e.g. "court lines (ITF)", "feet on the floor, body model height"


# How much the video supports a thing (an object, a camera, an event, a layer, or a span of an object's time):
#   observed         measured from the video (masks, points, lines) by a tool
#   estimated        derived from observations by a model or a fit; the default
#   standard         fixed by a published standard once the thing is recognised (court lines, net height)
#   hypothesis       not seen: placed by a rule, physics, an LLM or a generated picture; consistent with the
#                    video, not supported by it (was "assumed" up to schema 0.3.0)
#   human_confirmed  a person decided it
Status = Literal["observed", "estimated", "standard", "hypothesis", "human_confirmed"]

# What kind of support an Evidence entry is. None = not recorded (data from before schema 0.4.0).
EvidenceKind = Literal["observation", "standard", "rule", "physics", "llm", "generated_image", "human"]


class Evidence(Model):
    """Why a value was adopted: which observation and frames support it."""

    kind: EvidenceKind | None = None
    observation_id: Id | None = None
    shot_id: Id | None = None
    source_frames: list[int] = Field(default_factory=list)
    note: str = ""
