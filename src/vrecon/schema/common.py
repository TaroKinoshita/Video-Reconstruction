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


Status = Literal["estimated", "assumed", "human_confirmed"]


class Evidence(Model):
    """Why a value was adopted: which observation and frames support it."""

    observation_id: Id | None = None
    shot_id: Id | None = None
    source_frames: list[int] = Field(default_factory=list)
    note: str = ""
