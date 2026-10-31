"""Project state: the four layers of PLAN.md section 5.1.

    observations    raw analysis results (never edited by hand)
    reconstruction  the currently adopted scene (NOT ground truth)
    human           human edits and locks (highest priority, never overwritten)
    artifacts       files produced from the state (.blend, renders, passes, videos)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from .common import (
    Evidence,
    FileRef,
    Id,
    Intrinsics,
    Model,
    Pose,
    Provenance,
    Rational,
    Scale,
    Status,
    Vec3,
    utcnow,
)

SCHEMA_VERSION = "0.1.0"


# --- source and shots ---------------------------------------------------------


class SourceVideo(Model):
    file: FileRef
    fps: Rational
    frame_count: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


Transition = Literal["first", "continuous", "time_jump", "same_moment_other_view", "unknown"]


class Shot(Model):
    """A cut. Source frame range is [start, end)."""

    id: Id
    start: int = Field(ge=0)
    end: int
    transition_from_previous: Transition = "unknown"

    @model_validator(mode="after")
    def _range(self) -> Shot:
        if self.end <= self.start:
            raise ValueError(f"shot {self.id}: end ({self.end}) must be > start ({self.start})")
        return self


# --- layer 1: observations ---------------------------------------------------


ObservationKind = Literal[
    "shot_boundaries", "camera", "depth", "mask", "point_track", "object", "human_motion", "other"
]


class Observation(Model):
    id: Id
    kind: ObservationKind
    shot_id: Id | None = None
    files: list[FileRef] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)  # small values only; bulk data goes in files
    provenance: Provenance


# --- layer 2: adopted reconstruction --------------------------------------------


class TransformKey(Model):
    """Object transform at an event time (scene_time, seconds)."""

    time: float
    pose: Pose
    scale: Vec3 = (1.0, 1.0, 1.0)


class SceneObject(Model):
    id: Id
    name: str = ""
    kind: Literal["rigid", "proxy", "character", "set"] = "proxy"
    asset: FileRef | None = None  # mesh etc.; None = builder makes a proxy shape
    size: Vec3 | None = None  # proxy box size in meters
    rest: TransformKey = Field(default_factory=lambda: TransformKey(time=0.0, pose=Pose()))
    keys: list[TransformKey] = Field(default_factory=list)
    status: Status = "estimated"
    evidence: list[Evidence] = Field(default_factory=list)


class CameraKey(Model):
    """Camera pose at a shot frame. Intrinsics override the camera default when zooming."""

    shot_frame: int = Field(ge=0)
    pose: Pose
    intrinsics: Intrinsics | None = None


class Camera(Model):
    id: Id
    shot_id: Id
    intrinsics: Intrinsics
    keys: list[CameraKey] = Field(min_length=1)
    status: Status = "estimated"
    evidence: list[Evidence] = Field(default_factory=list)


class Event(Model):
    id: Id
    label: str
    subjects: list[Id] = Field(default_factory=list)
    start: float  # scene_time seconds
    end: float
    status: Status = "estimated"
    evidence: list[Evidence] = Field(default_factory=list)


class ShotMapping(Model):
    """Which scene_time range a shot shows (shot_frame 0 maps to scene_time_start)."""

    shot_id: Id
    scene_time_start: float


class ShotOverride(Model):
    """Per-shot correction to the shared set (e.g. furniture moved for a reverse shot)."""

    id: Id
    shot_id: Id
    target_id: Id
    pose: Pose | None = None
    note: str = ""


class Reconstruction(Model):
    scale: Scale = Field(default_factory=Scale)
    objects: list[SceneObject] = Field(default_factory=list)
    cameras: list[Camera] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    shot_mapping: list[ShotMapping] = Field(default_factory=list)
    overrides: list[ShotOverride] = Field(default_factory=list)


# --- layer 3: human edits and locks -------------------------------------------


class HumanEdit(Model):
    """A change a human made. Recorded, but does not by itself forbid further AI changes."""

    id: Id
    target_id: Id
    field: str  # dotted path, e.g. "rest.pose.translation" or "keys"
    old: Any = None
    new: Any = None
    source: Literal["blender_ingest", "cli", "other"] = "cli"
    at: datetime = Field(default_factory=utcnow)


class Lock(Model):
    """Fields the AI must not change. field="*" locks the whole target."""

    id: Id
    target_id: Id
    field: str = "*"
    reason: str = ""
    at: datetime = Field(default_factory=utcnow)


class HumanLayer(Model):
    edits: list[HumanEdit] = Field(default_factory=list)
    locks: list[Lock] = Field(default_factory=list)


# --- layer 4: artifacts ---------------------------------------------------------


class Artifact(Model):
    id: Id
    kind: Literal["blend", "render", "control_pass", "generated_video", "comparison", "report", "other"]
    file: FileRef
    revision: int = Field(ge=0, description="project revision the artifact was made from")
    provenance: Provenance


# --- project root ---------------------------------------------------------------


class Project(Model):
    schema_version: str = SCHEMA_VERSION
    id: Id
    revision: int = Field(default=0, ge=0, description="increments on every adopted change")
    source: SourceVideo
    shots: list[Shot] = Field(min_length=1)
    observations: list[Observation] = Field(default_factory=list)
    reconstruction: Reconstruction = Field(default_factory=Reconstruction)
    human: HumanLayer = Field(default_factory=HumanLayer)
    artifacts: list[Artifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> Project:
        errors: list[str] = []
        rec = self.reconstruction

        # unique ids across the whole project
        seen: dict[str, str] = {}
        for kind, items in [
            ("shot", self.shots),
            ("observation", self.observations),
            ("object", rec.objects),
            ("camera", rec.cameras),
            ("event", rec.events),
            ("override", rec.overrides),
            ("edit", self.human.edits),
            ("lock", self.human.locks),
            ("artifact", self.artifacts),
        ]:
            for item in items:
                if item.id in seen:
                    errors.append(f"duplicate id {item.id!r} ({seen[item.id]} and {kind})")
                seen[item.id] = kind

        # shots inside the video, ordered, not overlapping
        prev_end = 0
        for shot in self.shots:
            if shot.end > self.source.frame_count:
                errors.append(f"shot {shot.id} ends after the last frame ({self.source.frame_count})")
            if shot.start < prev_end:
                errors.append(f"shot {shot.id} overlaps or is out of order")
            prev_end = shot.end

        shot_ids = {s.id for s in self.shots}
        obs_ids = {o.id for o in self.observations}
        target_ids = {o.id for o in rec.objects} | {c.id for c in rec.cameras} | {e.id for e in rec.events}

        def need_shot(ref: str | None, where: str) -> None:
            if ref is not None and ref not in shot_ids:
                errors.append(f"{where}: unknown shot {ref!r}")

        for obs in self.observations:
            need_shot(obs.shot_id, f"observation {obs.id}")
        for cam in rec.cameras:
            need_shot(cam.shot_id, f"camera {cam.id}")
            shot = next((s for s in self.shots if s.id == cam.shot_id), None)
            frames = [k.shot_frame for k in cam.keys]
            if frames != sorted(set(frames)):
                errors.append(f"camera {cam.id}: keys must have increasing, unique shot_frame")
            if shot and frames and frames[-1] >= shot.end - shot.start:
                errors.append(f"camera {cam.id}: key at shot_frame {frames[-1]} is outside shot {shot.id}")
        for obj in rec.objects:
            times = [k.time for k in obj.keys]
            if times != sorted(set(times)):
                errors.append(f"object {obj.id}: keys must have increasing, unique time")
        for ev in rec.events:
            if ev.end < ev.start:
                errors.append(f"event {ev.id}: end before start")
            for s in ev.subjects:
                if s not in target_ids:
                    errors.append(f"event {ev.id}: unknown subject {s!r}")
        for m in rec.shot_mapping:
            need_shot(m.shot_id, "shot_mapping")
        for ov in rec.overrides:
            need_shot(ov.shot_id, f"override {ov.id}")
            if ov.target_id not in target_ids:
                errors.append(f"override {ov.id}: unknown target {ov.target_id!r}")
        for items in (rec.objects, rec.cameras, rec.events):
            for item in items:
                for ev in item.evidence:
                    if ev.observation_id is not None and ev.observation_id not in obs_ids:
                        errors.append(f"{item.id}: evidence refers to unknown observation {ev.observation_id!r}")
                    need_shot(ev.shot_id, f"{item.id} evidence")
        for lock in self.human.locks:
            if lock.target_id not in target_ids:
                errors.append(f"lock {lock.id}: unknown target {lock.target_id!r}")
        for edit in self.human.edits:
            if edit.target_id not in target_ids:
                errors.append(f"edit {edit.id}: unknown target {edit.target_id!r}")
        for art in self.artifacts:
            if art.revision > self.revision:
                errors.append(f"artifact {art.id}: revision {art.revision} is newer than project {self.revision}")

        if errors:
            raise ValueError("project is inconsistent:\n  - " + "\n  - ".join(errors))
        return self

    # --- helpers ------------------------------------------------------------

    def is_locked(self, target_id: str, field: str = "*") -> bool:
        """True if the AI must not change `field` of `target_id`.

        Fields are dotted paths. A change and a lock conflict when one path contains the other:
        a lock on "rest.pose" blocks changing "rest.pose.translation" and also changing "rest".
        "*" means the whole target.
        """
        for lock in self.human.locks:
            if lock.target_id != target_id:
                continue
            a, b = lock.field, field
            if a == "*" or b == "*" or a == b or b.startswith(a + ".") or a.startswith(b + "."):
                return True
        return False
