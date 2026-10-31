import json

import pytest
from pydantic import ValidationError

from vrecon.schema import (
    MIGRATIONS,
    SCHEMA_VERSION,
    Camera,
    CameraKey,
    FileRef,
    Intrinsics,
    Lock,
    Observation,
    Pose,
    Project,
    Provenance,
    Rational,
    SceneObject,
    SchemaVersionError,
    Shot,
    SourceVideo,
    TransformKey,
    json_schema,
    load_project,
    migrate,
    save_project,
)


def make_project() -> Project:
    intr = Intrinsics(width=640, height=360, fx=500, fy=500, cx=319.5, cy=179.5)
    return Project(
        id="proj_demo",
        source=SourceVideo(
            file=FileRef(id="file_src", path="input/clip.mp4"),
            fps=Rational(num=24, den=1),
            frame_count=48,
            width=640,
            height=360,
        ),
        shots=[Shot(id="shot_001", start=0, end=48, transition_from_previous="first")],
        observations=[
            Observation(id="obs_cam1", kind="camera", shot_id="shot_001", provenance=Provenance(tool="test"))
        ],
        reconstruction={
            "objects": [
                SceneObject(
                    id="obj_box",
                    size=(1, 1, 1),
                    keys=[
                        TransformKey(time=0.0, pose=Pose(translation=(0, 0, 0.5))),
                        TransformKey(time=1.0, pose=Pose(translation=(2, 0, 0.5))),
                    ],
                )
            ],
            "cameras": [
                Camera(
                    id="cam_001",
                    shot_id="shot_001",
                    intrinsics=intr,
                    keys=[CameraKey(shot_frame=0, pose=Pose(translation=(0, -5, 1)))],
                    evidence=[{"observation_id": "obs_cam1", "source_frames": [0, 10]}],
                )
            ],
        },
    )


def test_roundtrip(tmp_path):
    p = make_project()
    path = tmp_path / "project.json"
    save_project(p, path)
    loaded = load_project(path)
    assert loaded == p
    assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION


def test_unknown_field_rejected():
    data = make_project().model_dump()
    data["reconstruction"]["objects"][0]["colour"] = "red"
    with pytest.raises(ValidationError):
        Project.model_validate(data)


def test_bad_quaternion_rejected():
    with pytest.raises(ValidationError):
        Pose(rotation=(1, 1, 0, 0))


def test_bad_id_prefix_rejected():
    with pytest.raises(ValidationError):
        Shot(id="cut_001", start=0, end=10)


def test_relative_file_paths_only():
    with pytest.raises(ValidationError):
        FileRef(id="file_x", path="/etc/passwd")
    with pytest.raises(ValidationError):
        FileRef(id="file_x", path="../outside.mp4")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["shots"].append({"id": "shot_001", "start": 40, "end": 48}), "duplicate id"),
        (lambda d: d["shots"].__setitem__(0, {"id": "shot_001", "start": 0, "end": 99}), "ends after"),
        (lambda d: d["reconstruction"]["cameras"][0].__setitem__("shot_id", "shot_999"), "unknown shot"),
        (
            lambda d: d["reconstruction"]["cameras"][0]["keys"].append(
                {"shot_frame": 0, "pose": {"translation": [0, 0, 0]}}
            ),
            "increasing",
        ),
        (
            lambda d: d["reconstruction"]["cameras"][0]["keys"].append(
                {"shot_frame": 60, "pose": {"translation": [0, 0, 0]}}
            ),
            "outside shot",
        ),
        (
            lambda d: d["reconstruction"]["cameras"][0]["evidence"][0].__setitem__("observation_id", "obs_nope"),
            "unknown observation",
        ),
        (lambda d: d["human"]["locks"].append({"id": "lock_1", "target_id": "obj_ghost"}), "unknown target"),
    ],
)
def test_consistency_errors(mutate, message):
    data = make_project().model_dump()
    mutate(data)
    with pytest.raises(ValidationError, match=message):
        Project.model_validate(data)


def test_locks():
    p = make_project()
    p.human.locks.append(Lock(id="lock_1", target_id="obj_box", field="rest.pose"))
    assert p.is_locked("obj_box", "rest.pose")
    assert p.is_locked("obj_box", "rest.pose.translation")  # child of locked field
    assert p.is_locked("obj_box", "rest")  # parent contains locked field
    assert p.is_locked("obj_box", "*")
    assert not p.is_locked("obj_box", "keys")
    assert not p.is_locked("cam_001", "*")


def test_newer_schema_rejected():
    data = json.loads(make_project().model_dump_json())
    data["schema_version"] = "9.0.0"
    with pytest.raises(SchemaVersionError, match="newer"):
        migrate(data)


def test_migration_chain(monkeypatch):
    original = make_project()
    data = json.loads(original.model_dump_json())
    data["schema_version"] = "0.0.9"
    data.pop("revision")

    def old_to_current(raw):
        raw = dict(raw, schema_version=SCHEMA_VERSION)
        raw["revision"] = 0
        return raw

    monkeypatch.setitem(MIGRATIONS, "0.0.9", old_to_current)
    assert Project.model_validate(migrate(data)) == original


def test_missing_migration_rejected():
    data = json.loads(make_project().model_dump_json())
    data["schema_version"] = "0.0.1"
    with pytest.raises(SchemaVersionError, match="no migration"):
        migrate(data)


def test_failed_save_keeps_old_file(tmp_path):
    path = tmp_path / "project.json"
    save_project(make_project(), path)
    before = path.read_text()
    bad = make_project()
    bad.reconstruction.cameras[0].shot_id = "shot_999"  # passes field check, fails project check
    with pytest.raises(ValidationError):
        save_project(bad, path)
    assert path.read_text() == before
    assert list(tmp_path.iterdir()) == [path]  # no temp files left behind


def test_json_schema_exports():
    schema = json_schema()
    assert schema["title"] == "Project"
    assert "reconstruction" in schema["properties"]
