import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("scenedetect")

from vrecon.schema import load_project  # noqa: E402
from vrecon.video import new_project, probe  # noqa: E402


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def three_shot_video(path: Path, fps="24") -> Path:
    """Three 1-second shots with very different content."""
    parts = []
    for i, src in enumerate(["testsrc2=size=320x176", "mandelbrot=size=320x176", "smptebars=size=320x176"]):
        part = path.parent / f"part{i}.mp4"
        ffmpeg("-f", "lavfi", "-i", f"{src}:rate={fps}", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(part))
        parts.append(part)
    listing = path.parent / "list.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(path))
    return path


def test_probe_rational_fps(tmp_path):
    v = tmp_path / "a.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x176:rate=24000/1001", "-t", "1", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", str(v))
    info = probe(v)
    assert (info.width, info.height) == (320, 176)
    assert str(info.fps) == "24000/1001"
    assert info.frames == 24
    assert not info.variable_frame_rate


def test_new_project_detects_shots(tmp_path):
    video = three_shot_video(tmp_path / "clip.mp4")
    project = new_project(video, tmp_path / "proj")
    assert project.id == "proj_clip"
    assert project.source.frame_count == 72
    assert (project.source.fps.num, project.source.fps.den) == (24, 1)
    assert [(s.start, s.end) for s in project.shots] == [(0, 24), (24, 48), (48, 72)]
    assert project.shots[0].transition_from_previous == "first"
    obs = project.observations[0]
    assert obs.kind == "shot_boundaries" and obs.provenance.tool == "pyscenedetect"

    loaded = load_project(tmp_path / "proj" / "project.json")
    assert loaded == project
    assert (tmp_path / "proj" / project.source.file.path).is_file()
    record = json.loads((tmp_path / "proj" / "input" / "import.json").read_text())
    assert record["transcoded"] is False


def test_new_project_without_shot_detection(tmp_path):
    video = three_shot_video(tmp_path / "clip.mp4")
    project = new_project(video, tmp_path / "proj", project_id="proj_custom", shots=False)
    assert project.id == "proj_custom"
    assert [(s.start, s.end) for s in project.shots] == [(0, 72)]
    assert project.observations == []


def test_variable_frame_rate_is_transcoded(tmp_path):
    cfr = tmp_path / "cfr.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x176:rate=30", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           str(cfr))
    vfr = tmp_path / "vfr.mkv"  # drop every third frame and keep timestamps -> uneven frame spacing
    ffmpeg("-i", str(cfr), "-vf", "select='not(eq(mod(n\\,3)\\,2))'", "-fps_mode", "vfr", "-c:v", "libx264",
           str(vfr))
    assert probe(vfr).variable_frame_rate
    project = new_project(vfr, tmp_path / "proj", shots=False)
    record = json.loads((tmp_path / "proj" / "input" / "import.json").read_text())
    assert record["transcoded"] is True and record["reason"] == "variable frame rate"
    assert not probe(tmp_path / "proj" / project.source.file.path).variable_frame_rate


def test_refuses_to_overwrite(tmp_path):
    video = three_shot_video(tmp_path / "clip.mp4")
    new_project(video, tmp_path / "proj", shots=False)
    with pytest.raises(FileExistsError):
        new_project(video, tmp_path / "proj", shots=False)
