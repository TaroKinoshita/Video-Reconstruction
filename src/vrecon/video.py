"""Bring a reference video into a new project (PLAN.md 4.2, first step). No bpy.

    probe        ffprobe: size, frame rate (as a fraction), frame count, rotation, variable frame rate
    normalize    copy as is, or transcode to constant frame rate with rotation applied (CONVENTIONS.md 5.1)
    detect_shots PySceneDetect content detector -> shots [start, end) in source frames
    new_project  all of the above -> project.json + input/ folder
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from vrecon.schema import FileRef, Observation, Project, Provenance, Rational, Shot, SourceVideo, save_project


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    avg_fps: Fraction
    frames: int
    rotation: int  # degrees, as stored in the file's display matrix
    codec: str
    variable_frame_rate: bool  # judged from the actual frame timestamps, not container metadata


def _fraction(text: str) -> Fraction:
    num, _, den = text.partition("/")
    return Fraction(int(num), int(den or 1))


def _uneven_timestamps(path: str | Path, fps: Fraction, tolerance: float = 0.25) -> bool:
    """True if frame spacing deviates from 1/fps by more than `tolerance` of a frame anywhere.

    Container frame-rate fields can be misleading (e.g. Matroska reports a nominal rate), so this reads
    the packet timestamps themselves.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
         "-of", "csv=p=0", str(path)],
        capture_output=True, encoding="utf-8", errors="replace", check=True,
    )
    times = sorted(float(t) for t in out.stdout.split() if t not in ("", "N/A"))
    if len(times) < 3:
        return False
    step = 1 / float(fps)
    return any(abs((b - a) - step) > tolerance * step for a, b in zip(times, times[1:]))


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,codec_name:stream_tags=rotate"
         ":stream_side_data=rotation", "-of", "json", str(path)],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    if out.returncode != 0:
        raise ValueError(f"not a readable video: {path}: {out.stderr.strip()[-300:]}")
    streams = json.loads(out.stdout).get("streams") or []
    if not streams:
        raise ValueError(f"no video stream in {path}")
    s = streams[0]
    rotation = 0
    for sd in s.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    if "tags" in s and "rotate" in s["tags"]:
        rotation = int(s["tags"]["rotate"])
    fps = _fraction(s["r_frame_rate"])
    avg = s.get("avg_frame_rate", "0/0")
    avg_fps = _fraction(avg) if not avg.endswith("/0") else fps
    return VideoInfo(
        width=int(s["width"]),
        height=int(s["height"]),
        fps=fps,
        avg_fps=avg_fps,
        frames=int(s["nb_read_frames"]),
        rotation=rotation % 360,
        codec=s.get("codec_name", ""),
        variable_frame_rate=_uneven_timestamps(path, fps),
    )


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(src: Path, dst_dir: Path) -> tuple[Path, dict]:
    """Copy the video into dst_dir, or transcode it when it has a variable frame rate or a rotation flag."""
    info = probe(src)
    dst_dir.mkdir(parents=True, exist_ok=True)
    record = {"original_name": src.name, "original_sha256": sha256(src), "original_fps": str(info.fps),
              "original_avg_fps": str(info.avg_fps), "rotation": info.rotation}
    if not info.variable_frame_rate and info.rotation == 0:
        dst = dst_dir / f"source{src.suffix.lower()}"
        shutil.copyfile(src, dst)
        record["transcoded"] = False
        return dst, record
    fps = Fraction(info.fps).limit_denominator(1001)
    dst = dst_dir / "source.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-fps_mode", "cfr", "-r", str(fps),
           "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)  # ffmpeg applies the rotation flag while decoding
    record.update({"transcoded": True, "reason": "variable frame rate" if info.variable_frame_rate else "rotation",
                   "target_fps": str(fps), "ffmpeg_args": cmd[4:-1]})
    return dst, record


def detect_shots(path: str | Path, frames: int, threshold: float = 27.0, min_len: int = 15) -> tuple[list[tuple[int, int]], dict]:
    """Shot boundaries as [start, end) source frames, covering the whole video."""
    import scenedetect
    from scenedetect import ContentDetector, detect

    scenes = detect(str(path), ContentDetector(threshold=threshold, min_scene_len=min_len))
    def frame_of(tc):
        return tc.frame_num if hasattr(tc, "frame_num") else tc.get_frames()

    cuts = sorted({frame_of(s[0]) for s in scenes} - {0})
    bounds = [0] + [c for c in cuts if 0 < c < frames] + [frames]
    shots = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
    params = {"detector": "ContentDetector", "threshold": threshold, "min_scene_len": min_len,
              "version": scenedetect.__version__}
    return shots, params


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_").lower()
    return s or "video"


def new_project(video: str | Path, out_dir: str | Path, project_id: str | None = None,
                shots: bool = True, threshold: float = 27.0) -> Project:
    video, out = Path(video), Path(out_dir)
    if (out / "project.json").exists():
        raise FileExistsError(f"{out / 'project.json'} already exists")
    src, record = normalize(video, out / "input")
    info = probe(src)
    width, height = info.width, info.height
    if info.rotation in (90, 270):  # only possible if copied; normalize() bakes rotation otherwise
        width, height = height, width
    fps = Fraction(info.fps).limit_denominator(1001)

    observations = []
    if shots:
        bounds, params = detect_shots(src, info.frames, threshold=threshold)
        observations.append(Observation(
            id="obs_shots_001", kind="shot_boundaries",
            summary={"shots": [list(b) for b in bounds], "cuts": [b[0] for b in bounds[1:]]},
            provenance=Provenance(tool="pyscenedetect", tool_version=params["version"], params=params),
        ))
    else:
        bounds = [(0, info.frames)]

    project = Project(
        id=project_id or f"proj_{_slug(video.stem)}",
        source=SourceVideo(
            file=FileRef(id="file_source", path=src.relative_to(out).as_posix(), sha256=sha256(src)),
            fps=Rational(num=fps.numerator, den=fps.denominator),
            frame_count=info.frames,
            width=width,
            height=height,
        ),
        shots=[
            Shot(id=f"shot_{i:03d}", start=a, end=b, transition_from_previous="first" if i == 1 else "unknown")
            for i, (a, b) in enumerate(bounds, start=1)
        ],
        observations=observations,
    )
    (out / "input" / "import.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    save_project(project, out / "project.json")
    return project
