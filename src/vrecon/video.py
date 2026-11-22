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
        variable_frame_rate=_uneven_timestamps(path, fps if abs(float(avg_fps - fps)) <= 0.01 * float(fps)
                                               else avg_fps),
    )


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_rate(info: VideoInfo) -> Fraction:
    """The rate frames actually come at. r_frame_rate can be a multiple of it (tennis: 60/1 while 312 frames span
    10.46 s, avg 29.83): when the two differ by more than 1%, the average rate is used."""
    if abs(float(info.avg_fps - info.fps)) > 0.01 * float(info.fps):
        return Fraction(info.avg_fps).limit_denominator(1001)
    return Fraction(info.fps).limit_denominator(1001)


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
    fps = frame_rate(info)
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


def frame_changes(frames) -> tuple[list[float], list[float], list[float]]:
    """Per frame (vs the previous one): colour histogram change (0..2), mean pixel change (0..255), mean brightness."""
    import cv2
    import numpy as np

    small = [cv2.resize(np.asarray(f), (160, 90), interpolation=cv2.INTER_AREA) for f in frames]
    hists = []
    for f in small:
        h = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_RGB2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
        hists.append(h / max(h.sum(), 1.0))
    hist_d, pix_d = [0.0], [0.0]
    for i in range(1, len(small)):
        hist_d.append(float(np.abs(hists[i] - hists[i - 1]).sum()))
        pix_d.append(float(np.abs(small[i].astype(np.float32) - small[i - 1]).mean()))
    return hist_d, pix_d, [float(f.mean()) for f in small]


def detect_shots_fine(frames, hist_cut: float = 0.9, pix_cut: float = 25.0, transition_max: int = 3
                      ) -> tuple[list[tuple[int, int]], dict]:
    """Shot boundaries for edited footage (films): a cut where both the colour histogram and the pixels change at once
    (one test alone fires on fast motion or on lighting); no minimum shot length, so cuts a few frames apart are kept.
    A run of at most `transition_max` frames between two cuts (a white flash, a whip blur, a one-frame dissolve) is a
    transition, not a shot: it is listed in params["transitions"] and still covered by the returned list, so the shots
    tile the video. Returns ([start, end) per shot, params)."""
    hist_d, pix_d, lum = frame_changes(frames)
    n = len(hist_d)
    cuts = [i for i in range(1, n) if hist_d[i] >= hist_cut and pix_d[i] >= pix_cut]
    bounds = [0] + cuts + [n]
    shots = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
    transitions = [(a, b) for a, b in shots if b - a <= transition_max and 0 < a and b < n]
    params = {"detector": "fine (histogram + pixel change)", "hist_cut": hist_cut, "pix_cut": pix_cut,
              "transition_max": transition_max, "transitions": transitions,
              "cut_scores": {int(i): [round(hist_d[i], 2), round(pix_d[i], 1)] for i in cuts}}
    return shots, params


STANDARD_RATES = (Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30))


@dataclass
class CadenceInfo:
    period: int  # one repeated frame every `period` frames
    phase: int  # repeated frames are those with index % period == phase
    repeats: list[int]  # indices of the frames found to be copies of the one before
    share: float  # share of repeat candidates that fall on this phase
    frames: int = 0  # length of the clip

    @property
    def phase_frames(self) -> list[int]:
        """Every frame on the cadence's phase. Removing all of them (not only those found to be copies) keeps the
        rest evenly spaced; on tennis 54 of the 62 phase frames were found to be copies, the other 8 differ from
        their predecessor by about as much as ordinary frames (re-encoding)."""
        return [i for i in range(1, self.frames) if i % self.period == self.phase]


def detect_cadence(video, ratio: float = 0.35, min_share: float = 0.7) -> CadenceInfo | None:
    """Frame-rate padding (e.g. 24 -> 30 fps: every 5th frame a copy of the one before). video: [T, H, W(, C)]
    uint8. A frame is a repeat candidate when its difference from the previous one is under `ratio` times the
    median difference around it; it is a cadence only when at least `min_share` of the candidates fall on one
    phase of one period (4-6 frames) and that phase holds at least half its frames: a nearly still shot has small
    differences everywhere but no cadence. None: no cadence."""
    import numpy as np

    g = np.asarray(video).astype(np.int16)
    t_n = len(g)
    d = np.r_[np.inf, [float(np.abs(g[i] - g[i - 1]).mean()) for i in range(1, t_n)]]
    cand = []
    for i in range(1, t_n):
        lo, hi = max(1, i - 6), min(t_n, i + 7)
        if d[i] < ratio * float(np.median(d[lo:hi])):
            cand.append(i)
    if len(cand) < 3:
        return None
    best = None
    for per in (4, 5, 6):
        phases = np.bincount(np.array(cand) % per, minlength=per)
        ph = int(np.argmax(phases))
        share = phases[ph] / len(cand)
        if share >= min_share and phases[ph] >= 0.5 * len(range(ph, t_n, per)) and (best is None or share > best[2]):
            best = (per, ph, share)
    if best is None:
        return None
    return CadenceInfo(period=best[0], phase=best[1], repeats=[i for i in cand if i % best[0] == best[1]],
                       share=float(best[2]), frames=t_n)


def frame_times(video, fps: float, ratio: float = 0.35, min_share: float = 0.7):
    """Seconds per frame of a clip that may still contain padded frames: repeats get their predecessor's time, the
    other frames are spread evenly over the clip (detect_cadence). Without a cadence: i / fps."""
    import numpy as np

    t_n = len(video)
    rep = np.zeros(t_n, bool)
    cad = detect_cadence(video, ratio, min_share)
    if cad is not None:
        rep[cad.repeats] = True
    k = np.cumsum(~rep) - 1
    return k * ((t_n / fps) / max(int((~rep).sum()), 1))


def remove_padding(src: Path, dst_dir: Path, fps: Fraction) -> tuple[Path, CadenceInfo, list[int], Fraction] | None:
    """When src has a cadence: write the de-duplicated clip at the true rate (fps * kept / total) as
    dst_dir/source_dedup.mp4 and return it with the cadence, the kept original frames and the new rate."""
    import numpy as np

    from vrecon.control.core import encode_mp4, read_video_rgb

    frames = read_video_rgb(src)
    gray = frames.mean(axis=3).astype(np.uint8)
    cad = detect_cadence(gray)
    if cad is None:
        return None
    rep = set(cad.phase_frames)
    kept = [i for i in range(len(frames)) if i not in rep]
    new_fps = (fps * Fraction(len(kept), len(frames))).limit_denominator(1001)
    # a padded clip was made from a standard rate; container rates of padded files are often slightly off
    # (tennis: 29.827 fps -> 23.86), so snap to a standard rate within 1%
    near = min(STANDARD_RATES, key=lambda r: abs(float(r - new_fps)))
    if abs(float(near - new_fps)) < 0.01 * float(near):
        new_fps = near
    dst = dst_dir / "source_dedup.mp4"
    encode_mp4(frames[kept], dst, fps=f"{new_fps.numerator}/{new_fps.denominator}", crf=12)
    return dst, cad, kept, new_fps


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_").lower()
    return s or "video"


def new_project(video: str | Path, out_dir: str | Path, project_id: str | None = None,
                shots: bool = True, threshold: float = 27.0, dedup: bool = True) -> Project:
    """dedup: when the video repeats frames in a regular cadence (frame-rate padding), the project's source becomes
    the de-duplicated clip at the true rate and SourceVideo.cadence keeps the original and the frame map; observers
    then run on that clip (docs/REVIEW_INSTRUCTIONS.md F-22)."""
    from vrecon.schema import Cadence

    video, out = Path(video), Path(out_dir)
    if (out / "project.json").exists():
        raise FileExistsError(f"{out / 'project.json'} already exists")
    src, record = normalize(video, out / "input")
    info = probe(src)
    width, height = info.width, info.height
    if info.rotation in (90, 270):  # only possible if copied; normalize() bakes rotation otherwise
        width, height = height, width
    fps = frame_rate(info)
    cadence = None
    if dedup:
        found = remove_padding(src, out / "input", fps)
        if found is not None:
            dst, cad, kept, new_fps = found
            cadence = Cadence(original=FileRef(id="file_original", path=src.relative_to(out).as_posix(),
                                               sha256=sha256(src)),
                              original_fps=Rational(num=fps.numerator, den=fps.denominator),
                              original_frame_count=info.frames, period=cad.period, phase=cad.phase, kept_frames=kept)
            record["dedup"] = {"period": cad.period, "phase": cad.phase, "removed": len(cad.phase_frames),
                               "found_copies": len(cad.repeats),
                               "share": cad.share, "fps": f"{new_fps.numerator}/{new_fps.denominator}"}
            src, fps = dst, new_fps
            info = probe(src)

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
            cadence=cadence,
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
