import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vrecon.control.core import (
    canny_control,
    check_h3_constraints,
    depth_control,
    encode_mp4,
    probe_frames,
    snap_h3_frames,
)

ROOT = Path(__file__).resolve().parents[1]

try:
    import bpy  # noqa: F401

    HAS_BPY = True
except ImportError:
    HAS_BPY = False


def test_snap_and_constraints():
    assert snap_h3_frames(124) == 124
    assert snap_h3_frames(120) == 107
    assert snap_h3_frames(1000) == 243
    assert snap_h3_frames(2) == 5
    assert check_h3_constraints(1280, 704, 124, 24) == []
    probs = check_h3_constraints(1280, 720, 120, 30)
    assert len(probs) == 3


def test_depth_near_is_white_background_black():
    d = np.zeros((2, 4, 4), np.float32)
    d[:, :2, :] = 2.0  # near
    d[:, 2:, :2] = 8.0  # far
    # [:, 2:, 2:] stays 0 = background
    out, info = depth_control(d, mode="clip", lo_pct=0, hi_pct=100)
    assert out.shape == (2, 4, 4, 3) and out.dtype == np.uint8
    assert out[0, 0, 0, 0] == 255  # nearest -> white
    assert out[0, 3, 0, 0] == 0  # farthest geometry -> black
    assert out[0, 3, 3, 0] == 0  # background -> black (treated as far)
    assert info["mode"] == "clip"


def test_depth_clip_mode_keeps_brightness_meaning():
    d = np.full((2, 4, 4), 5.0, np.float32)
    d[1] = 3.0  # everything closer in frame 1
    clip, _ = depth_control(d + np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4), mode="clip")
    frame, _ = depth_control(d + np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4), mode="frame")
    assert clip[1].mean() > clip[0].mean()  # closer = brighter across frames
    assert abs(int(frame[1].mean()) - int(frame[0].mean())) <= 1  # per-frame normalisation hides it


def test_canny_white_edges():
    frames = np.zeros((1, 32, 32, 3), np.uint8)
    frames[0, 8:24, 8:24] = 200
    out, info = canny_control(frames)
    assert out.max() == 255 and out[0, 0, 0, 0] == 0
    assert info == {"low": 100, "high": 200}


def test_encode_roundtrip(tmp_path):
    frames = np.random.default_rng(0).integers(0, 255, (22, 64, 128, 3), dtype=np.uint8)
    path = encode_mp4(frames, tmp_path / "x.mp4")
    assert probe_frames(path) == 22


def make_jobs(tmp_path: Path, frames: int) -> Path:
    ctl = encode_mp4(np.zeros((frames, 64, 128, 3), np.uint8), tmp_path / "controls" / "a" / "control_depth.mp4")
    jobs = {
        "defaults": {"sample_size": [64, 128], "num_inference_steps": 2, "guidance_scale": 1.0,
                     "control_context_scale": 1.0, "seeds": [1, 2], "negative_prompt": ""},
        "jobs": [{"id": "a__depth", "scene": "a", "control": str(ctl.relative_to(tmp_path)),
                  "control_kind": "depth", "prompt": "x", "group": "main"}],
    }
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(jobs))
    return path


def run_dry(jobs: Path, out: Path):
    return subprocess.run(
        [sys.executable, str(ROOT / "gpu/h3_control/run_jobs.py"), "--jobs", str(jobs), "--out", str(out), "--dry-run"],
        capture_output=True, encoding="utf-8", errors="replace", timeout=120,
    )


def test_gpu_runner_dry_run_ok(tmp_path):
    proc = run_dry(make_jobs(tmp_path, 22), tmp_path / "out")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "2 runs planned" in proc.stdout and "a__depth__s1" in proc.stdout


def test_gpu_runner_dry_run_flags_bad_frames(tmp_path):
    proc = run_dry(make_jobs(tmp_path, 20), tmp_path / "out")
    assert proc.returncode == 1
    assert "not 17n+5" in proc.stdout


def test_gpu_runner_skips_finished(tmp_path):
    jobs = make_jobs(tmp_path, 22)
    done = tmp_path / "out" / "a__depth__s1"
    done.mkdir(parents=True)
    (done / "video.mp4").write_bytes(b"x")
    proc = run_dry(jobs, tmp_path / "out")
    assert "1 already done, 1 to run" in proc.stdout


@pytest.mark.skipif(not HAS_BPY, reason="needs bpy")
def test_export_controls_small_scene(tmp_path):
    from vrecon.schema import (Camera, CameraKey, FileRef, Intrinsics, Pose, Project, Rational, SceneObject, Shot,
                               SourceVideo, TransformKey)
    from vrecon.scene.controls import export_controls
    from vrecon.scene.convert import look_at_opencv

    intr = Intrinsics(width=128, height=64, fx=100, fy=100, cx=63.5, cy=31.5)
    eye = (0.0, -6.0, 0.0)
    project = Project(
        id="proj_ctl",
        source=SourceVideo(file=FileRef(id="file_src", path="a.mp4"), fps=Rational(num=24, den=1), frame_count=22,
                           width=128, height=64),
        shots=[Shot(id="shot_001", start=0, end=22)],
        reconstruction={
            "cameras": [Camera(id="cam_001", shot_id="shot_001", intrinsics=intr,
                               keys=[CameraKey(shot_frame=0, pose=Pose(translation=eye,
                                                                        rotation=look_at_opencv(eye, (0, 0, 0))))])],
            "objects": [SceneObject(id="obj_box", size=(1, 1, 1),
                                    rest=TransformKey(time=0, pose=Pose(translation=(0, 0, 0))))],
        },
    )
    m = export_controls(project, tmp_path, preview=False)
    assert m["h3_problems"] == []
    assert set(m["controls"]) == {"depth", "canny"}
    for f in m["controls"].values():
        assert probe_frames(tmp_path / f["path"]) == 22
    depth = np.load(tmp_path / "passes/depth.npz")["depth"]
    index = np.load(tmp_path / "passes/index.npz")["index"]
    assert depth.shape == (22, 64, 128)
    assert float(depth[0, 32, 64]) == pytest.approx(5.5, abs=0.01)  # box front face is 5.5 m from the camera
    assert m["passes"]["id_map"][str(int(index[0, 32, 64]))] == "obj_box"
    assert index[0, 0, 0] == 0 and depth[0, 0, 0] == 0  # background
