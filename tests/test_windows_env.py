"""The repo lives under a non-ASCII path on Windows, and pytest's tmp_path usually does not. Cover that here."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from vrecon.control.core import encode_mp4, load_frames, read_video_rgb


def _frames() -> np.ndarray:
    frames = np.zeros((5, 32, 64, 3), np.uint8)
    frames[:, :, 32:] = 200
    return frames


def test_png_under_non_ascii_path(tmp_path):
    cv2 = pytest.importorskip("cv2")
    d = tmp_path / "日本語"
    d.mkdir()
    path = d / "a.png"
    ok, buf = cv2.imencode(".png", _frames()[0])
    assert ok
    buf.tofile(path)
    assert load_frames([path]).shape == (1, 32, 64, 3)


def test_video_under_non_ascii_path(tmp_path):
    path = encode_mp4(_frames(), tmp_path / "日本語" / "a.mp4")
    got = read_video_rgb(path)
    assert got.shape == (5, 32, 64, 3)
    assert abs(int(got[0, 16, 48, 0]) - 200) < 8  # right half stays bright, left half dark
    assert int(got[0, 16, 8, 0]) < 8


def test_unreadable_video_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        read_video_rgb(bad)


def test_track_points_under_non_ascii_path(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from vrecon.optimize.tracks import track_points

    base = (np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(cv2.resize(base, (128, 128)), (5, 5), 0)
    frames = np.stack([np.repeat(np.roll(base, i, axis=1)[..., None], 3, axis=-1) for i in range(12)])
    tracks = track_points(encode_mp4(frames, tmp_path / "日本語" / "t.mp4"))
    assert tracks.size == (128, 128)
    assert tracks.xy.shape[1] == 12


def test_render_device_choice():
    pytest.importorskip("bpy")
    import bpy

    from vrecon.scene.builder import set_cycles_device

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    assert set_cycles_device(scene, "cpu") == "CPU" and scene.cycles.device == "CPU"
    used = set_cycles_device(scene, "auto")  # a GPU when there is one, CPU otherwise; never an error
    assert (used == "CPU") == (scene.cycles.device == "CPU")
    with pytest.raises(ValueError):
        set_cycles_device(scene, "metal")
