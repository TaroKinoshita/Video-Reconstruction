"""One time base (docs/REVIEW_INSTRUCTIONS.md WP-2): padded frames are removed at import, frames turn into seconds
in one place."""

from __future__ import annotations

import numpy as np
import pytest

from vrecon.schema import Rational, frame_time, time_frame
from vrecon.schema.project import source_frames_of
from vrecon.video import detect_cadence, frame_times


def _clip(n: int = 48, h: int = 96, w: int = 128) -> np.ndarray:
    """A textured square sliding left to right: every frame differs from the one before."""
    rng = np.random.default_rng(0)
    bg = rng.integers(40, 90, (h, w, 3), dtype=np.uint8)
    tex = rng.integers(150, 255, (24, 24, 3), dtype=np.uint8)
    out = []
    for i in range(n):
        f = bg.copy()
        x = 4 + int(i * (w - 32) / n)
        f[36:60, x:x + 24] = tex
        out.append(f)
    return np.stack(out)


def _pad_24_to_30(frames: np.ndarray) -> np.ndarray:
    """Every 4 frames get a copy of the 4th: 24 -> 30 fps padding (one repeat in every 5 frames)."""
    out = []
    for i, f in enumerate(frames):
        out.append(f)
        if i % 4 == 3:
            out.append(f.copy())
    return np.stack(out)


def test_detect_cadence_finds_the_padding():
    padded = _pad_24_to_30(_clip())
    assert len(padded) == 60
    cad = detect_cadence(padded.mean(axis=3).astype(np.uint8))
    assert cad is not None and cad.period == 5
    assert len(padded) - len(cad.repeats) == 48
    assert detect_cadence(_clip().mean(axis=3).astype(np.uint8)) is None


def test_frame_times_give_repeats_their_predecessors_time():
    padded = _pad_24_to_30(_clip())
    t = frame_times(padded.mean(axis=3).astype(np.uint8), fps=30.0)
    d = np.diff(t)
    assert np.sum(d == 0) == 12 and np.all(d >= 0)


def test_new_project_removes_padding(tmp_path):
    pytest.importorskip("cv2")
    from vrecon.control.core import FPS as H3_FPS
    from vrecon.control.core import encode_mp4
    from vrecon.video import new_project

    padded = _pad_24_to_30(_clip())
    video = encode_mp4(padded, tmp_path / "padded.mp4", fps=30, crf=12)
    p = new_project(video, tmp_path / "proj", shots=False)
    c = p.source.cadence
    assert c is not None and c.period == 5 and c.original_frame_count == 60
    assert p.source.frame_count == 48 == len(c.kept_frames)
    assert all(b > a for a, b in zip(c.kept_frames, c.kept_frames[1:]))
    # 30 * 48 / 60 = 24: the H3 rate, so an H3 export needs no resampling
    assert (p.source.fps.num, p.source.fps.den) == (H3_FPS, 1)
    times = [frame_time(p, f) for f in range(p.source.frame_count)]
    assert all(b > a for a, b in zip(times, times[1:]))
    assert time_frame(p, times[10]) == pytest.approx(10)
    # per-frame data made on the original padded clip maps onto the kept frames; the repeats drop out
    m = source_frames_of(p.source, 60)
    assert sum(x is None for x in m) == 12 and [x for x in m if x is not None] == list(range(48))
    assert source_frames_of(p.source, 48) == list(range(48))

    plain = new_project(encode_mp4(_clip(), tmp_path / "plain.mp4", fps=24), tmp_path / "proj2", shots=False)
    assert plain.source.cadence is None and plain.source.frame_count == 48


def test_rational_helpers():
    r = Rational(num=24000, den=1001)
    assert r.value == pytest.approx(23.976, abs=1e-3)
    assert r.frames(r.seconds(37)) == pytest.approx(37)
