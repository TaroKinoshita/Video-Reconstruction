import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("select_movers", Path(__file__).parents[1] / "gpu/observe/select_movers.py")
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)

T, H, W = 10, 40, 40


def box(x0, x1, frames=range(T)):
    m = np.zeros((T, H, W), bool)
    for t in frames:
        m[t, 10:20, x0:x1] = True
    return m


def test_two_separate_beats_one_merged():
    merged = sm.score_candidate({1: box(5, 30)})
    apart = sm.score_candidate({1: box(5, 12), 2: box(20, 27)})
    assert sm.best([("frame0", merged), ("frame5", apart)])[0] == "frame5"


def test_merged_track_that_splits_or_stops_loses_on_quality():
    merged = box(5, 12) | box(20, 27, frames=range(5, T))  # one track that becomes two pieces
    bad = sm.score_candidate({1: merged, 2: box(30, 36, frames=range(6))})
    good = sm.score_candidate({1: box(5, 12), 2: box(20, 27)})
    assert bad["tracks"] == good["tracks"] == 2
    assert sm.best([("bad", bad), ("good", good)])[0] == "good"


def test_short_huge_and_duplicate_tracks_are_not_counted():
    full = np.ones((T, H, W), bool)
    s = sm.score_candidate({1: box(5, 12), 2: box(5, 12), 3: box(20, 27, frames=range(3)), 4: full})
    assert s["tracks"] == 1 and s["kept"] == [1]


def test_tie_goes_to_steadier_then_first():
    jumpy = box(5, 12)
    jumpy[1::2] = np.roll(jumpy[1::2], 6, axis=2)
    steady, shaky = sm.score_candidate({1: box(5, 12)}), sm.score_candidate({1: jumpy})
    assert sm.best([("a", shaky), ("b", steady)])[0] == "b"
    assert sm.best([("a", steady), ("b", steady)])[0] == "a"
