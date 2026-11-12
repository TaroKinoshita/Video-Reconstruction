import json

import numpy as np
import pytest

from vrecon.analysis.triage import route
from vrecon.layers2d.estimate import estimate_layers, fit_keys
from vrecon.layers2d.model import ease, key, layer_state, render, states
from vrecon.layers2d.refine import refine_layers, silhouette_loss
from vrecon.schema import Layer2D, load_project


def test_easings_end_points():
    for name in ("linear", "ease_in", "ease_out", "ease_in_out"):
        assert float(ease(name, 0.0)) == pytest.approx(0.0)
        assert float(ease(name, 1.0)) == pytest.approx(1.0)
    assert float(ease("ease_in", 0.5)) < 0.5 < float(ease("ease_out", 0.5))
    assert float(ease("hold", 0.99)) == 0.0


def test_layer_state_interpolates_and_holds():
    lay = Layer2D(id="lay_a", size=(10, 10), keys=[key(1.0, 0, 0, easing="linear"), key(2.0, 10, 20, rotation=90)])
    assert layer_state(lay, 0.0)["x"] == 0  # before the first key: first key's values
    s = layer_state(lay, 1.5)
    assert (s["x"], s["y"], s["rotation"]) == pytest.approx((5, 10, 45))
    assert layer_state(lay, 9.0)["y"] == 20


def _sprite(w=40, h=24, seed=0):
    rng = np.random.default_rng(seed)
    rgb = rng.uniform(0.2, 1.0, (h // 4, w // 4, 3)).repeat(4, 0).repeat(4, 1)  # blocky texture: corners
    return np.dstack([rgb, np.ones((h, w))]).astype(np.float32)


def test_render_index_follows_z():
    a = Layer2D(id="lay_a", size=(40, 24), z=0, keys=[key(0, 50, 40)])
    b = Layer2D(id="lay_b", size=(40, 24), z=1, keys=[key(0, 60, 40)])
    _, idx = render([a, b], {"lay_a": _sprite(), "lay_b": _sprite(seed=1)}, 1, 24, (120, 80))
    assert idx[0, 40, 35] == 1 and idx[0, 40, 75] == 2 and idx[0, 40, 55] == 2  # overlap: b in front


def test_fit_keys_recovers_easing_and_timing():
    lay = Layer2D(id="lay_a", size=(10, 10), keys=[key(4 / 24, 10, 10, easing="ease_out"), key(20 / 24, 200, 60),
                                                    key(30 / 24, 200, 60, easing="ease_in_out"),
                                                    key(44 / 24, 200, 60, rotation=90)])
    keys = fit_keys(states(lay, 50, 24.0), 24.0)
    moves = [(round(a.time * 24), round(b.time * 24), a.easing) for a, b in zip(keys, keys[1:])
             if abs(a.x - b.x) + abs(a.rotation - b.rotation) > 1e-6]
    assert moves == [(4, 20, "ease_out"), (30, 44, "ease_in_out")]


def test_estimate_and_refine_on_a_small_clip():
    n, fps, size = 30, 24.0, (160, 100)
    truth = Layer2D(id="lay_a", size=(40, 24), keys=[key(5 / fps, 40, 50, easing="ease_in_out"),
                                                      key(25 / fps, 110, 55, rotation=30)])
    rng = np.random.default_rng(3)
    bg = np.clip(60 + rng.normal(0, 3, (size[1], size[0], 3)), 0, 255)
    frames, idx = render([truth], {"lay_a": _sprite()}, n, fps, size, background=bg)
    layers, sprites, _, _ = estimate_layers(frames, idx, {1: "lay_a"}, fps)
    got, want = states(layers[0], n, fps), states(truth, n, fps)
    assert np.abs(got[:, :2] - want[:, :2]).max() < 2.0
    d = got[:, 2] - want[:, 2]
    assert np.abs(d - np.median(d)).max() < 2.0
    before = silhouette_loss(layers, sprites, idx, {1: "lay_a"}, fps)
    refined, info = refine_layers(layers, sprites, idx, {1: "lay_a"}, fps, rounds=1, max_evals=60)
    assert info["loss_after"] <= before


def test_schema_0_1_0_projects_still_load(tmp_path):
    raw = {"schema_version": "0.1.0", "id": "proj_x",
           "source": {"file": {"id": "file_v", "path": "v.mp4"}, "fps": {"num": 24, "den": 1}, "frame_count": 10,
                      "width": 64, "height": 48},
           "shots": [{"id": "shot_001", "start": 0, "end": 10}]}
    p = tmp_path / "p.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    proj = load_project(p)
    assert proj.schema_version == "0.3.0" and proj.reconstruction.layers == []


def test_triage_route_rule():
    assert route({"frozen_pixels": 0.98}) == "undecided"
    assert route({"frozen_pixels": 0.4}) == "scene3d"
    assert route({"frozen_pixels": None}) == "unknown"
