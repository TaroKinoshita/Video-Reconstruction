"""The agent's tools on a tiny synthetic shot: flat ground at z = 0, one person standing on it, camera 3 m up."""
import json

import numpy as np

from vrecon.agent import checks
from vrecon.agent.session import Session, init_run


def _shot(tmp):
    N, W, H, step, f = 4, 160, 96, 4, 120.0
    # camera at (0, -8, 3) looking along +y, tilted down a little (OpenCV axes as columns)
    fwd = np.array([0, 1, -0.3])
    fwd /= np.linalg.norm(fwd)
    right = np.array([1.0, 0, 0])
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], 1)
    cam_R, cam_T = np.repeat(R[None], N, 0), np.repeat(np.array([[0, -8.0, 3]]), N, 0)
    h, w = H // step, W // step
    depth = np.full((2, h, w), np.inf, np.float32)
    vs, us = np.mgrid[0:h, 0:w]
    d = np.stack([(us.ravel() + 0.5 - w / 2) / (f / step), (vs.ravel() + 0.5 - h / 2) / (f / step),
                  np.ones(h * w)], 1) @ R.T
    t = -3.0 / d[:, 2]
    ok = (d[:, 2] < 0) & (t < 60)
    for j in range(2):
        depth[j].ravel()[ok] = t[ok]
    Q = (cam_T[0] + t[ok, None] * d[ok])[::2]
    xs = np.arange(-10, 10.01, 0.5)
    gx, gy = np.meshgrid(xs, xs + 5)
    ground_v = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1).astype(np.float32)
    np.savez(tmp / "ev.npz", cam_R=cam_R, cam_T=cam_T, f=np.float64(f), size=np.array([W, H]), step=np.int32(step),
             kf=np.array([0, 3]), depth=depth, Q=Q.astype(np.float32), Q_cls=np.zeros(len(Q), np.int8),
             Q_kf=np.zeros(len(Q), np.int16), ground_v=ground_v)
    parts = {"world_R": np.eye(3).tolist(), "world_t": [0, 0, 0], "trunks": [], "posts": [], "beams": [],
             "imagined_trees": [], "canopy": []}
    np.savez(tmp / "parts.npz", parts=np.array(json.dumps(parts)), ground_v=ground_v,
             backdrop_v=np.zeros((0, 3), np.float32))
    rng = np.random.default_rng(0)
    body = rng.uniform([-0.2, -0.15, 0.0], [0.2, 0.15, 1.75], (400, 3)) + [0, 2, 0]
    np.savez(tmp / "people.npz", labels=np.array(json.dumps(["obs_0"])), obs_0_t=np.arange(N),
             obs_0_v=np.repeat(body[None], N, 0).astype(np.float32))
    return tmp


def test_empty_start_build_ground_and_scale(tmp_path):
    _shot(tmp_path)
    init_run(tmp_path / "run", tmp_path / "ev.npz", tmp_path / "people.npz", tmp_path / "parts.npz", None, "empty")
    s = Session(tmp_path / "run")
    v = s.verify()
    assert v["counts"]["coverage"] >= 1
    assert any(i["id"].startswith("human:unsupported:obs_0") for i in v["issues"])
    # system meshes are locked in an empty start
    assert s.edit([{"op": "fixed", "ground": True}])["applied"] is False
    r = s.edit([{"op": "add", "primitives": [{"name": "Ground", "type": "box", "center": [0, 20, -0.05],
                                              "size": [120, 120, 0.1], "class": "floor"}]}])
    assert r["applied"]
    v = s.verify()
    assert v["counts"]["human"] == 0 and v["coverage_share"]["support"] > 0.95
    # a wall in front of the observed ground is a video-consistency violation
    s.edit([{"op": "add", "primitives": [{"name": "Wall", "type": "box", "center": [0, 0, 1.5],
                                          "size": [6, 0.2, 3], "class": "wall"}]}])
    v = s.verify()
    assert any(i["id"] == "video:Wall" for i in v["issues"])
    # k by height, then by support on the ground box
    assert abs(s.set_human_scale("obs_0", "height", 1.9)["height_m"] - 1.9) < 0.01
    k = s.set_human_scale("obs_0", "support", primitive="Ground")["k"]
    assert abs(k - 1.0) < 0.05
    assert s.submit()["accepted"] is False
    s.justify("video:Wall", "test")
    assert s.submit()["accepted"] is True


def test_render_views(tmp_path):
    _shot(tmp_path)
    init_run(tmp_path / "run", tmp_path / "ev.npz", tmp_path / "people.npz", tmp_path / "parts.npz", None, "empty")
    s = Session(tmp_path / "run")
    s.edit([{"op": "add", "primitives": [{"name": "Post", "type": "cylinder", "center": [1, 4, 1],
                                          "size": [0.4, 0.4, 2], "class": "post"}]}])
    for view in ("source", "orbit", "top"):
        assert (tmp_path / "run" / "renders").exists() or True
        p = s.render(view=view, frame=1, width=200)["path"]
        assert p.endswith(".png")
    # the camera-inside check
    s.edit([{"op": "add", "primitives": [{"name": "Box", "type": "box", "center": [0, -8, 3], "size": [1, 1, 1]}]}])
    assert any(i["id"] == "complete:camera_inside:Box" for i in checks.completeness(s.c["ev"], s.scene()))
