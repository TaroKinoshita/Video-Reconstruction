import numpy as np

from vrecon.analysis.proxies import board_proxy, near_band, plane_proxy


def test_plane_proxy_recovers_tilted_stands():
    rng = np.random.default_rng(0)
    # stands: 10 wide, rising 3 over 4 deep (37 deg back), seen from a camera at y = -10
    s, h = rng.uniform(-5, 5, 20000), rng.uniform(0, 1, 20000)
    p = np.c_[s, 5 + 4 * h, 3 * h] + rng.normal(0, 0.02, (20000, 3))
    c, q, size, r = plane_proxy(p, eye=np.array([0.0, -10.0, 1.5]))
    assert abs(r["tilt_deg"] - np.degrees(np.arctan2(3, 4))) < 2
    assert abs(size[0] - 10) < 0.6 and abs(size[1] - 5) < 0.4


def test_board_and_band():
    rng = np.random.default_rng(1)
    p = np.c_[rng.uniform(-3, 3, 5000), rng.normal(0, 0.01, 5000), rng.uniform(0, 1, 5000)]
    far = np.c_[rng.uniform(-3, 3, 500), np.full(500, 20.0), rng.uniform(0, 1, 500)]
    kept = near_band(np.r_[p, far], np.array([0.0, -10.0, 1.0]))
    assert (kept[:, 1] < 1).all()
    _, _, size, _ = board_proxy(kept)
    assert abs(size[0] - 6) < 0.4 and size[1] < 0.1 and abs(size[2] - 1) < 0.1


def test_ground_box_from_mask():
    from vrecon.analysis.proxies import ground_box

    # camera 1.5 up at y = -10 looking along +y (OpenCV: x right, y down, z forward)
    r = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
    c, k = np.array([0.0, -10.0, 1.5]), np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    # a person 0.5 wide, 1.8 tall standing at (1, 0): its front face projected
    def px(x, y, z):
        q = r.T @ (np.array([x, y, z]) - c)
        return 320 + 500 * q[0] / q[2], 240 + 500 * q[1] / q[2]
    (u0, v1), (u1, v0) = px(0.75, 0, 0), px(1.25, 0, 1.8)
    mask = np.zeros((480, 640), bool)
    mask[int(v0):int(v1), int(u0):int(u1)] = True
    centre, yaw, size = ground_box(mask, r, c, k, depth_ratio=0.0)
    assert abs(centre[0] - 1) < 0.05 and abs(centre[1]) < 0.1
    assert abs(size[0] - 0.5) < 0.05 and abs(size[2] - 1.8) < 0.08


def test_split_planes_two_walls():
    from vrecon.analysis.planes import split_planes

    rng = np.random.default_rng(2)
    back = np.c_[rng.uniform(-3, 3, 3000), 5 + rng.normal(0, 0.02, 3000), rng.uniform(0, 2, 3000)]
    side = np.c_[-3 + rng.normal(0, 0.02, 1500), rng.uniform(1, 5, 1500), rng.uniform(0, 2, 1500)]
    noise = rng.uniform([-3, 0, 0], [3, 5, 2], (300, 3))
    pieces, rest = split_planes(np.r_[back, side, noise], tol=0.08, cell=0.5)
    assert len(pieces) == 2
    normals = sorted(tuple(np.round(np.abs(p["normal"]), 1)) for p in pieces)
    assert normals == [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    assert rest[-300:].mean() > 0.8


def test_seated_figure_and_steps():
    from vrecon.analysis.proxies import seated_figure, stepped_mesh

    v, f = seated_figure(0.5, 0.6, 1.3)
    assert abs(v[:, 2].min() + 0.65) < 1e-9 and abs(v[:, 2].max() - 0.65) < 1e-9 and f.max() < len(v)
    # stands rising 35 deg, 10 wide, 8 up the slope: steps of about 0.45 m
    t = np.radians(35)
    a, b = np.array([1.0, 0, 0]), np.array([0, np.cos(t), np.sin(t)])
    r = np.c_[a, b, np.cross(a, b)]
    v, f, rows = stepped_mesh(r, (10.0, 8.0, 0.1), 0.45)
    assert rows == round(8 * np.sin(t) / 0.45)
    w = v @ r.T  # local -> world offsets
    assert abs(np.ptp(w[:, 2]) - 8 * np.sin(t)) < 1e-6


def test_enclose_pushes_copies_outside():
    from vrecon.run.complete import enclose
    from vrecon.schema import Pose, Project, Rational, SceneObject, Shot, SourceVideo, TransformKey
    from vrecon.schema import FileRef

    wall = SceneObject(id="obj_wall", kind="set", size=(10.0, 0.1, 3.0),
                       rest=TransformKey(time=0.0, pose=Pose(translation=(0.0, 5.0, 1.5))))
    p = Project(id="proj_t", source=SourceVideo(file=FileRef(id="file_v", path="v.mp4"), fps=Rational(num=24, den=1),
                                                frame_count=10, width=64, height=64),
                shots=[Shot(id="shot_001", start=0, end=10, transition_from_previous="first")],
                reconstruction={"objects": [wall]})
    # the camera stands at y = -8: the 180 deg copy (y = -5) would stand in front of it; it must go behind
    q, r = enclose(p, (0.0, 0.0, 0.0), ["obj_wall"], contain=[np.array([0.0, -8.0, 1.5])], margin=1.0)
    back = next(o for o in q.reconstruction.objects if o.id == "obj_wall_encl180")
    assert back.rest.pose.translation[1] <= -9.0 + 1e-6 and back.status == "hypothesis"
    side = next(o for o in q.reconstruction.objects if o.id == "obj_wall_encl090")
    assert abs(side.rest.pose.translation[0] + 5.0) < 1e-6  # turned +90: the wall at +y goes to -x


def test_camera_moves():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from camera_moves import apply_move

    r = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # looking along +y
    c, pivot = np.array([0.0, -10.0, 1.5]), np.zeros(3)
    r2, c2 = apply_move(r, c, "orbit90", 1.0, pivot, 1.0)
    assert np.allclose(c2, [10.0, 0.0, 1.5]) and np.allclose(r2[:, 2], [-1.0, 0, 0])
    _, c3 = apply_move(r, c, "dolly50", 1.0, pivot, 1.0)
    assert np.allclose(c3, [0.0, -5.0, 1.5])
    r4, c4 = apply_move(r, c, "crane5", 1.0, pivot, 1.0)
    assert np.allclose(c4, [0.0, -10.0, 6.5]) and r4[2, 2] < 0  # risen and looking down
    r5, _ = apply_move(r, c, "crane5", 0.0, pivot, 1.0)
    assert np.allclose(r5, r)
