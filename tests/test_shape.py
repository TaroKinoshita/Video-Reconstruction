"""Object shapes in the silhouette objective (optimize/shape.py): boxes unchanged, meshes drawn as meshes."""

import numpy as np

from vrecon.optimize import render_silhouettes
from vrecon.optimize.shape import BOX_FACES, BOX_U, object_shape, simplify
from vrecon.schema import Camera, CameraKey, FileRef, Intrinsics, Pose, SceneObject, TransformKey

from test_builder import make_project

LOOK = (0.7071068, 0.7071068, 0.0, 0.0)  # OpenCV camera looking along -Y


def _sphere(n=40):
    th, ph = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, 2 * n, endpoint=False), indexing="ij")
    v = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3) * 0.5
    f = []
    m = 2 * n
    for i in range(n - 1):
        for j in range(m):
            a, b, c, d = i * m + j, i * m + (j + 1) % m, (i + 1) * m + j, (i + 1) * m + (j + 1) % m
            f += [(a, c, b), (b, c, d)]
    return v, np.array(f)


def _scene(obj):
    intr = Intrinsics(width=200, height=200, fx=200, fy=200, cx=99.5, cy=99.5)
    cam = Camera(id="cam_001", shot_id="shot_001", intrinsics=intr,
                 keys=[CameraKey(shot_frame=0, pose=Pose(translation=(0, 4, 0), rotation=LOOK))])
    return make_project([cam], [obj], frames=1, size=(200, 200))


def test_box_is_default():
    o = SceneObject(id="obj_a", size=(1, 1, 1))
    u, f = object_shape(o)
    assert u is BOX_U and f is BOX_FACES


def test_mesh_silhouette_is_round(tmp_path):
    v, f = _sphere()
    np.savez(tmp_path / "ball.npz", verts=v, faces=f)
    box = SceneObject(id="obj_a", size=(1, 1, 1), rest=TransformKey(time=0, pose=Pose()))
    ball = box.model_copy(update={"asset": FileRef(id="file_ball", path=str(tmp_path / "ball.npz"))})
    a_box = (render_silhouettes(_scene(box), "shot_001", [0], ["obj_a"], (200, 200)) == 1).sum()
    a_ball = (render_silhouettes(_scene(ball), "shot_001", [0], ["obj_a"], (200, 200)) == 1).sum()
    assert abs(a_box / (200 / 3.5) ** 2 - 1) < 0.05  # the box's near face, 1 m at 3.5 m
    r = 200 * 0.5 / np.sqrt(4 ** 2 - 0.5 ** 2)  # a sphere's outline radius at 4 m
    assert abs(a_ball / (np.pi * r * r) - 1) < 0.05


def test_simplify_caps_faces():
    v, f = _sphere(80)
    v2, f2 = simplify(v, f, 600)
    assert len(f2) <= 600 and len(f2) > 100
    assert np.abs(v2.max(0) - v.max(0)).max() < 0.1
