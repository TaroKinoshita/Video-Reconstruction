"""analysis/shape_library.py: categories from labels, templates filling the box."""

import numpy as np

from vrecon.analysis.shape_library import TEMPLATES, apply_templates, categorize, template_mesh
from vrecon.schema import FileRef, Project, Rational, SceneObject, Shot, SourceVideo


def _obj(i, **kw):
    return SceneObject(id=i, kind=kw.pop("kind", "rigid"), size=kw.pop("size", (0.5, 0.5, 0.9)), **kw)


def test_categories_from_labels():
    assert categorize(_obj("obj_chair")) == "chair"
    assert categorize(_obj("obj_obs_3"), label="folding stool") == "chair"
    assert categorize(_obj("obj_thing", name="Ball boy")) is None or categorize(_obj("obj_x", name="ball kid")) in ("ball", "person")
    assert categorize(_obj("obj_p", kind="character")) == "person"
    assert categorize(_obj("obj_obs_1")) is None


def test_templates_fill_the_box():
    for name in TEMPLATES:
        pass
    for cat, size in (("chair", (0.5, 0.45, 0.9)), ("bench", (1.8, 0.4, 0.45)), ("person", (0.5, 0.3, 1.7)),
                      ("person", (0.6, 0.6, 1.1)), ("pole", (0.1, 0.1, 2.0)), ("ball", (0.3, 0.3, 0.3))):
        v, f = template_mesh(cat, size)
        assert np.allclose(v.max(0) - v.min(0), size, atol=1e-6) and np.allclose((v.max(0) + v.min(0)) / 2, 0, atol=1e-6)
        assert f.max() < len(v)


def test_apply_templates_only_touches_box_objects(tmp_path):
    p = Project(id="proj_s", source=SourceVideo(file=FileRef(id="file_v", path="v.mp4"), fps=Rational(num=24, den=1),
                                                frame_count=4, width=32, height=32),
                shots=[Shot(id="shot_001", start=0, end=4, transition_from_previous="first")],
                reconstruction={"objects": [_obj("obj_chair"), _obj("obj_floor", kind="set", size=(5, 5, 0.1)),
                                            _obj("obj_mystery"),
                                            _obj("obj_bag", asset=FileRef(id="file_bag", path="bag.npz"))]})
    q, rep = apply_templates(p, tmp_path)
    assert rep == {"obj_chair": "chair", "obj_mystery": "unknown", "obj_bag": "kept (has a mesh)"}
    chair = next(o for o in q.reconstruction.objects if o.id == "obj_chair")
    assert chair.asset.path == "obj_chair_template.npz" and (tmp_path / chair.asset.path).is_file()
    assert chair.evidence[-1].kind == "rule"


def test_locked_object_keeps_its_shape(tmp_path):
    """A human lock on an object's asset or size stops the template stage (same rule as the repair loop)."""
    from vrecon.analysis.shape_library import apply_templates
    from vrecon.schema import Lock, Pose, SceneObject, TransformKey

    from test_builder import make_project

    ball = SceneObject(id="obj_ball", name="ball", size=(0.2, 0.2, 0.2), rest=TransformKey(time=0, pose=Pose()))
    from vrecon.schema import Camera, CameraKey, Intrinsics

    cam = Camera(id="cam_001", shot_id="shot_001", intrinsics=Intrinsics(width=64, height=64, fx=50, fy=50, cx=31.5, cy=31.5),
                 keys=[CameraKey(shot_frame=0, pose=Pose())])
    p = make_project([cam], [ball])
    p.human.locks.append(Lock(id="lock_ball", target_id="obj_ball", field="asset"))
    out, rep = apply_templates(p, tmp_path, labels={"obj_ball": "ball"})
    assert next(o for o in out.reconstruction.objects if o.id == "obj_ball").asset is None
    assert rep["obj_ball"].startswith("kept")
