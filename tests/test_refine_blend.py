"""run/refine.py: flat static meshes become set pieces with the mask source's labels."""


from vrecon.schema import Pose, SceneObject, TransformKey

from vrecon.run import refine as rb


class _P:  # just what mark_set touches
    def __init__(self, objs):
        self.reconstruction = type("R", (), {"objects": objs})()


def _obj(oid, size, rot=(1.0, 0.0, 0.0, 0.0), keys=()):
    return SceneObject(id=oid, kind="proxy", size=size, rest=TransformKey(time=0, pose=Pose(rotation=rot)), keys=list(keys))


def test_floor_wall_and_movers():
    s = 0.5 ** 0.5
    objs = [_obj("obj_Ground", (20, 20, 0.01)), _obj("obj_Back", (10, 5, 0.01), rot=(s, s, 0, 0)),
            _obj("obj_Box", (1, 1, 1)),
            _obj("obj_Mover", (1, 1, 0.001), keys=[TransformKey(time=0, pose=Pose()), TransformKey(time=1, pose=Pose(translation=(1, 0, 0)))])]
    renamed = rb.mark_set(_P(objs))
    assert renamed == {"obj_Ground": "obj_floor", "obj_Back": "obj_wall"}
    assert [o.kind for o in objs] == ["set", "set", "proxy", "proxy"]  # a moving flat thing stays a mover
