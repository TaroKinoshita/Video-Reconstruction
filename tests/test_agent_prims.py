import numpy as np

from vrecon.agent import checks, prims


def P(typ, c=(0, 0, 0), s=(2, 2, 2), r=(0, 0, 0)):
    return {"name": "a", "type": typ, "center": list(c), "size": list(s), "rotation": list(r)}


def test_ray_hits_front_faces():
    o = np.array([0, 0, -10.0])
    d = np.array([[0, 0, 1.0], [0.0, 0.5, 1.0], [0, 0, -1.0]])
    for typ in ("box", "cylinder", "ellipsoid"):
        t = prims.intersect(P(typ), o, d)
        assert abs(t[0] - 9.0) < 1e-6, typ
        assert np.isinf(t[2]), typ


def test_cylinder_axis_and_rotation():
    # cylinder along x after a 90 degree turn about y: a ray down the x axis hits the end cap
    p = P("cylinder", s=(1, 1, 4), r=(0, 90, 0))
    t = prims.intersect(p, np.array([-10.0, 0, 0]), np.array([[1.0, 0, 0]]))
    assert abs(t[0] - 8.0) < 1e-6
    t = prims.intersect(p, np.array([0, 0, 10.0]), np.array([[0, 0, -1.0]]))
    assert abs(t[0] - 9.5) < 1e-6


def test_inside_hits_at_zero_and_sdf_sign():
    p = P("box", s=(2, 4, 6))
    assert prims.intersect(p, np.zeros(3), np.array([[1.0, 0, 0]]))[0] == 0.0
    s = prims.sdf(p, np.array([[0, 0, 0], [3, 0, 0], [0, 0, 3.5]]))
    assert s[0] < 0 and abs(s[1] - 2) < 1e-9 and abs(s[2] - 0.5) < 1e-9
    c = P("cylinder", s=(2, 2, 2))
    assert abs(prims.sdf(c, np.array([[3.0, 0, 0]]))[0] - 2) < 1e-9


def test_top_at_and_euler_roundtrip():
    p = P("box", c=(0, 0, 1), s=(2, 2, 2))
    z = prims.top_at(p, np.array([[0, 0.0], [5, 5]]))
    assert abs(z[0] - 2) < 1e-6 and np.isnan(z[1])
    R = prims.rot({"rotation": [20, -35, 70]})
    assert np.allclose(prims.rot({"rotation": prims.euler_from(R)}), R)


def test_from_parts_names_and_beam_axis():
    parts = {"trunks": [{"x": 1, "y": 2, "base": 0, "r": 0.3, "h": 10, "colour_lin": [0.1, 0.1, 0.1]}],
             "posts": [], "imagined_trees": [], "canopy": [],
             "beams": [{"a": [0, 0, 2], "b": [4, 0, 2], "r": 0.2, "colour_lin": [0.2, 0.2, 0.2]}]}
    ps = prims.from_parts(parts)
    assert [p["name"] for p in ps] == ["Trunk_000", "Beam_000"]
    beam = ps[1]
    lo, hi = prims.aabb(beam)
    assert abs((hi - lo)[0] - 4) < 1e-6 and abs((hi - lo)[2] - 0.4) < 1e-6


def test_clusters_split_far_groups():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 0.2, (300, 3)), rng.normal(10, 0.2, (200, 3))])
    cs = checks.clusters(X, size=0.6, min_pts=100)
    assert [len(c) for c in cs] == [300, 200]
