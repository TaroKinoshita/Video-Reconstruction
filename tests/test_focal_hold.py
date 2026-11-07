"""A focal estimate needs a loss that rises on both sides."""

from vrecon.run.policy import focal_is_weak


def test_sharp_minimum_is_evidence():
    assert not focal_is_weak(lambda v: abs(v - 0.2), 0.2)


def test_flat_valley_is_not():
    assert focal_is_weak(lambda v: 0.05 + 1e-4 * abs(v - 0.2), 0.2)


def test_rise_on_one_side_only_is_not():
    assert focal_is_weak(lambda v: 0.05 + max(0.0, 0.2 - v), 0.2)
