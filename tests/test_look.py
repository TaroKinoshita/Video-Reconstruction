"""Looker prototype: issue words map onto the fixed hypotheses; bad input is refused."""

import pytest

from vrecon.run import look


def test_every_object_issue_has_a_hypothesis_name():
    from vrecon.run.policy import hypotheses  # noqa: F401  (names checked below)

    assert set(look._HYP) == set(look.ISSUES) - {"camera"}


def test_unknown_issue_is_refused(tmp_path):
    with pytest.raises(ValueError):
        look.fix(str(tmp_path), "too_purple", "obj_00")
