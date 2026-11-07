"""Timing searched together with position must escape a minimum that position alone created."""

import numpy as np

from vrecon.optimize import Param, _scan_time


def test_scan_finds_shift_that_needs_a_position_refit():
    params = [Param("time", "o", step=4.0), Param("translation", "o", 0, 0.25)]

    def loss(x):  # best at shift -4, position +1; at shift 0, position 0 is a local minimum for each parameter alone
        shift, pos = x
        return min(abs(shift + 4) + abs(pos - 1), 1.0 + 3 * abs(shift) + 3 * abs(pos))

    x, fx, evals = _scan_time(loss, params, np.array([4.0, 0.25]), 40)
    assert x[0] == -4 and abs(x[1] - 1) < 1e-6 and fx < 1e-6
    assert evals > 0
