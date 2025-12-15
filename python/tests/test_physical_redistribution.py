from __future__ import annotations

import numpy as np

from carbokitten.box import Boundary
from carbokitten.redistribution import redistribute_3x3


def test_redistribution_conserves_mass_periodic() -> None:
    # One facies, small grid
    den = np.zeros((1, 5, 4), dtype=np.float64)
    den[0, 2, 1] = 10.0

    # water depth gradient
    w = np.arange(5 * 4, dtype=np.float64).reshape((5, 4))

    out = redistribute_3x3(denudation_mass_m=den, water_depth_m=w, dx_m=1.0, boundary=Boundary.PERIODIC)
    assert np.isclose(out.sum(), den.sum())


def test_redistribution_drops_mass_outside_coast_x() -> None:
    # Put mass at left edge; some weights point out of bounds in x
    den = np.zeros((1, 3, 3), dtype=np.float64)
    den[0, 0, 1] = 10.0

    w = np.zeros((3, 3), dtype=np.float64)
    w[0, :] = 0.0
    w[1, :] = 1.0
    w[2, :] = 2.0

    out = redistribute_3x3(denudation_mass_m=den, water_depth_m=w, dx_m=1.0, boundary=Boundary.COAST)

    # COAST ignores targets outside x; so output mass can be <= input mass
    assert out.sum() <= den.sum() + 1e-12
