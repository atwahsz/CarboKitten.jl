from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Boundary(str, Enum):
    """Boundary condition for 2D grids.

    The Julia code supports multiple boundary traits. For the first Python port we
    implement the one used by the ALCAP example: `COAST`.

    - COAST: reflect in x-direction, periodic in y-direction.
    - PERIODIC: periodic in both directions.
    - REFLECTED: reflected in both directions.

    Notes
    - All indexing in this Python port is 0-based.
    """

    COAST = "coast"
    PERIODIC = "periodic"
    REFLECTED = "reflected"


def _reflect_index(i: int, n: int) -> int:
    """Reflect index i into [0, n-1] (period 2n)."""

    if n <= 0:
        raise ValueError("n must be positive")
    period = 2 * n
    m = i % period
    return m if m < n else period - m - 1


def bounded_index(boundary: Boundary, i: int, n: int) -> int:
    """Map an index to a valid index under a boundary."""

    if boundary is Boundary.PERIODIC:
        return i % n
    if boundary is Boundary.REFLECTED:
        return _reflect_index(i, n)
    # COAST: reflect in x, periodic in y, so this helper is used per-axis.
    return _reflect_index(i, n)


@dataclass(frozen=True, slots=True)
class Box:
    """Spatial grid description.

    Parameters
    - grid_size: (nx, ny)
    - phys_scale_m: grid cell size in meters.
    - boundary: boundary condition (default: COAST).
    """

    grid_size: tuple[int, int]
    phys_scale_m: float
    boundary: Boundary = Boundary.COAST

    def axes_m(self) -> tuple[np.ndarray, np.ndarray]:
        """Return x and y axes in meters."""

        nx, ny = self.grid_size
        x = np.arange(nx, dtype=np.float64) * self.phys_scale_m
        y = np.arange(ny, dtype=np.float64) * self.phys_scale_m
        return x, y
