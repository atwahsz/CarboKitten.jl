"""carbokitten: a high-performance Python port of CarboKitten.jl.

This port focuses on:
- Fast array-based simulation using NumPy + Numba.
- Stateless, testable kernels for the heavy inner loops.

Units
- All values are stored as plain floats in SI-like units:
  - length: meters (m)
  - time: mega-years (Myr)
  - rates: meters per mega-year (m/Myr)
  - intensity: W/m^2

This is a work in progress.
"""

from .box import Box, Boundary
from .time import TimeProperties
from .run_model import run_model

__all__ = [
    "Boundary",
    "Box",
    "TimeProperties",
    "run_model",
]
