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
from .models_alcap import ALCAPInput, ALCAPModel
from .models_cap_bs92 import BS92Input, BS92Model, CAPInput, CAPModel
from .models_with_denudation import WithDenudationInput, WithDenudationModel
from .time import TimeProperties
from .run_model import run_model

__all__ = [
    "Boundary",
    "Box",
    "ALCAPInput",
    "ALCAPModel",
    "CAPInput",
    "CAPModel",
    "BS92Input",
    "BS92Model",
    "WithDenudationInput",
    "WithDenudationModel",
    "TimeProperties",
    "run_model",
]
