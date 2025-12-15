from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class WaveModel(Protocol):
    """Wave velocity/shear model.

    The Julia API is `wave_velocity(w) -> (v::Vec2, s::Vec2)` where `s` is the
    derivative of `v` w.r.t. water depth.

    In this Python port we return *fields* evaluated on the full grid for speed.

    Required units
    - v: m/Myr
    - s: 1/Myr
    """

    def fields(self, water_depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (vx, vy, sx, sy), each shaped (nx, ny)."""


@dataclass(frozen=True, slots=True)
class ZeroWave:
    """Default wave model: v=0, s=0."""

    def fields(self, water_depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = np.zeros_like(water_depth_m, dtype=np.float64)
        return z, z, z, z


@dataclass(frozen=True, slots=True)
class ConstantWave:
    """Constant wave velocity everywhere (shear=0)."""

    vx_m_myr: float
    vy_m_myr: float

    def fields(self, water_depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vx = np.full_like(water_depth_m, float(self.vx_m_myr), dtype=np.float64)
        vy = np.full_like(water_depth_m, float(self.vy_m_myr), dtype=np.float64)
        z = np.zeros_like(water_depth_m, dtype=np.float64)
        return vx, vy, z, z
