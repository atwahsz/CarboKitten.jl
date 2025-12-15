from __future__ import annotations

import math

import numpy as np
from numba import njit

from .box import Boundary
from .kernels import _idx_periodic, _idx_reflected


@njit(cache=True)
def _coast_target(ix: int, iy: int, nx: int, ny: int):
    """Target index for COAST boundary.

    Mirrors Julia `canonical(Coast, ...)` used by `offset_index`:
    - x out of bounds => None
    - y periodic
    """

    if ix < 0 or ix >= nx:
        return -1, -1, False
    return ix, iy % ny, True


@njit(cache=True)
def _periodic_target(ix: int, iy: int, nx: int, ny: int):
    x, y = _idx_periodic(ix, iy, nx, ny)
    return x, y, True


@njit(cache=True)
def _reflected_target(ix: int, iy: int, nx: int, ny: int):
    x, y = _idx_reflected(ix, iy, nx, ny)
    return x, y, True


@njit(cache=True)
def _redistribution_kernel(w3: np.ndarray, cellsize: float, out: np.ndarray) -> np.ndarray:
    """Julia PhysicalErosionMod.redistribution_kernel port.

    w3: (3,3) patch
    out: (3,3) weights
    """

    c = w3[1, 1]

    # s matrix (positive slopes only)
    s00 = (w3[0, 0] - c) / cellsize
    s01 = (w3[0, 1] - c) / (cellsize * math.sqrt(2.0))
    s02 = (w3[0, 2] - c) / cellsize

    s10 = (w3[1, 0] - c) / (cellsize * math.sqrt(2.0))
    s11 = 0.0
    s12 = (w3[1, 2] - c) / (cellsize * math.sqrt(2.0))

    s20 = (w3[2, 0] - c) / cellsize
    s21 = (w3[2, 1] - c) / (cellsize * math.sqrt(2.0))
    s22 = (w3[2, 2] - c) / cellsize

    if s00 < 0.0:
        s00 = 0.0
    if s01 < 0.0:
        s01 = 0.0
    if s02 < 0.0:
        s02 = 0.0
    if s10 < 0.0:
        s10 = 0.0
    if s12 < 0.0:
        s12 = 0.0
    if s20 < 0.0:
        s20 = 0.0
    if s21 < 0.0:
        s21 = 0.0
    if s22 < 0.0:
        s22 = 0.0

    sumslope = s00 + s01 + s02 + s10 + s11 + s12 + s20 + s21 + s22
    if sumslope == 0.0:
        for i in range(3):
            for j in range(3):
                out[i, j] = 0.0
        return out

    inv = 1.0 / sumslope
    out[0, 0] = s00 * inv
    out[0, 1] = s01 * inv
    out[0, 2] = s02 * inv
    out[1, 0] = s10 * inv
    out[1, 1] = s11 * inv
    out[1, 2] = s12 * inv
    out[2, 0] = s20 * inv
    out[2, 1] = s21 * inv
    out[2, 2] = s22 * inv
    return out


@njit(cache=True)
def _redistribute_3x3_coast(den_mass: np.ndarray, water_depth: np.ndarray, dx_m: float, out: np.ndarray) -> np.ndarray:
    n_f, nx, ny = den_mass.shape
    w3 = np.empty((3, 3), dtype=np.float64)
    weights = np.empty((3, 3), dtype=np.float64)

    for f in range(n_f):
        for x in range(nx):
            for y in range(ny):
                m = den_mass[f, x, y]
                if m == 0.0:
                    continue

                # build 3x3 patch of water_depth using offset_value-like behavior
                # PhysicalErosionMod.mass_erosion uses offset_value, which for Coast
                # effectively clamps x but wraps y. We reuse _idx_reflected for x and
                # periodic for y via (reflect x, periodic y) mapping.
                for i in range(3):
                    for j in range(3):
                        ix = x + (i - 1)
                        iy = y + (j - 1)
                        # for patch values: clamp/reflect x, periodic y
                        # this matches Julia offset_value(Coast) which clamps x to 1..n
                        # (reflection in our port is the same as clamp at distance 1)
                        # We implement by reflecting in x.
                        rx, ry = _idx_reflected(ix, iy, nx, ny)
                        w3[i, j] = water_depth[rx, ry]

                _redistribution_kernel(w3, dx_m, weights)

                # distribute to targets using canonical(Coast): x out-of-bounds ignored
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        tx, ty, ok = _coast_target(x + di, y + dj, nx, ny)
                        if not ok:
                            continue
                        out[f, tx, ty] += weights[di + 1, dj + 1] * m

    return out


@njit(cache=True)
def _redistribute_3x3_periodic(den_mass: np.ndarray, water_depth: np.ndarray, dx_m: float, out: np.ndarray) -> np.ndarray:
    n_f, nx, ny = den_mass.shape
    w3 = np.empty((3, 3), dtype=np.float64)
    weights = np.empty((3, 3), dtype=np.float64)

    for f in range(n_f):
        for x in range(nx):
            for y in range(ny):
                m = den_mass[f, x, y]
                if m == 0.0:
                    continue

                for i in range(3):
                    for j in range(3):
                        rx, ry = _idx_periodic(x + (i - 1), y + (j - 1), nx, ny)
                        w3[i, j] = water_depth[rx, ry]

                _redistribution_kernel(w3, dx_m, weights)

                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        tx, ty, _ = _periodic_target(x + di, y + dj, nx, ny)
                        out[f, tx, ty] += weights[di + 1, dj + 1] * m

    return out


@njit(cache=True)
def _redistribute_3x3_reflected(den_mass: np.ndarray, water_depth: np.ndarray, dx_m: float, out: np.ndarray) -> np.ndarray:
    n_f, nx, ny = den_mass.shape
    w3 = np.empty((3, 3), dtype=np.float64)
    weights = np.empty((3, 3), dtype=np.float64)

    for f in range(n_f):
        for x in range(nx):
            for y in range(ny):
                m = den_mass[f, x, y]
                if m == 0.0:
                    continue

                for i in range(3):
                    for j in range(3):
                        rx, ry = _idx_reflected(x + (i - 1), y + (j - 1), nx, ny)
                        w3[i, j] = water_depth[rx, ry]

                _redistribution_kernel(w3, dx_m, weights)

                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        tx, ty, _ = _reflected_target(x + di, y + dj, nx, ny)
                        out[f, tx, ty] += weights[di + 1, dj + 1] * m

    return out


def redistribute_3x3(
    *,
    denudation_mass_m: np.ndarray,
    water_depth_m: np.ndarray,
    dx_m: float,
    boundary: Boundary,
) -> np.ndarray:
    """Redistribute denuded sediment mass to neighboring cells.

    This ports Julia `Denudation/PhysicalErosionMod.total_mass_redistribution`.

    Parameters
    - denudation_mass_m: (n_f, nx, ny) meters
    - water_depth_m: (nx, ny) meters

    Returns
    - redistributed_mass_m: (n_f, nx, ny)

    Notes
    - Mass is conserved for PERIODIC/REFLECTED.
    - For COAST, mass that would move beyond x-boundary is dropped (matches Julia).
    """

    den = np.asarray(denudation_mass_m, dtype=np.float64)
    w = np.asarray(water_depth_m, dtype=np.float64)
    out = np.zeros_like(den)

    if boundary is Boundary.COAST:
        return _redistribute_3x3_coast(den, w, float(dx_m), out)
    if boundary is Boundary.PERIODIC:
        return _redistribute_3x3_periodic(den, w, float(dx_m), out)
    if boundary is Boundary.REFLECTED:
        return _redistribute_3x3_reflected(den, w, float(dx_m), out)
    raise ValueError(f"Unsupported boundary: {boundary}")
