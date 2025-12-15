from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _reflect_index(i: int, n: int) -> int:
    period = 2 * n
    m = i % period
    return m if m < n else period - m - 1


@njit(cache=True)
def _idx_coast(ix: int, iy: int, nx: int, ny: int) -> tuple[int, int]:
    """COAST boundary: reflect x, periodic y."""

    rx = _reflect_index(ix, nx)
    ry = iy % ny
    return rx, ry


@njit(cache=True)
def ca_step_coast(
    ca: np.ndarray,
    viability_lo: np.ndarray,
    viability_hi: np.ndarray,
    activation_lo: np.ndarray,
    activation_hi: np.ndarray,
    priority: np.ndarray,
) -> np.ndarray:
    """Apply the 5x5 CA stencil once (COAST boundary).

    Parameters
    - ca: (nx, ny) int32, values in [0..n_facies]
    - viability_lo/hi: (n_facies+1,) ints, index 0 unused
    - activation_lo/hi: (n_facies+1,) ints, index 0 unused
    - priority: (n_active,) int32, facies IDs to try in order

    Returns
    - next_ca: (nx, ny) int32
    """

    nx, ny = ca.shape
    out = np.empty_like(ca)

    for x in range(nx):
        for y in range(ny):
            cell = ca[x, y]

            # count facies occurrences in a 5x5 neighborhood
            # counts[facies_id] = occurrences
            # note: n_facies is small, so a Python-side allocation would be costly;
            # in Numba we can allocate per-cell safely.
            n_facies = viability_lo.shape[0] - 1
            counts = np.zeros(n_facies + 1, dtype=np.int32)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    ix, iy = _idx_coast(x + dx, y + dy, nx, ny)
                    f = ca[ix, iy]
                    if 0 <= f <= n_facies:
                        counts[f] += 1

            if cell == 0:
                chosen = 0
                for k in range(priority.size):
                    f = priority[k]
                    n = counts[f]
                    a = activation_lo[f]
                    b = activation_hi[f]
                    if a <= n <= b:
                        chosen = f
                        break
                out[x, y] = chosen
            else:
                n = counts[cell] - 1
                a = viability_lo[cell]
                b = viability_hi[cell]
                out[x, y] = cell if (a <= n <= b) else 0

    return out


@njit(cache=True)
def production_ca_gated(
    ca: np.ndarray,
    water_depth_m: np.ndarray,
    insolation_w_m2: float,
    dt_myr: float,
    max_growth_m_myr: np.ndarray,
    extinction_m_inv: np.ndarray,
    saturation_w_m2: np.ndarray,
) -> np.ndarray:
    """Compute CA-gated capped production.

    Returns
    - production_m: (n_facies, nx, ny)
    """

    nx, ny = water_depth_m.shape
    n_f = max_growth_m_myr.size
    out = np.zeros((n_f, nx, ny), dtype=np.float64)

    for x in range(nx):
        for y in range(ny):
            wd = water_depth_m[x, y]
            if wd <= 0.0:
                continue
            facies_id = ca[x, y]
            if facies_id <= 0 or facies_id > n_f:
                continue
            f = facies_id - 1  # 1-based -> 0-based

            I = insolation_w_m2 / saturation_w_m2[f]
            xdep = wd * extinction_m_inv[f]
            rate = max_growth_m_myr[f] * math.tanh(I * math.exp(-xdep))
            p = rate * dt_myr
            if p < 0.0:
                p = 0.0
            if p > wd:
                p = wd
            out[f, x, y] = p

    return out


@njit(cache=True)
def water_depth(
    t_myr: float,
    sea_level_m: float,
    eta0_m: np.ndarray,
    subsidence_rate_m_myr: float,
    t0_myr: float,
    sediment_height_m: np.ndarray,
) -> np.ndarray:
    """Compute water depth field (meters)."""

    return sea_level_m - eta0_m + subsidence_rate_m_myr * (t_myr - t0_myr) - sediment_height_m


@njit(cache=True)
def advection_coef_coast(
    w_m: np.ndarray,
    diffusivity_m_myr: float,
    dx_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute adv=(advx,advy) and rct for transport_dC (COAST boundary).

    This port currently implements wave_velocity == 0 and wave_shear == 0.

    Returns
    - advx: (nx, ny)
    - advy: (nx, ny)
    - rct: (nx, ny)
    """

    nx, ny = w_m.shape
    advx = np.empty((nx, ny), dtype=np.float64)
    advy = np.empty((nx, ny), dtype=np.float64)
    rct = np.empty((nx, ny), dtype=np.float64)

    for x in range(nx):
        for y in range(ny):
            x1, y1 = _idx_coast(x - 1, y, nx, ny)
            x2, y2 = _idx_coast(x + 1, y, nx, ny)
            wx1 = w_m[x1, y1]
            wx2 = w_m[x2, y2]

            x1, y1 = _idx_coast(x, y - 1, nx, ny)
            x2, y2 = _idx_coast(x, y + 1, nx, ny)
            wy1 = w_m[x1, y1]
            wy2 = w_m[x2, y2]

            dwx = (wx2 - wx1) / (2.0 * dx_m)
            dwy = (wy2 - wy1) / (2.0 * dx_m)
            ddw = (wx1 + wx2 + wy1 + wy2 - 4.0 * w_m[x, y]) / (dx_m * dx_m)

            advx[x, y] = diffusivity_m_myr * dwx
            advy[x, y] = diffusivity_m_myr * dwy
            rct[x, y] = -diffusivity_m_myr * ddw

    return advx, advy, rct


@njit(cache=True)
def max_dt(advx: np.ndarray, advy: np.ndarray, dx_m: float, courant_max: float) -> float:
    """Compute stability step (Myr) from adv field."""

    nx, ny = advx.shape
    m = 0.0
    for x in range(nx):
        for y in range(ny):
            u = abs(advx[x, y]) + abs(advy[x, y])
            v = u / dx_m
            if v > m:
                m = v
    # If adv is zero everywhere, pick a large dt.
    if m == 0.0:
        return 1.0e30
    return courant_max / m


@njit(cache=True)
def transport_dC_coast(
    advx: np.ndarray,
    advy: np.ndarray,
    rct: np.ndarray,
    C: np.ndarray,
    dx_m: float,
    dC: np.ndarray,
) -> np.ndarray:
    nx, ny = C.shape

    for x in range(nx):
        for y in range(ny):
            # upwind in x
            vx = advx[x, y]
            if vx < 0.0:
                x2, y2 = _idx_coast(x + 1, y, nx, ny)
                adv_term_x = vx * (C[x2, y2] - C[x, y]) / dx_m
            else:
                x1, y1 = _idx_coast(x - 1, y, nx, ny)
                adv_term_x = vx * (C[x, y] - C[x1, y1]) / dx_m

            # upwind in y
            vy = advy[x, y]
            if vy < 0.0:
                x2, y2 = _idx_coast(x, y + 1, nx, ny)
                adv_term_y = vy * (C[x2, y2] - C[x, y]) / dx_m
            else:
                x1, y1 = _idx_coast(x, y - 1, nx, ny)
                adv_term_y = vy * (C[x, y] - C[x1, y1]) / dx_m

            dC[x, y] = rct[x, y] * C[x, y] - adv_term_x - adv_term_y

    return dC


@njit(cache=True)
def rk4_step_transport(
    advx: np.ndarray,
    advy: np.ndarray,
    rct: np.ndarray,
    C: np.ndarray,
    t_myr: float,
    dt_myr: float,
    dx_m: float,
    k1: np.ndarray,
    k2: np.ndarray,
    k3: np.ndarray,
    k4: np.ndarray,
    tmp: np.ndarray,
) -> None:
    """In-place RK4 update for one facies concentration field."""

    # k1
    transport_dC_coast(advx, advy, rct, C, dx_m, k1)

    # k2
    tmp[:, :] = C + 0.5 * dt_myr * k1
    transport_dC_coast(advx, advy, rct, tmp, dx_m, k2)

    # k3
    tmp[:, :] = C + 0.5 * dt_myr * k2
    transport_dC_coast(advx, advy, rct, tmp, dx_m, k3)

    # k4
    tmp[:, :] = C + dt_myr * k3
    transport_dC_coast(advx, advy, rct, tmp, dx_m, k4)

    C[:, :] = C + (dt_myr / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # clamp negatives
    nx, ny = C.shape
    for x in range(nx):
        for y in range(ny):
            if C[x, y] < 0.0:
                C[x, y] = 0.0
