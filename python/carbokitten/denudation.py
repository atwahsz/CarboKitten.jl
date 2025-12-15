from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numba import njit

from .box import Boundary
from .kernels import _idx_coast, _idx_periodic, _idx_reflected


class DenudationModel(Protocol):
    """Protocol for denudation models."""

    def denudation_rate_m_myr(
        self,
        *,
        ca: np.ndarray,
        water_depth_m: np.ndarray,
        slope_deg: np.ndarray,
        facies_params: object,
    ) -> np.ndarray | None:  # (n_f, nx, ny)
        ...

    def redistribution_m(
        self,
        *,
        denudation_mass_m: np.ndarray,
        water_depth_m: np.ndarray,
        dx_m: float,
        boundary: Boundary,
    ) -> np.ndarray | None:
        ...


@njit(cache=True)
def slope_deg_kernel_coast(w: np.ndarray, dx_m: float, out: np.ndarray) -> np.ndarray:
    """Sobel-like slope kernel ported from Julia EmpiricalDenudationMod.slope_kernel.

    Boundary: COAST (reflect x, periodic y).

    Inputs are water depth in meters.
    """

    nx, ny = w.shape
    for x in range(nx):
        for y in range(ny):
            # gather 3x3 neighborhood
            nb = np.empty((3, 3), dtype=np.float64)
            kx = 0
            for dx in range(-1, 2):
                ky = 0
                for dy in range(-1, 2):
                    ix, iy = _idx_coast(x + dx, y + dy, nx, ny)
                    nb[kx, ky] = w[ix, iy]
                    ky += 1
                kx += 1

            dzdx = (
                (-nb[0, 0] - 2.0 * nb[1, 0] - nb[2, 0] + nb[0, 2] + 2.0 * nb[1, 2] + nb[2, 2])
                / (8.0 * dx_m)
            )
            dzdy = (
                (-nb[0, 0] - 2.0 * nb[0, 1] - nb[0, 2] + nb[2, 0] + 2.0 * nb[2, 1] + nb[0, 0])
                / (8.0 * dx_m)
            )

            # if center is the shallowest (or tied), treat as 0 slope
            c = abs(nb[1, 1])
            mn = c
            for i in range(3):
                for j in range(3):
                    v = abs(nb[i, j])
                    if v < mn:
                        mn = v
            if c <= mn:
                out[x, y] = 0.0
            else:
                out[x, y] = math.atan(math.sqrt(dzdx * dzdx + dzdy * dzdy)) * (180.0 / math.pi)

    return out


@njit(cache=True)
def slope_deg_kernel_periodic(w: np.ndarray, dx_m: float, out: np.ndarray) -> np.ndarray:
    nx, ny = w.shape
    for x in range(nx):
        for y in range(ny):
            nb = np.empty((3, 3), dtype=np.float64)
            kx = 0
            for ddx in range(-1, 2):
                ky = 0
                for ddy in range(-1, 2):
                    ix, iy = _idx_periodic(x + ddx, y + ddy, nx, ny)
                    nb[kx, ky] = w[ix, iy]
                    ky += 1
                kx += 1

            dzdx = (
                (-nb[0, 0] - 2.0 * nb[1, 0] - nb[2, 0] + nb[0, 2] + 2.0 * nb[1, 2] + nb[2, 2])
                / (8.0 * dx_m)
            )
            dzdy = (
                (-nb[0, 0] - 2.0 * nb[0, 1] - nb[0, 2] + nb[2, 0] + 2.0 * nb[2, 1] + nb[0, 0])
                / (8.0 * dx_m)
            )

            c = abs(nb[1, 1])
            mn = c
            for i in range(3):
                for j in range(3):
                    v = abs(nb[i, j])
                    if v < mn:
                        mn = v
            if c <= mn:
                out[x, y] = 0.0
            else:
                out[x, y] = math.atan(math.sqrt(dzdx * dzdx + dzdy * dzdy)) * (180.0 / math.pi)

    return out


@njit(cache=True)
def slope_deg_kernel_reflected(w: np.ndarray, dx_m: float, out: np.ndarray) -> np.ndarray:
    nx, ny = w.shape
    for x in range(nx):
        for y in range(ny):
            nb = np.empty((3, 3), dtype=np.float64)
            kx = 0
            for ddx in range(-1, 2):
                ky = 0
                for ddy in range(-1, 2):
                    ix, iy = _idx_reflected(x + ddx, y + ddy, nx, ny)
                    nb[kx, ky] = w[ix, iy]
                    ky += 1
                kx += 1

            dzdx = (
                (-nb[0, 0] - 2.0 * nb[1, 0] - nb[2, 0] + nb[0, 2] + 2.0 * nb[1, 2] + nb[2, 2])
                / (8.0 * dx_m)
            )
            dzdy = (
                (-nb[0, 0] - 2.0 * nb[0, 1] - nb[0, 2] + nb[2, 0] + 2.0 * nb[2, 1] + nb[0, 0])
                / (8.0 * dx_m)
            )

            c = abs(nb[1, 1])
            mn = c
            for i in range(3):
                for j in range(3):
                    v = abs(nb[i, j])
                    if v < mn:
                        mn = v
            if c <= mn:
                out[x, y] = 0.0
            else:
                out[x, y] = math.atan(math.sqrt(dzdx * dzdx + dzdy * dzdy)) * (180.0 / math.pi)

    return out


def slope_degrees(water_depth_m: np.ndarray, *, dx_m: float, boundary: Boundary) -> np.ndarray:
    """Compute slope in degrees using the Julia kernel semantics."""

    out = np.empty_like(water_depth_m, dtype=np.float64)
    if boundary is Boundary.COAST:
        return slope_deg_kernel_coast(water_depth_m, float(dx_m), out)
    if boundary is Boundary.PERIODIC:
        return slope_deg_kernel_periodic(water_depth_m, float(dx_m), out)
    if boundary is Boundary.REFLECTED:
        return slope_deg_kernel_reflected(water_depth_m, float(dx_m), out)
    raise ValueError(f"Unsupported boundary: {boundary}")


@dataclass(frozen=True, slots=True)
class NoDenudation:
    def denudation_rate_m_myr(
        self,
        *,
        ca: np.ndarray,
        water_depth_m: np.ndarray,
        slope_deg: np.ndarray,
        facies_params: object,
    ) -> np.ndarray | None:
        return None

    def redistribution_m(
        self,
        *,
        denudation_mass_m: np.ndarray,
        water_depth_m: np.ndarray,
        dx_m: float,
        boundary: Boundary,
    ) -> np.ndarray | None:
        return None


@dataclass(frozen=True, slots=True)
class EmpiricalDenudation:
    """Empirical denudation (Julia `EmpiricalDenudationMod`)."""

    precip_m: float

    def denudation_rate_m_myr(
        self,
        *,
        ca: np.ndarray,
        water_depth_m: np.ndarray,
        slope_deg: np.ndarray,
        facies_params: object,
    ) -> np.ndarray | None:
        # facies_params unused (matches Julia)
        precip = float(self.precip_m)

        a = 9.1363
        b = -0.008519
        c = 580.51
        d = 9.0156
        e = -0.1245
        f = 4.91086

        nx, ny = water_depth_m.shape
        n_f = int(np.max(ca))
        out = np.zeros((n_f, nx, ny), dtype=np.float64)

        # rate only for subaerial cells (water_depth<=0) and facies>0
        mm = precip * 1000.0
        for x in range(nx):
            for y in range(ny):
                fid = int(ca[x, y])
                if fid <= 0:
                    continue
                if water_depth_m[x, y] > 0.0:
                    continue
                s = float(slope_deg[x, y])
                rate = (a / (1.0 + math.exp(b * (mm - c)))) * (d / (1.0 + math.exp(e * (s - f))))
                out[fid - 1, x, y] = rate

        return out

    def redistribution_m(
        self,
        *,
        denudation_mass_m: np.ndarray,
        water_depth_m: np.ndarray,
        dx_m: float,
        boundary: Boundary,
    ) -> np.ndarray | None:
        return None


@dataclass(frozen=True, slots=True)
class PhysicalErosion:
    """Physical erosion (Julia `PhysicalErosionMod`)."""

    def denudation_rate_m_myr(
        self,
        *,
        ca: np.ndarray,
        water_depth_m: np.ndarray,
        slope_deg: np.ndarray,
        facies_params: object,
    ) -> np.ndarray | None:
        # facies_params: sequence with infiltration_coefficient and erodibility_m_yr
        facies = facies_params
        nx, ny = water_depth_m.shape
        n_f = len(facies)
        out = np.zeros((n_f, nx, ny), dtype=np.float64)

        for x in range(nx):
            for y in range(ny):
                fid = int(ca[x, y])
                if fid <= 0:
                    continue
                if water_depth_m[x, y] > 0.0:
                    continue
                f = facies[fid - 1]
                inf = float(f.infiltration_coefficient)
                erod_m_myr = float(f.erodibility_m_yr) * 1.0e6
                s = float(slope_deg[x, y])
                # Julia: -1 * -erodibility .* (1 - inf)^(1/3) * slope^(2/3)
                out[fid - 1, x, y] = erod_m_myr * (1.0 - inf) ** (1.0 / 3.0) * (s ** (2.0 / 3.0))

        return out

    def redistribution_m(
        self,
        *,
        denudation_mass_m: np.ndarray,
        water_depth_m: np.ndarray,
        dx_m: float,
        boundary: Boundary,
    ) -> np.ndarray | None:
        from .redistribution import redistribute_3x3

        return redistribute_3x3(
            denudation_mass_m=denudation_mass_m,
            water_depth_m=water_depth_m,
            dx_m=float(dx_m),
            boundary=boundary,
        )


@dataclass(frozen=True, slots=True)
class Dissolution:
    """Chemical dissolution (Julia `DissolutionMod`)."""

    temp_k: float
    precip_m: float
    pco2_atm: float
    reactionrate_m_yr: float

    def denudation_rate_m_myr(
        self,
        *,
        ca: np.ndarray,
        water_depth_m: np.ndarray,
        slope_deg: np.ndarray,
        facies_params: object,
    ) -> np.ndarray | None:
        facies = facies_params
        nx, ny = water_depth_m.shape
        n_f = len(facies)
        out = np.zeros((n_f, nx, ny), dtype=np.float64)

        temp = float(self.temp_k)
        precip = float(self.precip_m)
        pco2 = float(self.pco2_atm)
        alpha = float(self.reactionrate_m_yr)

        # helper functions ported from Julia (scalar)
        def karst_params(t: float):
            A = -0.4883 + 8.074e-4 * (t - 273.0)
            B = -0.3241 + 1.6e-4 * (t - 273.0)
            IA = 0.1
            K1 = 10 ** (-356.3094 - 0.06091964 * t + 21834.37 / t + 126.8339 * math.log10(t) - 1684915 / (t * t))
            K2 = 10 ** (-107.881 - 0.03252849 * t + 5151.79 / t + 38.92561 * math.log10(t) - 563713.9 / (t * t))
            KC = 10 ** (-171.9065 - 0.077993 * t + 2839.319 / t + 71.595 * math.log10(t))
            KH = 10 ** (108.3865 + 0.01985076 * t - 6919.53 / t - 40.4515 * math.log10(t) + 669365 / (t * t))
            activity_Ca = 10 ** (-4 * A * math.sqrt(IA) / (1 + 1e-8 * B * math.sqrt(IA)))
            activity_Alk = 10 ** (-A * math.sqrt(IA) / (1 + 5.4e-8 * B * math.sqrt(IA)))
            return K1, K2, KC, KH, activity_Ca, activity_Alk

        K1, K2, KC, KH, aCa, aAlk = karst_params(temp)

        for x in range(nx):
            for y in range(ny):
                fid = int(ca[x, y])
                if fid <= 0:
                    continue
                wd = float(water_depth_m[x, y])
                if wd > 0.0:
                    continue

                f = facies[fid - 1]
                reactive_surface = float(f.reactive_surface_m2_m3)
                mass_density = float(f.mass_density_kg_m3)
                infil = float(f.infiltration_coefficient)

                eq_c = (pco2 * (K1 * KC * KH) / (4.0 * K2 * aCa * (aAlk**2))) ** (1.0 / 3.0)
                eq_d = 1e6 * precip * infil * 40.0 * 1000.0 * eq_c / mass_density

                lam = precip * 100.0 * infil / (alpha * reactive_surface)
                # Julia: eq.denudation * (1 - (λ / -wd) * (1 - exp(wd/λ)))
                # wd is negative in subaerial cells
                if wd == 0.0:
                    out[fid - 1, x, y] = 0.0
                else:
                    out[fid - 1, x, y] = eq_d * (1.0 - (lam / (-wd)) * (1.0 - math.exp(wd / lam)))

        return out

    def redistribution_m(
        self,
        *,
        denudation_mass_m: np.ndarray,
        water_depth_m: np.ndarray,
        dx_m: float,
        boundary: Boundary,
    ) -> np.ndarray | None:
        return None


@dataclass(frozen=True, slots=True)
class DenudationFacies:
    """Per-facies denudation properties (from Julia `Components/Denudation.jl`)."""

    reactive_surface_m2_m3: float
    mass_density_kg_m3: float
    infiltration_coefficient: float
    erodibility_m_yr: float
