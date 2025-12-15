from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .box import Boundary, Box
from .denudation import DenudationFacies, DenudationModel, slope_degrees
from .kernels import (
    advection_coef_coast,
    advection_coef_periodic,
    advection_coef_reflected,
    ca_step_coast,
    ca_step_periodic,
    ca_step_reflected,
    max_dt,
    production_ca_gated,
    rk4_step_transport,
    water_depth as water_depth_kernel,
)
from .sediment_stack import pop_sediment, push_sediment
from .time import TimeProperties


@dataclass(frozen=True, slots=True)
class WithDenudationFacies(DenudationFacies):
    """Facies parameters required for WithDenudation (subset of Julia mixins)."""

    viability_range: tuple[int, int]
    activation_range: tuple[int, int]
    maximum_growth_rate_m_myr: float
    extinction_coefficient_m_inv: float
    saturation_intensity_w_m2: float
    diffusion_coefficient_m_myr: float
    active: bool = True


SeaLevelFn = Callable[[float], float]
InitialTopoFn = Callable[[float, float], float]


@dataclass(frozen=True, slots=True)
class WithDenudationInput:
    tag: str
    box: Box
    time: TimeProperties
    facies: tuple[WithDenudationFacies, ...]
    denudation: DenudationModel

    ca_interval: int = 1
    ca_random_seed: int = 0

    initial_topography: InitialTopoFn = lambda _x, _y: 0.0
    sea_level: SeaLevelFn = lambda _t: 0.0
    subsidence_rate_m_myr: float = 0.0

    intertidal_zone_m: float = 0.0
    disintegration_rate_m_myr: float = 50.0

    sediment_buffer_size: int = 50
    depositional_resolution_m: float = 0.5

    insolation_w_m2: float = 400.0


@dataclass(slots=True)
class WithDenudationState:
    step: int
    sediment_height_m: np.ndarray
    sediment_buffer: np.ndarray
    ca: np.ndarray
    ca_priority: np.ndarray


@dataclass(frozen=True, slots=True)
class WithDenudationFrame:
    production_m: np.ndarray
    disintegration_m: np.ndarray
    deposition_m: np.ndarray


class WithDenudationModel:
    """Runnable model port of Julia `Models/WithDenudation.jl`.

    Notes
    - This first port keeps the same structure but currently uses the same
      transport kernel as ALCAP (diffusion-only wave velocity).
    - Redistribution for PhysicalErosion is left as a TODO (currently None).
    """

    def __init__(self, input_: WithDenudationInput):
        self.input = input_
        self.steps = input_.time.steps

        self._eta0_m = self._initial_topography_grid()

        n_f = len(input_.facies)
        self._max_growth = np.array([f.maximum_growth_rate_m_myr for f in input_.facies], dtype=np.float64)
        self._extinction = np.array([f.extinction_coefficient_m_inv for f in input_.facies], dtype=np.float64)
        self._saturation = np.array([f.saturation_intensity_w_m2 for f in input_.facies], dtype=np.float64)
        self._diffusivity = np.array([f.diffusion_coefficient_m_myr for f in input_.facies], dtype=np.float64)

        self._viability_lo = np.zeros(n_f + 1, dtype=np.int32)
        self._viability_hi = np.zeros(n_f + 1, dtype=np.int32)
        self._activation_lo = np.zeros(n_f + 1, dtype=np.int32)
        self._activation_hi = np.zeros(n_f + 1, dtype=np.int32)
        active_ids: list[int] = []
        for i, f in enumerate(input_.facies, start=1):
            self._viability_lo[i] = int(f.viability_range[0])
            self._viability_hi[i] = int(f.viability_range[1])
            self._activation_lo[i] = int(f.activation_range[0])
            self._activation_hi[i] = int(f.activation_range[1])
            if f.active:
                active_ids.append(i)
        self._active_ids = np.array(active_ids, dtype=np.int32)

        nx, ny = input_.box.grid_size
        self._k1 = np.empty((nx, ny), dtype=np.float64)
        self._k2 = np.empty((nx, ny), dtype=np.float64)
        self._k3 = np.empty((nx, ny), dtype=np.float64)
        self._k4 = np.empty((nx, ny), dtype=np.float64)
        self._tmp = np.empty((nx, ny), dtype=np.float64)

        self._pop_out = np.empty((n_f, nx, ny), dtype=np.float64)

        if input_.box.boundary is Boundary.COAST:
            self._ca_step = ca_step_coast
            self._adv_coef = advection_coef_coast
        elif input_.box.boundary is Boundary.PERIODIC:
            self._ca_step = ca_step_periodic
            self._adv_coef = advection_coef_periodic
        elif input_.box.boundary is Boundary.REFLECTED:
            self._ca_step = ca_step_reflected
            self._adv_coef = advection_coef_reflected
        else:  # pragma: no cover
            raise ValueError(f"Unsupported boundary: {input_.box.boundary}")

    def _initial_topography_grid(self) -> np.ndarray:
        x, y = self.input.box.axes_m()
        nx, ny = self.input.box.grid_size
        eta0 = np.empty((nx, ny), dtype=np.float64)
        for i in range(nx):
            for j in range(ny):
                eta0[i, j] = float(self.input.initial_topography(float(x[i]), float(y[j])))
        return eta0

    def _rotate_priority(self, p: np.ndarray) -> np.ndarray:
        if p.size == 0:
            return p
        return np.roll(p, 1)

    def initial_state(self) -> WithDenudationState:
        inp = self.input
        nx, ny = inp.box.grid_size
        n_f = len(inp.facies)

        rng = np.random.default_rng(inp.ca_random_seed)
        choices = np.concatenate((np.array([0], dtype=np.int32), self._active_ids))
        ca = rng.choice(choices, size=(nx, ny)).astype(np.int32, copy=False)
        ca_priority = self._active_ids.copy()

        st = WithDenudationState(
            step=0,
            sediment_height_m=np.zeros((nx, ny), dtype=np.float64),
            sediment_buffer=np.zeros((inp.sediment_buffer_size, n_f, nx, ny), dtype=np.float64),
            ca=ca,
            ca_priority=ca_priority,
        )

        for _ in range(20):
            st.ca = self._ca_step(
                st.ca,
                self._viability_lo,
                self._viability_hi,
                self._activation_lo,
                self._activation_hi,
                st.ca_priority,
            )
            st.ca_priority = self._rotate_priority(st.ca_priority)

        return st

    def _water_depth(self, state: WithDenudationState) -> np.ndarray:
        inp = self.input
        t = inp.time.time_myr(state.step)
        sl = float(inp.sea_level(t))
        return water_depth_kernel(
            t_myr=float(t),
            sea_level_m=sl,
            eta0_m=self._eta0_m,
            subsidence_rate_m_myr=float(inp.subsidence_rate_m_myr),
            t0_myr=float(inp.time.t0_myr),
            sediment_height_m=state.sediment_height_m,
        )

    def _disintegrate(self, state: WithDenudationState, wd_m: np.ndarray) -> np.ndarray:
        inp = self.input
        max_h_m = float(inp.disintegration_rate_m_myr) * float(inp.time.dt_myr)

        wn = wd_m + float(inp.intertidal_zone_m)
        h = np.minimum(max_h_m, state.sediment_height_m)
        h = np.where(wn <= 0.0, 0.0, h)

        state.sediment_height_m -= h

        amount_units = h / float(inp.depositional_resolution_m)
        pop_sediment(state.sediment_buffer, amount_units, self._pop_out)
        return self._pop_out * float(inp.depositional_resolution_m)

    def _transport(self, active_layer_m: np.ndarray, wd_m: np.ndarray, step: int) -> np.ndarray:
        inp = self.input
        dx_m = float(inp.box.phys_scale_m)
        dt_myr = float(inp.time.dt_myr)
        wd = wd_m + float(inp.intertidal_zone_m)

        # transported sediment starts as the local active layer
        sediment = active_layer_m.copy()

        for f in range(len(inp.facies)):
            d = float(self._diffusivity[f])
            if d == 0.0:
                continue
            advx, advy, rct = self._adv_coef(wd, d, dx_m)
            m = max_dt(advx, advy, dx_m, courant_max=2.0)
            steps = int(math.ceil(dt_myr / m))
            if steps < 1:
                steps = 1
            subdt = dt_myr / float(steps)

            C = sediment[f, :, :]
            t = inp.time.time_myr(step)
            for _ in range(steps):
                rk4_step_transport(
                    advx,
                    advy,
                    rct,
                    C,
                    t_myr=float(t),
                    dt_myr=float(subdt),
                    dx_m=dx_m,
                    k1=self._k1,
                    k2=self._k2,
                    k3=self._k3,
                    k4=self._k4,
                    tmp=self._tmp,
                )

        return sediment

    def step(self, state: WithDenudationState) -> tuple[WithDenudationState, WithDenudationFrame]:
        inp = self.input

        if state.step % int(inp.ca_interval) == 0:
            state.ca = self._ca_step(
                state.ca,
                self._viability_lo,
                self._viability_hi,
                self._activation_lo,
                self._activation_hi,
                state.ca_priority,
            )
            state.ca_priority = self._rotate_priority(state.ca_priority)

        wd = self._water_depth(state)
        slope = slope_degrees(wd, dx_m=float(inp.box.phys_scale_m), boundary=inp.box.boundary)

        p = production_ca_gated(
            state.ca,
            wd,
            insolation_w_m2=float(inp.insolation_w_m2),
            dt_myr=float(inp.time.dt_myr),
            max_growth_m_myr=self._max_growth,
            extinction_m_inv=self._extinction,
            saturation_w_m2=self._saturation,
        )

        d = self._disintegrate(state, wd)
        active_layer = p + d

        sediment = self._transport(active_layer, wd, step=state.step)

        # subaerial denudation
        den_rate = inp.denudation.denudation_rate_m_myr(
            ca=state.ca,
            water_depth_m=wd,
            slope_deg=slope,
            facies_params=inp.facies,
        )

        if den_rate is not None:
            den_mass = den_rate * float(inp.time.dt_myr)
            # total mass per cell capped by available sediment height
            den_mass_cell = np.minimum(den_mass.sum(axis=0), state.sediment_height_m)
            state.sediment_height_m -= den_mass_cell

            amount_units = den_mass_cell / float(inp.depositional_resolution_m)
            pop_sediment(state.sediment_buffer, amount_units, self._pop_out)
            d = d + self._pop_out * float(inp.depositional_resolution_m)

            redist = inp.denudation.redistribution_m(
                denudation_mass_m=self._pop_out * float(inp.depositional_resolution_m),
                water_depth_m=wd,
                dx_m=float(inp.box.phys_scale_m),
                boundary=inp.box.boundary,
            )
            if redist is not None:
                sediment = sediment + redist

        push_sediment(state.sediment_buffer, sediment / float(inp.depositional_resolution_m))
        state.sediment_height_m += sediment.sum(axis=0)

        state.step += 1
        return state, WithDenudationFrame(production_m=p, disintegration_m=d, deposition_m=sediment)
