from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .box import Box, Boundary
from .kernels import (
    advection_coef_coast,
    ca_step_coast,
    max_dt,
    production_ca_gated,
    rk4_step_transport,
    water_depth as water_depth_kernel,
)
from .sediment_stack import pop_sediment, push_sediment
from .time import TimeProperties


@dataclass(frozen=True, slots=True)
class ALCAPFacies:
    """ALCAP facies parameters (merged from Julia mixins).

    Units
    - diffusion_coefficient_m_myr: meters per mega-year (m/Myr)
    - maximum_growth_rate_m_myr: meters per mega-year (m/Myr)
    - extinction_coefficient_m_inv: 1/meters (m^-1)
    - saturation_intensity_w_m2: W/m^2

    CA parameters
    - viability_range: (low, high) neighbor counts in a 5x5 window excluding center
    - activation_range: (low, high) neighbor counts in a 5x5 window including center
    """

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
class ALCAPInput:
    """Configuration for the ALCAP model.

    Units
    - sea_level(t): meters
    - initial_topography(x, y): meters
    - subsidence_rate_m_myr: m/Myr
    - disintegration_rate_m_myr: m/Myr
    - intertidal_zone_m: meters
    - depositional_resolution_m: meters
    - insolation_w_m2: W/m^2
    """

    tag: str
    box: Box
    time: TimeProperties
    facies: tuple[ALCAPFacies, ...]

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
class ALCAPState:
    step: int
    sediment_height_m: np.ndarray  # (nx, ny)
    sediment_buffer: np.ndarray  # (stack, n_f, nx, ny) in stack units
    active_layer_m: np.ndarray  # (n_f, nx, ny)
    ca: np.ndarray  # (nx, ny) int32
    ca_priority: np.ndarray  # (n_active,) int32


@dataclass(frozen=True, slots=True)
class ALCAPFrame:
    production_m: np.ndarray
    disintegration_m: np.ndarray
    deposition_m: np.ndarray


class ALCAPModel:
    """Runnable ALCAP model instance."""

    def __init__(self, input_: ALCAPInput):
        self.input = input_
        self.steps = input_.time.steps

        if input_.box.boundary is not Boundary.COAST:
            raise NotImplementedError("Only COAST boundary is implemented in this first port")

        self._eta0_m = self._initial_topography_grid()

        # pre-pack facies parameters into arrays for kernels
        n_f = len(input_.facies)
        self._max_growth = np.array([f.maximum_growth_rate_m_myr for f in input_.facies], dtype=np.float64)
        self._extinction = np.array([f.extinction_coefficient_m_inv for f in input_.facies], dtype=np.float64)
        self._saturation = np.array([f.saturation_intensity_w_m2 for f in input_.facies], dtype=np.float64)
        self._diffusivity = np.array([f.diffusion_coefficient_m_myr for f in input_.facies], dtype=np.float64)

        # CA ranges are 1-based indexed (0 is "no facies")
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

        # RK4 work arrays (reused)
        nx, ny = input_.box.grid_size
        self._k1 = np.empty((nx, ny), dtype=np.float64)
        self._k2 = np.empty((nx, ny), dtype=np.float64)
        self._k3 = np.empty((nx, ny), dtype=np.float64)
        self._k4 = np.empty((nx, ny), dtype=np.float64)
        self._tmp = np.empty((nx, ny), dtype=np.float64)

        # Disintegration output reuse
        self._pop_out = np.empty((n_f, nx, ny), dtype=np.float64)

    def _initial_topography_grid(self) -> np.ndarray:
        x, y = self.input.box.axes_m()
        nx, ny = self.input.box.grid_size
        eta0 = np.empty((nx, ny), dtype=np.float64)
        for i in range(nx):
            for j in range(ny):
                eta0[i, j] = float(self.input.initial_topography(float(x[i]), float(y[j])))
        return eta0

    def initial_state(self) -> ALCAPState:
        inp = self.input
        nx, ny = inp.box.grid_size
        n_f = len(inp.facies)

        rng = np.random.default_rng(inp.ca_random_seed)
        choices = np.concatenate((np.array([0], dtype=np.int32), self._active_ids))
        ca = rng.choice(choices, size=(nx, ny)).astype(np.int32, copy=False)

        # priority starts as active_ids and rotates each step
        ca_priority = self._active_ids.copy()

        sediment_height_m = np.zeros((nx, ny), dtype=np.float64)
        sediment_buffer = np.zeros((inp.sediment_buffer_size, n_f, nx, ny), dtype=np.float64)
        active_layer_m = np.zeros((n_f, nx, ny), dtype=np.float64)

        state = ALCAPState(
            step=0,
            sediment_height_m=sediment_height_m,
            sediment_buffer=sediment_buffer,
            active_layer_m=active_layer_m,
            ca=ca,
            ca_priority=ca_priority,
        )

        # Julia ALCAP does 20 warmup CA steps
        for _ in range(20):
            state = self._step_ca(state)

        # Initial sediment is zero in the example; hook left here for parity.
        return state

    def _rotate_priority(self, p: np.ndarray) -> np.ndarray:
        if p.size == 0:
            return p
        return np.roll(p, 1)

    def _step_ca(self, state: ALCAPState) -> ALCAPState:
        nxt = ca_step_coast(
            state.ca,
            self._viability_lo,
            self._viability_hi,
            self._activation_lo,
            self._activation_hi,
            state.ca_priority,
        )
        state.ca = nxt
        state.ca_priority = self._rotate_priority(state.ca_priority)
        return state

    def _water_depth(self, state: ALCAPState) -> np.ndarray:
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

    def _disintegrate(self, state: ALCAPState, wd_m: np.ndarray) -> np.ndarray:
        inp = self.input
        max_h_m = float(inp.disintegration_rate_m_myr) * float(inp.time.dt_myr)

        wn = wd_m + float(inp.intertidal_zone_m)
        h = np.minimum(max_h_m, state.sediment_height_m)
        h = np.where(wn <= 0.0, 0.0, h)

        state.sediment_height_m -= h

        amount_units = h / float(inp.depositional_resolution_m)
        pop_sediment(state.sediment_buffer, amount_units, self._pop_out)
        return self._pop_out * float(inp.depositional_resolution_m)

    def _transport(self, state: ALCAPState, wd_m: np.ndarray) -> None:
        inp = self.input
        dx_m = float(inp.box.phys_scale_m)
        dt_myr = float(inp.time.dt_myr)
        wd = wd_m + float(inp.intertidal_zone_m)

        for f in range(len(inp.facies)):
            d = float(self._diffusivity[f])
            if d == 0.0:
                continue

            advx, advy, rct = advection_coef_coast(wd, d, dx_m)
            m = max_dt(advx, advy, dx_m, courant_max=2.0)
            steps = int(math.ceil(dt_myr / m))
            if steps < 1:
                steps = 1
            subdt = dt_myr / float(steps)

            C = state.active_layer_m[f, :, :]
            t = inp.time.time_myr(state.step)
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

    def step(self, state: ALCAPState) -> tuple[ALCAPState, ALCAPFrame]:
        inp = self.input

        if state.step % int(inp.ca_interval) == 0:
            state = self._step_ca(state)

        wd = self._water_depth(state)

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

        state.active_layer_m += p
        state.active_layer_m += d

        self._transport(state, wd)

        # deposit
        deposit = state.active_layer_m.copy()
        push_sediment(state.sediment_buffer, deposit / float(inp.depositional_resolution_m))
        state.active_layer_m -= deposit
        state.sediment_height_m += deposit.sum(axis=0)

        state.step += 1
        return state, ALCAPFrame(production_m=p, disintegration_m=d, deposition_m=deposit)
