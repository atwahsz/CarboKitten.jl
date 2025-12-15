from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .box import Boundary, Box
from .kernels import (
    ca_step_coast,
    ca_step_periodic,
    ca_step_reflected,
    production_ca_gated,
    production_uniform,
    water_depth as water_depth_kernel,
)
from .time import TimeProperties


@dataclass(frozen=True, slots=True)
class CAPFacies:
    """CAP facies = CA parameters + production parameters."""

    # CA parameters
    viability_range: tuple[int, int] = (4, 10)
    activation_range: tuple[int, int] = (6, 10)
    active: bool = True

    # Production parameters
    maximum_growth_rate_m_myr: float = 0.0
    extinction_coefficient_m_inv: float = 0.0
    saturation_intensity_w_m2: float = 1.0


SeaLevelFn = Callable[[float], float]
InitialTopoFn = Callable[[float, float], float]


@dataclass(frozen=True, slots=True)
class CAPInput:
    """CAP model input (Julia `Models/CAP.jl`)."""

    tag: str
    box: Box
    time: TimeProperties
    facies: tuple[CAPFacies, ...]

    ca_interval: int = 1
    ca_random_seed: int = 0

    initial_topography: InitialTopoFn = lambda _x, _y: 0.0
    sea_level: SeaLevelFn = lambda _t: 0.0
    subsidence_rate_m_myr: float = 0.0

    insolation_w_m2: float = 400.0


@dataclass(slots=True)
class CAPState:
    step: int
    sediment_height_m: np.ndarray  # (nx, ny)
    ca: np.ndarray  # (nx, ny) int32
    ca_priority: np.ndarray  # (n_active,) int32


@dataclass(frozen=True, slots=True)
class CAPFrame:
    production_m: np.ndarray
    deposition_m: np.ndarray


@dataclass(frozen=True, slots=True)
class BS92Facies:
    """BS92 facies = production only."""

    maximum_growth_rate_m_myr: float
    extinction_coefficient_m_inv: float
    saturation_intensity_w_m2: float


@dataclass(frozen=True, slots=True)
class BS92Input:
    """BS92 model input (Julia `Models/BS92.jl`)."""

    tag: str
    box: Box
    time: TimeProperties
    facies: tuple[BS92Facies, ...]

    initial_topography: InitialTopoFn = lambda _x, _y: 0.0
    sea_level: SeaLevelFn = lambda _t: 0.0
    subsidence_rate_m_myr: float = 0.0

    insolation_w_m2: float = 400.0


@dataclass(slots=True)
class BS92State:
    step: int
    sediment_height_m: np.ndarray  # (nx, ny)


@dataclass(frozen=True, slots=True)
class BS92Frame:
    production_m: np.ndarray
    deposition_m: np.ndarray


class CAPModel:
    """Runnable CAP model (CA + production, no transport)."""

    def __init__(self, input_: CAPInput):
        self.input = input_
        self.steps = input_.time.steps

        self._eta0_m = self._initial_topography_grid()

        n_f = len(input_.facies)
        self._max_growth = np.array([f.maximum_growth_rate_m_myr for f in input_.facies], dtype=np.float64)
        self._extinction = np.array([f.extinction_coefficient_m_inv for f in input_.facies], dtype=np.float64)
        self._saturation = np.array([f.saturation_intensity_w_m2 for f in input_.facies], dtype=np.float64)

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

        if input_.box.boundary is Boundary.COAST:
            self._ca_step = ca_step_coast
        elif input_.box.boundary is Boundary.PERIODIC:
            self._ca_step = ca_step_periodic
        elif input_.box.boundary is Boundary.REFLECTED:
            self._ca_step = ca_step_reflected
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

    def _water_depth(self, step: int, sediment_height_m: np.ndarray) -> np.ndarray:
        inp = self.input
        t = inp.time.time_myr(step)
        sl = float(inp.sea_level(t))
        return water_depth_kernel(
            t_myr=float(t),
            sea_level_m=sl,
            eta0_m=self._eta0_m,
            subsidence_rate_m_myr=float(inp.subsidence_rate_m_myr),
            t0_myr=float(inp.time.t0_myr),
            sediment_height_m=sediment_height_m,
        )

    def initial_state(self) -> CAPState:
        inp = self.input
        nx, ny = inp.box.grid_size

        rng = np.random.default_rng(inp.ca_random_seed)
        choices = np.concatenate((np.array([0], dtype=np.int32), self._active_ids))
        ca = rng.choice(choices, size=(nx, ny)).astype(np.int32, copy=False)
        ca_priority = self._active_ids.copy()

        st = CAPState(step=0, sediment_height_m=np.zeros((nx, ny), dtype=np.float64), ca=ca, ca_priority=ca_priority)

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

    def step(self, state: CAPState) -> tuple[CAPState, CAPFrame]:
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

        wd = self._water_depth(state.step, state.sediment_height_m)

        prod = production_ca_gated(
            state.ca,
            wd,
            insolation_w_m2=float(inp.insolation_w_m2),
            dt_myr=float(inp.time.dt_myr),
            max_growth_m_myr=self._max_growth,
            extinction_m_inv=self._extinction,
            saturation_w_m2=self._saturation,
        )

        deposit = prod
        state.sediment_height_m += deposit.sum(axis=0)
        state.step += 1
        return state, CAPFrame(production_m=prod, deposition_m=deposit)


class BS92Model:
    """Runnable BS92 model (uniform production, no CA, no transport)."""

    def __init__(self, input_: BS92Input):
        self.input = input_
        self.steps = input_.time.steps

        self._eta0_m = self._initial_topography_grid()

        self._max_growth = np.array([f.maximum_growth_rate_m_myr for f in input_.facies], dtype=np.float64)
        self._extinction = np.array([f.extinction_coefficient_m_inv for f in input_.facies], dtype=np.float64)
        self._saturation = np.array([f.saturation_intensity_w_m2 for f in input_.facies], dtype=np.float64)

    def _initial_topography_grid(self) -> np.ndarray:
        x, y = self.input.box.axes_m()
        nx, ny = self.input.box.grid_size
        eta0 = np.empty((nx, ny), dtype=np.float64)
        for i in range(nx):
            for j in range(ny):
                eta0[i, j] = float(self.input.initial_topography(float(x[i]), float(y[j])))
        return eta0

    def _water_depth(self, step: int, sediment_height_m: np.ndarray) -> np.ndarray:
        inp = self.input
        t = inp.time.time_myr(step)
        sl = float(inp.sea_level(t))
        return water_depth_kernel(
            t_myr=float(t),
            sea_level_m=sl,
            eta0_m=self._eta0_m,
            subsidence_rate_m_myr=float(inp.subsidence_rate_m_myr),
            t0_myr=float(inp.time.t0_myr),
            sediment_height_m=sediment_height_m,
        )

    def initial_state(self) -> BS92State:
        nx, ny = self.input.box.grid_size
        return BS92State(step=0, sediment_height_m=np.zeros((nx, ny), dtype=np.float64))

    def step(self, state: BS92State) -> tuple[BS92State, BS92Frame]:
        inp = self.input
        wd = self._water_depth(state.step, state.sediment_height_m)

        prod = production_uniform(
            wd,
            insolation_w_m2=float(inp.insolation_w_m2),
            dt_myr=float(inp.time.dt_myr),
            max_growth_m_myr=self._max_growth,
            extinction_m_inv=self._extinction,
            saturation_w_m2=self._saturation,
        )
        deposit = prod
        state.sediment_height_m += deposit.sum(axis=0)
        state.step += 1
        return state, BS92Frame(production_m=prod, deposition_m=deposit)
