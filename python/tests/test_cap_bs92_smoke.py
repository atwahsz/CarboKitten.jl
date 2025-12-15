from __future__ import annotations

import math

from carbokitten.box import Box, Boundary
from carbokitten.models_cap_bs92 import BS92Facies, BS92Input, BS92Model, CAPFacies, CAPInput, CAPModel
from carbokitten.run_model import run_model
from carbokitten.time import TimeProperties


def test_cap_runs_smoke() -> None:
    facies = (
        CAPFacies(
            viability_range=(4, 10),
            activation_range=(6, 10),
            maximum_growth_rate_m_myr=100.0,
            extinction_coefficient_m_inv=0.1,
            saturation_intensity_w_m2=60.0,
        ),
    )

    inp = CAPInput(
        tag="cap",
        box=Box(grid_size=(10, 8), phys_scale_m=10.0, boundary=Boundary.PERIODIC),
        time=TimeProperties(dt_myr=0.001, steps=3),
        facies=facies,
        initial_topography=lambda x, _y: -x / 300.0,
        sea_level=lambda t: 1.0 + 0.1 * math.sin(2.0 * math.pi * t / 0.2),
        subsidence_rate_m_myr=0.0,
        insolation_w_m2=400.0,
    )

    st = run_model(CAPModel(inp))
    assert st.step == inp.time.steps


def test_bs92_runs_smoke() -> None:
    facies = (
        BS92Facies(
            maximum_growth_rate_m_myr=100.0,
            extinction_coefficient_m_inv=0.1,
            saturation_intensity_w_m2=60.0,
        ),
        BS92Facies(
            maximum_growth_rate_m_myr=50.0,
            extinction_coefficient_m_inv=0.05,
            saturation_intensity_w_m2=60.0,
        ),
    )

    inp = BS92Input(
        tag="bs92",
        box=Box(grid_size=(12, 6), phys_scale_m=10.0, boundary=Boundary.REFLECTED),
        time=TimeProperties(dt_myr=0.001, steps=3),
        facies=facies,
        initial_topography=lambda x, _y: -x / 300.0,
        sea_level=lambda t: 1.0,
        subsidence_rate_m_myr=0.0,
        insolation_w_m2=400.0,
    )

    st = run_model(BS92Model(inp))
    assert st.step == inp.time.steps
