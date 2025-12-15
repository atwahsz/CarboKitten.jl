from __future__ import annotations

import math

from carbokitten.box import Box, Boundary
from carbokitten.denudation import EmpiricalDenudation
from carbokitten.models_with_denudation import WithDenudationFacies, WithDenudationInput, WithDenudationModel
from carbokitten.run_model import run_model
from carbokitten.time import TimeProperties


def test_with_denudation_runs_smoke() -> None:
    facies = (
        WithDenudationFacies(
            # denudation facies params
            reactive_surface_m2_m3=1.0,
            mass_density_kg_m3=2700.0,
            infiltration_coefficient=0.5,
            erodibility_m_yr=1e-4,
            # CA + production + transport
            viability_range=(4, 10),
            activation_range=(6, 10),
            maximum_growth_rate_m_myr=100.0,
            extinction_coefficient_m_inv=0.1,
            saturation_intensity_w_m2=60.0,
            diffusion_coefficient_m_myr=0.0,
            active=True,
        ),
    )

    inp = WithDenudationInput(
        tag="with-denudation",
        box=Box(grid_size=(20, 10), phys_scale_m=10.0, boundary=Boundary.COAST),
        time=TimeProperties(dt_myr=0.001, steps=3),
        facies=facies,
        denudation=EmpiricalDenudation(precip_m=1.0),
        initial_topography=lambda x, _y: -x / 300.0,
        sea_level=lambda t: 1.0 + 0.1 * math.sin(2.0 * math.pi * t / 0.2),
        subsidence_rate_m_myr=0.0,
        disintegration_rate_m_myr=0.0,
        insolation_w_m2=400.0,
        sediment_buffer_size=10,
        depositional_resolution_m=0.5,
    )

    st = run_model(WithDenudationModel(inp))
    assert st.step == inp.time.steps
