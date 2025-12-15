from __future__ import annotations

import math

from carbokitten.box import Box, Boundary
from carbokitten.models_alcap import ALCAPFacies, ALCAPInput, ALCAPModel
from carbokitten.run_model import run_model
from carbokitten.time import TimeProperties


def test_alcap_runs_smoke() -> None:
    facies = (
        ALCAPFacies(
            viability_range=(4, 10),
            activation_range=(6, 10),
            maximum_growth_rate_m_myr=500.0,
            extinction_coefficient_m_inv=0.8,
            saturation_intensity_w_m2=60.0,
            diffusion_coefficient_m_myr=0.0,
        ),
    )

    inp = ALCAPInput(
        tag="smoke",
        box=Box(grid_size=(20, 10), phys_scale_m=150.0, boundary=Boundary.COAST),
        time=TimeProperties(dt_myr=0.001, steps=5),
        facies=facies,
        initial_topography=lambda x, _y: -x / 300.0,
        sea_level=lambda t: 4.0 * math.sin(2.0 * math.pi * t / 0.2),
        subsidence_rate_m_myr=0.0,
        disintegration_rate_m_myr=0.0,
        insolation_w_m2=400.0,
        sediment_buffer_size=10,
        depositional_resolution_m=0.5,
    )

    model = ALCAPModel(inp)
    st = run_model(model)
    assert st.step == inp.time.steps
    assert st.sediment_height_m.shape == inp.box.grid_size
