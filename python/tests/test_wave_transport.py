from __future__ import annotations

import numpy as np

from carbokitten.box import Boundary, Box
from carbokitten.models_alcap import ALCAPFacies, ALCAPInput, ALCAPModel
from carbokitten.run_model import run_model
from carbokitten.time import TimeProperties
from carbokitten.waves import ConstantWave


def test_wave_model_does_not_crash_transport() -> None:
    facies = (
        ALCAPFacies(
            viability_range=(4, 10),
            activation_range=(6, 10),
            maximum_growth_rate_m_myr=0.0,
            extinction_coefficient_m_inv=0.1,
            saturation_intensity_w_m2=60.0,
            diffusion_coefficient_m_myr=1.0e6,
            wave=ConstantWave(vx_m_myr=1.0, vy_m_myr=0.0),
        ),
    )

    inp = ALCAPInput(
        tag="wave",
        box=Box(grid_size=(10, 6), phys_scale_m=10.0, boundary=Boundary.PERIODIC),
        time=TimeProperties(dt_myr=0.001, steps=2),
        facies=facies,
        ca_interval=1,
        initial_topography=lambda _x, _y: 0.0,
        sea_level=lambda _t: 10.0,
        subsidence_rate_m_myr=0.0,
        disintegration_rate_m_myr=0.0,
        insolation_w_m2=400.0,
        sediment_buffer_size=5,
        depositional_resolution_m=0.5,
    )

    model = ALCAPModel(inp)
    st = model.initial_state()
    st.active_layer_m[0] = 1.0

    # run a couple steps
    for _ in range(2):
        st, _ = model.step(st)

    assert st.sediment_height_m.shape == inp.box.grid_size
    assert np.isfinite(st.sediment_height_m).all()
