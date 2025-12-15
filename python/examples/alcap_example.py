from __future__ import annotations

import math

from carbokitten.box import Box, Boundary
from carbokitten.models_alcap import ALCAPFacies, ALCAPInput, ALCAPModel
from carbokitten.run_model import run_model
from carbokitten.time import TimeProperties
from carbokitten.waves import ZeroWave


def main() -> None:
    facies = (
        ALCAPFacies(
            viability_range=(4, 10),
            activation_range=(6, 10),
            maximum_growth_rate_m_myr=500.0,
            extinction_coefficient_m_inv=0.8,
            saturation_intensity_w_m2=60.0,
            diffusion_coefficient_m_myr=50.0 * 1.0e6,  # 50 m/yr -> 50e6 m/Myr
            wave=ZeroWave(),
        ),
        ALCAPFacies(
            viability_range=(4, 10),
            activation_range=(6, 10),
            maximum_growth_rate_m_myr=400.0,
            extinction_coefficient_m_inv=0.1,
            saturation_intensity_w_m2=60.0,
            diffusion_coefficient_m_myr=25.0 * 1.0e6,
            wave=ZeroWave(),
        ),
        ALCAPFacies(
            viability_range=(4, 10),
            activation_range=(6, 10),
            maximum_growth_rate_m_myr=100.0,
            extinction_coefficient_m_inv=0.005,
            saturation_intensity_w_m2=60.0,
            diffusion_coefficient_m_myr=12.5 * 1.0e6,
            wave=ZeroWave(),
        ),
    )

    period_myr = 0.2
    amplitude_m = 4.0

    inp = ALCAPInput(
        tag="alcap-example",
        box=Box(grid_size=(100, 50), phys_scale_m=150.0, boundary=Boundary.COAST),
        time=TimeProperties(dt_myr=0.0002, steps=200),  # keep example fast
        facies=facies,
        ca_interval=1,
        initial_topography=lambda x, _y: -x / 300.0,
        sea_level=lambda t: amplitude_m * math.sin(2.0 * math.pi * t / period_myr),
        subsidence_rate_m_myr=50.0,
        disintegration_rate_m_myr=50.0,
        insolation_w_m2=400.0,
        sediment_buffer_size=50,
        depositional_resolution_m=0.5,
    )

    model = ALCAPModel(inp)

    def cb(step: int, _frame: object) -> None:
        if step % 50 == 0:
            print(f"step {step}")

    final_state = run_model(model, callback=cb)
    print("done", final_state.step)


if __name__ == "__main__":
    main()
