from __future__ import annotations

import math
import os

import numpy as np

from carbokitten.box import Box, Boundary
from carbokitten.h5writer import create_h5_output, h5_frame_writer, h5_state_writer
from carbokitten.models_cap_bs92 import CAPFacies, CAPInput, CAPModel
from carbokitten.output import OutputSpec, add_memory_data_set, memory_frame_writer, memory_state_writer, new_memory_output
from carbokitten.run_model import run_model_with_writers
from carbokitten.time import TimeProperties


def test_memory_output_smoke() -> None:
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
        box=Box(grid_size=(10, 8), phys_scale_m=10.0, boundary=Boundary.COAST),
        time=TimeProperties(dt_myr=0.001, steps=4),
        facies=facies,
        initial_topography=lambda x, _y: -x / 300.0,
        sea_level=lambda t: 1.0 + 0.1 * math.sin(2.0 * math.pi * t / 0.2),
        subsidence_rate_m_myr=0.0,
        insolation_w_m2=400.0,
    )

    x_m, y_m = inp.box.axes_m()
    t_myr = np.array([inp.time.time_myr(i) for i in range(inp.time.steps + 1)], dtype=np.float64)
    out = new_memory_output(tag=inp.tag, x_m=x_m, y_m=y_m, t_myr=t_myr, grid_size=inp.box.grid_size, n_facies=len(inp.facies))

    add_memory_data_set(out, "topography", OutputSpec(slice_=(slice(None), slice(None)), write_interval=2), steps=inp.time.steps)

    run_model_with_writers(
        CAPModel(inp),
        state_writer=memory_state_writer(out),
        frame_writer=memory_frame_writer(out),
    )

    ds = out.data_sets["topography"]
    assert ds.production_m.shape[-1] == inp.time.steps // 2


def test_h5_output_smoke(tmp_path) -> None:
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
        box=Box(grid_size=(10, 8), phys_scale_m=10.0, boundary=Boundary.COAST),
        time=TimeProperties(dt_myr=0.001, steps=4),
        facies=facies,
        initial_topography=lambda x, _y: -x / 300.0,
        sea_level=lambda t: 1.0,
        subsidence_rate_m_myr=0.0,
        insolation_w_m2=400.0,
    )

    x_m, y_m = inp.box.axes_m()
    t_myr = np.array([inp.time.time_myr(i) for i in range(inp.time.steps + 1)], dtype=np.float64)

    fname = os.path.join(tmp_path, "out.h5")
    output_sets = {"topography": OutputSpec(slice_=(slice(None), slice(None)), write_interval=2)}

    out = create_h5_output(
        fname,
        tag=inp.tag,
        x_m=x_m,
        y_m=y_m,
        t_myr=t_myr,
        grid_size=inp.box.grid_size,
        n_facies=len(inp.facies),
        output_sets=output_sets,
        steps=inp.time.steps,
    )

    try:
        run_model_with_writers(
            CAPModel(inp),
            state_writer=h5_state_writer(out, output_sets),
            frame_writer=h5_frame_writer(out, output_sets),
        )
    finally:
        out.fid.close()

    assert os.path.exists(fname)
