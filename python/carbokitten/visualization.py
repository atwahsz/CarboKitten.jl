from __future__ import annotations

"""Visualization helpers.

Julia CarboKitten uses Makie extensions for plotting. This Python port keeps
visualization optional.

If you want plots, install matplotlib and use the exported data from HDF5 or
MemoryOutput.
"""


def summary_plot(*_args, **_kwargs) -> None:
    raise RuntimeError(
        "Visualization is optional in the Python port. "
        "Install matplotlib and plot arrays from HDF5 (carbokitten.export) or MemoryOutput (carbokitten.output)."
    )
