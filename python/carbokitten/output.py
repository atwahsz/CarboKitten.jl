from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Slice1 = int | slice
Slice2 = tuple[Slice1, Slice1]


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """Output spec similar to Julia `OutputSpec`.

    Parameters
    - slice_: 2D slice over (x, y). Use ints or `slice(None)`.
    - write_interval: write every N steps.

    Notes
    - Python is 0-based. If you port Julia slices like `(:, 25)`, you probably
      want `(slice(None), 24)` here.
    """

    slice_: Slice2 = (slice(None), slice(None))
    write_interval: int = 1


@dataclass(slots=True)
class MemoryDataSet:
    slice_: Slice2
    write_interval: int
    production_m: np.ndarray  # (n_f, sx, sy, nw)
    disintegration_m: np.ndarray  # (n_f, sx, sy, nw)
    deposition_m: np.ndarray  # (n_f, sx, sy, nw)
    sediment_thickness_m: np.ndarray  # (sx, sy, nw+1)


@dataclass(slots=True)
class MemoryOutput:
    """In-memory output store (similar to Julia `MemoryWriter`)."""

    tag: str
    x_m: np.ndarray
    y_m: np.ndarray
    t_myr: np.ndarray
    grid_size: tuple[int, int]
    n_facies: int
    attributes: dict[str, Any]
    data_sets: dict[str, MemoryDataSet]


def _axis_size(s: Slice1, n: int) -> int:
    if isinstance(s, int):
        return 1
    if isinstance(s, slice):
        start, stop, step = s.indices(n)
        if step != 1:
            raise ValueError("step slices not supported")
        return max(0, stop - start)
    raise TypeError("invalid slice")


def new_memory_output(
    *,
    tag: str,
    x_m: np.ndarray,
    y_m: np.ndarray,
    t_myr: np.ndarray,
    grid_size: tuple[int, int],
    n_facies: int,
) -> MemoryOutput:
    return MemoryOutput(
        tag=tag,
        x_m=x_m,
        y_m=y_m,
        t_myr=t_myr,
        grid_size=grid_size,
        n_facies=n_facies,
        attributes={},
        data_sets={},
    )


def add_memory_data_set(out: MemoryOutput, name: str, spec: OutputSpec, *, steps: int) -> None:
    nx, ny = out.grid_size
    sx = _axis_size(spec.slice_[0], nx)
    sy = _axis_size(spec.slice_[1], ny)
    nw = steps // int(spec.write_interval)

    out.data_sets[name] = MemoryDataSet(
        slice_=spec.slice_,
        write_interval=int(spec.write_interval),
        production_m=np.zeros((out.n_facies, sx, sy, nw), dtype=np.float64),
        disintegration_m=np.zeros((out.n_facies, sx, sy, nw), dtype=np.float64),
        deposition_m=np.zeros((out.n_facies, sx, sy, nw), dtype=np.float64),
        sediment_thickness_m=np.zeros((sx, sy, nw + 1), dtype=np.float64),
    )


def memory_state_writer(out: MemoryOutput):
    """Return a `(idx, state) -> None` writer like Julia `state_writer`."""

    def _write(idx_1: int, state: Any) -> None:
        for ds in out.data_sets.values():
            if (idx_1 - 1) % ds.write_interval != 0:
                continue
            w = (idx_1 - 1) // ds.write_interval
            xsl, ysl = ds.slice_
            ds.sediment_thickness_m[:, :, w] = np.asarray(state.sediment_height_m[xsl, ysl], dtype=np.float64)

    return _write


def memory_frame_writer(out: MemoryOutput):
    """Return a `(idx, frame) -> None` writer like Julia `frame_writer`.

    This writer accumulates frame arrays into the per-interval bins.
    """

    def _write(idx_1: int, frame: Any) -> None:
        for ds in out.data_sets.values():
            if (idx_1 - 1) % ds.write_interval != 0:
                continue
            w = (idx_1 - 1) // ds.write_interval
            xsl, ysl = ds.slice_

            if getattr(frame, "production_m", None) is not None:
                ds.production_m[:, :, :, w] += np.asarray(frame.production_m[:, xsl, ysl], dtype=np.float64)
            if getattr(frame, "disintegration_m", None) is not None:
                ds.disintegration_m[:, :, :, w] += np.asarray(frame.disintegration_m[:, xsl, ysl], dtype=np.float64)
            if getattr(frame, "deposition_m", None) is not None:
                ds.deposition_m[:, :, :, w] += np.asarray(frame.deposition_m[:, xsl, ysl], dtype=np.float64)

    return _write
