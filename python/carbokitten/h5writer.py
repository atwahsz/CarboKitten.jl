from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from .output import OutputSpec


def _axis_size(s: int | slice, n: int) -> int:
    if isinstance(s, int):
        return 1
    start, stop, step = s.indices(n)
    if step != 1:
        raise ValueError("step slices not supported")
    return max(0, stop - start)


def _slice_to_str(s: int | slice) -> str:
    if isinstance(s, int):
        return str(s)
    # only support contiguous slices
    if s == slice(None):
        return ":"
    start, stop, step = s.start, s.stop, s.step
    if step not in (None, 1):
        raise ValueError("step slices not supported")
    if start is None or stop is None:
        raise ValueError("slice must be bounded or ':'")
    return f"{start}:{stop - 1}"  # mimic Julia's inclusive range text


@dataclass(slots=True)
class H5Output:
    """HDF5 output writer close to Julia `H5Writer` layout."""

    fid: h5py.File
    n_facies: int
    grid_size: tuple[int, int]


def create_h5_output(
    filename: str,
    *,
    tag: str,
    x_m: np.ndarray,
    y_m: np.ndarray,
    t_myr: np.ndarray,
    grid_size: tuple[int, int],
    n_facies: int,
    output_sets: dict[str, OutputSpec],
    steps: int,
) -> H5Output:
    fid = h5py.File(filename, "w")
    fid.require_group("input")

    # Minimal header; callers can add more datasets/attrs under input/.
    fid.attrs["tag"] = tag
    fid["input"].create_dataset("x_m", data=np.asarray(x_m, dtype=np.float64))
    fid["input"].create_dataset("y_m", data=np.asarray(y_m, dtype=np.float64))
    fid["input"].create_dataset("t_myr", data=np.asarray(t_myr, dtype=np.float64))

    out = H5Output(fid=fid, n_facies=n_facies, grid_size=grid_size)

    nx, ny = grid_size
    for name, spec in output_sets.items():
        grp = fid.require_group(str(name))
        grp.attrs["slice"] = ",".join([_slice_to_str(spec.slice_[0]), _slice_to_str(spec.slice_[1])])
        grp.attrs["write_interval"] = int(spec.write_interval)

        sx = _axis_size(spec.slice_[0], nx)
        sy = _axis_size(spec.slice_[1], ny)
        nw = steps // int(spec.write_interval)

        grp.create_dataset(
            "production",
            shape=(n_facies, sx, sy, nw),
            dtype=np.float64,
            chunks=(n_facies, sx, sy, 1),
            compression="gzip",
            compression_opts=3,
        )
        grp.create_dataset(
            "disintegration",
            shape=(n_facies, sx, sy, nw),
            dtype=np.float64,
            chunks=(n_facies, sx, sy, 1),
            compression="gzip",
            compression_opts=3,
        )
        grp.create_dataset(
            "deposition",
            shape=(n_facies, sx, sy, nw),
            dtype=np.float64,
            chunks=(n_facies, sx, sy, 1),
            compression="gzip",
            compression_opts=3,
        )
        grp.create_dataset(
            "sediment_thickness",
            shape=(sx, sy, nw + 1),
            dtype=np.float64,
            chunks=(sx, sy, 1),
            compression="gzip",
            compression_opts=3,
        )

    return out


def h5_state_writer(out: H5Output, output_sets: dict[str, OutputSpec]):
    def _write(idx_1: int, state: Any) -> None:
        nx, ny = out.grid_size
        for name, spec in output_sets.items():
            if (idx_1 - 1) % int(spec.write_interval) != 0:
                continue
            w = (idx_1 - 1) // int(spec.write_interval)
            xsl, ysl = spec.slice_
            sx = _axis_size(xsl, nx)
            sy = _axis_size(ysl, ny)

            a = np.asarray(state.sediment_height_m[xsl, ysl], dtype=np.float64)
            out.fid[str(name)]["sediment_thickness"][:, :, w] = a.reshape((sx, sy))

    return _write


def h5_frame_writer(out: H5Output, output_sets: dict[str, OutputSpec]):
    def _write(idx_1: int, frame: Any) -> None:
        nx, ny = out.grid_size
        for name, spec in output_sets.items():
            if (idx_1 - 1) % int(spec.write_interval) != 0:
                continue
            w = (idx_1 - 1) // int(spec.write_interval)
            xsl, ysl = spec.slice_
            sx = _axis_size(xsl, nx)
            sy = _axis_size(ysl, ny)

            grp = out.fid[str(name)]
            if getattr(frame, "production_m", None) is not None:
                grp["production"][:, :, :, w] += np.asarray(frame.production_m[:, xsl, ysl], dtype=np.float64).reshape(
                    (out.n_facies, sx, sy)
                )
            if getattr(frame, "disintegration_m", None) is not None:
                grp["disintegration"][:, :, :, w] += np.asarray(
                    frame.disintegration_m[:, xsl, ysl], dtype=np.float64
                ).reshape((out.n_facies, sx, sy))
            if getattr(frame, "deposition_m", None) is not None:
                grp["deposition"][:, :, :, w] += np.asarray(frame.deposition_m[:, xsl, ysl], dtype=np.float64).reshape(
                    (out.n_facies, sx, sy)
                )

    return _write
