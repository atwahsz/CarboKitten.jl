from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from .output import OutputSpec


@dataclass(frozen=True, slots=True)
class Header:
    tag: str
    x_m: np.ndarray
    y_m: np.ndarray
    t_myr: np.ndarray
    grid_size: tuple[int, int]
    n_facies: int
    time_steps: int
    attributes: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DataSet:
    name: str
    slice_spec: str
    write_interval: int
    production_m: np.ndarray
    disintegration_m: np.ndarray
    deposition_m: np.ndarray
    sediment_thickness_m: np.ndarray


def read_header(filename: str) -> Header:
    """Read header metadata from an output HDF5 file.

    This matches the Python writer in `carbokitten.h5writer` and is compatible with
    the basic Julia layout (axes under `input/x`, `input/y`, `input/t`).
    """

    with h5py.File(filename, "r") as f:
        g = f["input"]
        x = np.asarray(g["x"], dtype=np.float64)
        y = np.asarray(g["y"], dtype=np.float64)
        t = np.asarray(g["t"], dtype=np.float64)
        attrs = {k: g.attrs[k] for k in g.attrs.keys()}
        tag = str(attrs.get("tag", ""))
        n_facies = int(attrs.get("n_facies", 0))
        time_steps = int(attrs.get("time_steps", len(t) - 1))

    return Header(
        tag=tag,
        x_m=x,
        y_m=y,
        t_myr=t,
        grid_size=(int(x.size), int(y.size)),
        n_facies=n_facies,
        time_steps=time_steps,
        attributes=attrs,
    )


def list_datasets(filename: str) -> list[str]:
    with h5py.File(filename, "r") as f:
        return [k for k in f.keys() if k != "input"]


def read_dataset(filename: str, name: str) -> DataSet:
    with h5py.File(filename, "r") as f:
        g = f[name]
        slice_spec = str(g.attrs.get("slice", ""))
        write_interval = int(g.attrs.get("write_interval", 1))
        return DataSet(
            name=name,
            slice_spec=slice_spec,
            write_interval=write_interval,
            production_m=np.asarray(g["production"], dtype=np.float64),
            disintegration_m=np.asarray(g["disintegration"], dtype=np.float64),
            deposition_m=np.asarray(g["deposition"], dtype=np.float64),
            sediment_thickness_m=np.asarray(g["sediment_thickness"], dtype=np.float64),
        )


def write_csv_sediment_accumulation_curve(
    filename_csv: str,
    *,
    header: Header,
    thickness_m: np.ndarray,
    label: str = "loc",
) -> None:
    """Write a simple SAC CSV (pure stdlib, no pandas).

    thickness_m should be 1D with length time_steps+1.
    """

    import csv

    if thickness_m.ndim != 1:
        raise ValueError("thickness_m must be 1D")

    with open(filename_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestep", "time_myr", f"sac_{label}_m"])
        for i, (t, h) in enumerate(zip(header.t_myr, thickness_m, strict=False)):
            w.writerow([i, float(t), float(h)])
