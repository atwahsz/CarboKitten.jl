from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _push_column(col: np.ndarray, parcel: np.ndarray) -> None:
    """Push a parcel into one (stack, facies) column.

    Parameters
    - col: (stack_size, n_facies) float64, each stack level sums to <= 1
    - parcel: (n_facies,) float64, total mass is in "stack units" (dimensionless)

    This follows the Julia implementation in `src/SedimentStack.jl`.
    """

    mass = 0.0
    for f in range(parcel.size):
        mass += parcel[f]
    if mass == 0.0:
        return

    stack_size = col.shape[0]

    if mass > stack_size:
        # huge parcel: overwrite the column with the parcel fractions
        inv = 1.0 / mass
        for f in range(parcel.size):
            col[:, f] = 0.0
        for f in range(parcel.size):
            col[:, f] = parcel[f] * inv
        return

    bucket = 0.0
    for f in range(parcel.size):
        bucket += col[0, f]

    if bucket + mass < 1.0:
        for f in range(parcel.size):
            col[0, f] += parcel[f]
        return

    # fill remainder of top bucket with fraction of parcel
    inv = 1.0 / mass
    frac = parcel * inv
    rem = 1.0 - bucket
    for f in range(parcel.size):
        col[0, f] += frac[f] * rem

    mass = mass - rem
    n_full = int(mass)  # floor

    # shift down by n_full
    # Julia: col[n+2:end,:] .= col[1:end-n-1,:]
    # Here 0-based: move existing down.
    # target start = n_full+1
    for i in range(stack_size - 1, n_full, -1):
        for f in range(parcel.size):
            col[i, f] = col[i - (n_full + 1), f]

    # fill full layers 1..n_full with frac
    for i in range(1, n_full + 1):
        for f in range(parcel.size):
            col[i, f] = frac[f]

    mass = mass - float(n_full)

    # top bucket becomes remaining fraction
    for f in range(parcel.size):
        col[0, f] = frac[f] * mass


@njit(cache=True)
def _pop_fraction(col: np.ndarray, mass: float, out: np.ndarray) -> None:
    bucket = 0.0
    for f in range(col.shape[1]):
        bucket += col[0, f]

    if mass == 0.0 or bucket == 0.0:
        for f in range(out.size):
            out[f] = 0.0
        return

    # mass must be < bucket
    scale = mass / bucket
    for f in range(col.shape[1]):
        v = scale * col[0, f]
        out[f] = v
        col[0, f] -= v


@njit(cache=True)
def _pop_column(col: np.ndarray, amount: float, out: np.ndarray) -> None:
    """Pop an amount (in stack units) from a column into out."""

    bucket = 0.0
    for f in range(col.shape[1]):
        bucket += col[0, f]

    if amount < bucket:
        _pop_fraction(col, amount, out)
        return

    # start with the full top bucket
    for f in range(col.shape[1]):
        out[f] = col[0, f]

    amount = amount - bucket
    n_full = int(amount)

    stack_size = col.shape[0]
    if n_full > (stack_size - 2):
        # pop too much: empty
        for i in range(stack_size):
            for f in range(col.shape[1]):
                out[f] += col[i, f]
                col[i, f] = 0.0
        return

    # add full layers 1..n_full
    for i in range(1, n_full + 1):
        for f in range(col.shape[1]):
            out[f] += col[i, f]

    # shift remaining up
    # Julia: col[1:end-n-1, :] = col[n+2:end, :]
    for i in range(0, stack_size - (n_full + 1)):
        for f in range(col.shape[1]):
            col[i, f] = col[i + (n_full + 1), f]

    # zero out tail
    for i in range(stack_size - (n_full + 1), stack_size):
        for f in range(col.shape[1]):
            col[i, f] = 0.0

    amount = amount - float(n_full)

    # pop remaining fraction from new top
    tmp = np.empty(col.shape[1], dtype=np.float64)
    _pop_fraction(col, amount, tmp)
    for f in range(col.shape[1]):
        out[f] += tmp[f]


@njit(cache=True)
def push_sediment(buffer: np.ndarray, parcel: np.ndarray) -> None:
    """Push sediment parcels into the full 4D buffer.

    Parameters
    - buffer: (stack_size, n_facies, nx, ny)
    - parcel: (n_facies, nx, ny)
    """

    stack_size, n_f, nx, ny = buffer.shape
    for x in range(nx):
        for y in range(ny):
            col = buffer[:, :, x, y]
            p = parcel[:, x, y]
            _push_column(col, p)


@njit(cache=True)
def pop_sediment(buffer: np.ndarray, amount: np.ndarray, out: np.ndarray) -> None:
    """Pop sediment from the full 4D buffer.

    Parameters
    - buffer: (stack_size, n_facies, nx, ny)
    - amount: (nx, ny) in stack units
    - out: (n_facies, nx, ny) output in stack units
    """

    _, n_f, nx, ny = buffer.shape
    tmp = np.empty(n_f, dtype=np.float64)
    for x in range(nx):
        for y in range(ny):
            col = buffer[:, :, x, y]
            _pop_column(col, amount[x, y], tmp)
            for f in range(n_f):
                out[f, x, y] = tmp[f]
