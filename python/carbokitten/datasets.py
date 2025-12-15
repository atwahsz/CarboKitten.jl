from __future__ import annotations

"""Small dataset loading utilities.

The Julia package ships datasets via Julia Artifacts. In this Python port we only
provide thin wrappers around CSV/TSV reading.
"""

from dataclasses import dataclass

import numpy as np


def read_csv(path: str, *, delimiter: str = ",", skip_header: bool = True) -> np.ndarray:
    """Read a numeric CSV into a numpy array."""

    return np.genfromtxt(path, delimiter=delimiter, skip_header=1 if skip_header else 0)


def read_tsv(path: str, *, skip_header: bool = True) -> np.ndarray:
    """Read a numeric TSV into a numpy array."""

    return np.genfromtxt(path, delimiter="\t", skip_header=1 if skip_header else 0)
