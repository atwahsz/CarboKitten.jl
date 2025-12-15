from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimeProperties:
    """Time integration properties.

    Units
    - t0_myr: mega-years (Myr)
    - dt_myr: mega-years (Myr)

    Notes
    - The Julia implementation writes output on every step by default; this port
      keeps time properties minimal and leaves output scheduling to the caller.
    """

    dt_myr: float
    steps: int
    t0_myr: float = 0.0

    def time_myr(self, step: int) -> float:
        """Compute the model time at an integer step."""

        return self.t0_myr + float(step) * self.dt_myr
