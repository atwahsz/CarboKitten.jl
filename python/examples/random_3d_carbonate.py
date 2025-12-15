from __future__ import annotations

"""Random 3D carbonate model (grain size + RGT).

This script makes two 3D volumes:
- Grain size: random but smooth, like patchy facies.
- RGT (relative geological time): values from 0 (old) to 1 (young).

It also plots 3 slices for each volume:
- Inline (y fixed)
- Xline (x fixed)
- Time slice (t fixed)

How to run
- Install the python port: `python3 -m pip install -e /workspace/python`
- Install matplotlib: `python3 -m pip install matplotlib`
- Run: `python3 -m examples.random_3d_carbonate`

Units
- This is synthetic. Axes are sample indices, not meters.
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Volume3D:
    """A small container for a 3D volume."""

    data: np.ndarray  # (t, x, y)
    name: str
    cmap: str
    vmin: float | None = None
    vmax: float | None = None


def _smooth2d(a: np.ndarray, passes: int = 6) -> np.ndarray:
    """Simple smoothing without extra dependencies.

    This is a few rounds of a 3x3 weighted average.
    """

    if a.ndim != 2:
        raise ValueError("_smooth2d expects a 2D array")

    out = a.astype(np.float64, copy=True)
    for _ in range(int(passes)):
        # pad with edge values
        p = np.pad(out, pad_width=1, mode="edge")
        # weighted average (center weight 4)
        out = (
            p[1:-1, 1:-1] * 4.0
            + p[:-2, 1:-1]
            + p[2:, 1:-1]
            + p[1:-1, :-2]
            + p[1:-1, 2:]
            + p[:-2, :-2]
            + p[:-2, 2:]
            + p[2:, :-2]
            + p[2:, 2:]
        ) / 12.0
    return out


def make_grain_size_volume(
    *,
    nt: int,
    nx: int,
    ny: int,
    seed: int = 0,
) -> Volume3D:
    """Make a random 3D grain-size volume.

    Output
    - data: float64 array shaped (t, x, y)
    - values: 0..1 (coarser to finer, or vice versa)
    """

    rng = np.random.default_rng(seed)

    # build a time-varying spatial field
    vol = np.empty((nt, nx, ny), dtype=np.float64)

    base = rng.normal(size=(nx, ny)).astype(np.float64)
    base = _smooth2d(base, passes=10)
    base = (base - base.min()) / (base.max() - base.min() + 1e-12)

    for t in range(nt):
        layer = 0.65 * base + 0.35 * rng.normal(size=(nx, ny))
        layer = _smooth2d(layer, passes=6)
        layer = (layer - layer.min()) / (layer.max() - layer.min() + 1e-12)

        # add a gentle trend (like shallowing upward)
        trend = 0.15 * (t / max(1, nt - 1))
        vol[t] = np.clip(layer + trend, 0.0, 1.0)

    return Volume3D(data=vol, name="Grain size (random)", cmap="viridis", vmin=0.0, vmax=1.0)


def make_rgt_volume(
    *,
    nt: int,
    nx: int,
    ny: int,
    seed: int = 1,
) -> Volume3D:
    """Make a synthetic RGT (relative geological time) volume.

    RGT should be monotonic with time, but can be warped laterally.
    Values are 0..1.
    """

    rng = np.random.default_rng(seed)

    # Lateral time warp field (seconds-like, but in samples)
    warp = rng.normal(size=(nx, ny)).astype(np.float64)
    warp = _smooth2d(warp, passes=12)
    warp = warp / (np.max(np.abs(warp)) + 1e-12)
    warp *= 0.15 * (nt - 1)  # warp up to ~15% of total time range

    rgt = np.empty((nt, nx, ny), dtype=np.float64)
    for t in range(nt):
        # base is linear time
        base = float(t)
        # apply warp and normalize
        v = (base + warp) / max(1.0, float(nt - 1))
        rgt[t] = np.clip(v, 0.0, 1.0)

    # enforce monotonic increase along time axis
    rgt = np.maximum.accumulate(rgt, axis=0)

    return Volume3D(data=rgt, name="RGT (relative geological time)", cmap="magma", vmin=0.0, vmax=1.0)


def _plot_slices(
    *,
    vol: Volume3D,
    x_idx: int,
    y_idx: int,
    t_idx: int,
    out_png: Path | None,
) -> None:
    """Plot inline, xline, and time slice for a volume."""

    try:
        import matplotlib.pyplot as plt
        import matplotlib
    except ModuleNotFoundError:
        print("matplotlib is not installed. Run: python3 -m pip install matplotlib", file=sys.stderr)
        raise

    data = vol.data
    nt, nx, ny = data.shape

    x_idx = int(np.clip(x_idx, 0, nx - 1))
    y_idx = int(np.clip(y_idx, 0, ny - 1))
    t_idx = int(np.clip(t_idx, 0, nt - 1))

    inline = data[:, :, y_idx]  # (t, x)
    xline = data[:, x_idx, :]  # (t, y)
    tslice = data[t_idx, :, :]  # (x, y)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    fig.suptitle(vol.name)

    im0 = axes[0].imshow(inline, aspect="auto", origin="upper", cmap=vol.cmap, vmin=vol.vmin, vmax=vol.vmax)
    axes[0].set_title(f"Inline (y={y_idx})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("time")

    im1 = axes[1].imshow(xline, aspect="auto", origin="upper", cmap=vol.cmap, vmin=vol.vmin, vmax=vol.vmax)
    axes[1].set_title(f"Xline (x={x_idx})")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("time")

    im2 = axes[2].imshow(tslice, aspect="auto", origin="upper", cmap=vol.cmap, vmin=vol.vmin, vmax=vol.vmax)
    axes[2].set_title(f"Time slice (t={t_idx})")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("x")

    # one shared colorbar
    cbar = fig.colorbar(im2, ax=axes, shrink=0.85)
    cbar.set_label("value")

    backend = matplotlib.get_backend().lower()
    if out_png is not None or "agg" in backend:
        if out_png is None:
            out_png = Path(f"{vol.name.lower().replace(' ', '_')}.png")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"saved: {out_png}")
    else:
        plt.show()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Random 3D carbonate volume demo")
    parser.add_argument("--nt", type=int, default=120)
    parser.add_argument("--nx", type=int, default=160)
    parser.add_argument("--ny", type=int, default=120)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="If set, save PNGs into this folder (useful on headless machines).",
    )
    args = parser.parse_args(argv)

    nt, nx, ny = int(args.nt), int(args.nx), int(args.ny)

    grain = make_grain_size_volume(nt=nt, nx=nx, ny=ny, seed=42)
    rgt = make_rgt_volume(nt=nt, nx=nx, ny=ny, seed=7)

    # pick slice indices
    x_idx = nx // 2
    y_idx = ny // 2
    t_idx = nt // 2

    outdir = args.outdir
    grain_png = None if outdir is None else outdir / "grain_size.png"
    rgt_png = None if outdir is None else outdir / "rgt.png"

    _plot_slices(vol=grain, x_idx=x_idx, y_idx=y_idx, t_idx=t_idx, out_png=grain_png)
    _plot_slices(vol=rgt, x_idx=x_idx, y_idx=y_idx, t_idx=t_idx, out_png=rgt_png)


if __name__ == "__main__":
    main()
