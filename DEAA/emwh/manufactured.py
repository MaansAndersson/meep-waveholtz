"""Manufactured solution for the PEC cavity (paper Sec. III-C).

On the unit square with eps = mu = 1 and PEC walls, take

    u(x,y) = 16 x^2 (x-1)^2 y^2 (y-1)^2

which vanishes on the boundary, so it is PEC-compatible.  The WaveHoltz limit
solves  Lap(u) + omega^2 u = omega J, hence

    J = 16 omega x^2(x-1)^2 y^2(y-1)^2
        + (32/omega) [ (6x^2-6x+1) y^2(y-1)^2 + (6y^2-6y+1) x^2(x-1)^2 ]

which is exactly the forcing printed in the paper (verified numerically to
finite-difference truncation error).

MEEP cells are centred on the origin, so the paper's [0,1]^2 maps to
[-1/2, 1/2]^2 with x_paper = x_meep + 1/2.
"""

import math

import meep as mp
import numpy as np

SHIFT = 0.5


def exact(x, y):
    """u on paper coordinates (x, y) in [0,1]^2."""
    return 16.0 * x**2 * (x - 1.0) ** 2 * y**2 * (y - 1.0) ** 2


def forcing(x, y, omega):
    """J on paper coordinates."""
    fx = x**2 * (x - 1.0) ** 2
    fy = y**2 * (y - 1.0) ** 2
    return 16.0 * omega * fx * fy + (32.0 / omega) * (
        (6.0 * x**2 - 6.0 * x + 1.0) * fy + (6.0 * y**2 - 6.0 * y + 1.0) * fx
    )


def exact_on_grid(grid):
    """Evaluate u at a ComponentGrid's Yee points (MEEP coordinates)."""
    gx, gy = np.meshgrid(grid.xs + SHIFT, grid.ys + SHIFT, indexing="ij")
    return exact(gx, gy)


def manufactured_simulation(omega, resolution, courant=0.5):
    """PEC unit-square cavity driven by the manufactured volume source.

    sin-forcing: MEEP drives Re[amplitude * exp(-i omega t)], so amplitude=1j
    gives sin(omega t) * J, matching paper eq. (4a).
    """
    src = mp.ContinuousSource(
        frequency=omega / (2.0 * math.pi), width=0, start_time=0, end_time=1e20
    )

    def amp(p):
        return forcing(p.x + SHIFT, p.y + SHIFT, omega)

    return mp.Simulation(
        cell_size=mp.Vector3(1.0, 1.0, 0),
        resolution=resolution,
        sources=[
            mp.Source(
                src,
                component=mp.Ez,
                center=mp.Vector3(0, 0),
                size=mp.Vector3(1.0, 1.0, 0),
                amp_func=amp,
                amplitude=1j,
            )
        ],
        boundary_layers=[],
        Courant=courant,
        dimensions=2,
    )


def errors(num, ref, h):
    """(L2, Linf) on the interior, plus the same relative to ||ref||.

    L2 is the grid-weighted discrete norm sqrt(h^2 sum e^2), which approximates
    the continuous L2 norm and is therefore resolution independent.
    """
    e = (num - ref)[1:-1, 1:-1]
    r = ref[1:-1, 1:-1]
    l2 = math.sqrt(h * h * float(np.sum(e**2)))
    li = float(np.abs(e).max())
    l2r = math.sqrt(h * h * float(np.sum(r**2)))
    lir = float(np.abs(r).max())
    return l2, li, l2 / l2r, li / lir
