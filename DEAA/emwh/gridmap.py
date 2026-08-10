"""Mapping between MEEP Yee-grid positions and numpy array indices.

The EM-WaveHoltz iteration reads a field off the Yee grid, filters it, and
injects it back as initial data.  Both halves of that round trip must land on
*exactly* the same points: any interpolation in between silently replaces the
WaveHoltz operator with (interpolation o Pi), and the fixed point is no longer
the discrete frequency-domain solution.

So readback uses ``add_dft_fields(..., yee_grid=True)`` and injection uses
``Simulation.initialize_field``, which writes at the component's canonical Yee
locations.

Getting the two indexed consistently needs care, because MEEP will not simply
tell you where the Yee points are:

* ``get_array`` and ``get_array_metadata`` both work on the *centered* grid
  (16x16 for a 2x2 cell at resolution 8), not the Yee grid (17x17).  Passing
  ``dft_cell=`` to ``get_array_metadata`` still returns centered coordinates,
  so it cannot supply this mapping.
* ``initialize_field``'s callback, however, is invoked at exactly the Yee
  points.  Recording those positions recovers the lattice, and it matches the
  shape of ``get_dft_array(..., yee_grid=True)`` for every component tested
  (Ez at nodes, Hx offset half a cell in y, Hy offset half a cell in x), with
  and without PML.

One wrinkle: with PML the callback visits some points more than once (chunk
boundary copies -- 361 visits over a 289-point lattice in the 2x2/res-8 case).
Those are per-chunk copies of shared points, each written once in its own
chunk, so ``initialize_field``'s ``+=`` does not double-write; injecting the
constant 1 reads back as 1, not 2.
"""

import numpy as np

_ROUND = 9


class ComponentGrid:
    """Yee-point coordinates of a single field component."""

    def __init__(self, xs, ys, component=None):
        self.component = component
        self.xs = np.asarray(xs, dtype=float)
        self.ys = np.asarray(ys, dtype=float)
        self.shape = (self.xs.size, self.ys.size)
        self.x0, self.dx = self._origin_and_spacing(self.xs)
        self.y0, self.dy = self._origin_and_spacing(self.ys)

    @staticmethod
    def _origin_and_spacing(coords):
        if coords.size < 2:
            return (coords[0] if coords.size else 0.0), 1.0
        spacing = np.diff(coords)
        if not np.allclose(spacing, spacing[0], rtol=1e-7, atol=1e-10):
            raise ValueError(
                "non-uniform Yee coordinates: spacing in "
                f"[{spacing.min()!r}, {spacing.max()!r}]"
            )
        return coords[0], spacing[0]

    def index(self, x, y):
        return (
            int(round((x - self.x0) / self.dx)),
            int(round((y - self.y0) / self.dy)),
        )

    def positions(self):
        """(nx, ny, 2) array of the Yee coordinates, matching array indexing."""
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        return np.stack([gx, gy], axis=-1)

    def zeros(self):
        return np.zeros(self.shape, dtype=float)

    def __repr__(self):
        return (
            f"ComponentGrid({self.component}, shape={self.shape}, "
            f"x=[{self.xs[0]:.4f},{self.xs[-1]:.4f}], "
            f"y=[{self.ys[0]:.4f},{self.ys[-1]:.4f}])"
        )


def component_grid(sim, component):
    """Recover a component's Yee lattice from ``initialize_field``'s visits.

    Injects zeros, so it is safe to call on a live simulation.
    """
    pts = []

    def recorder(p):
        pts.append((p.x, p.y))
        return 0.0

    sim.initialize_field(component, recorder)
    if not pts:
        raise RuntimeError(f"initialize_field visited no points for {component}")
    arr = np.asarray(pts)
    xs = np.unique(np.round(arr[:, 0], _ROUND))
    ys = np.unique(np.round(arr[:, 1], _ROUND))
    return ComponentGrid(xs, ys, component=component)


def make_injector(grid, arr):
    """Build the ``initialize_field`` callback that writes ``arr`` onto ``grid``.

    Returns ``(callback, stats)``; ``stats['misses']`` counts callback positions
    that fell outside ``arr``, which would mean initial data is being silently
    dropped.  The tests assert it stays zero.
    """
    nx, ny = grid.shape
    x0, dx = grid.x0, grid.dx
    y0, dy = grid.y0, grid.dy
    stats = {"calls": 0, "misses": 0}

    def callback(p):
        stats["calls"] += 1
        i = int(round((p.x - x0) / dx))
        j = int(round((p.y - y0) / dy))
        if 0 <= i < nx and 0 <= j < ny:
            return complex(arr[i, j])
        stats["misses"] += 1
        return 0.0 + 0.0j

    return callback, stats
