"""Validate DFTFilter against an independent Python quadrature, and validate
the whole iteration against solve_cw on a PML-free PEC problem.

Layered on purpose, so a discrepancy can be attributed:
  A. DFTFilter vs ProbeFilter  -> MEEP's dt/sqrt(2pi) normalisation, E/H time
     staggering, and the Yee index mapping.
  B. PEC box vs solve_cw       -> the source convention and the fixed point,
     with no PML in play.
"""

import math
import sys

import numpy as np
import meep as mp

sys.path.insert(0, "/Users/appelo/Desktop/MEEP_STUFF")

from emwh.core import EMWaveHoltz, inject_state, tune_courant
from emwh.filters import ProbeFilter
from emwh.gridmap import make_injector
from emwh.problems import pec_box_simulation


def test_dft_matches_python_quadrature():
    print("\n=== A. DFTFilter vs independent Python quadrature ===")
    omega = 2 * math.pi * 0.4
    resolution, n_periods, size = 8, 2, 1.0
    courant, steps, period = tune_courant(omega, resolution, n_periods, 0.5)
    sim = pec_box_simulation(omega=omega, resolution=resolution, size=size,
                             courant=courant, with_sources=True)
    wh = EMWaveHoltz(sim, omega, components=(mp.Ez, mp.Hx, mp.Hy),
                     n_periods=n_periods)

    # A few interior Yee points per component.
    probes = []
    for c in (mp.Ez, mp.Hx, mp.Hy):
        g = wh.grids[c]
        for (i, j) in [(3, 4), (5, 2), (4, 4)]:
            probes.append((c, g.xs[i], g.ys[j], i, j))
    pf = ProbeFilter(sim, omega, wh.T, [(c, x, y) for c, x, y, _, _ in probes])

    # One window from zero initial data, sampling both filters together.
    sim.restart_fields()
    pf.start(wh.M)
    pf.sample(0)
    for n in range(1, wh.M + 1):
        sim.fields.step()
        pf.sample(n)
    dft_out = wh.filter.read(half_step_shift=False)
    probe_out = pf.read()

    print(f"  {wh.M} steps, {n_periods} periods")
    worst = 0.0
    for k, (c, x, y, i, j) in enumerate(probes):
        a = dft_out[c][i, j]
        b = probe_out[k]
        rel = abs(a - b) / max(abs(b), 1e-30)
        worst = max(worst, rel)
        name = {mp.Ez: "Ez", mp.Hx: "Hx", mp.Hy: "Hy"}[c]
        print(f"    {name}[{i},{j}] dft {a:+.10e}  probe {b:+.10e}  rel {rel:.2e}")
    print(f"  worst relative difference: {worst:.3e}")
    assert worst < 1e-9, "DFTFilter disagrees with the Python quadrature"
    print("  PASS: normalisation, staggering and index mapping all agree")


def test_pec_box_against_solve_cw(resolution=8, size=1.0, n_periods=3,
                                  omega=2 * math.pi * 0.4, tol=1e-11,
                                  maxiter=4000, forcing="cos"):
    print("\n=== B. PEC box: EM-WaveHoltz vs solve_cw (no PML) ===")
    courant, steps, period = tune_courant(omega, resolution, n_periods, 0.5)

    sim = pec_box_simulation(omega=omega, resolution=resolution, size=size,
                             courant=courant, forcing=forcing)
    wh = EMWaveHoltz(sim, omega, components=(mp.Ez, mp.Hx, mp.Hy),
                     n_periods=n_periods)
    nu, iters = wh.solve(tol=tol, maxiter=maxiter, verbose=False)
    print(f"  WaveHoltz converged in {iters} iterations "
          f"(final rel {wh.history[-1]:.2e})")

    sim.restart_fields()
    inject_state(sim, wh.grids, nu, (mp.Ez, mp.Hx, mp.Hy))
    ez_wh = np.asarray(sim.get_array(component=mp.Ez)).real.copy()

    sim2 = pec_box_simulation(omega=omega, resolution=resolution, size=size,
                              courant=courant, forcing=forcing)
    sim2.force_complex_fields = True
    sim2.init_sim()
    sim2.solve_cw(1e-10, 10000, 10)
    ez_cw = np.asarray(sim2.get_array(component=mp.Ez, cmplx=True))
    ref = ez_cw.real if forcing == "cos" else ez_cw.imag

    denom = np.abs(ref).max()
    scale = float(np.sum(ez_wh * ref) / np.sum(ref * ref))
    print(f"  max|WaveHoltz| {np.abs(ez_wh).max():.6e}, max|FDFD| {denom:.6e}")
    print(f"  best-fit scale WaveHoltz/FDFD = {scale:.10f}")
    print(f"  max rel diff (unscaled) = {np.abs(ez_wh - ref).max()/denom:.3e}")
    print(f"  max rel diff (scaled)   = "
          f"{np.abs(ez_wh - scale*ref).max()/denom:.3e}")
    return scale


if __name__ == "__main__":
    test_dft_matches_python_quadrature()
    test_pec_box_against_solve_cw()
