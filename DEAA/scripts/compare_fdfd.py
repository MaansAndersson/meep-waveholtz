"""Ring resonator (Fig. 1): EM-WaveHoltz vs MEEP's own FDFD solver.

Both solvers run on the *same* Simulation settings -- same grid, same subpixel
averaging, same PML, same ContinuousSource -- so this compares the methods, not
the discretizations.  MEEP's ``solve_cw`` finds the CW solution of the
discretized system with BiCGStab-L, which is the same discrete system
EM-WaveHoltz converges to, so agreement should be at solver-tolerance level
rather than merely at discretization-error level.
"""

import argparse
import math
import pathlib
import sys
import time

import numpy as np
import meep as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from emwh.core import (TM_FLUX_COMPONENTS, EMWaveHoltz, inject_state,
                       tune_courant)
from emwh.problems import OMEGA0, ring_simulation


def run_waveholtz(omega, resolution, dpml, n_periods, tol, maxiter, forcing):
    courant, steps, period = tune_courant(omega, resolution, n_periods, 0.5)
    sim = ring_simulation(
        omega=omega, resolution=resolution, dpml=dpml,
        courant=courant, forcing=forcing,
    )
    # flux state: the ring dielectric has eps != 1, so an (E,H) state
    # loses a factor of eps on injection and the iteration diverges
    wh = EMWaveHoltz(sim, omega, components=TM_FLUX_COMPONENTS,
                     n_periods=n_periods, backout=False)
    print(f"  grid {wh.grids[mp.Dz].shape}, {wh.M} steps/window "
          f"({n_periods} periods), Courant {courant:.6f}, dt {sim.fields.dt:.6f}")
    t0 = time.time()
    nu, iters = wh.solve(tol=tol, maxiter=maxiter, log_every=10)
    elapsed = time.time() - t0
    print(f"  EM-WaveHoltz: {iters} iterations, {elapsed:.1f} s")
    for c, name in ((mp.Dz, "Dz"), (mp.Bx, "Bx"), (mp.By, "By")):
        assert wh.inject_stats[c]["misses"] == 0, f"{name} injector missed"

    # Read the converged state back through get_array so it can be compared
    # with solve_cw on identical footing (both on the centered grid).
    sim.restart_fields()
    inject_state(sim, wh.grids, nu, TM_FLUX_COMPONENTS)
    ez = np.asarray(sim.get_array(component=mp.Ez)).real.copy()
    return wh, nu, ez, iters, elapsed


def run_fdfd(omega, resolution, dpml, tol, maxiters, L, forcing):
    courant, _, _ = tune_courant(omega, resolution, 1, 0.5)
    sim = ring_simulation(
        omega=omega, resolution=resolution, dpml=dpml,
        courant=courant, forcing=forcing, complex_fields=True,
    )
    sim.init_sim()
    t0 = time.time()
    sim.solve_cw(tol, maxiters, L)
    elapsed = time.time() - t0
    ez = np.asarray(sim.get_array(component=mp.Ez, cmplx=True))
    print(f"  MEEP FDFD (BiCGStab-{L}): {elapsed:.1f} s")
    return ez, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega-factor", type=float, default=1.0,
                    help="omega / omega_0 (paper uses 1.0, 2.24, 2.7)")
    ap.add_argument("--resolution", type=int, default=10,
                    help="points per unit length; N = 12*resolution over [-6,6]")
    ap.add_argument("--dpml", type=float, default=1.0)
    ap.add_argument("--periods", type=int, default=10)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--cw-tol", type=float, default=1e-8)
    ap.add_argument("--cw-L", type=int, default=10)
    ap.add_argument("--forcing", default="cos", choices=("cos", "sin"))
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    omega = args.omega_factor * OMEGA0
    print(f"\n### ring resonator, omega = {args.omega_factor} * omega_0 "
          f"= {omega:.6f}  (f = {omega / (2 * math.pi):.6f})")
    print(f"### resolution {args.resolution}, dpml {args.dpml}, "
          f"{args.periods} periods, forcing {args.forcing}")

    print("\n[1] EM-WaveHoltz (plain fixed point, no Krylov)")
    wh, nu, ez_wh, iters, t_wh = run_waveholtz(
        omega, args.resolution, args.dpml, args.periods,
        args.tol, args.maxiter, args.forcing,
    )

    print("\n[2] MEEP FDFD")
    ez_cw, t_cw = run_fdfd(
        omega, args.resolution, args.dpml, args.cw_tol, 10000,
        args.cw_L, args.forcing,
    )

    # cos-forcing converges to Re{E} (eq. 11); sin-forcing to Im{E} (eq. 9).
    ref = ez_cw.real if args.forcing == "cos" else ez_cw.imag
    print("\n[3] comparison of Ez")
    print(f"  max|WaveHoltz| = {np.abs(ez_wh).max():.6e}")
    print(f"  max|FDFD|      = {np.abs(ref).max():.6e}")
    denom = np.abs(ref).max()
    err = np.abs(ez_wh - ref).max()
    print(f"  max abs diff   = {err:.6e}")
    print(f"  max rel diff   = {err / denom:.6e}")
    # A pure scale factor would point at a source-normalisation mismatch.
    scale = float(np.sum(ez_wh * ref) / np.sum(ref * ref))
    resid = np.abs(ez_wh - scale * ref).max() / denom
    print(f"  best-fit scale = {scale:.10f}  (residual after scaling "
          f"{resid:.3e})")
    print(f"\n  timing: WaveHoltz {t_wh:.1f} s ({iters} iters), "
          f"FDFD {t_cw:.1f} s")

    if args.save:
        np.savez(args.save, ez_wh=ez_wh, ez_cw=ez_cw,
                 history=np.array(wh.history), omega=omega,
                 resolution=args.resolution)
        print(f"  saved -> {args.save}")


if __name__ == "__main__":
    main()
