"""Recover the correct Pi0 from the MEEP FDFD solution, and diff it against ours.

The fixed point satisfies  nu* = S nu* + Pi0,  so

    Pi0_target = (I - S) nu*

with nu* the discrete frequency-domain solution.  Both ingredients are already
verified independently:

  * S is the source-free operator (Pi with no sources, since then Pi 0 = 0),
    shown symmetric positive definite and correctly contractive;
  * nu* comes from MEEP's solve_cw, which notes/pec_convergence.tex shows
    converges at second order to the manufactured solution with amplitude -> 1.
    solve_cw finds the CW state of the *same* discretised stepper, so it is the
    exact discrete fixed point, not merely a second-order approximation.

Comparing Pi0_target with the Pi0 our code computes isolates the defect to the
forcing, and its structure says what kind of defect it is:

  * a constant ratio            -> source amplitude normalisation
  * a fit to a*Pi0_sin + b*Pi0_cos with small residual
                                -> the source phase (the exp(-i w dt/2)
                                   half-step offset mixing sin- and cos-forcing)
  * error localised where J lives -> source injection
"""

import math
import pathlib
import sys

import numpy as np
import meep as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from emwh.core import EMWaveHoltz
from emwh.manufactured import (SHIFT, errors, exact_on_grid, forcing,
                               manufactured_simulation)

OMEGA = 2 * math.pi * 0.4
COURANT = 0.5


def source_free(resolution):
    return mp.Simulation(
        cell_size=mp.Vector3(1.0, 1.0, 0), resolution=resolution,
        sources=[], boundary_layers=[], Courant=COURANT, dimensions=2,
    )


def pi0_of(sim, resolution):
    """Pi applied to zero initial data, i.e. the affine constant."""
    wh = EMWaveHoltz(sim, OMEGA, components=(mp.Ez,), n_periods=1)
    assert wh.M == 5 * resolution
    return wh, wh.apply_pi(wh.zero())[mp.Ez]


def fdfd_solution(resolution, grid):
    sim = manufactured_simulation(OMEGA, resolution, courant=COURANT)
    sim.force_complex_fields = True
    sim.init_sim()
    sim.solve_cw(1e-11, 20000, 10)
    v = np.empty(grid.shape, dtype=complex)
    for i, x in enumerate(grid.xs):
        for j, y in enumerate(grid.ys):
            v[i, j] = sim.fields.get_field(mp.Ez, mp.vec(float(x), float(y)))
    return v


def rel(a, b, h):
    return errors(a, b, h)[2]


def main(resolution=32):
    print(f"resolution {resolution}, omega {OMEGA:.6f}, Courant {COURANT}")

    # ---- our Pi0, for both source phases -----------------------------------
    wh_sin, pi0_sin = pi0_of(
        manufactured_simulation(OMEGA, resolution, courant=COURANT), resolution)
    grid = wh_sin.grids[mp.Ez]
    h = grid.dx

    sim_cos = manufactured_simulation(OMEGA, resolution, courant=COURANT)
    for s in sim_cos.sources:      # flip 1j -> 1, i.e. sin-forcing -> cos
        s.amplitude = 1.0
    _, pi0_cos = pi0_of(sim_cos, resolution)

    # ---- S, from the source-free simulation --------------------------------
    wh_free = EMWaveHoltz(source_free(resolution), OMEGA,
                          components=(mp.Ez,), n_periods=1)
    assert wh_free.grids[mp.Ez].shape == grid.shape
    z = wh_free.apply_pi(wh_free.zero())[mp.Ez]
    print(f"  sanity: |Pi 0| with no sources = {np.abs(z).max():.3e} (must be 0)")

    def S(v):
        return wh_free.apply_pi({mp.Ez: v})[mp.Ez]

    # ---- nu* from FDFD, and the target Pi0 ---------------------------------
    vc = fdfd_solution(resolution, grid)
    u = exact_on_grid(grid)
    for part, arr in (("Re", vc.real), ("Im", vc.imag)):
        print(f"  solve_cw {part}: rel err vs exact = {rel(arr, u, h):.4e}")
    nu_star = vc.real.copy()

    pi0_target = nu_star - S(nu_star)

    # ---- compare -----------------------------------------------------------
    print("\n  magnitudes")
    for name, a in (("Pi0 target (I-S)nu*", pi0_target),
                    ("Pi0 ours, sin-forcing", pi0_sin),
                    ("Pi0 ours, cos-forcing", pi0_cos)):
        print(f"    {name:24s} max {np.abs(a).max():.6e}  "
              f"L2 {math.sqrt(h*h*float(np.sum(a**2))):.6e}")

    denom = np.abs(pi0_target).max()
    d = np.abs(pi0_sin - pi0_target).max() / denom
    s1 = float(np.sum(pi0_sin * pi0_target) / np.sum(pi0_target**2))
    print(f"\n  ours(sin) vs target: max rel diff {d:.4e}, best-fit scale "
          f"{s1:.6f}, residual after scaling "
          f"{np.abs(pi0_sin/s1 - pi0_target).max()/denom:.4e}")

    # two-parameter fit: is the target a mix of the two source phases?
    A = np.stack([pi0_sin.ravel(), pi0_cos.ravel()], axis=1)
    coef, *_ = np.linalg.lstsq(A, pi0_target.ravel(), rcond=None)
    resid = np.abs(A @ coef - pi0_target.ravel()).max() / denom
    print(f"  target ~ a*Pi0_sin + b*Pi0_cos:  a = {coef[0]:+.6f}, "
          f"b = {coef[1]:+.6f},  residual {resid:.4e}")
    theta = OMEGA * wh_sin.sim.fields.dt / 2
    print(f"    (half-step phase: cos(w dt/2) = {math.cos(theta):.6f}, "
          f"sin(w dt/2) = {math.sin(theta):.6f})")
    print(f"    fitted angle atan2(b,a) = {math.atan2(coef[1], coef[0]):+.6f} rad"
          f"   vs w dt/2 = {theta:.6f}")

    # is the discrepancy localised where J is?
    gx, gy = np.meshgrid(grid.xs + SHIFT, grid.ys + SHIFT, indexing="ij")
    Jg = forcing(gx, gy, OMEGA)
    diff = pi0_sin - pi0_target
    cj = float(np.sum(diff * Jg) / np.sum(Jg * Jg))
    print(f"\n  diff vs J: best-fit {cj:+.6e}, residual "
          f"{np.abs(diff - cj*Jg).max()/np.abs(diff).max():.4e}")
    cu = float(np.sum(diff * u) / np.sum(u * u))
    print(f"  diff vs u: best-fit {cu:+.6e}, residual "
          f"{np.abs(diff - cu*u).max()/np.abs(diff).max():.4e}")

    # ---- and the payoff: iterate with the corrected Pi0 --------------------
    nu = np.zeros_like(pi0_target)
    for k in range(200):
        new = S(nu) + pi0_target
        if np.abs(new - nu).max() < 1e-14:
            nu = new
            break
        nu = new
    print(f"\n  iterating nu <- S nu + Pi0_target: {k+1} iters, "
          f"rel err vs exact {rel(nu, u, h):.4e}, vs nu* {rel(nu, nu_star, h):.4e}")
    print(f"  (ours, same grid, was {rel(wh_sin.solve(tol=1e-13, maxiter=3000, verbose=False)[0][mp.Ez], u, h):.4e})")


if __name__ == "__main__":
    main()
