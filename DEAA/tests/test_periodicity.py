"""The definitive correctness test, free of any solver-convention question.

EM-WaveHoltz claims its fixed point is the initial data of the T-periodic
solution of the forced time-domain problem.  That is checkable directly and
entirely inside MEEP: inject the converged nu, step one full window, and the
fields must come back to nu.

This validates the whole chain -- injection, time stepping with the forcing,
the filter, and the fixed-point iteration -- without needing to know how
solve_cw normalises or phases its output.
"""

import math
import pathlib
import sys

import numpy as np
import meep as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from emwh.core import (EMWaveHoltz, TM_FLUX_COMPONENTS, inject_state,
                       tune_courant)
from emwh.gridmap import make_injector
from emwh.problems import TM_COMPONENTS, pec_box_simulation, ring_simulation

NAMES = {mp.Ez: "Ez", mp.Hx: "Hx", mp.Hy: "Hy",
         mp.Dz: "Dz", mp.Bx: "Bx", mp.By: "By"}


def inject(sim, wh, nu):
    """Inject exactly the way apply_pi does, back-out included.

    The state is at integer time; MEEP's H slot holds H(-dt/2), so eq. (24)
    has to be applied here too or the comparison is between two different
    conventions.
    """
    sim.restart_fields()
    if wh.backout:
        nu = wh.half_step_backout(nu)
    inject_state(sim, wh.grids, nu, wh.components)


def snapshot(sim, components):
    return {c: np.asarray(sim.get_array(component=c)).real.copy()
            for c in components}


def check_periodicity(sim, wh, nu, label):
    """Inject nu, run one window, compare the state to nu.

    The "before" snapshot is taken from its own injection: reading H through
    get_array can leave MEEP's magnetic fields synchronized to integer time,
    which would corrupt the stepped run if done on the same pass.
    """
    inject(sim, wh, nu)
    before = snapshot(sim, wh.components)

    inject(sim, wh, nu)  # fresh state, untouched by the read above
    for _ in range(wh.M):
        sim.fields.step()
    after = snapshot(sim, wh.components)

    # Normalise by the norm of the whole state, not per component: for
    # sin-forcing in a lossless cavity nu_H is ~0 by construction, so a
    # per-component ratio divides by nothing and reports a meaningless O(1).
    print(f"  {label}: after one window of {wh.M} steps")
    scale = max(max(np.abs(before[c]).max() for c in wh.components), 1e-300)
    worst = 0.0
    for c in wh.components:
        rel = np.abs(after[c] - before[c]).max() / scale
        worst = max(worst, rel)
        print(f"    {NAMES[c]}: max|E(T) - nu| / max|nu| = {rel:.3e}   "
              f"(max|nu_{NAMES[c]}| = {np.abs(before[c]).max():.4e})")
    return worst


def test_pec_box_periodicity():
    print("\n=== T-periodicity of the fixed point: PEC box ===")
    omega = 2 * math.pi * 0.4
    resolution, n_periods = 8, 3
    courant, _, _ = tune_courant(omega, resolution, n_periods, 0.5)
    sim = pec_box_simulation(omega=omega, resolution=resolution, size=1.0,
                             courant=courant, forcing="sin")
    wh = EMWaveHoltz(sim, omega, components=TM_COMPONENTS, n_periods=n_periods)
    nu, iters = wh.solve(tol=1e-13, maxiter=4000, verbose=False)
    print(f"  converged in {iters} iterations, rel {wh.history[-1]:.2e}")
    worst = check_periodicity(sim, wh, nu, "PEC box")
    # Not exact: injection applies the eq. (24) half-step back-out, whose
    # stencil has no partner in the last row/column, so the state round trip
    # is inexact at the grid edge.  The residual defect is small and is not
    # what the convergence study measures.
    print(f"  worst relative periodicity defect: {worst:.3e}")
    assert worst < 5e-3, f"fixed point is far from T-periodic ({worst:.3e})"
    print("  PASS: the fixed point is T-periodic to the back-out edge error")


def test_ring_periodicity(resolution=5, n_periods=5, tol=1e-9, maxiter=1500):
    print("\n=== T-periodicity of the fixed point: ring resonator with PML ===")
    from emwh.problems import OMEGA0

    omega = OMEGA0
    courant, _, _ = tune_courant(omega, resolution, n_periods, 0.5)
    sim = ring_simulation(omega=omega, resolution=resolution, dpml=1.0,
                          courant=courant, forcing="cos")
    # The ring has eps = 3.4**2, so the state has to be the flux fields: an
    # (Ez,Hx,Hy) state drops a factor of eps on every injection and diverges
    # (61x per application, STATUS.md).
    wh = EMWaveHoltz(sim, omega, components=TM_FLUX_COMPONENTS,
                     n_periods=n_periods)
    nu, iters = wh.solve(tol=tol, maxiter=maxiter, verbose=False)
    print(f"  converged in {iters} iterations, rel {wh.history[-1]:.2e}")
    worst = check_periodicity(sim, wh, nu, "ring")
    # With PML the auxiliary variables are outside the filtered state and are
    # zeroed every window, so exact periodicity is not expected -- but the
    # defect should be small and should shrink as the filter window grows.
    print(f"  worst relative periodicity defect: {worst:.3e}")
    return worst


if __name__ == "__main__":
    test_pec_box_periodicity()
    test_ring_periodicity()
