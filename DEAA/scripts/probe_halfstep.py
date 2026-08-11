"""Is the half-step-late source still a bug, now that D/B injection is fixed?

Two separate questions, kept apart.

(A) Does the offset still exist?  It was measured from one step started from
    rest, which involves no injection at all, so the D/B fix should not have
    changed it -- but that is an assumption worth checking rather than
    asserting.

(B) Does it matter?  Earlier this was measured while the D/B bug dominated
    (errors stagnating at 21%), so the answer was meaningless.  With a working
    second-order solver it can be measured properly: run the manufactured PEC
    problem with and without the phase correction and compare against the exact
    solution.

Prediction worth testing.  A source applied at t = dt instead of t^{n-1/2}
forces sin(w t + phi) with phi = w dt / 2, and the resulting periodic solution
is a rotation in the (sin, cos) response basis:

    nu = cos(phi) * [sin-forced response] + sin(phi) * [cos-forced response].

For a *lossless PEC cavity* the cos-forced response is Re{E} = 0, so the
contamination term vanishes and only the amplitude is affected, by
cos(phi) = 1 - (w dt)^2/8 -- i.e. O(dt^2), the same order as the discretisation
error, so no order reduction and a fixed fraction of the total error.  For an
open problem Re{E} != 0 and the same argument predicts an O(dt) contamination
instead; that case is not tested here (see the note at the end).
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


def sim_with_phase(res, corrected):
    """Manufactured PEC problem; optionally shift the source back by dt/2."""
    sim = manufactured_simulation(OMEGA, res, courant=COURANT)
    if corrected:
        dt = COURANT / res
        for s in sim.sources:
            s.amplitude = 1j * np.exp(1j * OMEGA * dt / 2)
    return sim


def applied_current(res, amp):
    """-E/dt after one step from rest = the current MEEP actually applied."""
    sim = manufactured_simulation(OMEGA, res, courant=COURANT)
    for s in sim.sources:
        s.amplitude = amp
    wh = EMWaveHoltz(sim, OMEGA, components=(mp.Ez,), n_periods=1)
    g = wh.grids[mp.Ez]
    sim.restart_fields()
    sim.fields.step()
    e1 = np.array([[sim.fields.get_field(mp.Ez, mp.vec(float(x), float(y))).real
                    for y in g.ys] for x in g.xs])
    gx, gy = np.meshgrid(g.xs + SHIFT, g.ys + SHIFT, indexing="ij")
    J = forcing(gx, gy, OMEGA)
    m = np.zeros_like(J, dtype=bool)
    m[1:-1, 1:-1] = True          # PEC-clamped ring is not usable for the fit
    j = -e1 / sim.fields.dt
    r = float(np.sum(j[m] * J[m]) / np.sum(J[m] ** 2))
    resid = np.abs(j[m] - r * J[m]).max() / (abs(r) * np.abs(J[m]).max())
    return r, resid, sim.fields.dt


def main():
    print("=== (A) does the half-step offset still exist? ===")
    print(f"{'res':>4} {'dt':>10} {'A':>11} {'sinc':>10} {'t_eff/dt':>9} "
          f"{'fit resid':>10}")
    for res in (8, 16, 32, 64):
        rs, _, dt = applied_current(res, 1j)     # sin-forcing
        rc, resid, _ = applied_current(res, 1.0)  # cos-forcing
        A = math.hypot(rs, rc)
        t_eff = math.atan2(rs, rc) / OMEGA
        sinc = math.sin(OMEGA * dt / 2) / (OMEGA * dt / 2)
        print(f"{res:4d} {dt:10.6f} {A:11.7f} {sinc:10.6f} {t_eff/dt:9.5f} "
              f"{resid:10.2e}")
    print("  t_eff/dt = 1 means the current is applied a full step in, where")
    print("  eq. (23a) wants the half step (t_eff/dt = 0.5).")

    print("\n=== (B) does correcting it change the answer? ===")
    print(f"{'res':>4} {'relL2 as-is':>13} {'relL2 fixed':>13} {'change':>9} "
          f"{'|dnu|/|u| ':>11} {'predicted':>11}")
    prev = None
    for res in (8, 16, 32, 64, 96):
        out = {}
        for corrected in (False, True):
            sim = sim_with_phase(res, corrected)
            wh = EMWaveHoltz(sim, OMEGA, components=(mp.Ez,), n_periods=1)
            nu, _ = wh.solve(tol=1e-13, maxiter=4000, verbose=False)
            g = wh.grids[mp.Ez]
            u = exact_on_grid(g)
            out[corrected] = (nu[mp.Ez], errors(nu[mp.Ez], u, g.dx)[2], u, g.dx)
        (v0, e0, u, h), (v1, e1, _, _) = out[False], out[True]
        d = math.sqrt(h * h * float(np.sum((v1 - v0) ** 2)))
        nu_u = math.sqrt(h * h * float(np.sum(u ** 2)))
        dt = COURANT / res
        pred = 1.0 - math.cos(OMEGA * dt / 2)     # (1 - cos phi) * |u|
        print(f"{res:4d} {e0:13.4e} {e1:13.4e} {e1/e0:9.4f} "
              f"{d/nu_u:11.4e} {pred:11.4e}")

    print("\n  'change' < 1 means the correction helps; |dnu|/|u| is how much")
    print("  the two solutions differ, against the predicted (1 - cos(w dt/2)).")


if __name__ == "__main__":
    main()
