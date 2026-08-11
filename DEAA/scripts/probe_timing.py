"""Is dt right, and is the forcing evaluated at the right times, for Pi0?

Two checks.

(1) dt and the window.  dt must equal Courant/resolution and M*dt must equal
    the period exactly, or the filter's forced-mode multiplier is not 1.

(2) The applied current, measured rather than assumed.  From rest, one Yee
    step gives

        E^1 = E^0 + (dt/eps)(curl H^{1/2} - J^{1/2}) = -(dt/eps) J^{1/2},

    because E^0 = 0 makes H^{1/2} = 0 too.  So the field after exactly one step
    *is* the applied current, scaled by -dt.  Running the same thing with
    sin- and cos-forcing gives the two quadratures of the source, hence its
    amplitude and its effective evaluation time:

        J^{1/2} = A [ sin(w t_eff) or cos(w t_eff) ] J(x,y)
        A = hypot(r_sin, r_cos),   t_eff = atan2(r_sin, r_cos)/w

    The Yee convention (paper eq. 23a) wants t_eff = dt/2 and, given how MEEP
    differences its dipole moment, A = sinc(w dt/2).
"""

import math
import pathlib
import sys

import numpy as np
import meep as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from emwh.core import EMWaveHoltz
from emwh.manufactured import SHIFT, forcing, manufactured_simulation

COURANT = 0.5


def first_step_current(omega, res, forcing_kind):
    """-E after one step from rest, i.e. dt * J^{1/2}, on the Ez Yee grid."""
    sim = manufactured_simulation(omega, res, courant=COURANT)
    if forcing_kind == "cos":
        for s in sim.sources:
            s.amplitude = 1.0
    wh = EMWaveHoltz(sim, omega, components=(mp.Ez,), n_periods=1)
    grid = wh.grids[mp.Ez]
    sim.restart_fields()
    sim.fields.step()
    e1 = np.array([[sim.fields.get_field(mp.Ez, mp.vec(float(x), float(y))).real
                    for y in grid.ys] for x in grid.xs])
    return wh, grid, -e1 / sim.fields.dt      # = J^{1/2}


def main(res=32):
    print("=== (1) dt and the filter window ===")
    for f in (0.4, 0.25):
        omega = 2 * math.pi * f
        sim = manufactured_simulation(omega, res, courant=COURANT)
        wh = EMWaveHoltz(sim, omega, components=(mp.Ez,), n_periods=1)
        dt = sim.fields.dt
        T = 2 * math.pi / omega
        print(f"  f={f}: dt={dt!r}")
        print(f"        Courant/res = {COURANT/res!r}   diff {abs(dt-COURANT/res):.3e}")
        print(f"        M={wh.M}  M*dt={wh.M*dt!r}  T={T!r}  "
              f"M*dt-T = {wh.M*dt - T:+.3e}")

    print("\n=== (2) applied current, measured from one step ===")
    print(f"{'f':>6} {'omega':>8} {'dt':>9} {'A':>10} {'sinc(wdt/2)':>12} "
          f"{'t_eff':>10} {'dt/2':>10} {'t_eff/dt':>9} {'resid':>9}")
    for f in (0.1, 0.2, 0.25, 0.4, 0.5):
        omega = 2 * math.pi * f
        wh, grid, j_sin = first_step_current(omega, res, "sin")
        _, _, j_cos = first_step_current(omega, res, "cos")
        dt = wh.sim.fields.dt

        gx, gy = np.meshgrid(grid.xs + SHIFT, grid.ys + SHIFT, indexing="ij")
        Jg = forcing(gx, gy, omega)
        m = np.abs(Jg) > 1e-3 * np.abs(Jg).max()     # avoid dividing by ~0

        r_sin = float(np.sum(j_sin[m] * Jg[m]) / np.sum(Jg[m] ** 2))
        r_cos = float(np.sum(j_cos[m] * Jg[m]) / np.sum(Jg[m] ** 2))
        # how well is J^{1/2} actually proportional to J?
        resid = max(
            np.abs(j_sin[m] - r_sin * Jg[m]).max() / max(abs(r_sin), 1e-30),
            np.abs(j_cos[m] - r_cos * Jg[m]).max() / max(abs(r_cos), 1e-30),
        ) / np.abs(Jg[m]).max()

        A = math.hypot(r_sin, r_cos)
        t_eff = math.atan2(r_sin, r_cos) / omega
        sinc = math.sin(omega * dt / 2) / (omega * dt / 2)
        print(f"{f:6.3f} {omega:8.5f} {dt:9.6f} {A:10.6f} {sinc:12.6f} "
              f"{t_eff:10.6f} {dt/2:10.6f} {t_eff/dt:9.5f} {resid:9.2e}")


if __name__ == "__main__":
    main()
