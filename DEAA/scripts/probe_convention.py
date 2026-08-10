"""Which part of solve_cw's complex field does the WaveHoltz fixed point equal?

Paper convention is exp(+i w t) (eq. 1: i w eps E = curl H - J); MEEP drives
exp(-i w t).  For a lossless PEC cavity with real eps, mu, J the paper's
analysis says E is purely imaginary and H purely real, so sin-forcing should
recover Im{E} and cos-forcing should recover Re{H}.  Measure it rather than
argue about signs.
"""

import math
import sys

import numpy as np
import meep as mp

sys.path.insert(0, "/Users/appelo/Desktop/MEEP_STUFF")

from emwh.core import EMWaveHoltz, inject_state, tune_courant
from emwh.gridmap import make_injector
from emwh.problems import pec_box_simulation


def fit(a, b):
    """best-fit scale a ~ s*b, and the residual after scaling"""
    denom = float(np.sum(b * b))
    if denom == 0:
        return 0.0, float("inf")
    s = float(np.sum(a * b) / denom)
    scale = max(np.abs(b).max(), 1e-300)
    return s, float(np.abs(a - s * b).max() / scale)


def main(resolution=8, size=1.0, n_periods=3, omega=2 * math.pi * 0.4):
    courant, steps, period = tune_courant(omega, resolution, n_periods, 0.5)

    # FDFD reference once (the source phase does not change the CW solution
    # beyond an overall factor, but build it per forcing to be safe).
    for forcing in ("cos", "sin"):
        print(f"\n### forcing = {forcing}")
        sim = pec_box_simulation(omega=omega, resolution=resolution, size=size,
                                 courant=courant, forcing=forcing)
        wh = EMWaveHoltz(sim, omega, components=(mp.Ez, mp.Hx, mp.Hy),
                         n_periods=n_periods)
        nu, iters = wh.solve(tol=1e-12, maxiter=4000, verbose=False)
        print(f"  converged in {iters} iters, rel {wh.history[-1]:.2e}")

        sim.restart_fields()
        inject_state(sim, wh.grids, nu, (mp.Ez, mp.Hx, mp.Hy))
        wh_arr = {c: np.asarray(sim.get_array(component=c)).real.copy()
                  for c in (mp.Ez, mp.Hx, mp.Hy)}

        sim2 = pec_box_simulation(omega=omega, resolution=resolution, size=size,
                                  courant=courant, forcing=forcing)
        sim2.force_complex_fields = True
        sim2.init_sim()
        sim2.solve_cw(1e-10, 10000, 10)

        for c, name in ((mp.Ez, "Ez"), (mp.Hx, "Hx"), (mp.Hy, "Hy")):
            cw = np.asarray(sim2.get_array(component=c, cmplx=True))
            a = wh_arr[c]
            sr, rr = fit(a, cw.real)
            si, ri = fit(a, cw.imag)
            print(f"  {name}: max|WH| {np.abs(a).max():.4e}  "
                  f"max|Re cw| {np.abs(cw.real).max():.4e}  "
                  f"max|Im cw| {np.abs(cw.imag).max():.4e}")
            print(f"        vs Re: scale {sr:+.8f} resid {rr:.2e}   "
                  f"vs Im: scale {si:+.8f} resid {ri:.2e}")
        print(f"  omega = {omega:.6f}, 1/omega = {1/omega:.6f}, "
              f"dt = {sim.fields.dt:.6f}")


if __name__ == "__main__":
    main()
