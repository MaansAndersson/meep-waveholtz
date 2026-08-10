"""Build the exact discrete T-periodic solution and test Pi against it.

On a tiny grid, form the one-window propagator P and offset q of MEEP's own
time stepper (state -> state after M steps), so the exact discrete periodic
initial data solves (I - P) nu* = q.  Then ask whether Pi nu* == nu*.

That separates the two possibilities cleanly:
  * Pi nu* == nu*  -> the filter is right and the fixed point is the periodic
    solution (so a failing periodicity check would be the checker's fault);
  * Pi nu* != nu*  -> the filter, or the state convention, is wrong.
"""

import math
import sys

import numpy as np
import meep as mp

sys.path.insert(0, "/Users/appelo/Desktop/MEEP_STUFF")

from emwh.core import EMWaveHoltz, inject_state, tune_courant
from emwh.gridmap import make_injector
from emwh.problems import pec_box_simulation

C = (mp.Ez, mp.Hx, mp.Hy)
NAMES = {mp.Ez: "Ez", mp.Hx: "Hx", mp.Hy: "Hy"}


def flat(wh, d):
    return np.concatenate([d[c].ravel() for c in C])


def unflat(wh, v):
    out, k = {}, 0
    for c in C:
        n = wh.grids[c].shape[0] * wh.grids[c].shape[1]
        out[c] = v[k:k + n].reshape(wh.grids[c].shape).copy()
        k += n
    return out


def read_raw(sim, wh):
    """Raw Yee-slot values via pointwise get_field."""
    out = {}
    for c in C:
        g = wh.grids[c]
        a = np.empty(g.shape)
        for i, x in enumerate(g.xs):
            for j, y in enumerate(g.ys):
                a[i, j] = sim.fields.get_field(c, mp.vec(float(x), float(y))).real
        out[c] = a
    return out


def inject(sim, wh, nu):
    sim.restart_fields()
    inject_state(sim, wh.grids, nu, C)


def step_window(sim, wh, nu):
    inject(sim, wh, nu)
    for _ in range(wh.M):
        sim.fields.step()
    return read_raw(sim, wh)


def main(resolution=4, size=1.0, n_periods=1, omega=2 * math.pi * 0.4):
    courant, steps, _ = tune_courant(omega, resolution, n_periods, 0.5)
    sim = pec_box_simulation(omega=omega, resolution=resolution, size=size,
                             courant=courant, forcing="sin")
    wh = EMWaveHoltz(sim, omega, components=C, n_periods=n_periods)
    print(f"grids: " + ", ".join(f"{NAMES[c]}{wh.grids[c].shape}" for c in C))
    print(f"{wh.M} steps/window, dt = {sim.fields.dt:.6f}")

    # sanity: is get_field returning the raw slot value we injected?
    rng = np.random.default_rng(0)
    probe = {c: rng.standard_normal(wh.grids[c].shape) for c in C}
    inject(sim, wh, probe)
    back = read_raw(sim, wh)
    for c in C:
        d = np.abs(back[c] - probe[c])
        # boundary slots are clamped; compare the interior
        print(f"  get_field round trip {NAMES[c]}: max diff "
              f"{d.max():.2e}, interior {d[1:-1,1:-1].max():.2e}")

    n = sum(wh.grids[c].shape[0] * wh.grids[c].shape[1] for c in C)
    print(f"state dimension {n}")

    q = flat(wh, step_window(sim, wh, wh.zero()))
    P = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1.0
        P[:, k] = flat(wh, step_window(sim, wh, unflat(wh, e))) - q

    nu_star = np.linalg.solve(np.eye(n) - P, q)
    resid = np.abs((np.eye(n) - P) @ nu_star - q).max()
    print(f"\nexact periodic data: residual {resid:.2e}, "
          f"|nu*| max {np.abs(nu_star).max():.6e}")

    # Verify it really is periodic under the stepper.
    after = flat(wh, step_window(sim, wh, unflat(wh, nu_star)))
    print(f"periodicity of nu*: max|P nu* + q - nu*| = "
          f"{np.abs(after - nu_star).max():.3e}")

    # THE question: does the filter reproduce it?
    for shift in (True, False):
        inject(sim, wh, unflat(wh, nu_star))
        for _ in range(wh.M):
            sim.fields.step()
        out = flat(wh, wh.filter.read(half_step_shift=shift))
        err = np.abs(out - nu_star)
        print(f"\nhalf_step_shift={shift}: max|Pi nu* - nu*| = {err.max():.3e}")
        k = 0
        for c in C:
            m = wh.grids[c].shape[0] * wh.grids[c].shape[1]
            print(f"    {NAMES[c]}: {err[k:k+m].max():.3e}  "
                  f"(|nu*| {np.abs(nu_star[k:k+m]).max():.3e})")
            k += m

    # And where does our fixed point sit?
    nu_fp, iters = wh.solve(tol=1e-13, maxiter=4000, verbose=False)
    d = np.abs(flat(wh, nu_fp) - nu_star).max()
    print(f"\nfixed point vs nu*: max diff {d:.3e} "
          f"(|nu*| {np.abs(nu_star).max():.3e}), {iters} iters")


if __name__ == "__main__":
    main()
