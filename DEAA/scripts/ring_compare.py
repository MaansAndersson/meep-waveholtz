"""Ring resonator with PML: EM-WaveHoltz vs MEEP's FDFD, as contour plots.

Both solvers use the same cell, resolution, subpixel-averaged eps, PML and
ContinuousSource, so this compares the methods rather than the discretisations.

The WaveHoltz state is (Dz, Bx, By), not (Ez, Hx, Hy): the ring dielectric has
eps_r = 3.4^2, and MEEP time-steps the flux fields, so a state in E/H loses a
factor of eps on injection wherever the material is not vacuum.  E is recovered
at the end by injecting the converged state and reading Ez, where MEEP applies
E = D/eps.

Writes notes/ring_compare.png.
"""

import argparse
import math
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import meep as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from emwh.core import (TM_FLUX_COMPONENTS, EMWaveHoltz, inject_state,
                       tune_courant)
from emwh.problems import HALF, OMEGA0, RING_INNER, RING_OUTER, ring_simulation

DIVERGING = "RdBu_r"


def solve_waveholtz(omega, res, dpml, periods, tol, maxiter):
    courant, _, _ = tune_courant(omega, res, periods, 0.5)
    sim = ring_simulation(omega=omega, resolution=res, dpml=dpml,
                          courant=courant, forcing="cos")
    wh = EMWaveHoltz(sim, omega, components=TM_FLUX_COMPONENTS,
                     n_periods=periods, backout=False)
    print(f"  grid {wh.grids[mp.Dz].shape}, {wh.M} steps/window, "
          f"Courant {courant:.6f}")
    t0 = time.time()
    nu, iters = wh.solve(tol=tol, maxiter=maxiter, verbose=True, log_every=20)
    el = time.time() - t0
    for c in TM_FLUX_COMPONENTS:
        assert wh.inject_stats[c]["misses"] == 0
    # read E back: injecting D and reading Ez applies E = D/eps for us
    sim.restart_fields()
    inject_state(sim, wh.grids, nu, TM_FLUX_COMPONENTS)
    ez = np.asarray(sim.get_array(component=mp.Ez)).real.copy()
    print(f"  EM-WaveHoltz: {iters} iterations, {el:.1f} s")
    return ez, iters, el, wh


def solve_fdfd(omega, res, dpml, periods, tol, L):
    courant, _, _ = tune_courant(omega, res, periods, 0.5)
    sim = ring_simulation(omega=omega, resolution=res, dpml=dpml,
                          courant=courant, forcing="cos", complex_fields=True)
    sim.init_sim()
    t0 = time.time()
    sim.solve_cw(tol, 20000, L)
    el = time.time() - t0
    ez = np.asarray(sim.get_array(component=mp.Ez, cmplx=True))
    print(f"  MEEP FDFD (BiCGStab-{L}): {el:.1f} s")
    return ez, el


def panel(ax, X, Y, Z, title, vmax=None, cmap=DIVERGING):
    m = vmax if vmax is not None else np.abs(Z).max()
    lv = np.linspace(-m, m, 25)
    cf = ax.contourf(X, Y, Z, levels=lv, cmap=cmap, vmin=-m, vmax=m,
                     extend="both")
    ax.contour(X, Y, Z, levels=lv, colors="k", linewidths=0.2, alpha=0.3)
    for r in (RING_INNER, RING_OUTER):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="0.25",
                                lw=0.8, ls="--"))
    cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.formatter.set_powerlimits((-2, 3))
    cb.update_ticks()
    ax.set_title(title, fontsize=10, linespacing=1.4)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$", fontsize=8)
    ax.set_ylabel("$y$", fontsize=8)
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=10)
    ap.add_argument("--omega-factor", type=float, default=1.0)
    ap.add_argument("--dpml", type=float, default=1.0)
    ap.add_argument("--periods", type=int, default=10)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=600)
    args = ap.parse_args()

    omega = args.omega_factor * OMEGA0
    print(f"### ring, omega = {args.omega_factor} w0 = {omega:.6f}, "
          f"resolution {args.resolution}, {args.periods} periods")
    print("[1] EM-WaveHoltz")
    ez_wh, iters, t_wh, wh = solve_waveholtz(
        omega, args.resolution, args.dpml, args.periods, args.tol, args.maxiter)
    print("[2] MEEP FDFD")
    ez_cw, t_cw = solve_fdfd(omega, args.resolution, args.dpml, args.periods,
                             1e-8, 10)

    # cos-forcing converges to Re{E}; confirm against both parts rather than assume
    best = None
    for name, ref in (("Re", ez_cw.real), ("Im", ez_cw.imag)):
        s = float(np.sum(ez_wh * ref) / np.sum(ref * ref))
        r = np.abs(ez_wh - ref).max() / np.abs(ref).max()
        print(f"  vs {name}(cw): best-fit scale {s:.6f}, max rel diff {r:.4e}")
        if best is None or r < best[2]:
            best = (name, ref, r, s)
    part, ref, relerr, scale = best

    # plot only the physical domain [-HALF, HALF]^2, excluding the PML
    n = ez_wh.shape[0]
    ext = HALF + args.dpml
    xs = np.linspace(-ext, ext, n)
    keep = np.abs(xs) <= HALF
    X, Y = np.meshgrid(xs[keep], xs[keep], indexing="ij")
    A = ez_wh[np.ix_(keep, keep)]
    B = ref[np.ix_(keep, keep)]
    D = A - B
    vmax = max(np.abs(A).max(), np.abs(B).max())

    print(f"  on [-6,6]^2: max|WH| {np.abs(A).max():.4e}, "
          f"max|FDFD| {np.abs(B).max():.4e}, max|diff| {np.abs(D).max():.4e} "
          f"({np.abs(D).max()/np.abs(B).max():.3e} relative)")

    # The two point sources are delta functions: neither solver converges
    # pointwise there, and the max-norm difference is dominated by those two
    # cells.  Report the difference away from them as well.
    from emwh.problems import SOURCE_X
    rad = 0.6
    away = np.ones_like(D, dtype=bool)
    for sx in (SOURCE_X, -SOURCE_X):
        away &= ((X - sx) ** 2 + Y**2) > rad**2
    rel_away = np.abs(D[away]).max() / np.abs(B).max()
    print(f"  excluding disks of radius {rad} about the two point sources: "
          f"max|diff| {np.abs(D[away]).max():.4e} ({rel_away:.3e} relative)")
    return_info = (np.abs(D).max() / np.abs(B).max(), rel_away)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    panel(axes[0], X, Y, A, f"EM-WaveHoltz  $\\Re E_z$\n{iters} iterations, "
                            f"{t_wh:.1f} s", vmax)
    panel(axes[1], X, Y, B, f"MEEP FDFD (solve\\_cw)  ${part}\\,E_z$\n"
                            f"BiCGStab-10, {t_cw:.1f} s", vmax)
    panel(axes[2], X, Y, D, "difference\n"
                            f"max {np.abs(D).max()/np.abs(B).max():.1e} rel.; "
                            f"{rel_away:.1e} away from the sources")
    fig.suptitle(
        f"Ring resonator with PML, $\\omega = {args.omega_factor}\\,\\omega_0$, "
        f"resolution {args.resolution} ($N={12*args.resolution}$ over $[-6,6]$), "
        f"{args.periods} periods", fontsize=11)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig("notes/ring_compare.png", dpi=140)
    print("wrote notes/ring_compare.png")


if __name__ == "__main__":
    main()
