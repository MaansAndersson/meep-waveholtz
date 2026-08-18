"""Ring resonator with PML: MEEP's C++ WaveHoltz solver vs MEEP's FDFD, as
contour plots.

Both solvers use the same cell, resolution, subpixel-averaged eps, PML and
ContinuousSource, so this compares the methods rather than the discretisations.
Both are run with complex fields: the FDFD reference is MEEP's ``solve_cw``, the
WaveHoltz field comes from MEEP's native C++ solver (``fields::solve_waveholtz_cw``
in src/cw_fields.cpp).  The complex WaveHoltz field is compared against the FDFD
reference after a global complex scale fit (the fixed point carries a small
uniform phase/amplitude offset from the resonant transient).

Writes notes/ring_compare.png.
"""

import argparse
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import meep as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from emwh.core import tune_courant
from emwh.problems import (HALF, OMEGA0, RING_INNER, RING_OUTER, SOURCE_X,
                           ring_simulation)

DIVERGING = "RdBu_r"


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


def solve_meep_waveholtz(omega, res, dpml, periods, tol, maxiter, L=2):
    """MEEP's native C++ WaveHoltz solver (fields::solve_waveholtz_cw in
    src/cw_fields.cpp), run with complex fields.  Returns the complex Ez array,
    the wall time and the convergence flag."""
    courant, _, _ = tune_courant(omega, res, periods, 0.5)
    sim = ring_simulation(omega=omega, resolution=res, dpml=dpml,
                          courant=courant, forcing="cos", complex_fields=True)
    sim.init_sim()
    t0 = time.time()
    ok = sim.solve_waveholtz_cw(tol, maxiter, L)
    el = time.time() - t0
    if not ok:
        print("  !! solve_waveholtz_cw reported non-convergence")
    ez = np.asarray(sim.get_array(component=mp.Ez, cmplx=True))
    print(f"  MEEP WaveHoltz (solve\\_waveholtz\\_cw): {el:.1f} s, "
          f"converged={ok}")
    return ez, el, ok


def complex_fit(a, b):
    """Best-fit complex scale s with a ~ s*b (least squares)."""
    num = np.sum(np.conj(b) * a)
    den = np.sum(np.conj(b) * b)
    return num / den if den else 0.0


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
    print("[1] MEEP FDFD")
    ez_cw, t_cw = solve_fdfd(omega, args.resolution, args.dpml, args.periods,
                             1e-8, 10)
    print("[2] MEEP WaveHoltz (solve_waveholtz_cw)")
    ez_mwh, t_mwh, ok_mwh = solve_meep_waveholtz(
        omega, args.resolution, args.dpml, args.periods, args.tol,
        args.maxiter)

    # plot only the physical domain [-HALF, HALF]^2, excluding the PML
    n = ez_cw.shape[0]
    ext = HALF + args.dpml
    xs = np.linspace(-ext, ext, n)
    keep = np.abs(xs) <= HALF
    X, Y = np.meshgrid(xs[keep], xs[keep], indexing="ij")
    Am = ez_mwh[np.ix_(keep, keep)]
    Bm = ez_cw[np.ix_(keep, keep)]
    vmax = max(np.abs(Am).max(), np.abs(Bm).max())

    print(f"  on [-{HALF},{HALF}]^2: max|FDFD| {np.abs(Bm).max():.4e}, "
          f"max|WH| {np.abs(Am).max():.4e}")

    # The two point sources are delta functions: neither solver converges
    # pointwise there, and the max-norm difference is dominated by those two
    # cells.  Report the difference away from them as well.
    rad = 0.6
    away = np.ones_like(Bm, dtype=bool)
    for sx in (SOURCE_X, -SOURCE_X):
        away &= ((X - sx) ** 2 + Y**2) > rad**2

    # compare the complex WaveHoltz field against the complex FDFD reference,
    # fitting a global complex scale (the fixed point carries a small uniform
    # phase/amplitude offset from the resonant transient)
    s_mwh = complex_fit(Am, Bm)
    Dm = Am - s_mwh * Bm
    rel_all_mwh = np.abs(Dm).max() / np.abs(Bm).max()
    rel_away_mwh = np.abs(Dm[away]).max() / np.abs(Bm).max()
    print(f"  MEEP WaveHoltz vs FDFD: complex fit scale "
          f"{s_mwh.real:.5f}{s_mwh.imag:+.5f}i, "
          f"max rel diff {rel_all_mwh:.3e} "
          f"({rel_away_mwh:.3e} away from the sources)")

    failures = []
    if not ok_mwh:
        failures.append(
            f"solve_waveholtz_cw did not converge (tol {args.tol},"
            f" maxiter {args.maxiter})")
    if rel_away_mwh > 0.05:
        failures.append(
            f"solve_waveholtz_cw vs solve_cw: max rel diff {rel_away_mwh:.2e} "
            "away from the sources exceeds 5%")
    if failures:
        print("test solve_waveholtz_cw: FAIL")
        for f in failures:
            print("  - " + f)
    else:
        print(f"test solve_waveholtz_cw: PASS (agrees with solve_cw to "
              f"{rel_away_mwh:.2e}, away from the sources)")

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 10.2))
    panel(axes[0, 0], X, Y, Bm.real, f"MEEP FDFD (solve\\_cw)  $\\Re E_z$\n"
                                     f"BiCGStab-10, {t_cw:.1f} s", vmax)
    panel(axes[0, 1], X, Y, Am.real, f"MEEP WaveHoltz (solve\\_waveholtz\\_cw)\n"
                                     f"$\\Re(s\\,E_z)$, {t_mwh:.1f} s", vmax)
    panel(axes[1, 0], X, Y, Dm.real, "difference $\\Re$ (MEEP WH $-$ FDFD)\n"
                                     f"max {rel_all_mwh:.1e} rel.; "
                                     f"{rel_away_mwh:.1e} away from the sources",
          vmax)
    panel(axes[1, 1], X, Y, Dm.imag, "difference $\\Im$ (MEEP WH $-$ FDFD)\n"
                                     f"max {rel_all_mwh:.1e} rel.; "
                                     f"{rel_away_mwh:.1e} away from the sources",
          vmax)
    fig.suptitle(
        f"Ring resonator with PML, $\\omega = {args.omega_factor}\\,\\omega_0$, "
        f"resolution {args.resolution} ($N={12*args.resolution}$ over $[-6,6]$), "
        f"{args.periods} periods: C++ WaveHoltz vs FDFD",
        fontsize=11)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig("notes/ring_compare_mpi.png", dpi=140)
    print("wrote notes/ring_compare_mpi.png")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
