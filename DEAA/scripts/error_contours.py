"""Contour plots of the PEC-cavity error fields.

The norms in notes/pec_convergence.tex say the error stagnates; these show
*where* it lives and what shape it has, which the norms cannot.

Signed fields use a diverging map with the neutral midpoint pinned to zero
(symmetric limits), so sign is readable and zero is unambiguous.  The solution
panels use a single-hue sequential map.  No rainbow anywhere.

Writes notes/pec_error_contours.png.
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import meep as mp

sys.path.insert(0, "/Users/appelo/Desktop/MEEP_STUFF")

from emwh.core import EMWaveHoltz
from emwh.manufactured import (SHIFT, errors, exact_on_grid,
                               manufactured_simulation)

OMEGA = 2 * math.pi * 0.4
COURANT = 0.5
RES = 64

DIVERGING = "RdBu_r"      # two hues, neutral midpoint
SEQUENTIAL = "Blues"      # single hue, light -> dark


def source_free(resolution):
    return mp.Simulation(
        cell_size=mp.Vector3(1.0, 1.0, 0), resolution=resolution,
        sources=[], boundary_layers=[], Courant=COURANT, dimensions=2,
    )


def sample(sim, grid):
    return np.array([[sim.fields.get_field(mp.Ez, mp.vec(float(x), float(y)))
                      for y in grid.ys] for x in grid.xs])


def compute():
    sim = manufactured_simulation(OMEGA, RES, courant=COURANT)
    wh = EMWaveHoltz(sim, OMEGA, components=(mp.Ez,), n_periods=1)
    grid = wh.grids[mp.Ez]
    u = exact_on_grid(grid)
    pi0_ours = wh.apply_pi(wh.zero())[mp.Ez].copy()
    nu, _ = wh.solve(tol=1e-13, maxiter=4000, verbose=False)
    nu = nu[mp.Ez]

    sim_cw = manufactured_simulation(OMEGA, RES, courant=COURANT)
    sim_cw.force_complex_fields = True
    sim_cw.init_sim()
    sim_cw.solve_cw(1e-11, 20000, 10)
    nu_star = sample(sim_cw, grid).real.copy()

    wh_free = EMWaveHoltz(source_free(RES), OMEGA, components=(mp.Ez,),
                          n_periods=1)

    def S(v):
        return wh_free.apply_pi({mp.Ez: v})[mp.Ez]

    pi0_target = nu_star - S(nu_star)
    nu_fix = np.zeros_like(pi0_target)
    for _ in range(400):
        new = S(nu_fix) + pi0_target
        if np.abs(new - nu_fix).max() < 1e-14:
            nu_fix = new
            break
        nu_fix = new

    s = float(np.sum(nu * u) / np.sum(u * u))
    return dict(grid=grid, u=u, nu=nu, nu_fix=nu_fix, s=s,
                pi0_ours=pi0_ours, pi0_target=pi0_target,
                h=grid.dx)


def panel(ax, X, Y, Z, title, cmap, diverging, subtitle=None):
    if diverging:
        m = np.abs(Z).max()
        lv = np.linspace(-m, m, 25) if m > 0 else np.linspace(-1, 1, 25)
        kw = dict(levels=lv, cmap=cmap, vmin=-m, vmax=m)
    else:
        kw = dict(levels=25, cmap=cmap)
    cf = ax.contourf(X, Y, Z, **kw)
    ax.contour(X, Y, Z, levels=kw["levels"], colors="k",
               linewidths=0.25, alpha=0.35)
    cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.formatter.set_powerlimits((-2, 3))
    cb.update_ticks()
    # subtitle goes inside the title block; placing it below the axes made the
    # top row collide with the bottom row's titles
    ax.set_title(title + ("\n" + subtitle if subtitle else ""), fontsize=9.5,
                 linespacing=1.5)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=7, length=2, color="0.6")
    for sp in ax.spines.values():
        sp.set_color("0.75")


def main():
    os.makedirs("notes", exist_ok=True)
    d = compute()
    g = d["grid"]
    X, Y = np.meshgrid(g.xs + SHIFT, g.ys + SHIFT, indexing="ij")
    u, nu, nu_fix, s, h = d["u"], d["nu"], d["nu_fix"], d["s"], d["h"]

    e_raw = nu - u
    e_scaled = nu / s - u
    e_fix = nu_fix - u
    d_pi0 = d["pi0_ours"] - d["pi0_target"]

    def rl2(e):
        return math.sqrt(h * h * np.sum(e[1:-1, 1:-1] ** 2)) / math.sqrt(
            h * h * np.sum(u[1:-1, 1:-1] ** 2))

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2))
    panel(axes[0, 0], X, Y, u, "exact  $u$", SEQUENTIAL, False,
          f"max {np.abs(u).max():.4f}")
    panel(axes[0, 1], X, Y, nu, r"EM-WaveHoltz  $\nu$", SEQUENTIAL, False,
          f"max {np.abs(nu).max():.4f}   best-fit scale {s:.4f}")
    panel(axes[0, 2], X, Y, e_raw, r"error  $\nu - u$", DIVERGING, True,
          f"rel $L_2$ = {rl2(e_raw):.3e}")
    panel(axes[1, 0], X, Y, e_scaled,
          r"error, amplitude removed  $\nu/s - u$", DIVERGING, True,
          f"rel $L_2$ = {rl2(e_scaled):.3e}")
    panel(axes[1, 1], X, Y, e_fix,
          r"error with $\Pi_0=(I-S)\nu^*$", DIVERGING, True,
          f"rel $L_2$ = {rl2(e_fix):.3e}   (pure $O(h^2)$)")
    panel(axes[1, 2], X, Y, d_pi0,
          r"forcing defect  $\Pi_0^{\rm ours}-\Pi_0^{\rm target}$",
          DIVERGING, True,
          f"max {np.abs(d_pi0).max():.4e}")

    fig.suptitle(
        f"PEC cavity, manufactured solution — resolution {RES}, "
        f"Courant {COURANT}, $\\omega=2\\pi(0.4)$", fontsize=11)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig("notes/pec_error_contours.png", dpi=140)
    print("wrote notes/pec_error_contours.png")
    for name, e in (("nu - u", e_raw), ("nu/s - u", e_scaled),
                    ("fixed Pi0", e_fix)):
        print(f"  {name:12s} rel L2 {rl2(e):.4e}  max {np.abs(e).max():.4e}")
    def shape_fit(label, a, b):
        c = float(np.sum(a * b) / np.sum(b * b))
        r = np.abs(a - c * b).max() / np.abs(a).max()
        print(f"  {label:34s} fit {c:+.5e}, residual {r:.3e}")

    print("\n  shape comparisons (residual ~0 means the same spatial pattern)")
    # nu/s - u is orthogonal to u by construction, so that fit is vacuous; the
    # informative question is whether the leftover shape error is just an
    # amplified copy of the O(h^2) discretisation error.
    shape_fit("(nu/s - u) vs (nu_fix - u):", e_scaled, e_fix)
    shape_fit("(nu - u)   vs u:", e_raw, u)
    shape_fit("Pi0 defect vs u:", d_pi0, u)
    shape_fit("Pi0 defect vs Pi0_target:", d_pi0, d["pi0_target"])


if __name__ == "__main__":
    main()
