"""Grid convergence for the PEC cavity manufactured solution, Courant fixed.

With f = 0.4 and Courant = 0.5 the window is M = 5*resolution timesteps, an
integer at every resolution, so the Courant number is held *exactly* fixed
while the grid is refined -- dt shrinks in proportion to h, and the temporal
error of Theorem 1 is O(h^2) alongside the spatial error.

Four series:

  1. EM-WaveHoltz as implemented
  2. the same, with the best-fit amplitude divided out (separates a pure
     amplitude error from a shape error)
  3. EM-WaveHoltz driven by the *manufactured* forcing term
     Pi0 = (I - S) nu*, with nu* taken from MEEP's FDFD solver.  Since
     nu* = S nu* + Pi0 defines the fixed point, this is the correct Pi0 by
     construction, and it isolates whether the defect lives in S or in Pi0.
  4. MEEP's FDFD solver solve_cw, the control.

If (3) tracks (4) while (1) stagnates, S is sound and the whole defect is in
the forcing term.

Writes notes/pec_convergence.png and notes/pec_convergence_data.txt.
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
from emwh.manufactured import errors, exact_on_grid, manufactured_simulation

OMEGA = 2 * math.pi * 0.4
COURANT = 0.5
RESOLUTIONS = (8, 16, 24, 32, 48, 64, 96)


def fit_scale(v, u):
    return float(np.sum(v * u) / np.sum(u * u))


def source_free(resolution):
    """Same grid and timestep, no sources: applying Pi here gives exactly S."""
    return mp.Simulation(
        cell_size=mp.Vector3(1.0, 1.0, 0), resolution=resolution,
        sources=[], boundary_layers=[], Courant=COURANT, dimensions=2,
    )


def sample_on_grid(sim, grid, component=mp.Ez):
    """Pointwise read on the Yee lattice (get_array would give centered grid)."""
    out = np.empty(grid.shape, dtype=complex)
    for i, x in enumerate(grid.xs):
        for j, y in enumerate(grid.ys):
            out[i, j] = sim.fields.get_field(component, mp.vec(float(x), float(y)))
    return out


def run_resolution(res):
    # --- 1. EM-WaveHoltz as implemented ---------------------------------
    sim = manufactured_simulation(OMEGA, res, courant=COURANT)
    wh = EMWaveHoltz(sim, OMEGA, components=(mp.Ez,), n_periods=1)
    assert wh.M == 5 * res, (wh.M, res)
    grid = wh.grids[mp.Ez]
    h = grid.dx
    u = exact_on_grid(grid)
    nu_ours, iters = wh.solve(tol=1e-13, maxiter=4000, verbose=False)
    nu_ours = nu_ours[mp.Ez]

    # --- 4. control: MEEP FDFD ------------------------------------------
    sim_cw = manufactured_simulation(OMEGA, res, courant=COURANT)
    sim_cw.force_complex_fields = True
    sim_cw.init_sim()
    sim_cw.solve_cw(1e-11, 20000, 10)
    vc = sample_on_grid(sim_cw, grid)
    # pick the part that represents the solution (empirically Re throughout)
    part = "Re" if errors(vc.real, u, h)[2] < errors(vc.imag, u, h)[2] else "Im"
    nu_star = (vc.real if part == "Re" else vc.imag).copy()

    # --- 3. EM-WaveHoltz with the manufactured Pi0 = (I - S) nu* --------
    wh_free = EMWaveHoltz(source_free(res), OMEGA, components=(mp.Ez,),
                          n_periods=1)
    assert wh_free.grids[mp.Ez].shape == grid.shape
    zero_check = np.abs(wh_free.apply_pi(wh_free.zero())[mp.Ez]).max()

    def S(v):
        return wh_free.apply_pi({mp.Ez: v})[mp.Ez]

    pi0_target = nu_star - S(nu_star)
    nu_fix = np.zeros_like(pi0_target)
    for k in range(500):
        new = S(nu_fix) + pi0_target
        step = np.abs(new - nu_fix).max()
        nu_fix = new
        if step < 1e-14:
            break
    iters_fix = k + 1

    s = fit_scale(nu_ours, u)
    _, _, l2r, lir = errors(nu_ours, u, h)
    _, _, l2rs, lirs = errors(nu_ours / s, u, h)
    _, _, f_l2, f_li = errors(nu_fix, u, h)
    _, _, c_l2, c_li = errors(nu_star, u, h)
    _, _, gap, _ = errors(nu_fix, nu_star, h)

    return dict(res=res, h=h, iters=iters, scale=s, l2r=l2r, lir=lir,
                l2rs=l2rs, lirs=lirs, iters_fix=iters_fix, fix_l2=f_l2,
                fix_li=f_li, cw_part=part, cw_l2=c_l2, cw_li=c_li,
                gap=gap, zero_check=zero_check)


def rates(rows, key):
    out = [float("nan")]
    for a, b in zip(rows[:-1], rows[1:]):
        out.append(math.log(a[key] / b[key]) / math.log(a["h"] / b["h"]))
    return out


def main():
    os.makedirs("notes", exist_ok=True)
    print(f"omega = {OMEGA:.6f}, Courant = {COURANT} (fixed), M = 5*resolution")
    rows = []
    for res in RESOLUTIONS:
        r = run_resolution(res)
        rows.append(r)
        print(f"  res {r['res']:3d} h={r['h']:.5f}  ours {r['l2r']:.4e}/"
              f"{r['lir']:.4e} (scale {r['scale']:.4f}, {r['iters']} it)  |  "
              f"fixed-Pi0 {r['fix_l2']:.4e}/{r['fix_li']:.4e} "
              f"({r['iters_fix']} it, gap to nu* {r['gap']:.2e})  |  "
              f"CW({r['cw_part']}) {r['cw_l2']:.4e}/{r['cw_li']:.4e}")

    keys = ("l2r", "lir", "l2rs", "lirs", "fix_l2", "fix_li", "cw_l2", "cw_li")
    for k in keys:
        for r, q in zip(rows, rates(rows, k)):
            r[k + "_rate"] = q

    with open("notes/pec_convergence_data.txt", "w") as fh:
        fh.write(f"omega={OMEGA!r} Courant={COURANT!r} M=5*resolution\n")
        fh.write("res h iters scale " + " ".join(f"{k} {k}_rate" for k in keys)
                 + " iters_fix gap_to_nustar\n")
        for r in rows:
            fh.write(f"{r['res']} {r['h']:.6f} {r['iters']} {r['scale']:.6f} "
                     + " ".join(f"{r[k]:.6e} {r[k+'_rate']:.3f}" for k in keys)
                     + f" {r['iters_fix']} {r['gap']:.3e}\n")

    hs = np.array([r["h"] for r in rows])
    series = [
        ("l2r", "lir", "EM-WaveHoltz", "tab:red", "o", "-"),
        ("l2rs", "lirs", "EM-WaveHoltz (best-fit scale removed)",
         "tab:orange", "s", "--"),
        ("fix_l2", "fix_li", r"EM-WaveHoltz with $\Pi_0=(I-S)\nu^*$",
         "tab:green", "D", "-"),
        ("cw_l2", "cw_li", "MEEP FDFD solve_cw (control)", "tab:blue", "^", ":"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, idx, title in ((axes[0], 0, r"relative $L_2$ error"),
                           (axes[1], 1, r"relative $L_\infty$ error")):
        for l2key, likey, label, color, marker, ls in series:
            vals = np.array([r[(l2key, likey)[idx]] for r in rows])
            ax.loglog(hs, vals, marker=marker, ls=ls, color=color, lw=1.6,
                      ms=7 if marker == "D" else 5, mfc="none"
                      if marker == "D" else color, label=label)
        anchor = rows[0][(series[-1][0], series[-1][1])[idx]]
        ax.loglog(hs, anchor * (hs / hs[0]) ** 2, "k:", lw=1.2,
                  label=r"$O(h^2)$ reference")
        ax.set_xlabel("$h$")
        ax.set_ylabel(title)
        ax.set_title(title + ", Courant fixed at 0.5")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("notes/pec_convergence.png", dpi=140)
    print("\nwrote notes/pec_convergence.png and notes/pec_convergence_data.txt")


if __name__ == "__main__":
    main()
