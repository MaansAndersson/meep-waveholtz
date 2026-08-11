"""Grid refinement study for the ring resonator, in the discrete L2 norm.

There is no exact solution here, so two independent measures are reported.

1. Error against a fine reference. MEEP's ``solve_cw`` is cheap and second-order
   convergent, so a highly resolved FDFD solve is a legitimate reference for
   both methods. EM-WaveHoltz and ``solve_cw`` are each measured against it.
2. Self-convergence, which needs no reference at all: successive differences
   ||u_r - u_{2r}|| between EM-WaveHoltz solutions on doubling grids.

Both are evaluated on a *fixed* point set -- the Ez Yee lattice of the coarsest
resolution -- which is a subset of the Yee lattice of every finer resolution
used (all are multiples of it). Every solution is therefore sampled exactly at
those points, with no interpolation error entering the comparison. Resolutions
must all be multiples of ``BASE``.

    L2(e) = sqrt( h_eval^2 * sum e^2 ),   reported relative to the reference.

The two point sources are delta functions, so the solution has a log
singularity at each and no method converges pointwise there. L2 is reported
both over the whole physical domain and excluding disks about the sources.

Writes notes/ring_refine.png and notes/ring_refine_data.txt.
"""

import argparse
import math
import os
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import meep as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from emwh.core import TM_FLUX_COMPONENTS, EMWaveHoltz, inject_state, tune_courant
from emwh.problems import HALF, OMEGA0, SOURCE_X, ring_simulation

BASE = 5           # evaluation lattice; every resolution must be a multiple
DPML = 1.0
PERIODS = 10


def eval_points(dpml=DPML):
    """Ez Yee points of the BASE grid lying in the physical domain."""
    ext = HALF + dpml
    xs = np.array([-ext + i / BASE for i in range(14 * BASE + 1)])
    xs = xs[np.abs(xs) <= HALF + 1e-12]
    return xs


def sample(sim, xs):
    return np.array([[sim.fields.get_field(mp.Ez, mp.vec(float(x), float(y)))
                      for y in xs] for x in xs])


def waveholtz(omega, res, xs, tol, maxiter):
    courant, _, _ = tune_courant(omega, res, PERIODS, 0.5)
    sim = ring_simulation(omega=omega, resolution=res, dpml=DPML,
                          courant=courant, forcing="cos")
    wh = EMWaveHoltz(sim, omega, components=TM_FLUX_COMPONENTS,
                     n_periods=PERIODS, backout=False)
    t0 = time.time()
    nu, iters = wh.solve(tol=tol, maxiter=maxiter, verbose=False)
    el = time.time() - t0
    sim.restart_fields()
    inject_state(sim, wh.grids, nu, TM_FLUX_COMPONENTS)
    return sample(sim, xs).real, iters, el


def fdfd(omega, res, xs, tol=1e-10):
    courant, _, _ = tune_courant(omega, res, PERIODS, 0.5)
    sim = ring_simulation(omega=omega, resolution=res, dpml=DPML,
                          courant=courant, forcing="cos", complex_fields=True)
    sim.init_sim()
    t0 = time.time()
    sim.solve_cw(tol, 20000, 10)
    el = time.time() - t0
    return sample(sim, xs).real, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--reference", type=int, default=80)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--maxiter", type=int, default=400)
    args = ap.parse_args()

    for r in args.resolutions + [args.reference]:
        assert r % BASE == 0, f"resolution {r} is not a multiple of {BASE}"

    os.makedirs("notes", exist_ok=True)
    omega = OMEGA0
    xs = eval_points()
    h = 1.0 / BASE
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    away = np.ones_like(X, dtype=bool)
    for sx in (SOURCE_X, -SOURCE_X):
        away &= ((X - sx) ** 2 + Y**2) > 0.6**2

    def l2(e, mask=None):
        v = e if mask is None else e[mask]
        return math.sqrt(h * h * float(np.sum(v**2)))

    print(f"evaluation lattice {xs.size}x{xs.size} (BASE={BASE}), "
          f"reference solve_cw at resolution {args.reference}")
    t0 = time.time()
    ref, t_ref = fdfd(omega, args.reference, xs)
    print(f"  reference done in {t_ref:.1f} s, "
          f"||ref||_2 = {l2(ref):.6e}, max = {np.abs(ref).max():.4e}")
    nref, nref_a = l2(ref), l2(ref, away)

    rows = []
    sols = {}
    for res in args.resolutions:
        wh_u, iters, t_wh = waveholtz(omega, res, xs, args.tol, args.maxiter)
        cw_u, t_cw = fdfd(omega, res, xs)
        sols[res] = wh_u
        row = dict(res=res, iters=iters, t_wh=t_wh, t_cw=t_cw,
                   wh=l2(wh_u - ref) / nref, cw=l2(cw_u - ref) / nref,
                   wh_a=l2(wh_u - ref, away) / nref_a,
                   cw_a=l2(cw_u - ref, away) / nref_a)
        rows.append(row)
        print(f"  res {res:3d}: WH {iters:4d} it {t_wh:7.1f}s  "
              f"relL2 {row['wh']:.4e} (away {row['wh_a']:.4e})   |  "
              f"CW {t_cw:6.1f}s  relL2 {row['cw']:.4e} (away {row['cw_a']:.4e})",
              flush=True)

    def rate(key):
        out = [float("nan")]
        for a, b in zip(rows[:-1], rows[1:]):
            out.append(math.log(a[key] / b[key]) / math.log(b["res"] / a["res"]))
        return out

    for k in ("wh", "cw", "wh_a", "cw_a"):
        for r, q in zip(rows, rate(k)):
            r[k + "_rate"] = q

    # reference-free: successive differences between WaveHoltz solutions
    print("\n  self-convergence (no reference):")
    self_rows = []
    for a, b in zip(args.resolutions[:-1], args.resolutions[1:]):
        d = l2(sols[a] - sols[b]) / nref
        da = l2(sols[a] - sols[b], away) / nref_a
        self_rows.append((a, b, d, da))
        print(f"    ||u_{a} - u_{b}||_2 / ||ref||_2 = {d:.4e} "
              f"(away from sources {da:.4e})")
    # Fit the order from the successive differences.  For a p-th order method
    #     ||u_a - u_b|| ~ C a^-p (1 - (a/b)^p),
    # dominated by the coarse grid.  A plain log-ratio of two differences drops
    # the (1 - (a/b)^p) factor and, when the refinement ratio is not constant
    # (here 2, 1.5, 1.333), overstates the order badly -- it reports ~3.8 for
    # data that is in fact second order.
    if len(self_rows) >= 2:
        def model(a, b, q):
            return a ** (-q) * (1.0 - (a / b) ** q)

        for i in (0, 1):
            def fit(idx):
                (a0, b0, d0, _), (a1, b1, d1, _) = self_rows[idx:idx + 2]
                lo, hi = 0.5, 5.0
                for _ in range(200):
                    mid = 0.5 * (lo + hi)
                    f = model(a0, b0, mid) / model(a1, b1, mid) - d0 / d1
                    flo = model(a0, b0, lo) / model(a1, b1, lo) - d0 / d1
                    lo, hi = (mid, hi) if f * flo > 0 else (lo, mid)
                return 0.5 * (lo + hi)
            if i + 2 <= len(self_rows):
                a0, b0 = self_rows[i][:2]
                a1, b1 = self_rows[i + 1][:2]
                print(f"    fitted order from ({a0},{b0}) vs ({a1},{b1}): "
                      f"p = {fit(i):.3f}")

    with open("notes/ring_refine_data.txt", "w") as fh:
        fh.write(f"omega={omega!r} periods={PERIODS} dpml={DPML} "
                 f"reference=solve_cw@{args.reference} base={BASE}\n")
        fh.write("res iters t_wh t_cw relL2_WH rate relL2_CW rate "
                 "relL2_WH_away rate relL2_CW_away rate\n")
        for r in rows:
            fh.write(f"{r['res']} {r['iters']} {r['t_wh']:.1f} {r['t_cw']:.1f} "
                     + " ".join(f"{r[k]:.6e} {r[k+'_rate']:.3f}"
                                for k in ("wh", "cw", "wh_a", "cw_a")) + "\n")
        for a, b, d, da in self_rows:
            fh.write(f"# self {a} {b} {d:.6e} {da:.6e}\n")

    hs = np.array([1.0 / r["res"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, (kw, kc, title) in zip(axes, (
            ("wh", "cw", "whole physical domain"),
            ("wh_a", "cw_a", "excluding disks about the sources"))):
        ax.loglog(hs, [r[kw] for r in rows], "o-", color="tab:red", lw=1.6,
                  ms=6, label="EM-WaveHoltz")
        ax.loglog(hs, [r[kc] for r in rows], "^:", color="tab:blue", lw=1.6,
                  ms=6, label="MEEP FDFD (solve\\_cw)")
        anchor = rows[0][kc]
        ax.loglog(hs, anchor * (hs / hs[0]) ** 2, "k:", lw=1.2,
                  label=r"$O(h^2)$ reference")
        ax.set_xlabel("$h$")
        ax.set_ylabel(r"relative $L_2$ error")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Ring resonator, grid refinement at $\\omega_0$ "
                 f"({PERIODS} periods; reference: solve\\_cw at "
                 f"resolution {args.reference})", fontsize=11)
    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig("notes/ring_refine.png", dpi=140)
    print(f"\nwrote notes/ring_refine.png  (total {time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
