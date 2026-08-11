"""Zero forcing, random initial data: does ||Pi^k nu|| decay to zero?

With no sources Pi 0 = 0, so Pi is the *linear* operator S of eq. (15) and the
iteration is pure power iteration.  The paper's Appendix B (eq. 57) says the
spectral radius of S is strictly below 1, so any initial data must be damped
away.  This is the cleanest check of the transient-filtering half of the
method, and it is independent of the forcing -- which matters here, because
the forced fixed point is currently wrong (see STATUS.md) while every
homogeneous-operator test passes.

Produces decay.png.
"""

import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import meep as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from emwh.core import TM_FLUX_COMPONENTS, EMWaveHoltz, tune_courant
from emwh.problems import OMEGA0, TM_COMPONENTS, pec_box_simulation, ring_simulation


def norm(nu, comps):
    return math.sqrt(sum(float(np.sum(nu[c] ** 2)) for c in comps))


def run_case(label, make_sim, omega, comps, n_periods, iters, seed=0):
    wh = EMWaveHoltz(make_sim(), omega, components=comps, n_periods=n_periods)
    rng = np.random.default_rng(seed)
    nu = {c: rng.standard_normal(wh.grids[c].shape) for c in comps}

    n0 = norm(nu, comps)
    norms = [1.0]
    print(f"\n{label}: {wh.M} steps/window, "
          f"dims {sum(wh.grids[c].shape[0]*wh.grids[c].shape[1] for c in comps)}")
    for k in range(1, iters + 1):
        nu = wh.apply_pi(nu)
        r = norm(nu, comps) / n0
        norms.append(r)
        if k <= 3 or k % max(1, iters // 6) == 0:
            rate = norms[-1] / norms[-2] if norms[-2] > 0 else float("nan")
            print(f"   k={k:4d}  ||nu||/||nu_0|| = {r:.6e}   ratio {rate:.5f}")
        if r < 1e-15:
            break
    return np.array(norms), wh


def asymptotic_rate(norms, tail=10):
    """Geometric rate from the last usable decade of the decay."""
    good = norms[(norms > 1e-14) & (norms > 0)]
    if len(good) < tail + 2:
        tail = max(2, len(good) // 2)
    seg = good[-tail:]
    if len(seg) < 2:
        return float("nan")
    return float(np.exp(np.diff(np.log(seg)).mean()))


def main():
    cases = []

    # A. closed PEC cavity, full (nu_E, nu_H) state
    omega = 2 * math.pi * 0.4
    c, _, _ = tune_courant(omega, 8, 1, 0.5)
    cases.append((
        "PEC cavity, (Ez,Hx,Hy)",
        lambda c=c, omega=omega: pec_box_simulation(
            omega=omega, resolution=8, size=1.0, courant=c, with_sources=False),
        omega, TM_COMPONENTS, 1, 60,
    ))

    # B. same cavity, Ez-only reduction of eq. (12)
    c, _, _ = tune_courant(omega, 8, 1, 0.5)
    cases.append((
        "PEC cavity, Ez only (eq. 12)",
        lambda c=c, omega=omega: pec_box_simulation(
            omega=omega, resolution=8, size=1.0, courant=c, with_sources=False),
        omega, (mp.Ez,), 1, 60,
    ))

    # C. same cavity, filtering over 5 periods
    c, _, _ = tune_courant(omega, 8, 5, 0.5)
    cases.append((
        "PEC cavity, Ez only, 5 periods",
        lambda c=c, omega=omega: pec_box_simulation(
            omega=omega, resolution=8, size=1.0, courant=c, with_sources=False),
        omega, (mp.Ez,), 5, 60,
    ))

    # D. open domain: ring resonator with PML, in flux variables.  The ring
    # dielectric has eps_r = 3.4^2, and MEEP time-steps D and B, so an (E,H)
    # state loses a factor of eps on injection and diverges; (Dz,Bx,By) is the
    # correct state.  (An earlier note here blamed PML -- that was wrong, PML
    # injection is fine.)
    c, _, _ = tune_courant(OMEGA0, 5, 5, 0.5)
    cases.append((
        "Ring resonator + PML, (Dz,Bx,By)",
        lambda c=c: ring_simulation(
            omega=OMEGA0, resolution=5, dpml=1.0, courant=c, with_sources=False),
        OMEGA0, TM_FLUX_COMPONENTS, 5, 150,
    ))

    results = []
    for label, make_sim, om, comps, npd, iters in cases:
        norms, wh = run_case(label, make_sim, om, comps, npd, iters)
        results.append((label, norms, asymptotic_rate(norms)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.8, len(results)))

    for (label, norms, rate), col in zip(results, colors):
        k = np.arange(len(norms))
        ax1.semilogy(k, np.maximum(norms, 1e-18), "o-", ms=3, lw=1.4,
                     color=col, label=f"{label}  ($\\rho\\approx${rate:.3f})")
        ratios = norms[1:] / np.where(norms[:-1] > 0, norms[:-1], np.nan)
        valid = norms[:-1] > 1e-14
        ax2.plot(np.arange(1, len(norms))[valid], ratios[valid], "o-", ms=3,
                 lw=1.4, color=col, label=label)

    ax1.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax1.set_xlabel("iteration $k$")
    ax1.set_ylabel(r"$\|\Pi^k \nu_0\| \, / \, \|\nu_0\|$")
    ax1.set_title("Zero forcing, random initial data:\nWaveHoltz damps all transients")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(1e-17, 5)

    ax2.axhline(1.0, color="crimson", lw=1.0, ls="--", label="contractivity limit")
    ax2.axhline(0.5, color="0.45", lw=1.0, ls=":",
                label=r"$|\beta(0)|=1/2$ (static mode)")
    ax2.set_xlabel("iteration $k$")
    ax2.set_ylabel(r"$\|\nu_{k}\| / \|\nu_{k-1}\|$")
    ax2.set_title("Observed contraction factor")
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig("decay.png", dpi=140)
    print("\nwrote decay.png")

    print("\nsummary")
    for label, norms, rate in results:
        print(f"  {label:34s} final {norms[-1]:.3e} after {len(norms)-1} iters, "
              f"rho ~ {rate:.4f}")
        assert norms[-1] < 1e-5 * norms[0], f"{label}: did not decay"
        assert rate < 1.0, f"{label}: not contractive"
    print("  PASS: every case is contractive and decays to zero")


if __name__ == "__main__":
    main()
