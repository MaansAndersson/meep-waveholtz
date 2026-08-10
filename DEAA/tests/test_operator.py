"""Structural tests on the discrete EM-WaveHoltz operator.

The sharpest available end-to-end check.  For the energy-conserving PEC case
the paper proves (Appendix B, eq. 59) that I - S is self-adjoint and positive
definite, and verifies it numerically in Sec. III-J.  Reproducing that here
exercises injection, readback, the DFT normalisation and the Yee index mapping
all at once: a mapping off by one cell, or a wrong sqrt(2*pi), destroys the
symmetry immediately.
"""

import math
import sys

import numpy as np
import meep as mp

sys.path.insert(0, "/Users/appelo/Desktop/MEEP_STUFF")

from emwh.core import EMWaveHoltz, tune_courant
from emwh.problems import pec_box_simulation


def build_operator(omega=2 * math.pi * 0.4, resolution=6, size=1.0, n_periods=1):
    """Form S column by column on a tiny source-free PEC cavity.

    With no sources Pi 0 = 0, so Pi is linear and Pi == S.
    """
    courant, steps, period = tune_courant(omega, resolution, n_periods, 0.5)
    sim = pec_box_simulation(
        omega=omega, resolution=resolution, size=size,
        courant=courant, with_sources=False,
    )
    wh = EMWaveHoltz(sim, omega, components=(mp.Ez,), n_periods=n_periods)
    grid = wh.grids[mp.Ez]
    n = grid.shape[0] * grid.shape[1]
    print(f"  grid {grid.shape} -> {n} unknowns, {steps} steps/window, "
          f"Courant {courant:.6f}")

    S = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1.0
        out = wh.apply_pi({mp.Ez: e.reshape(grid.shape)})
        S[:, k] = out[mp.Ez].ravel()
    return wh, S, grid


def test_operator_structure():
    print("\n=== EM-WaveHoltz operator structure (PEC, source free) ===")
    wh, S, grid = build_operator()

    misses = wh.inject_stats[mp.Ez]["misses"]
    calls = wh.inject_stats[mp.Ez]["calls"]
    print(f"  injector: {calls} callback calls, {misses} misses")
    assert misses == 0, f"{misses} injected values fell outside the array"

    nx, ny = grid.shape
    A = np.eye(S.shape[0]) - S
    scale = np.abs(A).max()

    # The outermost ring of the Yee array is a ghost layer, not an owned
    # degree of freedom: MEEP never time-steps it, so a value injected there
    # just sits still and is filtered by the static multiplier beta(0) = -1/2.
    # Those rows decouple, converge to zero, and are excluded from the
    # symmetry statement of Appendix B.
    interior = np.ones((nx, ny), dtype=bool)
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
    k = interior.ravel()

    Ai = A[np.ix_(k, k)]
    asym_int = np.abs(Ai - Ai.T).max()
    asym_all = np.abs(A - A.T).max()
    print(f"  max|A - A^T|: interior {asym_int:.3e}, full grid {asym_all:.3e} "
          f"(max|A| = {scale:.3e})")

    # The outer ring splits into PEC-clamped nodes (MEEP never updates them,
    # but they do enter the curl stencil) and dead slots (S column identically
    # zero).  Hand-injecting into a clamped node leaks into the interior, but
    # its self-response is the static beta(0) = -1/2, so it is contractive --
    # and test_boundary_stays_zero below shows the real iteration, started from
    # nu = 0, never puts anything there in the first place.
    Sg = S.reshape(nx, ny, nx, ny)
    ghost = [(i, j) for i in range(nx) for j in range(ny) if not interior[i, j]]
    self_resp = np.array([Sg[i, j, i, j] for i, j in ghost])
    leak = max(np.abs(Sg[:, :, i, j].ravel()[k]).max() for i, j in ghost)
    print(f"  outer ring: self-response in "
          f"{sorted(set(np.round(self_resp, 12)))}, interior leak {leak:.3e}")

    evals = np.linalg.eigvalsh(0.5 * (Ai + Ai.T))
    print(f"  eigenvalues of I-S (interior): min {evals.min():.6e}, "
          f"max {evals.max():.6e}")
    print(f"  condition number: {evals.max() / evals.min():.4f}")

    rho = np.abs(np.linalg.eigvals(S)).max()
    print(f"  spectral radius of S: {rho:.6f}  (must be < 1)")

    assert asym_int < 1e-10 * max(scale, 1.0), "I - S is not symmetric"
    assert np.abs(self_resp).max() <= 0.5 + 1e-12, "outer ring not contractive"
    assert evals.min() > 0, "I - S is not positive definite"
    assert rho < 1.0, "S is not contractive"
    print("  PASS: I-S symmetric positive definite on owned DOFs, S contractive")


def test_filter_multiplier_on_forced_mode():
    """The filter must reproduce a pure cos(omega t) signal exactly.

    This is what makes the frequency-domain solution a fixed point, and it is
    the property the rectangle-rule DFT has to share with the paper's composite
    trapezoid.  Checked directly on the quadrature, independent of MEEP.
    """
    print("\n=== filter multiplier on the forced mode ===")
    for n_periods in (1, 3, 10):
        for M_per in (17, 64, 137):
            M = M_per * n_periods
            omega = 2 * math.pi * 0.37
            T = n_periods * 2 * math.pi / omega
            dt = T / M
            n = np.arange(1, M + 1)  # MEEP samples after t += 1
            t = n * dt
            kernel = np.cos(omega * t) - 0.25
            mult = (2.0 / T) * np.sum(kernel * np.cos(omega * t)) * dt
            # A static component picks up (2/T) * int(-1/4) dt = -1/2: the
            # -1/4 term is exactly what damps zero frequency.
            const = (2.0 / T) * np.sum(kernel * 1.0) * dt
            assert abs(mult - 1.0) < 1e-12, (n_periods, M_per, mult)
            assert abs(const + 0.5) < 1e-12, (n_periods, M_per, const)
    print("  PASS: multiplier is exactly 1 on cos(omega t), and -1/2 on a")
    print("        static component, for the rectangle rule over whole periods")


def test_boundary_stays_zero():
    """Started from nu = 0, the iteration never populates the clamped ring.

    MEEP holds Ez on a PEC wall at zero, so the filter reads zero there and the
    injector writes zero back: the ring is inert for the whole iteration, and
    the leakage seen when hand-injecting into it cannot arise in practice.
    """
    print("\n=== clamped ring stays zero under the real iteration ===")
    omega = 2 * math.pi * 0.4
    resolution, n_periods = 6, 1
    courant, steps, _ = tune_courant(omega, resolution, n_periods, 0.5)
    sim = pec_box_simulation(omega=omega, resolution=resolution, size=1.0,
                             courant=courant, with_sources=True)
    wh = EMWaveHoltz(sim, omega, components=(mp.Ez,), n_periods=n_periods)
    nu = wh.zero()
    worst = 0.0
    for _ in range(8):
        nu = wh.apply_pi(nu)
        a = nu[mp.Ez]
        worst = max(worst, np.abs(a[0, :]).max(), np.abs(a[-1, :]).max(),
                    np.abs(a[:, 0]).max(), np.abs(a[:, -1]).max())
    print(f"  max |nu| on the outer ring over 8 iterations: {worst:.3e}")
    print(f"  max |nu| overall: {np.abs(nu[mp.Ez]).max():.3e}")
    assert worst == 0.0, "the clamped ring became nonzero"
    print("  PASS: outer ring is identically zero throughout")


if __name__ == "__main__":
    test_filter_multiplier_on_forced_mode()
    test_operator_structure()
    test_boundary_stays_zero()
