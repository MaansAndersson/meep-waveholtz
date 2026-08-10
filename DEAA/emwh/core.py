"""The EM-WaveHoltz operator and fixed-point driver, driven by MEEP.

Implements the Yee-EM-WaveHoltz method of Sec. II-A of Peng & Appelo,
"EM-WaveHoltz: A flexible frequency-domain method built from time-domain
solvers", using MEEP's FDTD core as the time-domain solver.

State-vector convention
-----------------------
The WaveHoltz state ``nu`` lives at *integer* time for both E and H, as in the
paper.  Readback needs no correction: MEEP samples H at t^n - dt/2 with a
phase evaluated at the same instant, so its DFT already reports the
integer-time coefficient H_0.  Injection does, because MEEP's array slot holds
H(-dt/2) -- see :meth:`EMWaveHoltz.half_step_backout`, which is paper eq. (24).

Skipping that back-out (treating MEEP's raw slots as the state) gives an
operator that is still stationary, affine and convergent, but whose fixed point
is *not* the T-periodic solution -- caught by ``tests/test_periodicity.py``.
"""

import math
import time

import meep as mp
import numpy as np

from .filters import DFTFilter, ProbeFilter
from .gridmap import component_grid, make_injector


# Initial data must be written to the *flux* fields D and B, never to E and H.
# MEEP time-steps D and B and recomputes E = D/eps, H = B/mu through update_eh
# every step, so a write to E or H is silently discarded on the very next step.
# Measured: injecting a Gaussian into Ez in a lossless PEC box loses 98% of its
# energy on step 1 (sum Ez^2: 13.85 -> 0.039); injecting the same data into Dz
# conserves it (13.85 -> 12.82 -> 11.05 -> ...).
#
# Reading stays on E and H: those are the physical fields, and update_eh has
# already refreshed them by the time the DFT samples.
#
# For eps != 1 the state must BE (D, B), not (E, H).  Filtering E and injecting
# the result into D silently drops a factor of eps wherever the material is not
# vacuum: on the ring resonator (eps_r = 3.4^2) that made the iteration diverge
# at 61x per application, while the same run with a (Dz, Bx, By) state
# converges at 0.821.  Since eps and mu are time independent, filtering D = eps E
# is just eps times the filter of E, so the D/B formulation is the same method;
# recover E by reading it back after injection, where MEEP applies E = D/eps.
INJECT_AS = {
    mp.Ex: mp.Dx, mp.Ey: mp.Dy, mp.Ez: mp.Dz,
    mp.Hx: mp.Bx, mp.Hy: mp.By, mp.Hz: mp.Bz,
    # a D/B state injects itself
    mp.Dx: mp.Dx, mp.Dy: mp.Dy, mp.Dz: mp.Dz,
    mp.Bx: mp.Bx, mp.By: mp.By, mp.Bz: mp.Bz,
}

#: 2D TM state in flux variables -- the correct choice whenever eps != 1.
TM_FLUX_COMPONENTS = (mp.Dz, mp.Bx, mp.By)


def inject_state(sim, grids, nu, components=None):
    """Write a WaveHoltz state onto the grid, into D/B rather than E/H.

    The single place initial data is written, so the E/H vs D/B distinction
    cannot drift between the solver and the tests.  Returns the per-component
    injector stats.
    """
    stats = {}
    for c in (components if components is not None else nu):
        arr = nu.get(c)
        if arr is None or not arr.any():
            continue
        if arr.shape != grids[c].shape:
            raise ValueError(
                f"component {c}: initial data has shape {arr.shape}, "
                f"expected {grids[c].shape}"
            )
        callback, st = make_injector(grids[c], arr)
        # initialize_field accumulates (f += val), so the caller must have
        # zeroed the fields (restart_fields) first.
        sim.initialize_field(INJECT_AS[c], callback)
        stats[c] = st
    return stats


def tune_courant(omega, resolution, n_periods=1, courant_target=0.5):
    """Pick a Courant number giving an integer number of steps per window.

    MEEP has no direct control over dt, but dt = Courant / resolution, so the
    Courant number is the free parameter.  An exact integer M = T/dt matters:
    the "multiplier is exactly one on the forced mode" property of the filter
    relies on summing over whole periods.

    Returns ``(courant, M, T)`` with T = n_periods * 2*pi/omega.
    """
    period = n_periods * 2.0 * math.pi / omega
    steps = int(round(period * resolution / courant_target))
    if steps < 1:
        raise ValueError("resolution too coarse for even one step per window")
    courant = period * resolution / steps
    return courant, steps, period


def meep_source_scale(omega, dt):
    """Amplitude MEEP actually drives for a ``ContinuousSource``.

    MEEP's ``src_time`` returns a dipole moment and differences it, so the
    driven current is

        sinc(omega*dt/2) * exp(-i*omega*(t + dt/2))

    (verified to 8e-15 in ``scripts/probe_api.py``).  Two consequences:

    * the current is evaluated at the half step, exactly the sin(omega t^{n+1/2})
      convention of paper eq. (23a);
    * the amplitude carries a factor sin(omega dt/2)/(omega dt/2), which is
      precisely omega_tilde/omega of eq. (31).

    The factor is identical for WaveHoltz and for ``solve_cw`` -- both go
    through the same ``src_time`` -- so it cancels in that comparison, but it
    matters when checking against an analytic solution.
    """
    half = 0.5 * omega * dt
    return math.sin(half) / half


class EMWaveHoltz:
    """Fixed-point EM-WaveHoltz iteration on top of an ``mp.Simulation``.

    The simulation must already carry the forcing (a ``ContinuousSource`` with
    no turn-on ramp) and must have been built with a Courant number from
    :func:`tune_courant`.
    """

    def __init__(
        self,
        sim,
        omega,
        components,
        n_periods=1,
        filter="dft",
        where=None,
        backout=True,
    ):
        self.sim = sim
        self.omega = omega
        self.components = tuple(components)
        self.n_periods = n_periods
        self.backout = backout
        self.T = n_periods * 2.0 * math.pi / omega

        if sim.fields is None:
            sim.init_sim()

        dt = sim.fields.dt
        steps = self.T / dt
        self.M = int(round(steps))
        if abs(steps - self.M) > 1e-9 * max(1.0, self.M):
            raise ValueError(
                f"window T={self.T!r} is not an integer number of timesteps "
                f"(T/dt = {steps!r}); build the Simulation with a Courant "
                "number from tune_courant()"
            )

        # One DFT accumulator per component: each sits on its own staggered
        # Yee lattice.
        if filter != "dft":
            raise ValueError(
                f"unknown filter {filter!r}; ProbeFilter is a validation tool, "
                "use it directly (see tests/test_filter.py)"
            )
        self.filter = DFTFilter(sim, omega, self.T, self.components, where=where)

        # Recover each component's Yee lattice.  get_array_metadata reports the
        # centered grid even for a yee_grid=True DFT cell, so the lattice comes
        # from the positions initialize_field actually visits.
        self.grids = {c: component_grid(sim, c) for c in self.components}
        for c in self.components:
            got = np.asarray(sim.get_dft_array(self.filter.dfts[c], c, 0)).shape
            want = self.grids[c].shape
            if got != want:
                raise RuntimeError(
                    f"component {c}: DFT array shape {got} does not match the "
                    f"Yee lattice {want} recovered from initialize_field; the "
                    "index mapping would be wrong"
                )

        self.inject_stats = {}
        self.pi_zero = None
        self.history = []

    # ------------------------------------------------------------------
    # the operator
    # ------------------------------------------------------------------
    def zero(self):
        return {c: self.grids[c].zeros() for c in self.components}

    def half_step_backout(self, nu):
        """Paper eq. (24): convert H at integer time to MEEP's H^{-1/2} slot.

        The WaveHoltz state lives at integer time for both E and H -- that is
        what the filter returns, since MEEP samples H at t^n - dt/2 with a
        matching phase and so reports the integer-time coefficient H_0.  But
        MEEP's array *slot* holds H(-dt/2), so injection must step H back half
        a step:

            Hx^{-1/2} = Hx^0 + (dt/2mu) dEz/dy
            Hy^{-1/2} = Hy^0 - (dt/2mu) dEz/dx

        Assumes mu = 1 (true for both problems here) and the 2D TM component
        set.  Inside PML the H update is not the plain curl, so there the
        back-out is only approximate.
        """
        if not (mp.Ez in nu and mp.Hx in nu and mp.Hy in nu):
            return nu
        dt = self.sim.fields.dt
        ez = nu[mp.Ez]
        out = dict(nu)
        hx = nu[mp.Hx].copy()
        hy = nu[mp.Hy].copy()
        # Hx[i,j] sits between Ez[i,j] and Ez[i,j+1]; Hy[i,j] between
        # Ez[i,j] and Ez[i+1,j].  The final row/column has no partner and
        # lies outside the domain, so it is left alone.
        dy = self.grids[mp.Hx].dy
        dx = self.grids[mp.Hy].dx
        hx[:, :-1] += (0.5 * dt / dy) * (ez[:, 1:] - ez[:, :-1])
        hy[:-1, :] -= (0.5 * dt / dx) * (ez[1:, :] - ez[:-1, :])
        out[mp.Hx] = hx
        out[mp.Hy] = hy
        return out

    def apply_pi(self, nu):
        """One application of the filtered time-domain solution operator."""
        sim = self.sim
        sim.restart_fields()

        if self.backout:
            nu = self.half_step_backout(nu)

        # eps = mu = 1 here, so D and B carry the same values as E and H; for
        # eps != 1 the filtered E would need scaling by eps first.
        self.inject_stats.update(
            inject_state(sim, self.grids, nu, self.components))

        for _ in range(self.M):
            sim.fields.step()

        return self.filter.read(half_step_shift=False)

    # ------------------------------------------------------------------
    # the iteration
    # ------------------------------------------------------------------
    def solve(self, tol=1e-7, maxiter=5000, verbose=True, log_every=1,
              pi0=None):
        """Iterate nu^{n+1} = Pi nu^n from nu^0 = 0.

        Stops on ||nu^{n+1} - nu^n|| / ||nu^1 - nu^0|| < tol.

        ``pi0`` supplies the affine constant externally, for the case where the
        forcing term is computed some other way (see :mod:`emwh.pi0`, or
        Pi0 = (I-S)nu* from a known solution).  The simulation must then carry
        **no** sources, so that ``apply_pi`` is exactly S and the forcing is not
        counted twice; that is checked rather than assumed.
        """
        if pi0 is not None:
            leak = max(float(np.abs(self.apply_pi(self.zero())[c]).max())
                       for c in self.components)
            if leak > 1e-12:
                raise ValueError(
                    f"pi0 was supplied but the simulation is not source free "
                    f"(|Pi 0| = {leak:.3e}); the forcing would be counted twice"
                )

        nu = self.zero()
        first = None
        self.history = []
        t_start = time.time()

        for k in range(1, maxiter + 1):
            new = self.apply_pi(nu)
            if pi0 is not None:
                new = {c: new[c] + pi0[c] for c in self.components}
            if k == 1:
                # Pi applied to zero initial data: the constant term of the
                # affine operator, and the right-hand side a Krylov driver
                # would need for (I - S) nu = Pi 0.
                self.pi_zero = {c: new[c].copy() for c in self.components}

            diff = math.sqrt(
                sum(float(np.sum((new[c] - nu[c]) ** 2)) for c in self.components)
            )
            if first is None:
                first = diff
            rel = diff / first if first > 0 else 0.0
            self.history.append(rel)
            nu = new

            if verbose and (k % log_every == 0 or rel < tol):
                rate = (
                    self.history[-1] / self.history[-2]
                    if len(self.history) > 1 and self.history[-2] > 0
                    else float("nan")
                )
                print(
                    f"  iter {k:5d}   rel {rel:.3e}   rate {rate:.4f}   "
                    f"{time.time() - t_start:7.1f}s",
                    flush=True,
                )
            if rel < tol:
                return nu, k

        return nu, maxiter
