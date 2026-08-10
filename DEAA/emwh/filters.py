"""Discretizations of the EM-WaveHoltz filter.

The filter of eq. (7)/(13) is

    Pi v = (2/T) int_0^T (cos(w t) - 1/4) field(t) dt

Split the kernel:

    (2/T) int cos(wt) f dt   -   (1/(2T)) int f dt

The second term is nothing but a DFT at *frequency zero*, so the entire filter
is two of MEEP's native DFT accumulators and costs no per-timestep Python.
That is :class:`DFTFilter`, the production path.

:class:`ProbeFilter` is the reference implementation.  It samples a handful of
individual Yee points every timestep through ``fields.get_field`` and applies
the quadrature weights explicitly in Python.  It exists to validate the parts
of :class:`DFTFilter` that are easy to get silently wrong -- MEEP's
``dt/sqrt(2*pi)`` DFT normalisation, the E/H time staggering, and the index
mapping -- without the cost of pulling whole grids every step.  (A whole-grid
Python reference is not available: ``get_array`` returns the *centered* grid,
not the Yee grid, so it cannot be compared against the Yee arrays the
iteration actually uses.)
"""

import math

import numpy as np

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _is_magnetic(component):
    import meep as mp

    return component in (mp.Hx, mp.Hy, mp.Hz, mp.Bx, mp.By, mp.Bz)


class DFTFilter:
    """Filter via MEEP's native DFT accumulators (production path).

    Three subtleties are handled here:

    * ``decimation_factor=1`` is mandatory.  MEEP's default (0) picks a
      decimation from the Nyquist rate, which would subsample the quadrature
      and break the property that the forced omega-mode is filtered with
      multiplier exactly one.

    * MEEP exposes no way to zero a DFT accumulator, and ``restart_fields``
      deliberately leaves the transforms running.  But ``restart_fields`` also
      resets ``t`` to 0, so every window replays an identical phase sequence
      and the integral over window k is the *difference* of the running totals
      after windows k and k-1.

    * ``DftObj`` is lazy; it must be instantiated before ``get_dft_array``
      will accept it, and the component must be allocated before the DFT is
      added at all.
    """

    def __init__(self, sim, omega, period, components, where=None):
        self.sim = sim
        self.omega = omega
        self.T = period
        self.components = tuple(components)
        freq = omega / (2.0 * math.pi)
        # index 0 -> the cos(wt) term; index 1 -> the -1/4 (frequency zero) term
        self.freqs = [freq, 0.0]
        self.dfts = {}
        for c in self.components:
            sim.fields.require_component(c)
        for c in self.components:
            kwargs = dict(yee_grid=True, decimation_factor=1, persist=True)
            if where is not None:
                kwargs["where"] = where
            dft = sim.add_dft_fields([c], self.freqs, **kwargs)
            dft.swigobj_attr("where")  # force instantiation
            self.dfts[c] = dft
        self._running = {c: None for c in self.components}

    def reset(self):
        self._running = {c: None for c in self.components}

    def read(self, half_step_shift=True):
        """Pi applied over the window that just finished, per component.

        Magnetic components need a half-step phase rotation.  MEEP accumulates
        their DFT at s_n = t_n - dt/2 (``update_dfts(time(), time()-0.5*dt, t)``),
        so the filter returns the coefficient H_0 = H(0) at *integer* time --
        but the array slot the injector writes into holds H(-dt/2).  Feeding
        H(0) back into an H(-dt/2) slot is a half-step inconsistent round trip,
        and the resulting fixed point is not the T-periodic solution.

        Writing the periodic field as H = H_0 cos(wt) + H_1 sin(wt), the value
        actually wanted is

            H(-dt/2) = H_0 cos(w dt/2) - H_1 sin(w dt/2)
                     = Re[(H_0 + i H_1) exp(+i w dt/2)]

        and H_0 + i H_1 is exactly the (rescaled) complex transform already
        accumulated, so the correction is a single phase rotation.  Electric
        components need none: they are sampled at integer times and their slot
        holds the integer-time value.
        """
        out = {}
        dt = self.sim.fields.dt
        rot = np.exp(1j * self.omega * dt / 2.0)
        for c in self.components:
            dft = self.dfts[c]
            f_omega = np.asarray(self.sim.get_dft_array(dft, c, 0))
            f_zero = np.asarray(self.sim.get_dft_array(dft, c, 1))
            prev = self._running[c]
            if prev is None:
                d_omega, d_zero = f_omega, f_zero
            else:
                d_omega = f_omega - prev[0]
                d_zero = f_zero - prev[1]
            self._running[c] = (f_omega.copy(), f_zero.copy())
            if half_step_shift and _is_magnetic(c):
                d_omega = d_omega * rot
            # MEEP accumulates sum f exp(+i w t) dt / sqrt(2 pi); undo that to
            # recover the plain time integrals.  The frequency-zero term is
            # time-origin independent, so it needs no rotation.
            int_cos = SQRT_2PI * d_omega.real
            int_one = SQRT_2PI * d_zero.real
            out[c] = (2.0 / self.T) * int_cos - (1.0 / (2.0 * self.T)) * int_one
        return out


class ProbeFilter:
    """Reference filter evaluated at a few Yee points (validation only).

    ``quadrature='rectangle'`` reproduces what MEEP's DFT does -- samples at
    n = 1..M with uniform weight dt, since ``update_dfts`` runs after ``t += 1``
    inside ``fields::step``.  ``quadrature='trapezoid'`` is the composite rule
    of eqs. (25)-(26): n = 0..M with half weights at the ends.

    Over an integer number of periods both give multiplier exactly one on the
    forced omega-mode, so they share a fixed point and differ only in the
    transient contraction factor beta_h(lambda).
    """

    def __init__(self, sim, omega, period, probes, quadrature="rectangle"):
        if quadrature not in ("rectangle", "trapezoid"):
            raise ValueError(f"unknown quadrature {quadrature!r}")
        self.sim = sim
        self.omega = omega
        self.T = period
        self.probes = list(probes)  # (component, x, y)
        self.quadrature = quadrature
        self.M = None
        self._acc = None

    def start(self, nsteps):
        self.M = nsteps
        self._acc = np.zeros(len(self.probes))

    def _weight(self, n):
        if self.quadrature == "rectangle":
            return 0.0 if n == 0 else 1.0
        return 0.5 if (n == 0 or n == self.M) else 1.0

    def sample(self, n):
        import meep as mp

        weight = self._weight(n)
        if weight == 0.0:
            return
        dt = self.sim.fields.dt
        t_e = n * dt
        # H lags E by half a step; match MEEP's own DFT staggering,
        # update_dfts(time(), time() - 0.5*dt, t).
        t_h = t_e - 0.5 * dt
        for k, (c, x, y) in enumerate(self.probes):
            t = t_h if _is_magnetic(c) else t_e
            val = self.sim.fields.get_field(c, mp.vec(float(x), float(y))).real
            self._acc[k] += weight * dt * (math.cos(self.omega * t) - 0.25) * val

    def read(self):
        return (2.0 / self.T) * self._acc
