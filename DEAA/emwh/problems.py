"""Test problems, principally the ring resonator of Fig. 1.

Geometry follows Sec. III-A: computational domain [-6,6]^2, a ring resonator
with eps_r = 3.4^2 occupying 1 <= sqrt(x^2+y^2) <= 2, eps = 1 outside, mu = 1,
and two Ez point sources at (1.1, 0) with magnitude 1 and (-1.1, 0) with
magnitude -1.  omega_0 = 0.118 * 2*pi, i.e. MEEP frequency f_0 = 0.118.

The paper closes the domain with a double absorbing boundary layer; here MEEP's
PML stands in for it, placed *outside* [-6,6]^2 so the physical domain is
unchanged.
"""

import math

import meep as mp

OMEGA0 = 0.118 * 2.0 * math.pi
HALF = 6.0
RING_OUTER = 2.0
RING_INNER = 1.0
RING_EPS = 3.4**2
SOURCE_X = 1.1

TM_COMPONENTS = (mp.Ez, mp.Hx, mp.Hy)


def ring_geometry():
    return [
        mp.Cylinder(
            radius=RING_OUTER,
            height=mp.inf,
            material=mp.Medium(epsilon=RING_EPS),
        ),
        mp.Cylinder(radius=RING_INNER, height=mp.inf, material=mp.air),
    ]


def ring_sources(omega, forcing="cos"):
    """Two antisymmetric Ez point sources.

    MEEP drives the real-field current with Re[amplitude * exp(-i*omega*t)], so
    a real amplitude gives cos-forcing (eq. 10) and an amplitude of 1j gives
    sin-forcing (eq. 4).  ``width=0`` suppresses the turn-on ramp: the forcing
    must be a pure sinusoid from t = 0 for the periodic problem to be the one
    we think it is.
    """
    if forcing == "cos":
        phase = 1.0
    elif forcing == "sin":
        phase = 1j
    else:
        raise ValueError(f"unknown forcing {forcing!r}")

    src = mp.ContinuousSource(
        frequency=omega / (2.0 * math.pi), width=0, start_time=0, end_time=1e20
    )
    return [
        mp.Source(
            src,
            component=mp.Ez,
            center=mp.Vector3(SOURCE_X, 0),
            amplitude=phase * 1.0,
        ),
        mp.Source(
            src,
            component=mp.Ez,
            center=mp.Vector3(-SOURCE_X, 0),
            amplitude=phase * -1.0,
        ),
    ]


def ring_simulation(
    omega=OMEGA0,
    resolution=10,
    dpml=1.0,
    courant=0.5,
    forcing="cos",
    complex_fields=False,
    with_sources=True,
):
    """Build the ring-resonator ``mp.Simulation``.

    ``resolution`` is points per unit length, so the paper's N = 120, 240, 480
    over [-6,6] correspond to resolution 10, 20, 40.
    """
    cell = mp.Vector3(2 * (HALF + dpml), 2 * (HALF + dpml), 0)
    return mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        geometry=ring_geometry(),
        sources=ring_sources(omega, forcing) if with_sources else [],
        boundary_layers=[mp.PML(dpml)] if dpml > 0 else [],
        Courant=courant,
        force_complex_fields=complex_fields,
        dimensions=2,
    )


def pec_box_simulation(
    omega=OMEGA0,
    resolution=10,
    size=2.0,
    courant=0.5,
    forcing="cos",
    with_sources=True,
):
    """Small closed PEC cavity: the energy-conserving case of Sec. I-A.

    No PML and no dispersion, so the discrete operator is exactly the one
    Theorem 1 and Appendix B describe -- I - S should come out symmetric
    positive definite.  Used by the unit tests.
    """
    src = mp.ContinuousSource(
        frequency=omega / (2.0 * math.pi), width=0, start_time=0, end_time=1e20
    )
    sources = (
        [
            mp.Source(
                src,
                component=mp.Ez,
                center=mp.Vector3(0.1, 0.15),
                amplitude=1.0 if forcing == "cos" else 1j,
            )
        ]
        if with_sources
        else []
    )
    return mp.Simulation(
        cell_size=mp.Vector3(size, size, 0),
        resolution=resolution,
        sources=sources,
        boundary_layers=[],
        Courant=courant,
        dimensions=2,
    )
