"""Empirically check the MEEP behaviours the EM-WaveHoltz design relies on.

Run this first; every later piece assumes these answers.
"""

import math
import pathlib
import sys

import numpy as np
import meep as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def probe_source_waveform():
    print("\n=== 1. What current does ContinuousSource(width=0) actually drive? ===")
    f = 0.118
    omega = 2 * np.pi * f
    dt = 0.5 / 10
    s = mp.continuous_src_time(f, 0, 0, 1e20, 3.0)
    ts = [0.0, 0.5, 1.0, 2.0, 4.0]

    # dipole() is the dipole moment; the driven current is its time difference.
    print(f"  dipole(1.0)  = {s.dipole(1.0):.6f}   (|.| = {abs(s.dipole(1.0)):.6f})")
    cur = [s.current(t, dt) for t in ts]
    # Hypothesis: current = sinc(w dt/2) * exp(-i w (t + dt/2))
    scale = 2 * np.sin(omega * dt / 2) / (omega * dt)
    pred = [scale * np.exp(-1j * omega * (t + dt / 2)) for t in ts]
    err = max(abs(a - b) for a, b in zip(cur, pred))
    print(f"  current(1.0) = {cur[2]:.6f}")
    print(f"  predicted    = {pred[2]:.6f}")
    print(f"  max|current - sinc(w dt/2)*exp(-i w (t+dt/2))| = {err:.3e}")
    print(f"  amplitude factor sin(w dt/2)/(w dt/2) = {scale:.10f}")
    print("  -> note this is exactly omega_tilde/omega of paper eq. (31)")


def probe_grids():
    print("\n=== 2. Yee grids: get_array vs get_dft_array(yee_grid=True) ===")
    sim = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 0), resolution=8, dimensions=2, Courant=0.5
    )
    sim.init_sim()
    for c in (mp.Ez, mp.Hx, mp.Hy):
        sim.fields.require_component(c)
    dfts = {
        c: sim.add_dft_fields([c], [0.3, 0.0], yee_grid=True,
                              decimation_factor=1, persist=True)
        for c in (mp.Ez, mp.Hx, mp.Hy)
    }
    sim.init_sim()
    for c, name in ((mp.Ez, "Ez"), (mp.Hx, "Hx"), (mp.Hy, "Hy")):
        xs, ys, zs, w = sim.get_array_metadata(dft_cell=dfts[c])
        arr_dft = np.asarray(sim.get_dft_array(dfts[c], c, 0))
        arr_get = np.asarray(sim.get_array(component=c))
        print(f"  {name}: dft_array {arr_dft.shape}  get_array {arr_get.shape}  "
              f"metadata ({len(xs)},{len(ys)})")
        print(f"        x[0..2] = {np.asarray(xs)[:3]}  y[0..2] = {np.asarray(ys)[:3]}")
    return sim, dfts


def probe_roundtrip():
    print("\n=== 3. initialize_field -> get_array round trip ===")
    sim = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 0), resolution=8, dimensions=2, Courant=0.5
    )
    sim.init_sim()

    def f(p):
        return math.sin(3.0 * p.x) * math.cos(2.0 * p.y)

    sim.initialize_field(mp.Ez, f)
    arr = np.asarray(sim.get_array(component=mp.Ez)).real
    xs, ys, _, _ = sim.get_array_metadata(center=mp.Vector3(), size=mp.Vector3(2, 2, 0))
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    exact = np.sin(3.0 * xs)[:, None] * np.cos(2.0 * ys)[None, :]
    print(f"  get_array shape {arr.shape}, metadata ({len(xs)},{len(ys)})")
    print(f"  max|get_array - exact_at_metadata_points| = "
          f"{np.abs(arr - exact).max():.3e}")
    print("  -> if ~0, get_array returns raw Yee values for Ez (no interpolation)")


def probe_restart_pml():
    print("\n=== 4. restart_fields() clears PML auxiliary state ===")
    sim = mp.Simulation(
        cell_size=mp.Vector3(6, 6, 0),
        resolution=8,
        dimensions=2,
        Courant=0.5,
        boundary_layers=[mp.PML(1.0)],
    )
    sim.init_sim()
    # Drive a pulse into the PML by hand, no sources involved.
    sim.initialize_field(mp.Ez, lambda p: math.exp(-4.0 * (p.x**2 + p.y**2)))
    for _ in range(120):
        sim.fields.step()
    before = np.abs(np.asarray(sim.get_array(component=mp.Ez)).real).max()
    sim.restart_fields()
    after_restart = np.abs(np.asarray(sim.get_array(component=mp.Ez)).real).max()
    # If any auxiliary (PML/conductivity) state survived, stepping from zero
    # initial data with no sources would regenerate a nonzero field.
    for _ in range(40):
        sim.fields.step()
    after_steps = np.abs(np.asarray(sim.get_array(component=mp.Ez)).real).max()
    print(f"  |Ez| before restart      = {before:.6e}")
    print(f"  |Ez| right after restart = {after_restart:.6e}")
    print(f"  |Ez| after 40 more steps = {after_steps:.6e}   (must be exactly 0)")
    print(f"  fields.t after restart   = {sim.fields.t}")


def probe_dft_across_restart():
    print("\n=== 5. DFT accumulators survive restart_fields and keep accumulating ===")
    omega = 2.0 * math.pi * 0.25
    resolution = 8
    period = 2.0 * math.pi / omega
    steps = int(round(period * resolution / 0.5))
    courant = period * resolution / steps
    sim = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 0), resolution=resolution,
        dimensions=2, Courant=courant,
    )
    sim.init_sim()
    sim.fields.require_component(mp.Ez)
    dft = sim.add_dft_fields([mp.Ez], [omega / (2 * math.pi), 0.0],
                             yee_grid=True, decimation_factor=1, persist=True)
    sim.init_sim()
    print(f"  dt = {sim.fields.dt:.8f}, steps/period = {period / sim.fields.dt:.8f}")

    totals = []
    for window in range(3):
        sim.restart_fields()
        sim.initialize_field(mp.Ez, lambda p: math.exp(-4.0 * (p.x**2 + p.y**2)))
        for _ in range(steps):
            sim.fields.step()
        arr = np.asarray(sim.get_dft_array(dft, mp.Ez, 0))
        totals.append(np.abs(arr).max())
        print(f"  after window {window}: max|cumulative DFT| = {totals[-1]:.8e}, "
              f"fields.t = {sim.fields.t}")
    d1 = totals[1] - totals[0]
    print(f"  increments: {totals[0]:.8e}, {d1:.8e}, {totals[2] - totals[1]:.8e}")
    print("  -> increments should be equal: identical windows, cumulative totals")


if __name__ == "__main__":
    probe_source_waveform()
    probe_grids()
    probe_roundtrip()
    probe_restart_pml()
    probe_dft_across_restart()
