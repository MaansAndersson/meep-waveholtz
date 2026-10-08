"""FDTD vs solve_cw vs solve_waveholtz_cw on the wh.cpp case-1 cavity (eps=2, PEC, no PML).

Manufactured solution: Ez = poly cos(w t), poly = (x^2-1)(y^2-1), forced by Jz_poly sin(w t).
"""
import os
import sys
import time
import numpy as np
import meep as mp

res = int(sys.argv[1]) if len(sys.argv) > 1 else 20
omega = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
ring = bool(sys.argv[3]) if len(sys.argv) > 3 else False
wh_restart = int(os.environ.get("CAVITY_WH_RESTART", 10))   # GMRES restart length for WH
skip_cw = bool(os.environ.get("CAVITY_SKIP_CW"))            # WH only: no CW solve, no plot
eps = 1.0
T = 2 * np.pi / omega
Nt = int(T * res * 2)                  # same choice as wh.cpp: T/dt integral, Courant ~ 0.5
courant = T / Nt * res
# Relative nudge to the Courant number. solve_waveholtz_cw recomputes Nt = ceil(T/dt): a nudge of
# +1e-10 keeps T/dt just below Nt, so the C++ side cannot round up to Nt+1.
courant *= 1 + float(os.environ.get("CAVITY_COURANT_NUDGE", 0))
dt = courant / res

poly = lambda x, y: (x**2 - 1) * (y**2 - 1)
jz = lambda p: omega * eps * poly(p.x, p.y) + (2 * (p.y**2 - 1) + 2 * (p.x**2 - 1)) / omega
x = 0.5 * omega * dt
amp = 1j * np.exp(1j * x) / (np.sin(x) / x)   # driven current -> sin(w t), as add_cw_source


def run(solve, width=0, complex_fields=False):
    src = mp.ContinuousSource(frequency=omega / (2 * np.pi), width=width)
    sim = mp.Simulation(cell_size=mp.Vector3(2, 2), resolution=res, Courant=courant,
                        default_material=mp.Medium(epsilon=eps), force_complex_fields=complex_fields,
                        sources=[mp.Source(src, mp.Ez, center=mp.Vector3(), size=mp.Vector3(2, 2),
                                           amp_func=jz, amplitude=amp)])
    sim.init_sim()
    solve(sim)
    return np.real(sim.get_array(component=mp.Ez)), sim.get_array_metadata()[:2]

def run_pml(solve, width=0, complex_fields=False):
    src = mp.ContinuousSource(frequency=omega / (2 * np.pi), width=width)
    dpml = 0.5
    sim = mp.Simulation(cell_size=mp.Vector3(2+dpml, 2+dpml), resolution=res, Courant=courant,
                        default_material=mp.Medium(epsilon=eps),boundary_layers=[mp.PML(dpml/2)], force_complex_fields=complex_fields,
                        sources=[mp.Source(src, mp.Ez, center=mp.Vector3(), size=mp.Vector3(2, 2),
                                           amp_func=jz, amplitude=amp)])
    sim.init_sim()
    solve(sim)
    return np.real(sim.get_array(component=mp.Ez)), sim.get_array_metadata()[:2]

def run_ring(solve, width=0, complex_fields=False):
    
    n = 3.4  # refractive index of ring
    w = 1  # width of ring
    r = 1  # inner radius of ring
    pad = 4  # padding between outer ring and PML
    dpml = 2  # PML thickness
    
    sxy = 2 * (r + w + pad + dpml)
    cell_size = mp.Vector3(sxy, sxy)
    
    pml_layers = [mp.PML(dpml)]
    
    nonpml_vol = mp.Volume(
        center=mp.Vector3(),
        size=mp.Vector3(sxy - 2 * dpml, sxy - 2 * dpml),
    )
    
    geometry = [
        mp.Cylinder(radius=r + w, material=mp.Medium(index=n)),
        mp.Cylinder(radius=r),
    ]
    
    fcen = 0.118  # frequency of resonant mode
    
    # src = mp.ContinuousSource(frequency=omega / (2 * np.pi),
    #                          width=width)
    src = [
        mp.Source(
            mp.ContinuousSource(fcen, width=width),
            component=mp.Ez,
            center=mp.Vector3(r + 0.1),
        ),
        mp.Source(
            mp.ContinuousSource(fcen, width=width),
            component=mp.Ez,
            center=mp.Vector3(-(r + 0.1)),
            amplitude=-1,
        ),
    ]

    symmetries = [
        mp.Mirror(mp.X, phase=-1),
        mp.Mirror(mp.Y, phase=+1),
    ]

    sim = mp.Simulation(
        resolution=res,
        cell_size=cell_size,
        geometry=geometry,
        sources=src,
        force_complex_fields=complex_fields,
        symmetries=symmetries,
        boundary_layers=pml_layers,
    )
    
    sim.init_sim()
    solve(sim)
    return np.real(sim.get_array(component=mp.Ez)), sim.get_array_metadata()[:2]



def timed(name, solve):
    """Wrap a solver call so its wall time (solve only, not structure setup) is printed."""
    def f(sim):
        t0 = time.perf_counter()
        solve(sim)
        print(f"{name} solve time: {time.perf_counter() - t0:.2f} s", flush=True)
    return f


runner = run_ring if ring else run
if os.environ.get("CAVITY_SKIP_WH"):  # CW only: solve, optionally save, and stop
    ez_cw, (xs, ys) = runner(timed("cw", lambda s: s.solve_cw(1e-10, 2000, 10)), complex_fields=True)
    if os.environ.get("CAVITY_SAVE_CW"):
        np.savez(os.environ["CAVITY_SAVE_CW"], ez=ez_cw, xs=xs, ys=ys)
    sys.exit(0)
# WH first, and its error is printed before CW starts, so it survives if the CW run is killed.
results = {"wh": runner(timed("wh", lambda s: s.solve_waveholtz_cw(1e-10, 400, wh_restart)))}
xs, ys = results["wh"][1]
exact = poly(*np.meshgrid(xs, ys, indexing="ij"))
nrm = lambda a: np.linalg.norm(a) / np.linalg.norm(exact)
print(f"wh    rel. L2 error vs exact: {nrm(results['wh'][0] - exact):.3e}", flush=True)
if os.environ.get("CAVITY_SAVE_WH"):  # save WH field + grid + exact for offline plotting
    np.savez(os.environ["CAVITY_SAVE_WH"], ez=results["wh"][0], xs=xs, ys=ys, exact=exact)
if skip_cw:
    sys.exit(0)

results["cw"] = runner(timed("cw", lambda s: s.solve_cw(1e-10, 2000, 10)), complex_fields=True)
if os.environ.get("CAVITY_SAVE_CW"):
    np.savez(os.environ["CAVITY_SAVE_CW"], ez=results["cw"][0], xs=xs, ys=ys)
print(f"cw    rel. L2 error vs exact: {nrm(results['cw'][0] - exact):.3e}", flush=True)
# results["fdtd"] = runner(lambda s: s.run(until=50 * T), width=4 * T)   # smooth turn-on, sample at t = 50T
# print(f"fdtd  rel. L2 error vs exact: {nrm(results['fdtd'][0] - exact):.3e}")
print(f"cw-wh rel. L2 diff: {nrm(results['cw'][0] - results['wh'][0]):.3e}")
# normalized by the CW field itself (meaningful for the ring, where `exact` is not a solution)
cw_wh = np.linalg.norm(results['cw'][0] - results['wh'][0]) / np.linalg.norm(results['cw'][0])
print(f"cw-wh rel. L2 diff vs |cw|: {cw_wh:.3e}")
# print(f"fdtd-wh rel. L2 diff: {nrm(results['fdtd'][0] - results['wh'][0]):.3e}")
# print(f"fdtd-cw rel. L2 diff: {nrm(results['fdtd'][0] - results['cw'][0]):.3e}")


def plot_ez(results, eps_arr, ref="cw", fname="notes/cavity_ez.png"):
    """Top row: Ez per solver. Bottom row: Ez - Ez[ref] per solver. eps contours overlaid."""
    import matplotlib.pyplot as plt
    names = list(results)
    fig, ax = plt.subplots(2, len(names), figsize=(4 * len(names), 7), squeeze=False)
    for j, k in enumerate(names):
        ez, (xs, ys) = results[k]
        ext = [xs[0], xs[-1], ys[0], ys[-1]]
        for i, (a, t) in enumerate([(ez, f"Ez {k}"), (ez - results[ref][0], f"Ez {k} - {ref}")]):
            im = ax[i, j].imshow(a.T, origin="lower", extent=ext, cmap="RdBu_r")
            if np.ptp(eps_arr) > 0:  # uniform eps (cavity) has no interfaces to draw
                ax[i, j].contour(xs, ys, eps_arr.T, colors="k", linewidths=0.5)
            ax[i, j].set_title(t)
            fig.colorbar(im, ax=ax[i, j])
    fig.tight_layout()
    fig.savefig(fname, dpi=150)


_eps = []
(run_ring if ring else run)(lambda s: _eps.append(s.get_array(component=mp.Dielectric)))
plot_ez(results, _eps[0])
