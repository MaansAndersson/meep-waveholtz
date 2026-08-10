"""Ground truth: which Yee points does initialize_field visit, and do they
match the shape of get_dft_array(..., yee_grid=True)?"""

import sys

import numpy as np
import meep as mp

sys.path.insert(0, "/Users/appelo/Desktop/MEEP_STUFF")


def force_dft(sim, dft):
    """DftObj is lazy; touching an attribute instantiates the swig object."""
    dft.swigobj_attr("where")
    return dft


def record_positions(sim, c):
    pts = []

    def rec(p):
        pts.append((p.x, p.y))
        return 0.0

    sim.initialize_field(c, rec)
    return np.array(pts)


def main(size=2.0, res=8, dpml=0.0):
    sim = mp.Simulation(
        cell_size=mp.Vector3(size, size, 0), resolution=res,
        dimensions=2, Courant=0.5,
        boundary_layers=[mp.PML(dpml)] if dpml else [],
    )
    sim.init_sim()
    for c in (mp.Ez, mp.Hx, mp.Hy):
        sim.fields.require_component(c)
    dfts = {}
    for c in (mp.Ez, mp.Hx, mp.Hy):
        d = sim.add_dft_fields([c], [0.3, 0.0], yee_grid=True,
                               decimation_factor=1, persist=True)
        dfts[c] = force_dft(sim, d)

    print(f"\n### cell {size} x {size}, resolution {res}, dpml {dpml}, dx = {1/res}")
    for c, name in ((mp.Ez, "Ez"), (mp.Hx, "Hx"), (mp.Hy, "Hy")):
        pts = record_positions(sim, c)
        xs = np.unique(np.round(pts[:, 0], 10))
        ys = np.unique(np.round(pts[:, 1], 10))
        arr = np.asarray(sim.get_dft_array(dfts[c], c, 0))
        print(f"\n{name}:")
        print(f"  initialize_field visited {len(pts)} points -> lattice "
              f"({len(xs)}, {len(ys)})")
        print(f"    x: [{xs[0]:.5f} ... {xs[-1]:.5f}] step {xs[1]-xs[0]:.5f}")
        print(f"    y: [{ys[0]:.5f} ... {ys[-1]:.5f}] step {ys[1]-ys[0]:.5f}")
        print(f"  get_dft_array shape {arr.shape}   MATCH: "
              f"{arr.shape == (len(xs), len(ys))}")
        expect = np.array([(x, y) for x in xs for y in ys])
        rowmajor = (len(pts) == len(expect)
                    and np.allclose(np.round(pts, 10), expect))
        print(f"  visit order row-major (x outer, y inner): {rowmajor}")


if __name__ == "__main__":
    main(2.0, 8, 0.0)
    main(2.0, 8, 0.5)
    main(3.0, 10, 0.0)
