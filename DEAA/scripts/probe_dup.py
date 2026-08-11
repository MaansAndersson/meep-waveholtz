"""With PML, initialize_field visits some Yee points more than once.
Since it accumulates (f += val), does that double-write the initial data?"""

import pathlib
import sys
from collections import Counter

import numpy as np
import meep as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def check(dpml):
    size, res = 2.0, 8
    sim = mp.Simulation(
        cell_size=mp.Vector3(size, size, 0), resolution=res,
        dimensions=2, Courant=0.5,
        boundary_layers=[mp.PML(dpml)] if dpml else [],
    )
    sim.init_sim()
    sim.fields.require_component(mp.Ez)

    pts = []

    def rec(p):
        pts.append((round(p.x, 9), round(p.y, 9)))
        return 0.0

    sim.initialize_field(mp.Ez, rec)
    counts = Counter(pts)
    dups = {p: n for p, n in counts.items() if n > 1}
    print(f"\n=== dpml = {dpml} ===")
    print(f"  visits {len(pts)}, distinct {len(counts)}, duplicated {len(dups)}")
    if dups:
        mult = Counter(dups.values())
        print(f"  multiplicity histogram: {dict(mult)}")
        sample = sorted(dups)[:4]
        print(f"  sample duplicated points: {sample}")

    # Now inject the constant 1 and read the field back pointwise.
    sim.restart_fields()
    sim.initialize_field(mp.Ez, lambda p: 1.0)
    xs = np.round(np.unique([p[0] for p in pts]), 9)
    ys = np.round(np.unique([p[1] for p in pts]), 9)
    vals = np.array([[sim.fields.get_field(mp.Ez, mp.vec(float(x), float(y))).real
                      for y in ys] for x in xs])
    print(f"  after injecting constant 1.0: min {vals.min():.6f}, "
          f"max {vals.max():.6f}")
    bad = np.argwhere(np.abs(vals - 1.0) > 1e-9)
    print(f"  points differing from 1.0: {len(bad)} of {vals.size}")
    if len(bad):
        for i, j in bad[:5]:
            print(f"    ({xs[i]:+.5f},{ys[j]:+.5f}) = {vals[i, j]:.6f}")


if __name__ == "__main__":
    check(0.0)
    check(0.5)
