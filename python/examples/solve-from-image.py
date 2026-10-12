"""
Define a 2d structure from an IMAGE, solve it in the frequency domain with the
WaveHoltz solver (Simulation.solve_wh), and plot the structure and the field.

Two ways to turn an image into materials are available:

  --method grid  (default)
      A mp.MaterialGrid inside a mp.Block that covers the physical extent of the
      image.  Pixel values in [0,1] interpolate between two media (0 -> air,
      1 -> material of index n).  With smoothing on (default) MEEP applies its
      level-set subpixel smoothing with the interface at pixel value 0.5, so a
      binary image gives a sharp, accurately averaged boundary.  --no-smooth uses
      plain bilinear interpolation of the pixel values instead.

  --method h5
      Convert the image to an epsilon array, write it to an HDF5 file and pass it
      as Simulation(epsilon_input_file=...).  The dataset is stretched over the
      WHOLE cell (PML included), so the image is first padded to the cell size.

Image conventions: grayscale (RGB is converted), white = material (index n),
black = air.  Image row 0 is the TOP of the cell (large y).  The physical width of
the image is given by --image-size (its height follows from the aspect ratio).
If --image is not given, a synthetic ring.png is generated so the example is
self-contained.

Run inside the meep-serial or meep-mpi env:

    python solve-from-image.py                                  # synthetic ring, MaterialGrid
    python solve-from-image.py --image my_design.png --image-size 6 --fcen 0.2
    mpirun -np 4 python solve-from-image.py --image my_design.png --image-size 6

Then:  open solve_from_image.png
"""

import argparse
import time

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import meep as mp

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--image", default=None, help="PNG/JPG to use (default: generate ring.png)")
parser.add_argument("--image-size", type=float, default=6.0, help="physical width of the image (default 6)")
parser.add_argument("--pixels-per-unit", type=int, default=100, help="resolution of the generated image")
parser.add_argument("--index", type=float, default=3.4, help="refractive index of the white regions")
parser.add_argument("--method", choices=["grid", "h5"], default="grid")
parser.add_argument("--no-smooth", action="store_true", help="grid method: plain bilinear interpolation")
parser.add_argument("--beta", type=float, default=1000.0, help="grid method: level-set projection sharpness")
parser.add_argument("--solver", choices=["gmres", "bicgstab"], default="gmres")
parser.add_argument("--tol", type=float, default=1e-6)
parser.add_argument("--maxiters", type=int, default=10000)
parser.add_argument("--L", type=int, default=10)
parser.add_argument("--periods", type=int, default=1)
parser.add_argument("--fcen", type=float, default=0.118,
                    help="source frequency (default 0.118, a resonance of the synthetic ring)")
parser.add_argument("--resolution", type=int, default=20)
parser.add_argument("--out", default="solve_from_image.png")
args = parser.parse_args()

# ---------------------------------------------------------------- cell, sources (same as the ring examples)
n = args.index  # refractive index of the material (white pixels)
w = 1  # width of the synthetic ring
r = 1  # inner radius of the synthetic ring
pad = 4  # padding between the image region and PML
dpml = 2  # PML thickness
fcen = args.fcen

sxy = 2 * (r + w + pad + dpml)
cell_size = mp.Vector3(sxy, sxy)
nonpml_vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(sxy - 2 * dpml, sxy - 2 * dpml))
material = mp.Medium(index=n)


def say(*a):
    if mp.am_master():
        print(*a, flush=True)


# ---------------------------------------------------------------- 1. get an image
def make_ring_image(L, ppu):
    """Synthetic white ring on black, L x L units at ppu pixels per unit, row 0 = top."""
    N = int(round(L * ppu))
    x = (np.arange(N) + 0.5) / ppu - L / 2  # pixel centres
    X, Y = np.meshgrid(x, x)  # rows run along y, columns along x
    R = np.hypot(X, Y)
    img = ((R >= r) & (R <= r + w)).astype(np.uint8) * 255
    return np.flipud(img)  # image convention: first row is the top


if args.image is None:
    image = make_ring_image(args.image_size, args.pixels_per_unit)
    if mp.am_master():
        Image.fromarray(image).save("ring.png")
        print("generated ring.png", flush=True)
else:
    image = np.asarray(Image.open(args.image).convert("L"))

# pixel values -> weights in [0,1], indexed [ix, iy] with y increasing (MEEP's convention)
weights = np.flipud(image.astype(np.float64) / 255.0).T
Nx, Ny = weights.shape
Lx = args.image_size
Ly = Lx * Ny / Nx
say(f"image {Nx} x {Ny} pixels covering {Lx:g} x {Ly:g} units ({Nx / Lx:g} pixels per unit)")
say(f"fcen={fcen:g}  resolution={args.resolution} ({args.resolution / (fcen * n):.1f} pixels per wavelength in the material)")

# ---------------------------------------------------------------- 2. image -> MEEP material
geometry = []
epsilon_input_file = ""
if args.method == "grid":
    smooth = not args.no_smooth
    matgrid = mp.MaterialGrid(
        mp.Vector3(Nx, Ny),
        mp.air,  # pixel value 0
        material,  # pixel value 1
        weights=weights,
        do_averaging=smooth,
        beta=args.beta if smooth else 0,
        eta=0.5,
    )
    geometry = [mp.Block(center=mp.Vector3(), size=mp.Vector3(Lx, Ly, 0), material=matgrid)]
    say(f"method: MaterialGrid in a {Lx:g} x {Ly:g} Block, "
        + ("subpixel smoothing with level-set projection" if smooth else "plain bilinear interpolation"))
else:
    import h5py

    # epsilon on the image pixels, then pad with air to the full cell (the file spans the whole cell)
    px = Lx / Nx
    Ncx, Ncy = int(round(sxy / px)), int(round(sxy / px))
    eps_cell = np.ones((Ncx, Ncy))
    x0, y0 = (Ncx - Nx) // 2, (Ncy - Ny) // 2
    eps_cell[x0 : x0 + Nx, y0 : y0 + Ny] = 1 + (n**2 - 1) * weights
    epsilon_input_file = "image_eps.h5:eps"
    if mp.am_master():
        with h5py.File("image_eps.h5", "w") as f:
            f["eps"] = eps_cell  # shape (nx, ny), x is the slow index, as MEEP expects
    mp.all_wait()  # make sure the file exists before the other ranks read it
    say(f"method: epsilon_input_file, dataset {Ncx} x {Ncy} stretched over the {sxy:g} x {sxy:g} cell")

# ---------------------------------------------------------------- 3. solve
src = [
    mp.Source(mp.ContinuousSource(fcen), component=mp.Ez, center=mp.Vector3(r + 0.1)),
    mp.Source(mp.ContinuousSource(fcen), component=mp.Ez, center=mp.Vector3(-(r + 0.1)), amplitude=-1),
]
# no mirror symmetries: a user-supplied image is in general not symmetric
sim = mp.Simulation(
    resolution=args.resolution,
    cell_size=cell_size,
    geometry=geometry,
    epsilon_input_file=epsilon_input_file,
    sources=src,
    boundary_layers=[mp.PML(dpml)],
)
sim.init_sim()
solver = mp.GMRES if args.solver == "gmres" else mp.BICGSTAB
t0 = time.time()
ok = sim.solve_wh(args.tol, args.maxiters, args.L, args.periods, solver)
say(f"solve_wh ({args.solver.upper()}, tol={args.tol:g}, L={args.L}, periods={args.periods}): "
    f"converged={ok}  wall time {time.time() - t0:.1f} s")

eps = sim.get_array(vol=nonpml_vol, component=mp.Dielectric)
ez = np.real(sim.get_array(vol=nonpml_vol, component=mp.Ez))

# ---------------------------------------------------------------- 4. plot (master only)
if mp.am_master():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    im = ax1.imshow(eps.transpose(), origin="lower", cmap="binary", interpolation="none")
    ax1.set_title(f"epsilon from image ({args.method})")
    fig.colorbar(im, ax=ax1, shrink=0.8)
    vmax = np.max(np.abs(ez))
    ax2.imshow(eps.transpose(), origin="lower", cmap="binary", interpolation="spline36")
    im = ax2.imshow(ez.transpose(), origin="lower", cmap="RdBu", vmin=-vmax, vmax=vmax, alpha=0.9,
                    interpolation="spline36")
    ax2.set_title(f"WaveHoltz $E_z$ at f = {fcen:g}")
    fig.colorbar(im, ax=ax2, shrink=0.8)
    for ax in (ax1, ax2):
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}", flush=True)
