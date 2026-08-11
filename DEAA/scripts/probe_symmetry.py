"""Where does the asymmetry of I - S live, and which inner product fixes it?"""

import math
import pathlib
import sys

import numpy as np
import meep as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tests.test_operator import build_operator


def main():
    wh, S, grid = build_operator()
    nx, ny = grid.shape
    A = np.eye(S.shape[0]) - S
    D = A - A.T
    Dg = np.abs(D).reshape(nx, ny, nx, ny)

    print(f"\ngrid {grid.shape}, max|A-A^T| = {np.abs(D).max():.4e}")
    print(f"x coords: {grid.xs}")

    # Which rows carry the asymmetry?
    row = np.abs(D).max(axis=1).reshape(nx, ny)
    print("\nmax|A-A^T| by row (i,j):")
    for i in range(nx):
        print("  " + " ".join(f"{row[i, j]:7.4f}" for j in range(ny)))

    # Are boundary rows/cols of S zero (PEC-constrained DOFs)?
    Sg = S.reshape(nx, ny, nx, ny)
    edge = [(0, j) for j in range(ny)] + [(nx - 1, j) for j in range(ny)]
    edge += [(i, 0) for i in range(nx)] + [(i, ny - 1) for i in range(nx)]
    edge = sorted(set(edge))
    emax_row = max(np.abs(Sg[i, j]).max() for i, j in edge)
    emax_col = max(np.abs(Sg[:, :, i, j]).max() for i, j in edge)
    print(f"\nboundary DOFs: max|S row| = {emax_row:.3e}, "
          f"max|S col| = {emax_col:.3e}")

    # Candidate Yee quadrature weights: half cell on faces, quarter at corners.
    w = np.ones((nx, ny))
    w[0, :] *= 0.5
    w[-1, :] *= 0.5
    w[:, 0] *= 0.5
    w[:, -1] *= 0.5
    W = np.diag(w.ravel())
    WA = W @ A
    print(f"\nEuclidean : max|A - A^T|   = {np.abs(A - A.T).max():.4e}")
    print(f"Yee weight: max|WA - (WA)^T| = {np.abs(WA - WA.T).max():.4e}")

    # And with the boundary ring dropped entirely.
    keep = np.ones((nx, ny), dtype=bool)
    keep[0, :] = keep[-1, :] = keep[:, 0] = keep[:, -1] = False
    k = keep.ravel()
    Ai = A[np.ix_(k, k)]
    print(f"interior only: max|A - A^T| = {np.abs(Ai - Ai.T).max():.4e}  "
          f"({k.sum()} dofs)")


if __name__ == "__main__":
    main()
