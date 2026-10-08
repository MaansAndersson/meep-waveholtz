/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include <math.h>
#include <string.h>

#include "meep/mympi.hpp"
#include "gmres.hpp"

#include "config.h"

/* gmres() implements the restarted generalized minimal residual method GMRES(m) for
   non-symmetric linear operators, as described in:

      Youcef Saad and Martin H. Schultz, "GMRES: A generalized minimal residual algorithm for
      solving nonsymmetric linear systems," SIAM J. Sci. Stat. Comput. 7, 856-869 (1986).

   The Arnoldi basis is orthogonalized with modified Gram-Schmidt and the Hessenberg least-squares
   problem is solved progressively with Givens rotations, which gives a residual estimate at every
   iteration.  Convergence is only accepted on the true residual |b - Ax|, which is recomputed at
   every restart, since the Givens estimate is prone to rounding errors. */

using namespace std;

namespace meep {

static double dot(size_t n, const realnum *x, const realnum *y) {
  double sum = 0;
  for (size_t i = 0; i < n; ++i)
    sum += x[i] * y[i];
  return sum_to_all(sum);
}

static double norm2(size_t n, const realnum *x) {
  // note: we don't just do sqrt(dot(n, x, x)) in order to avoid overflow
  size_t i;
  double xmax = 0, scale;
  long double sum = 0;
  for (i = 0; i < n; ++i) {
    double xabs = fabs(x[i]);
    if (xabs > xmax) xmax = xabs;
  }
  xmax = max_to_all(xmax);
  if (xmax == 0) return 0;
  scale = 1.0 / xmax;
  for (i = 0; i < n; ++i) {
    double xs = scale * x[i];
    sum += xs * xs;
  }
  return xmax * sqrt(sum_to_all(sum));
}

static void xpay(size_t n, realnum *x, double a, const realnum *y) {
  for (size_t m = 0; m < n; ++m)
    x[m] += a * y[m];
}

#define MEEP_MIN_OUTPUT_TIME 4.0 // output no more often than this many seconds

typedef realnum *prealnum; // grr, ISO C++ forbids new (double*)[...]

/* GMRES(m) algorithm for the n-by-n problem Ax = b */
ptrdiff_t gmres(const int m, const size_t n, realnum *x, gmres_op A, void *Adata,
                const realnum *b, const double tol, int *iters, realnum *work,
                const bool quiet) {
  if (!work) return (m + 2) * n; // required workspace

  // Arnoldi basis V[0..m] and a scratch vector w
  prealnum *V = new prealnum[m + 1];
  for (int i = 0; i <= m; ++i)
    V[i] = work + i * n;
  realnum *w = work + (m + 1) * n;

  // (m+1) x m upper-Hessenberg matrix, column-major: H(i,j) = H[j * (m + 1) + i]
  double *H = new double[(m + 1) * m];
#define H(i, j) H[(j) * (m + 1) + (i)]
  double *cs = new double[m]; // Givens rotations
  double *sn = new double[m];
  double *g = new double[m + 1]; // rotated right-hand side |r| e_1
  double *y = new double[m];

  double bnrm = norm2(n, b);
  if (bnrm == 0.0) bnrm = 1.0;

  int iter = 0;
  double last_output_wall_time = wall_time();

  int ierr = 0; // error code to return, if any

  double resid;
  while (1) {
    // V[0] = r = b - Ax
    A(x, w, Adata);
    for (size_t k = 0; k < n; ++k)
      V[0][k] = b[k] - w[k];
    resid = norm2(n, V[0]);
    if (resid <= tol * bnrm) break;
    if (iter >= *iters) {
      ierr = 1;
      break;
    }

    {
      double s = 1.0 / resid;
      for (size_t k = 0; k < n; ++k)
        V[0][k] *= s;
    }
    g[0] = resid;
    for (int i = 1; i <= m; ++i)
      g[i] = 0;

    int j = 0; // number of Arnoldi vectors used in this cycle
    for (; j < m && iter < *iters; ++j) {
      ++iter;
      A(V[j], w, Adata);
      for (int i = 0; i <= j; ++i) { // modified Gram-Schmidt
        H(i, j) = dot(n, w, V[i]);
        xpay(n, w, -H(i, j), V[i]);
      }
      H(j + 1, j) = norm2(n, w);
      if (H(j + 1, j) != 0) {
        double s = 1.0 / H(j + 1, j);
        for (size_t k = 0; k < n; ++k)
          V[j + 1][k] = s * w[k];
      }

      for (int i = 0; i < j; ++i) { // apply previous rotations to column j
        double t = cs[i] * H(i, j) + sn[i] * H(i + 1, j);
        H(i + 1, j) = -sn[i] * H(i, j) + cs[i] * H(i + 1, j);
        H(i, j) = t;
      }
      double rho = hypot(H(j, j), H(j + 1, j));
      if (rho == 0) { // A is singular on the Krylov subspace
        ierr = -1;
        break;
      }
      cs[j] = H(j, j) / rho;
      sn[j] = H(j + 1, j) / rho;
      H(j, j) = rho;
      H(j + 1, j) = 0;
      g[j + 1] = -sn[j] * g[j];
      g[j] = cs[j] * g[j];

      if (!quiet && wall_time() > last_output_wall_time + MEEP_MIN_OUTPUT_TIME) {
        master_printf("residual[%d] = %g\n", iter, fabs(g[j + 1]) / bnrm);
        last_output_wall_time = wall_time();
      }
      if (fabs(g[j + 1]) <= tol * bnrm) {
        ++j;
        break;
      }
    }

    // solve the j x j upper-triangular system H y = g and update x += V y
    for (int i = j - 1; i >= 0; --i) {
      double t = g[i];
      for (int k = i + 1; k < j; ++k)
        t -= H(i, k) * y[k];
      y[i] = t / H(i, i);
    }
    for (int i = 0; i < j; ++i)
      xpay(n, x, y[i], V[i]);

    if (ierr) break;
  }
#undef H

  if (!quiet && ierr >= 0) master_printf("final residual = %g\n", resid / bnrm);

  delete[] y;
  delete[] g;
  delete[] sn;
  delete[] cs;
  delete[] H;
  delete[] V;

  *iters = iter;
  return ierr;
}

} // namespace meep
