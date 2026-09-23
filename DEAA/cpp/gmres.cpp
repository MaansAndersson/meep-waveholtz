// Restarted GMRES(m) for distributed vectors, using MEEP's MPI wrappers.
//
// Meant to be #included into another translation unit (wh.cpp does
// `#include "gmres.cpp"`). The standalone Poisson test at the bottom is only
// compiled with -DGMRES_MAIN, so including this file adds no main().
//
// Each rank passes only its own slice of the vectors (length n, which may
// differ between ranks); inner products and norms reduce with sum_to_all /
// max_to_all, so every rank computes identical Krylov coefficients. The
// operator is any callable A(const T *x, T *y) computing y = A x on the local
// slice (doing whatever communication it needs internally). Every rank must call
// gmres collectively.
//
// Algorithm: modified Gram-Schmidt Arnoldi + Givens rotations on the Hessenberg
// least-squares problem, restarted every m iterations.
//   Y. Saad, M. H. Schultz, SIAM J. Sci. Stat. Comput. 7(3), 856-869 (1986).
//   R. Barrett et al., Templates for the Solution of Linear Systems, SIAM (1994), Sec. 2.3.4.

#ifndef DEAA_GMRES_CPP
#define DEAA_GMRES_CPP

#include <math.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "meep/mympi.hpp"

namespace meep {

namespace gmres_detail {

template <typename T> double dot(size_t n, const T *x, const T *y) {
  double sum = 0;
  for (size_t i = 0; i < n; ++i)
    sum += (double)x[i] * (double)y[i];
  return sum_to_all(sum);
}

template <typename T> double norm2(size_t n, const T *x) {
  // note: we don't just do sqrt(dot(n, x, x)) in order to avoid overflow
  size_t i;
  double xmax = 0, scale;
  long double sum = 0;
  for (i = 0; i < n; ++i) {
    double xabs = fabs((double)x[i]);
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

template <typename T> void xpay(size_t n, T *x, double a, const T *y) {
  for (size_t m = 0; m < n; ++m)
    x[m] += a * y[m];
}

} // namespace gmres_detail

struct gmres_result {
  int iters;     // total Arnoldi iterations (applications of A, excluding residuals)
  double relres; // true ||b - A x|| / ||b|| at exit
  bool converged;
};

// Solve A x = b starting from the initial guess in x (overwritten with the
// solution). Converges when ||b - A x|| <= tol ||b||; m is the restart length,
// maxit caps the total number of Arnoldi iterations. verbose prints the
// residual at every restart on the master rank.
template <typename T, typename Op>
gmres_result gmres(size_t n, Op &&A, const T *b, T *x, int m, double tol, int maxit,
                   bool verbose = true) {
  using namespace gmres_detail;
  std::vector<std::vector<T> > V(m + 1, std::vector<T>(n));
  std::vector<T> w(n);
  std::vector<double> H((size_t)(m + 1) * m), cs(m), sn(m), g(m + 1), yv(m);
  auto Hij = [&](int i, int j) -> double & { return H[(size_t)j * (m + 1) + i]; };

  const double bnorm = norm2(n, b);
  if (bnorm == 0) {
    memset(x, 0, n * sizeof(T));
    return {0, 0, true};
  }

  int it = 0;
  while (it < maxit) {
    // r = b - A x
    A(x, w.data());
    for (size_t k = 0; k < n; ++k)
      V[0][k] = b[k] - w[k];
    const double beta = norm2(n, V[0].data());
    if (verbose) master_printf("gmres: iter %4d  |r|/|b| = %.3e  (restart)\n", it, beta / bnorm);
    if (beta <= tol * bnorm) return {it, beta / bnorm, true};
    for (size_t k = 0; k < n; ++k)
      V[0][k] /= beta;
    std::fill(g.begin(), g.end(), 0.0);
    g[0] = beta;

    int j = 0;
    for (; j < m && it < maxit; ++j, ++it) {
      A(V[j].data(), w.data());
      for (int i = 0; i <= j; ++i) { // modified Gram-Schmidt
        Hij(i, j) = dot(n, w.data(), V[i].data());
        xpay(n, w.data(), -Hij(i, j), V[i].data());
      }
      Hij(j + 1, j) = norm2(n, w.data());
      if (Hij(j + 1, j) != 0)
        for (size_t k = 0; k < n; ++k)
          V[j + 1][k] = w[k] / Hij(j + 1, j);

      for (int i = 0; i < j; ++i) { // apply previous rotations to column j
        const double t = cs[i] * Hij(i, j) + sn[i] * Hij(i + 1, j);
        Hij(i + 1, j) = -sn[i] * Hij(i, j) + cs[i] * Hij(i + 1, j);
        Hij(i, j) = t;
      }
      const double a = Hij(j, j), c = Hij(j + 1, j), rho = hypot(a, c);
      cs[j] = a / rho;
      sn[j] = c / rho;
      Hij(j, j) = rho;
      Hij(j + 1, j) = 0;
      g[j + 1] = -sn[j] * g[j];
      g[j] = cs[j] * g[j];

      if (fabs(g[j + 1]) <= tol * bnorm) {
        ++j, ++it;
        break;
      }
    }

    // Solve the j x j upper-triangular system H y = g and update x += V y.
    for (int i = j - 1; i >= 0; --i) {
      double t = g[i];
      for (int k = i + 1; k < j; ++k)
        t -= Hij(i, k) * yv[k];
      yv[i] = t / Hij(i, i);
    }
    for (int i = 0; i < j; ++i)
      xpay(n, x, yv[i], V[i].data());

    if (fabs(g[j]) <= tol * bnorm) break;
  }

  // True residual, not the (rounding-prone) Givens estimate.
  A(x, w.data());
  for (size_t k = 0; k < n; ++k)
    w[k] = b[k] - w[k];
  const double relres = norm2(n, w.data()) / bnorm;
  return {it, relres, relres <= tol * 1.01};
}

} // namespace meep

#ifdef GMRES_MAIN

// Restarted GMRES(m) on a matrix-free 5-point finite-difference Poisson problem,
// parallelized with MEEP's MPI wrappers (meep/mympi.hpp) only.
//
//   -Laplace(u) = f  on (0,1)^2,  u = 0 on the boundary,
//   u = e^x sin(pi x) sin(pi y)  (manufactured; deliberately not an eigenvector
//   of the discrete Laplacian, which would make GMRES converge in one step),
//   f = e^x sin(pi y) [ (2 pi^2 - 1) sin(pi x) - 2 pi cos(pi x) ].
//
// The N x N interior grid is split into horizontal slabs of rows, one per rank.
// Applying A needs one ghost row from each neighbouring rank; the exchange uses
// meep::send in two odd/even phases per direction so pairs never serialize.
// The solver is the generic meep::gmres above; this part only supplies the operator.
//
//   make gmres && mpirun -n 4 ./gmres [N] [restart] [tol]   (built with -DGMRES_MAIN)
//
// The discrete error vs the exact u should fall like h^2 (4x per doubling of N+1).

#include <stdlib.h>

using namespace std;

namespace meep {

// Row-slab decomposition of the N x N interior grid. Local unknown (i, j) is
// global point (i + 1, j0 + j + 1) * h, stored at index j * N + i.
struct slab {
  int N;      // interior points per direction
  double h;   // 1 / (N + 1)
  int j0, ny; // first global row owned, number of rows owned
  int rank, nproc;
  size_t n() const { return (size_t)ny * N; }

  explicit slab(int N_) : N(N_), h(1.0 / (N_ + 1)), rank(my_rank()), nproc(count_processors()) {
    if (N < nproc) abort("need N >= number of processes (N=%d, P=%d)\n", N, nproc);
    const int base = N / nproc, rem = N % nproc;
    ny = base + (rank < rem);
    j0 = rank * base + (rank < rem ? rank : rem);
  }
};

// Fill ghost rows g[0] (below) and g[ny+1] (above) of the padded array g,
// which has (ny + 2) * N entries with the owned rows at g[N .. (ny+1)*N).
// Physical boundary ghosts stay zero (homogeneous Dirichlet).
static void exchange_halo(const slab &s, double *g) {
  const int N = s.N, r = s.rank, P = s.nproc;
  double *bot_ghost = g, *bot_row = g + N;
  double *top_row = g + (size_t)s.ny * N, *top_ghost = g + (size_t)(s.ny + 1) * N;
  memset(bot_ghost, 0, N * sizeof(double));
  memset(top_ghost, 0, N * sizeof(double));

  // Each phase pairs (r, r+1) with r % 2 == parity, so each rank is at most one
  // side of one blocking send/recv and the pairs proceed concurrently.
  for (int parity = 0; parity < 2; ++parity) {
    if (r % 2 == parity && r + 1 < P) { // pair (r, r+1): I am the lower rank
      send(r, r + 1, top_row, N);       // my top row -> their bottom ghost
      send(r + 1, r, top_ghost, N);     // their bottom row -> my top ghost
    }
    else if (r >= 1 && (r - 1) % 2 == parity) { // pair (r-1, r): I am the upper rank
      send(r - 1, r, bot_ghost, N);
      send(r, r - 1, bot_row, N);
    }
  }
}

// y = A x with A = -Laplace_h (5-point stencil, SPD). Matrix-free.
static void Ax(const slab &s, const double *x, double *y, vector<double> &pad) {
  const int N = s.N;
  memcpy(pad.data() + N, x, s.n() * sizeof(double));
  exchange_halo(s, pad.data());
  const double ih2 = 1.0 / (s.h * s.h);
  for (int j = 0; j < s.ny; ++j) {
    const double *c = pad.data() + (size_t)(j + 1) * N; // row j, with rows j-1/j+1 at c -/+ N
    double *yr = y + (size_t)j * N;
    for (int i = 0; i < N; ++i) {
      const double w = i > 0 ? c[i - 1] : 0, e = i < N - 1 ? c[i + 1] : 0;
      yr[i] = (4 * c[i] - w - e - c[i - N] - c[i + N]) * ih2;
    }
  }
}

} // namespace meep

int main(int argc, char **argv) {
  meep::initialize mpi(argc, argv);
  const int N = argc > 1 ? atoi(argv[1]) : 63;
  const int m = argc > 2 ? atoi(argv[2]) : 30;
  const double tol = argc > 3 ? atof(argv[3]) : 1e-10;

  meep::slab s(N);
  const size_t n = s.n();
  const double pi = 3.14159265358979323846;
  vector<double> b(n), x(n, 0.0), uex(n);
  for (int j = 0; j < s.ny; ++j)
    for (int i = 0; i < N; ++i) {
      const double xx = (i + 1) * s.h, yy = (s.j0 + j + 1) * s.h;
      uex[(size_t)j * N + i] = exp(xx) * sin(pi * xx) * sin(pi * yy);
      b[(size_t)j * N + i] =
          exp(xx) * sin(pi * yy) * ((2 * pi * pi - 1) * sin(pi * xx) - 2 * pi * cos(pi * xx));
    }

  meep::master_printf("Poisson %dx%d on %d rank(s), GMRES(%d), tol %.1e\n", N, N, s.nproc, m,
                      tol);
  const double t0 = meep::wall_time();
  vector<double> pad((size_t)(s.ny + 2) * N, 0.0);
  auto A = [&](const double *in, double *out) { meep::Ax(s, in, out, pad); };
  meep::gmres_result res = meep::gmres(n, A, b.data(), x.data(), m, tol, 20 * N * N);
  const double t1 = meep::wall_time();

  // Discrete max-norm and grid-L2 errors against the exact solution.
  double emax = 0, e2 = 0;
  for (size_t k = 0; k < n; ++k) {
    const double e = fabs(x[k] - uex[k]);
    emax = e > emax ? e : emax;
    e2 += e * e;
  }
  emax = meep::max_to_all(emax);
  e2 = sqrt(meep::sum_to_all(e2)) * s.h;

  meep::master_printf("%s after %d iterations, true |r|/|b| = %.3e, %.3f s\n",
                      res.converged ? "converged" : "NOT converged", res.iters, res.relres,
                      t1 - t0);
  meep::master_printf("h = %.4e   max error = %.4e   L2 error = %.4e\n", s.h, emax, e2);
  return res.converged ? 0 : 1;
}

#endif // GMRES_MAIN

#endif // DEAA_GMRES_CPP
