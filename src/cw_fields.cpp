/* Copyright (C) 2005-2025 Massachusetts Institute of Technology
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

#include "meep_internals.hpp"
#include "bicgstab.hpp"

#include <vector>

using namespace std;

namespace meep {

static void fields_to_array(const fields &f, complex<realnum> *x) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c) || is_B(c)) {
          realnum *fr, *fi;
#define COPY_FROM_FIELD(fld)                                                                       \
  if ((fr = f.chunks[i]->fld[0]) && (fi = f.chunks[i]->fld[1]))                                    \
    LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx)                                                   \
  x[ix++] = complex<double>(fr[idx], fi[idx]);
          COPY_FROM_FIELD(f[c]);
          COPY_FROM_FIELD(f_u[c]);
          COPY_FROM_FIELD(f_cond[c]);
          COPY_FROM_FIELD(f_bfast[c]);
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          COPY_FROM_FIELD(f_w[c2]);
          if (f.chunks[i]->f_w[c2][0]) COPY_FROM_FIELD(f[c2]);
#undef COPY_FROM_FIELD
        }
      }
}

static void array_to_fields(const complex<realnum> *x, fields &f) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c) || is_B(c)) {
          realnum *fr, *fi;
#define COPY_TO_FIELD(fld)                                                                         \
  if ((fr = f.chunks[i]->fld[0]) && (fi = f.chunks[i]->fld[1]))                                    \
    LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx) {                                                 \
      fr[idx] = real(x[ix]);                                                                       \
      fi[idx] = imag(x[ix++]);                                                                     \
    }
          COPY_TO_FIELD(f[c]);
          COPY_TO_FIELD(f_u[c]);
          COPY_TO_FIELD(f_cond[c]);
          COPY_TO_FIELD(f_bfast[c]);
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          COPY_TO_FIELD(f_w[c2]);
          if (f.chunks[i]->f_w[c2][0]) COPY_TO_FIELD(f[c2]);
#undef COPY_TO_FIELD
        }
      }

  f.step_boundaries(D_stuff);
  f.update_eh(E_stuff, true);
  f.step_boundaries(E_stuff);

  /* done in f.step before updating D:
  f.step_boundaries(B_stuff);
  f.update_eh(H_stuff);
  f.step_boundaries(H_stuff); */
}


static void fields_sum_to_array_real(const fields &f, realnum *x, const double scale_D,
                                const double scale_B) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c)) {
          realnum *fr;
#define COPY_FROM_FIELD(fld)                                                                       \
  if ((fr = f.chunks[i]->fld[0])) LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx)                     \
  x[ix++] += scale_D * fr[idx];
          COPY_FROM_FIELD(f[c]);
          COPY_FROM_FIELD(f_u[c]);
          COPY_FROM_FIELD(f_cond[c]);
          COPY_FROM_FIELD(f_bfast[c]);
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          COPY_FROM_FIELD(f_w[c2]);
          if (f.chunks[i]->f_w[c2][0]) COPY_FROM_FIELD(f[c2]);
#undef COPY_FROM_FIELD
        }
        else if (is_B(c)) {
          realnum *fr;
#define COPY_FROM_FIELD(fld)                                                                       \
  if ((fr = f.chunks[i]->fld[0])) LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx)                     \
  x[ix++] += scale_B * fr[idx];
          COPY_FROM_FIELD(f[c]);
          COPY_FROM_FIELD(f_u[c]);
          COPY_FROM_FIELD(f_cond[c]);
          COPY_FROM_FIELD(f_bfast[c]);
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          COPY_FROM_FIELD(f_w[c2]);
          if (f.chunks[i]->f_w[c2][0]) COPY_FROM_FIELD(f[c2]);
#undef COPY_FROM_FIELD
        }
      }
}

static void fields_to_array_real(const fields &f, realnum *x) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c) || is_B(c)) {
          realnum *fr;
#define COPY_FROM_FIELD(fld)                                                                       \
  if ((fr = f.chunks[i]->fld[0])) LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx)                     \
  x[ix++] = fr[idx];
          COPY_FROM_FIELD(f[c]);
          COPY_FROM_FIELD(f_u[c]);
          COPY_FROM_FIELD(f_cond[c]);
          COPY_FROM_FIELD(f_bfast[c]);
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          COPY_FROM_FIELD(f_w[c2]);
          if (f.chunks[i]->f_w[c2][0]) COPY_FROM_FIELD(f[c2]);
#undef COPY_FROM_FIELD
        }
      }
}

static void array_to_fields_real(const realnum *x, fields &f) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c) || is_B(c)) {
          realnum *fr;
#define COPY_TO_FIELD(fld)                                                                         \
  if ((fr = f.chunks[i]->fld[0])) LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx) {                   \
      fr[idx] = x[ix++];                                                                           \
    }
          COPY_TO_FIELD(f[c]);
          COPY_TO_FIELD(f_u[c]);
          COPY_TO_FIELD(f_cond[c]);
          COPY_TO_FIELD(f_bfast[c]);
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          COPY_TO_FIELD(f_w[c2]);
          if (f.chunks[i]->f_w[c2][0]) COPY_TO_FIELD(f[c2]);
#undef COPY_TO_FIELD
        }
      }

  f.step_boundaries(D_stuff);
  f.update_eh(E_stuff, true);
  f.step_boundaries(E_stuff);

  /* done in f.step before updating D:
  f.step_boundaries(B_stuff);
  f.update_eh(H_stuff);
  f.step_boundaries(H_stuff); */
}



typedef struct {
  size_t n;
  fields *f;
  complex<double> iomega;
} fieldop_data;

static void fieldop(const realnum *xr, realnum *yr, void *data_) {
  const complex<realnum> *x = reinterpret_cast<const complex<realnum> *>(xr);
  complex<realnum> *y = reinterpret_cast<complex<realnum> *>(yr);
  fieldop_data *data = (fieldop_data *)data_;
  array_to_fields(x, *data->f);
  data->f->step();
  fields_to_array(*data->f, y);
  size_t n = data->n;
  realnum dt_inv = 1.0 / data->f->dt;
  complex<realnum> iomega = complex<realnum>(real(data->iomega), imag(data->iomega));
  for (size_t i = 0; i < n; ++i)
    y[i] = (y[i] - x[i]) * dt_inv + iomega * x[i];
}

// Rayleigh-quotient estimate <x,Ax>/<x,x> for eigenfrequency given approximate eigenvector x
// (length n), overwriting x with Ax and b with x/|x|.
static complex<double> estimate_eigfreq(complex<realnum> *b, complex<realnum> *x, size_t n,
                                        fieldop_data *data) {
  memcpy(b, x, n * sizeof(complex<realnum>));
  fieldop(reinterpret_cast<realnum *>(b), reinterpret_cast<realnum *>(x), (void *)data);
  complex<double> bdotx(0, 0);
  double bnorm2 = 0;
  for (size_t i = 0; i < n; ++i) {
    complex<realnum> bi = b[i];
    bnorm2 += real(bi) * real(bi) + imag(bi) * imag(bi);
    complex<realnum> bx = conj(bi) * x[i];
    bdotx += complex<double>(real(bx), imag(bx));
  }
  bnorm2 = sum_to_all(bnorm2);
  bdotx = sum_to_all(bdotx);
  double bnorminv = 1 / sqrt(bnorm2);
  for (size_t i = 0; i < n; ++i) {
    b[i] *= bnorminv; // normalize b for subsequent shift-and-invert iterations
  }
  complex<double> iomega = data->iomega - bdotx / bnorm2; // unshifted eigenvalue
  // now, invert: iomega = (1 - exp(-i * (2 * pi * frequency) * dt)) / dt)
  // to get frequency = log(1 - iomega * dt) / (-2 pi i * dt)
  double dt = data->f->dt;
  return log(1.0 - iomega * dt) / complex<double>(0, -2 * pi * dt);
}

/* Solve for the CW (constant frequency) field response at the given
   frequency to the sources (with amplitude given by the current sources
   at the current time).  The solver halts at a fractional convergence
   of tol, or when maxiters is reached, or when convergence fails;
   returns true if convergence succeeds and false if it fails.

   The parameter L determines the order of the iterative algorithm
   that is used.  L should always be positive and should normally be
   >= 2.  Larger values of L will often lead to faster convergence, at
   the expense of more memory and more work per iteration.

   If the optional argument eigfreq is non-NULL, then the solver is used for a
   shift-and-invert power iteration to find the closest eigenfrequency and
   eigenvector to frequency: the solver is iterated up to eigiters times,
   or until the estimated eigenfreq stops changing by <= eigtol (relative). */
bool fields::solve_cw(double tol, int maxiters, complex<double> frequency, int L,
                      complex<double> *eigfreq, double eigtol, int eigiters) {
  if (is_real) meep::abort("solve_cw is incompatible with use_real_fields()");
  if (L < 1) meep::abort("solve_cw called with L = %d < 1", L);
  int tsave = t; // save time (gets incremented by iterations)
  int iters;

  set_solve_cw_omega(2 * pi * frequency);

  step(); // step once to make sure everything is allocated

  size_t N = 0; // size of linear system (on this processor, at least)
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) {
      FOR_COMPONENTS(c) {
        if (chunks[i]->f[c][0] && (is_D(c) || is_B(c))) {
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          /* unknowns are just D and B in non-PML regions, but in PML
             regions the E, U, W, and C fields are also unknowns (in
             principle, we might be able to compute these extra fields
             in frequency domain via scalinb by the appropriate s
             factors, rather than storing them, but I had some
             problems getting that working) */
          N += 2 * chunks[i]->gv.nowned(c) *
               (1 + (chunks[i]->f_u[c][0] != NULL) + (chunks[i]->f_w[c2][0] != NULL) * 2 +
                (chunks[i]->f_cond[c][0] != NULL) + (chunks[i]->f_bfast[c][0] != NULL));
        }
      }
    }

  iters = maxiters;
  size_t nwork = (size_t)bicgstabL(L, N, 0, 0, 0, 0, tol, &iters, 0, true);
  realnum *work = new realnum[nwork + 2 * N];
  complex<realnum> *x = reinterpret_cast<complex<realnum> *>(work + nwork);
  complex<realnum> *b = reinterpret_cast<complex<realnum> *>(work + nwork + N);

  fields_to_array(*this, x); // initial guess = initial fields

  // get J amplitudes from current time step
  zero_fields(); // note that we've saved the fields in x above
  calc_sources(time());
  step_source(B_stuff, true);
  step_boundaries(B_stuff);
  update_eh(H_stuff);
  calc_sources(time() + 0.5 * dt);
  step_source(D_stuff, true);
  step_boundaries(D_stuff);
  update_eh(E_stuff);
  fields_to_array(*this, b);
  double mdt_inv = -1.0 / dt;
  for (size_t i = 0; i < N / 2; ++i)
    b[i] *= mdt_inv;
  {
    double bmax = 0;
    for (size_t i = 0; i < N / 2; ++i) {
      double babs = abs(b[i]);
      if (babs > bmax) bmax = babs;
    }
    am_now_working_on(MpiAllTime);
    if (max_to_all(bmax) == 0.0) meep::abort("zero current amplitudes in solve_cw");
    finished_working();
  }

  fieldop_data data;
  data.f = this;
  data.n = N / 2;
  data.iomega = ((1.0 - exp(complex<double>(0., -1.) * (2 * pi * frequency) * dt)) * (1.0 / dt));
  iters = maxiters;

  int ierr = (int)bicgstabL(L, N, reinterpret_cast<realnum *>(x), fieldop, &data,
                            reinterpret_cast<realnum *>(b), tol, &iters, work, verbosity == 0);

  if (verbosity > 0) {
    master_printf("Finished solve_cw after %d CG iters (~ %d timesteps).\n", iters, iters * 2 * L);
    if (ierr) master_printf(" -- CONVERGENCE FAILURE (%d) in solve_cw!\n", ierr);
  }

  // do additional shift-and-invert iterations to find eigenfrequency
  if (eigfreq) {
    *eigfreq = estimate_eigfreq(b, x, data.n, &data);
    if (verbosity > 0) {
      master_printf("Initial eigen-frequency estimate = %g%+gi\n", real(*eigfreq), imag(*eigfreq));
    }
    for (int eigiter = 0; eigiter < eigiters; ++eigiter) {
      iters = maxiters;
      int ierr = (int)bicgstabL(L, N, reinterpret_cast<realnum *>(x), fieldop, &data,
                                reinterpret_cast<realnum *>(b), tol, &iters, work, verbosity == 0);
      complex<double> newfreq = estimate_eigfreq(b, x, data.n, &data);
      complex<double> dfreq = newfreq - *eigfreq;
      if (verbosity > 0) {
        master_printf("Eigensolver step %d: %d CG iters, freq = %g%+gi (change = %g%+gi).\n",
                      eigiter + 1, iters, real(newfreq), imag(newfreq), real(dfreq), imag(dfreq));
        if (ierr) master_printf(" -- CONVERGENCE FAILURE (%d) in solve_cw!\n", ierr);
      }
      *eigfreq = newfreq;
      if (abs(dfreq) <= eigtol * abs(newfreq)) break; // converged
    }
    memcpy(x, b, N * sizeof(realnum));
  }

  array_to_fields(x, *this);
  step(); // ensure H/B are updated and synced with E/D

  delete[] work;
  t = tsave;

  unset_solve_cw_omega();
  update_dfts();

  return !ierr;
}

/* as solve_cw, but infers frequency from sources */
bool fields::solve_cw(double tol, int maxiters, int L, complex<double> *eigfreq, double eigtol,
                      int eigiters) {
  complex<double> freq = 0.0;
  for (src_time *s = sources; s; s = s->next) {
    complex<double> sf = s->frequency();
    if (sf != freq && freq != 0.0 && sf != 0.0)
      meep::abort("must pass frequency to solve_cw if sources do not agree");
    if (sf != 0.0) freq = sf;
  }
  if (freq == 0.0) meep::abort("must pass frequency to solve_cw if sources do not specify one");
  return solve_cw(tol, maxiters, freq, L, eigfreq, eigtol, eigiters);
}



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

namespace gmres_detail {

// Shared body of gmres (Flexible = false; M unused, no Z storage) and fgmres.
template <bool Flexible, typename T, typename Op, typename Prec>
gmres_result gmres_impl(size_t n, Op &&A, Prec &&M, const T *b, T *x, int m, double tol,
                        int maxit, bool verbose, const char *name) {
  std::vector<std::vector<T> > V(m + 1, std::vector<T>(n));
  std::vector<std::vector<T> > Z(Flexible ? m : 0, std::vector<T>(n)); // z_j = M_j(v_j)
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
    if (verbose) master_printf("%s: iter %4d  |r|/|b| = %.3e  (restart)\n", name, it, beta / bnorm);
    if (beta <= tol * bnorm) return {it, beta / bnorm, true};
    for (size_t k = 0; k < n; ++k)
      V[0][k] /= beta;
    std::fill(g.begin(), g.end(), 0.0);
    g[0] = beta;

    int j = 0;
    for (; j < m && it < maxit; ++j, ++it) {
      if constexpr (Flexible) {
        M(V[j].data(), Z[j].data());
        A(Z[j].data(), w.data());
      }
      else
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

    // Solve the j x j upper-triangular system H y = g and update x += V y
    // (x += Z y for the flexible variant).
    for (int i = j - 1; i >= 0; --i) {
      double t = g[i];
      for (int k = i + 1; k < j; ++k)
        t -= Hij(i, k) * yv[k];
      yv[i] = t / Hij(i, i);
    }
    for (int i = 0; i < j; ++i)
      xpay(n, x, yv[i], (Flexible ? Z[i] : V[i]).data());

    if (fabs(g[j]) <= tol * bnorm) break;
  }

  // True residual, not the (rounding-prone) Givens estimate.
  A(x, w.data());
  for (size_t k = 0; k < n; ++k)
    w[k] = b[k] - w[k];
  const double relres = norm2(n, w.data()) / bnorm;
  return {it, relres, relres <= tol * 1.01};
}


} // namespace gmres_detail

// Solve A x = b starting from the initial guess in x (overwritten with the
// solution). Converges when ||b - A x|| <= tol ||b||; m is the restart length,
// maxit caps the total number of Arnoldi iterations. verbose prints the
// residual at every restart on the master rank.
template <typename T, typename Op>
gmres_result gmres(size_t n, Op &&A, const T *b, T *x, int m, double tol, int maxit,
                   bool verbose = true) {
  auto none = [](const T *, T *) {};
  return gmres_detail::gmres_impl<false>(n, A, none, b, x, m, tol, maxit, verbose, "gmres");
}

// Flexible GMRES: as gmres, but right-preconditioned by M(const T *in, T *out),
// out ~= A^{-1} in, which may be nonlinear and may change on every call (e.g. an
// inexact inner solve, or a stateful lambda). Stores m extra vectors z_j = M(v_j).
// Residuals and the stopping test are on the true ||b - A x||. iters counts outer
// iterations only; work done inside M is not included.
template <typename T, typename Op, typename Prec>
gmres_result fgmres(size_t n, Op &&A, Prec &&M, const T *b, T *x, int m, double tol, int maxit,
                    bool verbose = true) {
  return gmres_detail::gmres_impl<true>(n, A, M, b, x, m, tol, maxit, verbose, "fgmres");
}



void pi_single(const double omega, const double Tend, const size_t Nt, const int N_iwh, fields &f,
               realnum *wh_vector, size_t N) {
  // waveholtz
  double filt = 0.0;
  double t = 0.0;
  f.t = t;

  // Transfer initial guess to field
  array_to_fields_real(wh_vector, f);

  // Set quadrature sum to zeoro
  for (size_t i = 0; i < N; i++) {
    wh_vector[i] = 0.0;
  }

  filt = std::cos(t * omega) - 0.25;
  fields_sum_to_array_real(f, wh_vector, filt * 0.5, 0.0);

  // Time-loop
  for (size_t it = 1; it <= Nt; it++) {
    f.step();
    t = it * f.dt;
    filt = std::cos(t * omega) - 0.25;
    const double wD = (it == Nt) ? 0.5 : 1.0;
    fields_sum_to_array_real(f, wh_vector, wD * filt, filt);
  }

  // double scale = f.dt * 2 / Tend;
  // fields_sum_to_array(f, wh_vector, filt*(scale - 0.5), filt*(scale - 1));
  for (size_t i = 0; i < N; i++)
    wh_vector[i] *= f.dt * 2 / Tend;

  array_to_fields_real(wh_vector, f);
}

bool fields::solve_waveholtz_cw(double tol, int maxiters, int restart, int periods,
                                complex<double> *eigfreq, double eigtol, int eigiters) {
  complex<double> freq = 0.0;
  for (src_time *s = sources; s; s = s->next) {
    complex<double> sf = s->frequency();
    if (sf != freq && freq != 0.0 && sf != 0.0)
      meep::abort("must pass frequency to solve_waveholtz_cw if sources do not agree");
    if (sf != 0.0) freq = sf;
  }
  if (freq == 0.0)
    meep::abort("must pass frequency to solve_waveholtz_cw if sources do not specify one");
  return solve_waveholtz_cw(tol, maxiters, freq, restart, periods, eigfreq, eigtol, eigiters);
}

bool fields::solve_waveholtz_cw(double tol, int maxiters, complex<double> frequency, int restart, int periods,
                                complex<double> *eigfreq, double eigtol, int eigiters) {
  if (!is_real) meep::abort("solve_cw is incompatible with use_complex_fields()");
  (void)eigfreq; // eigenfrequency estimation is not implemented for WaveHoltz
  (void)eigtol;
  (void)eigiters;

  const double freq = real(frequency);
  if (freq <= 0.0 || imag(frequency) != 0.0)
    meep::abort("solve_waveholtz_cw requires a real positive frequency (got %g%+gi)", freq,
                imag(frequency));
  const double omega = 2 * pi * freq; // angular frequency of the harmonic response
  const double T = periods * 1.0 / freq;        // one source period

  // The filter must span exactly T = Nt*dt. dt is stored in the fields, in every fields_chunk
  // (which is what step() uses) and in every structure_chunk, and it is baked into the PML
  // sigma and the conductivity factor, so all of them must change together; assigning only
  // fields::dt makes the filter and the time stepping disagree.
  auto set_dt = [&](double new_dt) {
    if (new_dt == dt) return;
    const double r = new_dt / dt;
    for (int i = 0; i < num_chunks; i++) {
      fields_chunk *fc = chunks[i];
      fc->changing_structure(); // un-share the structure_chunk before modifying it
      fc->dt = new_dt;
      fc->Courant = fc->a * new_dt;
      structure_chunk *sc = fc->s;
      if (sc->is_mine())
        FOR_DIRECTIONS(d) {
          if (!sc->sig[d]) continue;
          for (int k = 0; k < sc->sigsize[d]; k++) {
            sc->sig[d][k] *= r; // sig = 0.5*dt*prefac*s is linear in dt
            sc->siginv[d][k] = 1 / (sc->kap[d][k] + sc->sig[d][k]);
          }
        }
      sc->dt = new_dt;
      sc->Courant = sc->a * new_dt;
      sc->condinv_stale = true;
      sc->update_condinv(); // condinv = 1/(1 + conductivity*dt/2)
    }
    dt = new_dt;
  };

  const int Nt = (int)std::ceil(T / dt - 1e-9); // round up => new dt <= old dt, so still stable
  const double dt_user = dt;
  set_dt(T / Nt);
  if (fabs(Nt * dt - T) > 1e-12 * T)
    meep::abort("solve_waveholtz_cw: Nt*dt = %.15g != T = %.15g", Nt * dt, T);
  if (verbosity > 0 && dt != dt_user)
    master_printf("solve_waveholtz_cw: dt %.12g -> %.12g (%d steps per period)\n", dt_user, dt, Nt);

  use_real_fields();
  step(); // MEEP allocates PML auxiliary arrays (f_u, f_w, ...) lazily on the first step
  zero_fields();
  t = 0;

  size_t N_D = 0; // Size of the D field
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) {
      FOR_COMPONENTS(c) {
        if (chunks[i]->f[c][0] && (is_D(c))) {
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          N_D += chunks[i]->gv.nowned(c) *
                 (1 + (chunks[i]->f_u[c][0] != NULL) + (chunks[i]->f_w[c2][0] != NULL) * 2 +
                  (chunks[i]->f_cond[c][0] != NULL) + (chunks[i]->f_bfast[c][0] != NULL));
        }
      }
    }

  size_t N_B = 0; // Size of the B field
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) {
      FOR_COMPONENTS(c) {
        if (chunks[i]->f[c][0] && (is_B(c))) {
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          N_B += chunks[i]->gv.nowned(c) *
                 (1 + (chunks[i]->f_u[c][0] != NULL) + (chunks[i]->f_w[c2][0] != NULL) * 2 +
                  (chunks[i]->f_cond[c][0] != NULL) + (chunks[i]->f_bfast[c][0] != NULL));
        }
      }
    }

  realnum *d_wh_vector = new realnum[N_D];
  realnum *b_wh_vector = new realnum[N_B];

  const size_t N = N_D + N_B;
  realnum *Pi0 = new realnum[N]();
  realnum *xfgmres = new realnum[N](); // initial guess 0, overwritten with the solution
  realnum *xgmres = new realnum[N]();  // initial guess 0, overwritten with the solution
	


  // Create the r.h.s. of the solver.
  zero_fields();
  pi_single(omega, T, Nt, 1, *this, Pi0, N); // Pi0 = Pi(0), sources on
  // we need to pass the sources
	remove_sources();
  zero_fields();


  // A = I-S. with no source.
  auto A = [&](const realnum *in, realnum *out) {
    for (size_t i = 0; i < N; i++)
      out[i] = in[i];
    pi_single(omega, T, Nt, 1, *this, out, N); // out = S in, in place
    for (size_t i = 0; i < N; i++)
      out[i] = in[i] - out[i];
  };

	gmres_result r = gmres(N, A, Pi0, xgmres, restart, tol, maxiters);
	array_to_fields_real(xgmres, *this);
  if (verbosity > 0)
    master_printf("solve_waveholtz_cw: %d GMRES iters, true |r|/|b| = %.3e%s\n", r.iters, r.relres,
                  r.converged ? "" : "  -- CONVERGENCE FAILURE");
  set_dt(dt_user); // restore, so later time stepping uses the configured dt

  return r.converged;
}



} // namespace meep
