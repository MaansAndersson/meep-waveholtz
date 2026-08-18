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


/* WaveHoltz: time-harmonic solve by filtered fixed-point iteration; see
   solve_waveholtz_cw below.  The iteration state is the physical flux state
   (D, B) only; the PML auxiliary fields are rebuilt from zero at the start
   of every window, exactly as the reference implementation in DEAA/emwh
   restarts the simulation before each window. */

static void axpy_fields_to_array_DB(const fields &f, complex<realnum> *x, complex<double> scale) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c) || is_B(c)) {
          realnum *fr, *fi;
#define COPY_FROM_FIELD(fld)                                                                       \
   if ((fr = f.chunks[i]->fld[0]) && (fi = f.chunks[i]->fld[1]))                                   \
    LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx) {                                                 \
      x[ix] += complex<realnum>(scale * complex<double>(fr[idx], fi[idx]));                        \
      ix++;                                                                                        \
    }
          COPY_FROM_FIELD(f[c]);
#undef COPY_FROM_FIELD
        }
      }
}

static void array_DB_to_fields(const complex<realnum> *x, fields &f) {
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
#undef COPY_TO_FIELD
        }
      }
  f.step_boundaries(D_stuff);
  f.step_boundaries(B_stuff);
}

/* Solve for the CW (constant frequency) field response at the given
   frequency by the WaveHoltz fixed-point iteration (no Krylov subspace).

   Each window evolves the FDTD fields -- with the sources running -- for K
   periods of T = 1/Re(frequency), where K is the number of periods (default
   10, as in the reference; a single period is too coarse a filter) for which
   K*T is an integral number of timesteps, accumulating the trajectory with
   the complex-exponential WaveHoltz kernel

       (1/(K*T)) e^{+i*2*pi*Re(frequency)*t}

   by the composite trapezoidal rule; the accumulated state is fed back as
   the initial data of the next window.  The fixed point of this affine
   iteration is the complex amplitude of the time-harmonic response.  (The
   complex kernel is needed with complex fields: the real cosine kernel
   would also project the conjugate e^{+i omega t} homogeneous mode with
   multiplier one, leaving a spurious component that never decays.)  The
   iteration state is the physical flux state (D, B) only -- the PML
   auxiliary fields are rebuilt from zero at the start of every window, as
   in the reference implementation -- so the method is exactly the
   EM-WaveHoltz iteration of DEAA/emwh.

   Notes: complex fields are required (use_real_fields unsupported, like
   solve_cw), and the sources must be a pure sinusoid (e.g.
   ContinuousSource with width=0) so that the forcing is exactly
   time-harmonic.

   Iterates until the state stops changing by < tol relative to its first
   change, or until maxiters windows; returns true iff converged. */

/* as solve_waveholtz_cw, but infers frequency from sources */
bool fields::solve_waveholtz_cw(double tol, int maxiters, int L, complex<double> *eigfreq,
                                double eigtol, int eigiters) {
  complex<double> freq = 0.0;
  for (src_time *s = sources; s; s = s->next) {
    complex<double> sf = s->frequency();
    if (sf != freq && freq != 0.0 && sf != 0.0)
      meep::abort("must pass frequency to solve_waveholtz_cw if sources do not agree");
    if (sf != 0.0) freq = sf;
  }
  if (freq == 0.0)
    meep::abort("must pass frequency to solve_waveholtz_cw if sources do not specify one");
  return solve_waveholtz_cw(tol, maxiters, freq, L, eigfreq, eigtol, eigiters);
}

bool fields::solve_waveholtz_cw(double tol, int maxiters, complex<double> frequency, int L,
                                complex<double> *eigfreq, double eigtol, int eigiters) {
  (void)eigfreq; // eigenfrequency estimation is not implemented for WaveHoltz
  (void)eigtol;
  (void)eigiters;
  if (is_real) meep::abort("solve_waveholtz_cw is incompatible with use_real_fields()");
  if (L < 1) meep::abort("solve_waveholtz_cw called with L = %d < 1", L);

  const double freq = real(frequency);
  if (freq <= 0.0 || imag(frequency) != 0.0)
    meep::abort("solve_waveholtz_cw requires a real positive frequency (got %g%+gi)", freq,
                imag(frequency));
  const double omega = 2 * pi * freq; // angular frequency of the harmonic response
  const double T = 1.0 / freq;        // one source period

  const int tsave = t;
  step(); // step once to make sure everything is allocated

  /* number of real unknowns: the physical (D, B) state on this processor */
  size_t N = 0;
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) {
      FOR_COMPONENTS(c) {
        if (chunks[i]->f[c][0] && (is_D(c) || is_B(c))) {
          N += 2 * chunks[i]->gv.nowned(c);
        }
      }
    }
  const size_t n = N / 2; // number of complex unknowns

  /* the window must contain an integral number of timesteps, otherwise the
     trapezoidal quadrature does not give multiplier exactly one on the
     forced harmonic.  A single period is too coarse a filter: modes at
     frequencies within a fraction of omega of the drive leak into the fixed
     point, so prefer the reference implementation's 10-period window, and
     fall back to the largest integral window of at most one period below
     (or above) that. */
  int M = 0, K = 0;
  for (int Ktry = 10; Ktry >= 1 && !K; --Ktry) {
    const double M_d = Ktry * T / dt;
    const int M_try = (int)round(M_d);
    if (fabs(M_d - M_try) <= 1e-6 * std::max(1.0, M_d)) { K = Ktry; M = M_try; }
  }
  for (int Ktry = 11; Ktry <= 64 && !K; ++Ktry) {
    const double M_d = Ktry * T / dt;
    const int M_try = (int)round(M_d);
    if (fabs(M_d - M_try) <= 1e-6 * std::max(1.0, M_d)) { K = Ktry; M = M_try; }
  }
  if (K == 0)
    meep::abort("solve_waveholtz_cw: no integer number of periods K <= 64 makes the window "
                "K*T an integral number of timesteps (T/dt = %g); choose the Courant number "
                "so that it is (e.g. with DEAA.emwh.tune_courant)",
                T / dt);

  complex<realnum> *x = new complex<realnum>[n > 0 ? n : 1]; // filtered state (new iterate)
  complex<realnum> *y = new complex<realnum>[n > 0 ? n : 1]; // previous iterate

  zero_fields(); // initial guess v^0 = 0
  t = tsave;     // every window starts at the same phase

  memset(y, 0, n * sizeof(complex<realnum>)); // y = v^0 = 0
  double first_diff = 0.0;
  bool converged = false;
  int iters = 0;
  while (iters < maxiters && !converged) {
    ++iters;
    for (size_t i = 0; i < n; ++i)
      x[i] = 0;

    /* composite trapezoidal rule for the complex-exponential filter
       (1/(K*T)) int_0^{K*T} e^{+i omega t} u dt.  With complex fields the real
       cosine kernel would not only project the driven e^{-i omega t} mode but
       also the conjugate e^{+i omega t} homogeneous mode with multiplier one, so
       a spurious component would never decay; the complex kernel projects only
       the driven mode.  Sampled at t = 0 (weight 1/2), t = dt..K*T-dt (weight 1)
       and t = K*T (weight 1/2). */
    t = tsave;
    double cw = cos(omega * time()), sw = sin(omega * time());
    complex<double> w = 0.5 * dt * complex<double>(cw, sw);
    axpy_fields_to_array_DB(*this, x, w);
    for (int m = 1; m <= M; ++m) {
      step();
      cw = cos(omega * time());
      sw = sin(omega * time());
      w = dt * (m == M ? 0.5 : 1.0) * complex<double>(cw, sw);
      axpy_fields_to_array_DB(*this, x, w);
    }

    double diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
      x[i] *= (realnum)(1.0 / (K * T)); // apply the filter prefactor
      const double dre = real(x[i] - y[i]), dim = imag(x[i] - y[i]);
      diff += dre * dre + dim * dim;
    }
    diff = sqrt(sum_to_all(diff));
    if (first_diff == 0.0) first_diff = diff;
    const double rel = diff / std::max(first_diff, 1e-300);
    if (verbosity > 0)
      master_printf("WaveHoltz window %d: ||v^{k+1}-v^k|| = %g (%g relative)\n", iters, diff, rel);

    /* install the filtered state as the next window's initial data: keep only
       the physical (D, B) state and zero everything else (E/H and the PML
       auxiliary fields), exactly as the reference implementation restarts the
       simulation before every window */
    for (size_t i = 0; i < n; ++i)
      y[i] = x[i];
    zero_fields();
    array_DB_to_fields(x, *this);

    converged = rel < tol;
  }

  if (verbosity > 0) {
    master_printf("Finished solve_waveholtz_cw after %d windows (~ %d timesteps).\n", iters,
                  iters * M);
    if (!converged) master_printf(" -- CONVERGENCE FAILURE in solve_waveholtz_cw!\n");
  }

  /* refresh E = D/eps (and the PML E/W fields) for readback, as the state
     itself holds only D and B */
  update_eh(E_stuff);
  step_boundaries(E_stuff);

  delete[] x;
  delete[] y;
  t = tsave;
  update_dfts();

  return converged;
}



} // namespace meep
