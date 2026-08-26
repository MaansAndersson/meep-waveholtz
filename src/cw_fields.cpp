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
   restarts the simulation before each window.

   The filter replicates MEEP's time-domain DFT (fields::update_dfts, as
   accumulated for the dft_fields monitors used by the reference
   implementation's DFTFilter): during each window every FDTD step n = 1..M
   contributes weight dt with phase e^{+i omega t}, where D components are
   sampled at the integer time t_n = n*dt and B components at t_n - dt/2
   (MEEP's H/B fields live half a step behind E/D).  The frequency-omega
   term is the cos projection and a frequency-zero term is the -1/4
   DC-suppression term of EM-WaveHoltz.  With complex fields the filter
   returns the full complex phasor (identical to solve_cw); with real fields
   it returns its real part (the cos-forced response), exactly as the DEAA
   reference implementation. */

// x[ix] += scale * f[c] for all owned (D/B) points of all owned chunks
static void axpy_fields_to_array_DB(const fields &f, complex<realnum> *x, complex<double> scale) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c) || is_B(c)) {
          realnum *fr = f.chunks[i]->f[c][0];
          realnum *fi = f.chunks[i]->f[c][1];
          if (!fr) continue;
          if (fi)
            LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx) {
              complex<double> fc(fr[idx], fi[idx]);
              x[ix++] += complex<realnum>(scale * fc);
            }
          else
            LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx)
            x[ix++] += complex<realnum>(scale * double(fr[idx]));
        }
      }
}

// x[ix] += dt * e^{i omega t_c} * f[c], t_c = tE for D, tB for B (MEEP's
// update_dfts samples H/B one half step behind E/D)
static void axpy_phased_fields_to_array_DB(const fields &f, complex<realnum> *x, double dt,
                                           double omega, double tE, double tB) {
  size_t ix = 0;
  const complex<double> phaseE = dt * exp(complex<double>(0.0, omega * tE));
  const complex<double> phaseB = dt * exp(complex<double>(0.0, omega * tB));
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c) || is_B(c)) {
          realnum *fr = f.chunks[i]->f[c][0];
          realnum *fi = f.chunks[i]->f[c][1];
          if (!fr) continue;
          const complex<double> phase = is_D(c) ? phaseE : phaseB;
          if (fi)
            LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx) {
              complex<double> fc(fr[idx], fi[idx]);
              x[ix++] += complex<realnum>(phase * fc);
            }
          else
            LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx)
            x[ix++] += complex<realnum>(phase * double(fr[idx]));
        }
      }
}

static void array_DB_to_fields(const complex<realnum> *x, fields &f) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c) || is_B(c)) {
          realnum *fr = f.chunks[i]->f[c][0];
          realnum *fi = f.chunks[i]->f[c][1];
          if (!fr) continue;
          if (fi)
            LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx) {
              fr[idx] = real(x[ix]);
              fi[idx] = imag(x[ix++]);
            }
          else
            LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx) fr[idx] = real(x[ix++]);
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
   MEEP's DFT kernels

       complex fields:  (1/(K*T)) int_0^{K*T} e^{+i omega t} u dt - (1/4) mean
       real fields:     (2/(K*T)) int_0^{K*T} cos(omega t)       u dt - (1/2) mean

   at every FDTD step (rectangle quadrature, D at integer and B at
   half-integer times, exactly as the reference implementation's DFTFilter).
   The accumulated state is fed back as the initial data of the next window;
   the fixed point of this affine iteration is the time-harmonic response --
   with complex fields the full complex phasor, identical to solve_cw; with
   real fields its cos-projection (matching the DEAA EM-WaveHoltz).  The
   iteration state is the physical flux state (D, B) only -- the PML
   auxiliary fields are rebuilt from zero at the start of every window, as
   in the reference implementation.  The -1/4 term suppresses the DC
   component; the complex kernel additionally kills the conjugate e^{+i omega
   t} homogeneous mode.

   Sources must be a pure sinusoid (e.g. ContinuousSource with width=0) so
   that the forcing is exactly time-harmonic.  Iterates until the state stops
   changing by < tol relative to its first change, or until maxiters
   windows; returns true iff converged. */

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
  if (L < 1) meep::abort("solve_waveholtz_cw called with L = %d < 1", L);

  const double freq = real(frequency);
  if (freq <= 0.0 || imag(frequency) != 0.0)
    meep::abort("solve_waveholtz_cw requires a real positive frequency (got %g%+gi)", freq,
                imag(frequency));
  const double omega = 2 * pi * freq; // angular frequency of the harmonic response
  const double T = 1.0 / freq;        // one source period

  const int tsave = t;
  step(); // step once to make sure everything is allocated

  /* number of unknowns: one (complex) amplitude per owned (D/B) point */
  size_t n = 0;
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) {
      FOR_COMPONENTS(c) {
        if (chunks[i]->f[c][0] && (is_D(c) || is_B(c))) {
          n += chunks[i]->gv.nowned(c);
        }
      }
    }

  /* the window must contain an integral number of timesteps, otherwise the
     DFT does not give multiplier exactly one on the forced harmonic.  A
     single period is too coarse a filter: modes at frequencies within a
     fraction of omega of the drive leak into the fixed point, so prefer the
     reference implementation's 10-period window, and fall back to the
     largest integral window of at most one period below (or above) that. */
  int M = 0, K = 0;
  for (int Ktry = 10; Ktry >= 1 && !K; --Ktry) {
    const double M_d = Ktry * T / dt;
    const int M_try = (int)round(M_d);
    if (fabs(M_d - M_try) <= 1e-6 * std::max(1.0, M_d)) { K = Ktry; M = M_try; }
  }
  for (int Ktry = 11; Ktry <= 256 && !K; ++Ktry) {
    const double M_d = Ktry * T / dt;
    const int M_try = (int)round(M_d);
    if (fabs(M_d - M_try) <= 1e-6 * std::max(1.0, M_d)) { K = Ktry; M = M_try; }
  }
  if (K == 0)
    meep::abort("solve_waveholtz_cw: no integer number of periods K <= 256 makes the window "
                "K*T an integral number of timesteps (T/dt = %g); choose the Courant number "
                "so that it is (e.g. with DEAA.emwh.tune_courant)",
                T / dt);

  const double window = K * T; // length of the filter window (seconds)
  complex<realnum> *v = new complex<realnum>[n > 0 ? n : 1];   // iterate / solution
  complex<realnum> *vn = new complex<realnum>[n > 0 ? n : 1];  // window filter output Pi(v)
  complex<realnum> *pi0 = new complex<realnum>[n > 0 ? n : 1]; // affine constant Pi(0)
  complex<realnum> *ct = new complex<realnum>[n > 0 ? n : 1];  // e^{+i omega t} accumulator
  complex<realnum> *ot = new complex<realnum>[n > 0 ? n : 1];  // frequency-zero accumulator

  /* One WaveHoltz window: out = Pi(vin), the DFT filter of the trajectory
     evolved from initial data vin (physical D/B only) with the sources
     running.  The state arrays are per-rank (each lattice point owned by
     exactly one chunk), so the window itself needs no communication. */
  auto apply_window = [&](const complex<realnum> *vin, complex<realnum> *out) {
    zero_fields();
    array_DB_to_fields(vin, *this);
    t = tsave;
    memset(ct, 0, n * sizeof(complex<realnum>));
    memset(ot, 0, n * sizeof(complex<realnum>));
    for (int m = 1; m <= M; ++m) {
      step();
      const double tE = time();        // integer-time sample instant
      const double tB = tE - 0.5 * dt; // H/B fields live half a step behind
      axpy_phased_fields_to_array_DB(*this, ct, dt, omega, tE, tB);
      axpy_fields_to_array_DB(*this, ot, dt); // frequency-zero term
    }
    if (is_real) // real fields: cos projection (DEAA reference filter)
      for (size_t i = 0; i < n; ++i) {
        const double re =
            (2.0 / window) * std::real(ct[i]) - (0.5 / window) * std::real(ot[i]);
        out[i] = complex<realnum>(realnum(re), 0);
      }
    else // complex fields: full phasor, identical to solve_cw
      for (size_t i = 0; i < n; ++i)
        out[i] =
            complex<realnum>((1.0 / window) * ct[i] - (1.0 / (4.0 * window)) * ot[i]);
  };

  zero_fields(); // initial guess v^0 = 0
  t = tsave;     // every window starts at the same phase

  memset(v, 0, n * sizeof(complex<realnum>));   // v = v^0 = 0
  memset(pi0, 0, n * sizeof(complex<realnum>));
  apply_window(v, pi0); // pi0 = Pi(0): the affine constant (one window)

  bool converged = false;
  int iters = 0;
  if (n == 0) {
    converged = true; // nothing to solve on this problem
  }
  else if (L < 2) {
    /* Plain fixed-point iteration v <- Pi(v) (used when the GMRES restart
       dimension L < 2).  Convergence is measured on the iterate increment
       ||v^{k+1}-v^k|| relative to the first one, exactly as the reference
       implementation in DEAA/emwh does. */
    double first_diff = 0.0;
    while (iters < maxiters && !converged) {
      ++iters;
      apply_window(v, vn);
      double diff = 0.0;
      for (size_t i = 0; i < n; ++i) {
        const double dre = std::real(vn[i] - v[i]), dim = std::imag(vn[i] - v[i]);
        diff += dre * dre + dim * dim;
      }
      diff = sqrt(sum_to_all(diff));
      if (first_diff == 0.0) first_diff = diff;
      const double rel = diff / std::max(first_diff, 1e-300);
      if (verbosity > 0 && iters % 10 == 0)
        master_printf("WaveHoltz window %d: ||v^{k+1}-v^k|| = %g (%g relative)\n", iters, diff,
                      rel);
      memcpy(v, vn, n * sizeof(complex<realnum>));
      converged = rel < tol;
    }
  }
  else {
    /* Restarted GMRES(restart) on the linear system

           (I - S) v = Pi0,   with   A v = (I - S) v = v - Pi(v) + Pi0,

       whose solution is the fixed point of the WaveHoltz iteration.  The
       residual is r = Pi0 - A v = Pi(v) - v, i.e. exactly the fixed-point
       increment, and GMRES minimizes it over the Krylov space
       span{Pi0, A Pi0, A^2 Pi0, ...}.  Each matrix-vector product is one
       WaveHoltz window (apply_window), so the cost per GMRES step equals
       one fixed-point iteration, while the convergence is far better for
       the slowly-decaying (near-resonant) modes.  All inner products and
       norms are MPI reductions (sum_to_all), so the Krylov sequence and
       the number of windows are identical on every rank.

       The restart dimension is floored at 10: restarted GMRES with very
       small restarts stalls on the non-normal filtered operator, and the
       result must not depend on L (only the convergence speed may). */
    const int restart = std::max(L, 10);
    std::vector<complex<realnum> *> V(restart + 1);
    for (int k = 0; k <= restart; ++k) V[k] = new complex<realnum>[n > 0 ? n : 1];
    std::vector<std::vector<complex<double>>> H(restart + 1,
                                                std::vector<complex<double>>(restart, 0.0));
    std::vector<double> cs(restart, 0.0);              // real part of the givens
    std::vector<complex<double>> sn(restart, 0.0);     // (complex) givens sine
    std::vector<complex<double>> g(restart + 1, 0.0);  // accumulated rhs of the LS problem
    std::vector<complex<double>> y(restart, 0.0);      // LS solution
    complex<realnum> *w = V[restart];                  // scratch = A V[j]

    // r0 = Pi0 - A*0 = Pi0
    double beta = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double re = std::real(pi0[i]), im = std::imag(pi0[i]);
      beta += re * re + im * im;
    }
    beta = sqrt(sum_to_all(beta));
    const double tol_abs = tol * beta; // relative tolerance on ||Pi(v)-v||

    int nw = 0; // matrix-vector products (windows) used, pi0 not counted
    bool done = (beta == 0.0); // zero forced response: v* = 0
    if (!done) {
      int cycle = 0;
      while (!done && nw < maxiters) {
        ++cycle;
        // restart: V[0] = r / ||r||
        const double binv = 1.0 / beta;
        for (size_t i = 0; i < n; ++i) V[0][i] = complex<realnum>(pi0[i] * binv);
        g[0] = beta;
        for (int k = 1; k <= restart; ++k) g[k] = 0.0;

        int mdone = 0; // Arnoldi steps completed in this cycle
        for (int j = 0; j < restart && nw < maxiters; ++j) {
          // w = A V[j] = V[j] - Pi(V[j]) + Pi0   (one window)
          apply_window(V[j], vn);
          ++nw;
          for (size_t i = 0; i < n; ++i)
            w[i] = complex<realnum>(complex<double>(V[j][i]) - complex<double>(vn[i]) +
                                    complex<double>(pi0[i]));

          // Arnoldi (modified Gram-Schmidt) against V[0..j]
          for (int i = 0; i <= j; ++i) {
            complex<double> h(0.0, 0.0);
            for (size_t k = 0; k < n; ++k)
              h += std::conj(complex<double>(V[i][k])) * complex<double>(w[k]);
            h = sum_to_all(h);
            H[i][j] = h;
            for (size_t k = 0; k < n; ++k)
              w[k] = complex<realnum>(complex<double>(w[k]) - h * complex<double>(V[i][k]));
          }
          double nrm = 0.0;
          for (size_t k = 0; k < n; ++k) {
            const double re = std::real(w[k]), im = std::imag(w[k]);
            nrm += re * re + im * im;
          }
          H[j + 1][j] = sqrt(sum_to_all(nrm));
          mdone = j + 1;
          if (H[j + 1][j] == 0.0) break; // happy breakdown
          for (size_t k = 0; k < n; ++k)
            V[j + 1][k] = complex<realnum>(complex<double>(w[k]) * (1.0 / H[j + 1][j]));

          // apply the previous rotations to the new Hessenberg column
          for (int i = 0; i < j; ++i) {
            const double c = cs[i];
            const complex<double> s = sn[i];
            const complex<double> a = H[i][j], b = H[i + 1][j];
            H[i][j] = c * a + s * b;
            H[i + 1][j] = -std::conj(s) * a + c * b;
          }
          // new complex givens rotation zeroing H[j+1][j]
          const complex<double> a = H[j][j], b = H[j + 1][j];
          const double aa = std::abs(a), bb = std::abs(b);
          if (bb == 0.0) { cs[j] = 1.0; sn[j] = 0.0; }
          else if (aa == 0.0) { cs[j] = 0.0; sn[j] = std::conj(b) / bb; }
          else {
            const double t = sqrt(aa * aa + bb * bb);
            cs[j] = aa / t;
            sn[j] = (a / aa) * std::conj(b) / t;
          }
          H[j][j] = cs[j] * a + sn[j] * b;
          H[j + 1][j] = 0.0;
          g[j + 1] = -std::conj(sn[j]) * g[j];
          g[j] = cs[j] * g[j];
          const double resid = std::abs(g[j + 1]);
          if (verbosity > 1)
            master_printf("WaveHoltz GMRES cycle %d, step %d: ||r|| = %g (%g rel.)\n", cycle,
                          j + 1, resid, resid / beta);
          if (resid <= tol_abs || nw >= maxiters) break;
        }

        // least-squares solve: H(0:mdone,0:mdone) y = g(0:mdone)
        for (int i = mdone - 1; i >= 0; --i) {
          complex<double> s = g[i];
          for (int k = i + 1; k < mdone; ++k)
            s -= H[i][k] * y[k];
          y[i] = s / H[i][i];
        }
        // x += V[0:mdone] y
        for (int j = 0; j < mdone; ++j)
          for (size_t k = 0; k < n; ++k)
            v[k] = complex<realnum>(complex<double>(v[k]) + y[j] * complex<double>(V[j][k]));

        // true residual r = Pi(v) - v (one window), then restart
        apply_window(v, vn);
        ++nw;
        for (size_t i = 0; i < n; ++i)
          pi0[i] = complex<realnum>(complex<double>(vn[i]) - complex<double>(v[i]));
        beta = 0.0;
        for (size_t i = 0; i < n; ++i) {
          const double re = std::real(pi0[i]), im = std::imag(pi0[i]);
          beta += re * re + im * im;
        }
        beta = sqrt(sum_to_all(beta));
        if (verbosity > 0)
          master_printf("WaveHoltz GMRES cycle %d: ||Pi(v)-v|| = %g (%g rel.), %d windows\n",
                        cycle, beta, beta / std::max(tol_abs, 1e-300), nw);
        done = (beta <= tol_abs);
      }
      converged = (beta <= tol_abs);
      iters = nw + 1; // + the pi0 window
    }
    else
      converged = true;
    for (int k = 0; k <= restart; ++k) delete[] V[k];
  }

  /* install the solution as the fields' physical (D, B) state for readback */
  zero_fields();
  array_DB_to_fields(v, *this);

  if (verbosity > 0) {
    master_printf("Finished solve_waveholtz_cw after %d windows (~ %d timesteps).\n", iters,
                  iters * M);
    if (!converged) master_printf(" -- CONVERGENCE FAILURE in solve_waveholtz_cw!\n");
  }

  /* refresh E = D/eps (and the PML E/W fields) for readback, as the state
     itself holds only D and B */
  update_eh(E_stuff);
  step_boundaries(E_stuff);

  delete[] v;
  delete[] vn;
  delete[] pi0;
  delete[] ct;
  delete[] ot;
  t = tsave;

  return converged;
}



} // namespace meep
