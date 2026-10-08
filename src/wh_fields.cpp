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

#include "meep_internals.hpp"
#include "bicgstab.hpp"
#include "gmres.hpp"

using namespace std;

namespace meep {

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

  // f.step redoes this before updating D, but the final write-back of the solution needs it:
  // without it, B on chunk-boundary (non-owned) points and all of H keep stale values
  f.step_boundaries(B_stuff);
  f.update_eh(H_stuff, true);
  f.step_boundaries(H_stuff);
}



// One application of the WaveHoltz filter map Pi: starting from the fields in wh_vector (length
// N), step the fields over [0, Tend] in Nt steps and overwrite wh_vector with the trapezoidal
// approximation of (2/Tend) int_0^Tend (cos(omega t) - 1/4) u(t) dt.  The filtered solution is
// also left in the fields.  Pi is affine: Pi(x) = S x + Pi(0), with Pi(0) due to the sources.
static void pi_single(const double omega, const double Tend, const int Nt, fields &f,
                      realnum *wh_vector, size_t N) {
  double filt = 0.0;
  double t = 0.0;
  f.t = 0;

  // Transfer initial guess to field
  array_to_fields_real(wh_vector, f);

  // Set quadrature sum to zero
  for (size_t i = 0; i < N; i++)
    wh_vector[i] = 0.0;

  filt = cos(t * omega) - 0.25;
  fields_sum_to_array_real(f, wh_vector, filt * 0.5, 0.0);

  // Time-loop
  for (int it = 1; it <= Nt; it++) {
    f.step();
    t = it * f.dt;
    filt = cos(t * omega) - 0.25;
    const double wD = (it == Nt) ? 0.5 : 1.0;
    fields_sum_to_array_real(f, wh_vector, wD * filt, filt);
  }

  for (size_t i = 0; i < N; i++)
    wh_vector[i] *= f.dt * 2 / Tend;

  array_to_fields_real(wh_vector, f);
}

typedef struct {
  size_t n;
  fields *f;
  double omega; // angular frequency of the harmonic response
  double T;     // length of the filter window
  int Nt;       // number of time steps in the filter window
} whop_data;

// y = (I - S) x, where S is the WaveHoltz filter map Pi with the sources removed.
static void whop(const realnum *x, realnum *y, void *data_) {
  whop_data *data = (whop_data *)data_;
  size_t n = data->n;
  memcpy(y, x, n * sizeof(realnum));
  pi_single(data->omega, data->T, data->Nt, *data->f, y, n); // y = S x, in place
  for (size_t i = 0; i < n; ++i)
    y[i] = x[i] - y[i];
}

/* Solve for the CW (constant frequency) field response at the given real frequency with the
   WaveHoltz method: the fixed point x = Pi(x) of the time-domain filter map Pi (see pi_single)
   is the time-harmonic solution, so we solve the linear system (I - S) x = Pi(0) with an iterative
   Krylov solver.  Pi is applied over a window of `periods` source periods, and dt is reduced
   slightly (and restored afterwards) so that the window is an integer number of time steps.

   solver selects BiCGSTAB(L) or GMRES(L): L is the subspace dimension for BiCGSTAB(L) and the
   restart length for GMRES.  The solver halts at a fractional convergence of tol, or when
   maxiters is reached; returns true if convergence succeeds and false if it fails.  Each
   iteration costs `periods` source periods of time stepping (2L of them per BiCGSTAB(L)
   iteration).  Requires real fields. */
bool fields::solve_wh(double tol, int maxiters, complex<double> frequency, int L, int periods,
                      linear_solver solver) {
  if (!is_real) meep::abort("solve_wh is incompatible with use_complex_fields()");
  if (L < 1) meep::abort("solve_wh called with L = %d < 1", L);
  if (periods < 1) meep::abort("solve_wh called with periods = %d < 1", periods);

  const double freq = real(frequency);
  if (freq <= 0.0 || imag(frequency) != 0.0)
    meep::abort("solve_wh requires a real positive frequency (got %g%+gi)", freq,
                imag(frequency));
  const double omega = 2 * pi * freq; // angular frequency of the harmonic response
  const double T = periods / freq;    // filter window

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
      if (sc->is_mine()) FOR_DIRECTIONS(d) {
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

  const int Nt = (int)ceil(T / dt - 1e-9); // round up => new dt <= old dt, so still stable
  const double dt_user = dt;

  use_real_fields();
  step(); // MEEP allocates PML auxiliary arrays (f_u, f_w, ...) lazily on the first step
  set_dt(T / Nt);
  if (fabs(Nt * dt - T) > 1e-12 * T)
    meep::abort("solve_wh: Nt*dt = %.15g != T = %.15g", Nt * dt, T);
  if (verbosity > 0 && dt != dt_user)
    master_printf("solve_wh: dt %.12g -> %.12g (%d steps per window)\n", dt_user, dt, Nt);
  zero_fields();
  t = 0;

  size_t N = 0; // size of linear system (on this processor, at least)
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) {
      FOR_COMPONENTS(c) {
        if (chunks[i]->f[c][0] && (is_D(c) || is_B(c))) {
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          N += chunks[i]->gv.nowned(c) *
               (1 + (chunks[i]->f_u[c][0] != NULL) + (chunks[i]->f_w[c2][0] != NULL) * 2 +
                (chunks[i]->f_cond[c][0] != NULL) + (chunks[i]->f_bfast[c][0] != NULL));
        }
      }
    }

  // BiCGSTAB(L) and GMRES(L) share the same calling sequence
  typedef ptrdiff_t (*solver_func)(const int, const size_t, realnum *, bicgstab_op, void *,
                                   const realnum *, const double, int *, realnum *, const bool);
  solver_func solve = solver == GMRES ? gmres : bicgstabL;
  const char *solver_name = solver == GMRES ? "GMRES" : "BiCGSTAB";

  int iters = maxiters;
  size_t nwork = (size_t)solve(L, N, 0, 0, 0, 0, tol, &iters, 0, true);
  realnum *work = new realnum[nwork + 2 * N];
  realnum *x = work + nwork;     // initial guess 0, overwritten with the solution
  realnum *b = work + nwork + N; // right-hand side Pi(0)
  memset(x, 0, N * sizeof(realnum));
  memset(b, 0, N * sizeof(realnum));

  // Create the r.h.s. of the solver: b = Pi(0), with the sources on
  pi_single(omega, T, Nt, *this, b, N);
  // the operator S = Pi - Pi(0) is source free
  remove_sources();
  zero_fields();

  whop_data data;
  data.n = N;
  data.f = this;
  data.omega = omega;
  data.T = T;
  data.Nt = Nt;

  int ierr = (int)solve(L, N, x, whop, &data, b, tol, &iters, work, verbosity == 0);

  array_to_fields_real(x, *this);
  if (verbosity > 0) {
    master_printf("Finished solve_wh after %d %s iters.\n", iters, solver_name);
    if (ierr) master_printf(" -- CONVERGENCE FAILURE (%d) in solve_wh!\n", ierr);
  }
  set_dt(dt_user); // restore, so later time stepping uses the configured dt

  delete[] work;

  return !ierr;
}

/* as solve_wh, but infers frequency from sources */
bool fields::solve_wh(double tol, int maxiters, int L, int periods, linear_solver solver) {
  complex<double> freq = 0.0;
  for (src_time *s = sources; s; s = s->next) {
    complex<double> sf = s->frequency();
    if (sf != freq && freq != 0.0 && sf != 0.0)
      meep::abort("must pass frequency to solve_wh if sources do not agree");
    if (sf != 0.0) freq = sf;
  }
  if (freq == 0.0) meep::abort("must pass frequency to solve_wh if sources do not specify one");
  return solve_wh(tol, maxiters, freq, L, periods, solver);
}

} // namespace meep
