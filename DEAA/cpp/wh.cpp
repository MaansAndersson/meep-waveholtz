#include <climits>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>

#include <meep.hpp>

#include "gmres.cpp"

#include <fstream>
#include <iomanip>

// using namespace meep;
using std::complex;

static const double PI = 3.14159265358979323846;

using namespace meep;

// Should probably be templated?

static void complex_fields_to_array(const fields &f, complex<realnum> *x) {
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

static void array_to_complex_fields(const complex<realnum> *x, fields &f) {
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

static void fields_sum_to_array(const fields &f, realnum *x, const double scale_D,
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

static void fields_to_array(const fields &f, realnum *x) {
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

static void array_to_fields(const realnum *x, fields &f) {
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

static void physical_mask(fields &f, realnum *mask, size_t N) {
  f.zero_fields();
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if ((is_D(c) || is_B(c)) && f.chunks[i]->f[c][0])
          std::fill(f.chunks[i]->f[c][0], f.chunks[i]->f[c][0] + f.chunks[i]->gv.ntot(), 1.0);
      }
  std::fill(mask, mask + N, 0.0);
  fields_sum_to_array(f, mask, 1.0, 1.0);
  f.zero_fields();
}

void print_to_file(std::string name, const complex<realnum> *vector, size_t N) {
  std::ofstream outfile;
  if (am_master()) {
    outfile.open(name);
    outfile << std::scientific << std::setprecision(16);
  }

  for (size_t i = 0; i < N; i++) {
    auto a = vector[i];
    if (am_master()) outfile << a << std::endl;
  }

  if (am_master()) outfile.close();
}
// Functions
static double g_eamp = 1.0; // Cannot be passed as an argument?
static complex<double> dz_init(const vec &p) {
  return g_eamp * std::sin(p.x() * PI) * std::sin(p.y() * PI);
}
static complex<double> set_xplusy(const vec &p) { return p.x() + p.y(); }
static complex<double> find_x(const vec &p) { return p.x(); }
static complex<double> find_y(const vec &p) { return p.y(); }
static complex<double> jz_profile(const vec &p) {
  return std::sin(p.x() * PI) * std::sin(p.y() * PI);
}
static double g_bamp = 0.0;
static complex<double> bx_init(const vec &p) {
  return g_bamp * std::sin(p.x() * PI) * std::cos(p.y() * PI);
}
static complex<double> by_init(const vec &p) {
  return -g_bamp * std::cos(p.x() * PI) * std::sin(p.y() * PI);
}
static complex<double> init_zero(const vec &p) { return 0.0; }
static complex<double> init_one(const vec &p) { return 1.0; }

static complex<double> poly(const vec &p) {
  return (p.x() - 1) * (p.x() + 1) * (p.y() - 1) * (p.y() + 1);
}

static double poly_xx(const vec &p) { return (2) * (p.y() - 1) * (p.y() + 1); }

static double poly_yy(const vec &p) { return (2) * (p.x() - 1) * (p.x() + 1); }

// 0: vacuum, 1: eps=2, 2: mu=2, 3: smooth eps(x,y), mu(x,y), 4: eps jump at x=0
static int g_case = 0;

static double eps_fn(const vec &p) {
  switch (g_case) {
    case 1: return 2.0;
    case 3: return 2.0 + 0.5 * std::sin(PI * p.x() / 2) * std::sin(PI * p.y() / 2);
    case 4: return p.x() < 0 ? 1.0 : 3.0;
    default: return 1.0;
  }
}

static double mu_fn(const vec &p) {
  switch (g_case) {
    case 2: return 2.0;
    case 3: return 1.5 + 0.5 * p.x() * p.y();
    default: return 1.0;
  }
}

static double mu_x(const vec &p) { return g_case == 3 ? 0.5 * p.y() : 0.0; }
static double mu_y(const vec &p) { return g_case == 3 ? 0.5 * p.x() : 0.0; }

static double poly_x(const vec &p) { return 2 * p.x() * (p.y() * p.y() - 1); }
static double poly_y(const vec &p) { return 2 * p.y() * (p.x() * p.x() - 1); }

static complex<double> eps_poly(const vec &p) { return eps_fn(p) * poly(p); }

static double g_omega = 0.0;
static complex<double> Jz_poly(const vec &p) {
  const double mu = mu_fn(p);
  const double div_mu_grad =
      (poly_xx(p) + poly_yy(p)) / mu - (mu_x(p) * poly_x(p) + mu_y(p) * poly_y(p)) / (mu * mu);
  return g_omega * eps_fn(p) * poly(p) + div_mu_grad / g_omega;
}

static complex<double> Jz_bump(const vec &p) {
  const double dx = p.x() - 0.1, dy = p.y() + 0.05;
  return PI * g_omega * std::exp(-g_omega * g_omega * (dx * dx + dy * dy));
}
static complex<double> (*g_jz)(const vec &) = Jz_poly;

/* Reference B for the manufactured solution, at the *staggered* time level.

   The exact time-harmonic solution is E_z = poly cos(w t), hence
   B_x = -poly_y sin(w t)/w and B_y = +poly_x sin(w t)/w, whose WaveHoltz
   filters both vanish -- which is why this used to be init_zero.  But the Yee
   scheme stores B a half step behind D: after stepping to t, f[Bx] holds
   B(t - dt/2).  The discrete filter therefore converges to B(-dt/2), which is
   O(dt) and *not* zero.  Comparing it against zero makes the reported error
   first order in dt even though the operator itself is second order, so use
   the half-step-shifted values instead:

     B_x(-dt/2) = +poly_y(p) sin(w dt/2)/w,   poly_y = 2 y (x^2 - 1)
     B_y(-dt/2) = -poly_x(p) sin(w dt/2)/w,   poly_x = 2 x (y^2 - 1)

   (Same convention as g_bamp / bx_init / by_init in main.) */
static double g_bref = 0.0;
static complex<double> poly_bx(const vec &p) { return g_bref * 2 * p.y() * (p.x() * p.x() - 1); }
static complex<double> poly_by(const vec &p) { return -g_bref * 2 * p.x() * (p.y() * p.y() - 1); }

void get_full_solution(fields &f, realnum *sol_vector) {

  f.zero_fields();
  f.initialize_field(Dz, eps_poly);
  g_bref = std::sin(g_omega * f.dt / 2) / g_omega;
  f.initialize_field(Bx, poly_bx);
  f.initialize_field(By, poly_by);
  fields_to_array(f, sol_vector);
  f.zero_fields();
}

static double g_sin_omega = 0.0;
static double g_dt = 0.0;
static complex<double> sin_current(double t, void *) { return std::sin(g_sin_omega * t); }

// Not used atm, should be tested

static void add_cw_source(fields &f, const std::string &mode, double omega_src, double scale) {
  g_omega = omega_src;
  g_dt = f.dt;
  if (mode == "none") return;

  if (mode == "sin") {
    continuous_src_time src(omega_src / (2 * PI), 0.0, 0.0);
    src.is_integrated = false;
    const double x = 0.5 * omega_src * f.dt;
    const double sinc = (x == 0.0) ? 1.0 : std::sin(x) / x;
    const complex<double> amp = complex<double>(0, 1) * std::exp(complex<double>(0, x)) / sinc;
    f.add_volume_source(Ez, src, f.v, g_jz, scale * amp);
  }
  else if (mode == "sin") {
    g_sin_omega = omega_src;
    custom_src_time src(sin_current, NULL, -meep::infinity,
                        meep::infinity); //,
                                         // omega_src); // / (2 * PI));
    src.is_integrated = false;
    f.add_volume_source(Ez, src, f.v, Jz_poly, scale);
  }
  else
    meep::abort("src-mode must be \"none\", \"cw\" or \"cos\" (got \"%s\")", mode.c_str());
}

void pi_single(const double omega, const double Tend, const size_t Nt, const int N_iwh, fields &f,
               realnum *wh_vector, size_t N) {

  // waveholtz
  double filt = 0.0;
  double t = 0.0;
  f.t = t;

  // Transfer initial guess to field
  array_to_fields(wh_vector, f);

  // Set quadrature sum to zeoro
  for (size_t i = 0; i < N; i++) {
    wh_vector[i] = 0.0;
  }

  filt = std::cos(t * omega) - 0.25;
  fields_sum_to_array(f, wh_vector, filt * 0.5, 0.0);

  // Time-loop
  for (size_t it = 1; it <= Nt; it++) {
    f.step();
    t = it * f.dt;
    filt = std::cos(t * omega) - 0.25;
    const double wD = (it == Nt) ? 0.5 : 1.0;
    fields_sum_to_array(f, wh_vector, wD * filt, filt);
  }

  // double scale = f.dt * 2 / Tend;
  // fields_sum_to_array(f, wh_vector, filt*(scale - 0.5), filt*(scale - 1));
  for (size_t i = 0; i < N; i++)
    wh_vector[i] *= f.dt * 2 / Tend;

  array_to_fields(wh_vector, f);
}

void pi_post(const double omega, const double Tend, const size_t Nt, fields &f, realnum *wh_cos,
             realnum *wh_sin, size_t N) {
	
	f.remove_sources();
  add_cw_source(f, "sin", omega, 1.);
  // waveholtz
  double Cfilt = 0.0;
  double Sfilt = 0.0;
  double t = 0.0;
  f.t = t;

  array_to_fields(wh_cos, f);

  for (size_t i = 0; i < N; i++) {
    wh_cos[i] = 0.0;
    wh_sin[i] = 0.0;
  }

  Cfilt = std::cos(t * omega) - 0.25;
  // Sfilt = std::sin(t * omega);
  fields_sum_to_array(f, wh_cos, Cfilt * 0.5, 0.0);
  // fields_sum_to_array(f, wh_sin, Sfilt * 0.5, 0.0);

  // Time-loop
  for (size_t it = 1; it <= Nt; it++) {
    f.step();
    t = it * f.dt;
    const double wD = (it == Nt) ? 0.5 : 1.0;
    Cfilt = std::cos(t * omega) - 0.25;
    Sfilt = std::sin(t * omega);
    fields_sum_to_array(f, wh_cos, Cfilt * wD, Cfilt);
    fields_sum_to_array(f, wh_sin, Sfilt * wD, Sfilt);
  }

  for (size_t i = 0; i < N; i++) {
    wh_cos[i] *= f.dt * 2 / Tend;
    wh_sin[i] *= f.dt * 2 / Tend;
  }

  f.zero_fields();
}

void pi_steps_in_place(const double omega, const double Tend, const size_t Nt, const int N_iwh,
                       fields &f, realnum *wh_vector, size_t N) {

  realnum *wh_old_vector = new realnum[N];
  realnum *sol_vector = new realnum[N];

  for (size_t i = 0; i < N; i++) {
    wh_old_vector[i] = 0.0;
  }
  get_full_solution(f, sol_vector);
  f.zero_fields();

  array_to_fields(wh_vector, f);

  // waveholtz
  for (int iwh = 0; iwh < N_iwh; iwh++) {
    double filt = 0.0;
    double t = 0.0;
    // implicit int it = 0;
    double rD2 = 0.0;
    double eD = 0.0;

    filt = std::cos(t * omega) - 0.25;
    for (size_t i = 0; i < N; i++) {
      wh_vector[i] = 0.0;
    }
    fields_sum_to_array(f, wh_vector, filt * 0.5, 0.0);

    // Time-loop
    for (size_t it = 1; it <= Nt; it++) {
      f.step();
      t = it * f.dt;
      filt = std::cos(t * omega) - 0.25;
      const double wD = (it == Nt) ? 0.5 : 1.0;
      fields_sum_to_array(f, wh_vector, wD * filt, filt);
    }
    rD2 = 0.0;
    for (size_t i = 0; i < N; i++) {
      wh_vector[i] *= f.dt * 2 / Tend;
      rD2 += std::pow(wh_old_vector[i] - wh_vector[i], 2);
      wh_old_vector[i] = wh_vector[i];
      if (eD < abs(sol_vector[i] - wh_vector[i])) eD = abs(sol_vector[i] - wh_vector[i]);
      // eD += std::pow(real(dz_sol_vector[i] - dz_wh_vector[i]), 2) / N_D;
    }

    array_to_fields(wh_vector, f);
    // Tol should be parameter for solver
    if (std::sqrt(rD2) < 1e-12) break;

  } // End of wh

  delete[] wh_old_vector;
  delete[] sol_vector;
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 1;

  const double wx = 2.0, wy = 2.0;
  const double res = (argc > 1) ? atof(argv[1]) : 32; // * PI;
  const double omega = (argc > 2) ? atof(argv[2]) : 1;
  const int periods = (argc > 3) ? atof(argv[3]) : 1;
  g_case = (argc > 4) ? atoi(argv[4]) : 0;
  const double dpml = (argc > 5) ? atof(argv[5]) : 0.0;
  if (dpml > 0) g_jz = Jz_bump;

  const double Tend = periods * 2 * PI / omega;
  const size_t Nt = Tend * res * 2;
  const double dt_temp = Tend / (double)Nt;
  const double courant = 1 * dt_temp * res;

  /* Start phase: MEEP's t = 0 is physical time t0.  t0 = 0 starts at a field
     maximum (E maximal, B zero); t0 = T/4 = 0.5*pi/omega starts at a field
     null (E zero, B maximal). */
  const double t0 = 0.0;

  /* vol2d puts the origin at a corner; center_origin() matches what
     Simulation does (python/simulation.py:1691-1694). */
  grid_volume gv = vol2d(wx, wy, res);
  gv.center_origin();

  /* Default boundary_region() == no_pml().  Boundary conditions are *not* set
     here: the fields constructor makes every real boundary Metallic (PEC,
     which it is not -- so deliberately no use_bloch here. */
  structure s(gv, eps_fn, dpml > 0 ? pml(dpml) : no_pml(), identity(), 0, courant);
  if (g_case == 2 || g_case == 3) s.set_mu(mu_fn, false);
  fields f(&s);

  f.use_real_fields();

  add_cw_source(f, "sin", omega, 1.);

  f.require_component(Dz);
  f.require_component(Bx);
  f.require_component(By);

  f.initialize_field(Bx, find_x);
  f.step(); // MEEP allocates PML auxiliary arrays (f_u, f_w, ...) lazily on the first step
            // Needed when running with PML
  f.zero_fields();
  f.t = 0;

  master_printf("dt: %e, dt temp: %e \n", f.dt, dt_temp);
  if (abs(f.dt - dt_temp) > 1e-9) { meep::abort("Dt fails"); }
  if (abs(Nt * dt_temp - Tend) > 1e-9) { meep::abort("Inccorect number of iterations"); }

  master_printf("t = %e\n", f.time());

  master_printf("2D PEC cavity %.3g x %.3g, resolution %.0f, Courant %.3g\n", wx, wy, res, courant);
  master_printf("dt = %.17g, omega = %.17g, T = %.17g\n", f.dt, omega, 2 * PI / omega);
  master_printf("Nt = %zu, t0 = %.17g (cos(omega t0) = %+.6f)\n", Nt, t0, g_eamp);

  size_t N_D = 0; // Size of the D field
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) {
      FOR_COMPONENTS(c) {
        if (f.chunks[i]->f[c][0] && (is_D(c))) {
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          N_D += f.chunks[i]->gv.nowned(c) *
                 (1 + (f.chunks[i]->f_u[c][0] != NULL) + (f.chunks[i]->f_w[c2][0] != NULL) * 2 +
                  (f.chunks[i]->f_cond[c][0] != NULL) + (f.chunks[i]->f_bfast[c][0] != NULL));
        }
      }
    }

  size_t N_B = 0; // Size of the B field
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) {
      FOR_COMPONENTS(c) {
        if (f.chunks[i]->f[c][0] && (is_B(c))) {
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          N_B += f.chunks[i]->gv.nowned(c) *
                 (1 + (f.chunks[i]->f_u[c][0] != NULL) + (f.chunks[i]->f_w[c2][0] != NULL) * 2 +
                  (f.chunks[i]->f_cond[c][0] != NULL) + (f.chunks[i]->f_bfast[c][0] != NULL));
        }
      }
    }

  master_printf("Tot spatial unkowns N = %zu, N_D = %zu, N_B = %zu \n", N_D + N_B, N_D, N_B);
  // THIS FUNCTION SETS FIELDS TO ZERO!
  // coordinates2file(f, N_D, N_B);

  realnum *d_wh_vector = new realnum[N_D];
  realnum *b_wh_vector = new realnum[N_B];

  // f.zero_fields();
  // g_eamp = std::cos(omega * t0);
  // f.initialize_field(Dz, dz_init);

  // g_bamp = (PI / omega) * std::sin(omega * (f.dt / 2 - t0));
  // f.initialize_field(Bx, bx_init);
  // f.initialize_field(By, by_init);

  f.zero_fields();
  //   ABOVE THIS SHOULD BE ALLOCATED OUTSIDE PI

  const size_t N = N_D + N_B;
  const int m = 10; // GMRES restart length
  realnum *Pi0 = new realnum[N]();
  realnum *xfgmres = new realnum[N](); // initial guess 0, overwritten with the solution
  realnum *xgmres = new realnum[N]();  // initial guess 0, overwritten with the solution

  // When testing pml
  realnum *xref = NULL, *phys = NULL;
  if (dpml > 0) {
    xref = new realnum[N]();
    realnum *xprev = new realnum[N]();
    phys = new realnum[N];
    physical_mask(f, phys, N);
    int k = 0;
    double change = 1;
    for (; k < 5 && change > 1e-11; k++) {
      for (size_t it = 0; it < Nt; it++)
        f.step();
      fields_to_array(f, xref);
      double dmax = 0, xmax = 0;
      for (size_t i = 0; i < N; i++) {
        if (phys[i] != 0) {
          dmax = std::max(dmax, (double)std::abs(xref[i] - xprev[i]));
          xmax = std::max(xmax, (double)std::abs(xref[i]));
        }
        xprev[i] = xref[i];
      }
      change = max_to_all(dmax) / max_to_all(xmax);
    }
    master_printf("time-domain reference: %d periods, last relative change %.3e\n", k, change);
    delete[] xprev;
  }

  // Create the r.h.s. of the solver.
  f.zero_fields();
  pi_steps_in_place(omega, Tend, Nt, 1, f, Pi0, N); // Pi0 = Pi(0), sources on
  f.remove_sources();
  f.zero_fields();

  // A = I-S. with no source.
  auto A = [&](const realnum *in, realnum *out) {
    for (size_t i = 0; i < N; i++)
      out[i] = in[i];
    pi_single(omega, Tend, Nt, 1, f, out, N); // out = S in, in place
    for (size_t i = 0; i < N; i++)
      out[i] = in[i] - out[i];
  };

  // No-op preconditioner for fmgres
  // Nested GMRES solver (S-I)d = r;
//  auto M = [&](const double *in, double *out) {
//    for (size_t i = 0; i < N; i++) {
//      out[i] = in[i];
//    }
//    meep::gmres_result sol3 = meep::gmres(N, A, in, out, 5, 1e-4, 5, false);
//  };
//
//  const double wt0 = meep::wall_time();
//  meep::gmres_result sol = meep::fgmres(N, A, M, Pi0, xfgmres, m, 1e-13, 100);
//  const double wt1 = meep::wall_time();
//  master_printf("GMRES %s after %d iterations, |r|/|b| = %.3e, %.3f s\n",
//                sol.converged ? "converged" : "NOT converged", sol.iters, sol.relres, wt1 - wt0);
//
  const double wt2 = meep::wall_time();
  meep::gmres_result sol2 = meep::gmres(N, A, Pi0, xgmres, m, 1e-13, 100);
  const double wt3 = meep::wall_time();

  master_printf("GMRES %s after %d iterations, |r|/|b| = %.3e, %.3f s\n",
                sol2.converged ? "converged" : "NOT converged", sol2.iters, sol2.relres, wt3 - wt2);
  if (xref) {
    double e = 0, e2 = 0, xmax = 0, eaux = 0;
    for (size_t i = 0; i < N; i++) {
      if (phys[i] == 0) {
        eaux = std::max(eaux, (double)std::abs(xgmres[i] - xref[i]));
        continue;
      }
      //e = std::max(e, (double)std::abs(xfgmres[i] - xref[i]));
      e2 = std::max(e2, (double)std::abs(xgmres[i] - xref[i]));
      xmax = std::max(xmax, (double)std::abs(xref[i]));
    }
    xmax = max_to_all(xmax);
    //master_printf("case %d pml %g FGMRES rel. max diff vs time-domain: %.3e\n", g_case, dpml,
    //              max_to_all(e) / xmax);
    master_printf("case %d pml %g GMRES  rel. max diff vs time-domain: %.3e\n", g_case, dpml,
                  max_to_all(e2) / xmax);
    master_printf("case %d pml %g PML auxiliary (static) diff: %.3e\n", g_case, dpml,
                  max_to_all(eaux) / xmax);
    delete[] xref;
    delete[] phys;
  }
  else {
    realnum *exact = new realnum[N];
    realnum *isD = new realnum[N]();
    f.zero_fields();
    f.initialize_field(Dz, init_one);
    fields_sum_to_array(f, isD, 1.0, 0.0);
    get_full_solution(f, exact);
    double eD = 0, eB = 0, eD2 = 0, eB2 = 0;
    for (size_t i = 0; i < N; i++) {
      const double e = std::abs(xfgmres[i] - exact[i]);
			const double e2 = std::abs(xgmres[i] - exact[i]);
      if (isD[i] != 0) {
        eD = std::max(eD, e);
        eD2 = std::max(eD2, e2);
      }
      else {
        eB = std::max(eB, e);
        eB2 = std::max(eB2, e2);
      }
    }
    master_printf("case %d FGMRES max error D: %.3e  B: %.3e\n", g_case, max_to_all(eD),
                  max_to_all(eB));
    master_printf("case %d GMRES  max error D: %.3e  B: %.3e\n", g_case, max_to_all(eD2),
                  max_to_all(eB2));
    delete[] isD;
    delete[] exact;
  }

  // The error checks above zero the fields; write the solution back first.
  array_to_fields(xgmres, f);

	//realnum *wh_cos = new realnum[N];
	realnum *wh_sin = new realnum[N];

	pi_post(omega, Tend, Nt, f, xgmres, wh_sin, N);

  fields f_complex(&s);

  complex<realnum> *Dhat = new complex<realnum>[2 * N_D];
  complex<realnum> *Bhat = new complex<realnum>[2 * N_B];
  // f_complex

  for (int i = 0; i < N_D; i++) {
    Dhat[i] = std::complex(xgmres[i], -wh_sin[i]);
  }

	double theta = f.dt * omega / 2;
  for (int i = 0; i < N_B; i++) {
    Bhat[i] = std::complex(std::cos(theta)*xgmres[i]+std::sin(theta)*wh_sin[i], std::sin(theta)*xgmres[i]-std::cos(theta)*wh_sin[i]);
  }

	add_cw_source(f_complex,"sin", omega, 1.); // 0.5*omega/PI, 1.);
	auto ok = f_complex.solve_cw(1e-6, 1000, 10);
	
	if (!ok)
		meep::abort("BiCGStab did not converge!");
  
  h5file *dz_CW_file = f_complex.open_h5file("dz_CW", h5file::WRITE, 0, false);
  f_complex.output_hdf5(Dz, f_complex.v, dz_CW_file);
  delete dz_CW_file;

  h5file *dz_CW_file_i = f_complex.open_h5file("dz_CW_i", h5file::WRITE, 0, false);
  f_complex.output_hdf5(Dz, f_complex.v, dz_CW_file_i);
  delete dz_CW_file_i;

	array_to_fields(xgmres, f);
  h5file *dz_file = f.open_h5file("dz", h5file::WRITE, 0, false);
  f.output_hdf5(Dz, f.v, dz_file);
  delete dz_file;

  // h5file *bx_file = f.open_h5file("bx", h5file::WRITE, 0, false);
  // f.output_hdf5(Bx, f.v, bx_file);
  // delete bx_file;

  // h5file *by_file = f.open_h5file("by", h5file::WRITE, 0, false);
  // f.output_hdf5(By, f.v, by_file);
  // delete by_file;

  // h5file *eps_file = f.open_h5file("eps", h5file::WRITE, 0, false);
  // f.output_hdf5(Dielectric, f.v, eps_file);
  // delete eps_file;

  // h5file *mu_file = f.open_h5file("mu", h5file::WRITE, 0, false);
  // f.output_hdf5(Permeability, f.v, mu_file);
  // delete mu_file;

  return 0;
}
