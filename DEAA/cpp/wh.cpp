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

#include <fstream>
#include <iomanip>

// using namespace meep;
using std::complex;

static const double PI = 3.14159265358979323846;

using namespace meep;

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
  if (fr = f.chunks[i]->fld[0]) LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx)                       \
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

static void D_sum_to_array(const fields &f, realnum *x, const double scale) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c)) {
          realnum *fr;
#define COPY_FROM_FIELD(fld)                                                                       \
  if ((fr = f.chunks[i]->fld[0])) LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx)                     \
  x[ix++] += scale * fr[idx];
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

static void B_sum_to_array(const fields &f, realnum *x, const double scale) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_B(c)) {
          realnum *fr;
#define COPY_FROM_FIELD(fld)                                                                       \
  if (fr = f.chunks[i]->fld[0]) LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx)                       \
  x[ix++] += scale * fr[idx];
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

static void D_to_array(const fields &f, realnum *x) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c)) {
          realnum *fr;
#define COPY_FROM_FIELD(fld)                                                                       \
  if (fr = f.chunks[i]->fld[0]) LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx)                       \
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

static void B_to_array(const fields &f, realnum *x) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_B(c)) {
          realnum *fr;
#define COPY_FROM_FIELD(fld)                                                                       \
  if (fr = f.chunks[i]->fld[0]) LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx)                       \
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

static void array_to_D(const realnum *x, fields &f) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c)) {
          realnum *fr;
#define COPY_TO_FIELD(fld)                                                                         \
  if (fr = f.chunks[i]->fld[0]) LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx) {                     \
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

  /* done in f.step before updating D: */
  // f.step_boundaries(B_stuff);
  // f.update_eh(H_stuff);
  // f.step_boundaries(H_stuff);
}

static void array_to_B(const realnum *x, fields &f) {
  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_B(c)) {
          realnum *fr;
#define COPY_TO_FIELD(fld)                                                                         \
  if (fr = f.chunks[i]->fld[0]) LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx) {                     \
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

  //* done in f.step before updating D:
  // f.step_boundaries(B_stuff);
  // f.update_eh(H_stuff);
  // f.step_boundaries(H_stuff);
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
static double eps_one(const vec &) { return 1.0; }
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

static double g_omega = 0.0;
static complex<double> Jz_poly(const vec &p) {
  return (g_omega * poly(p) + poly_xx(p) / g_omega + poly_yy(p) / g_omega);
}

void getsolution(fields &f, realnum *dz_sol_vector) {

  f.zero_fields();
  f.initialize_field(Dz, poly);
  D_to_array(f, dz_sol_vector);
  f.zero_fields();
}

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
  f.initialize_field(Dz, poly);
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

  if (mode == "cw") {
    continuous_src_time src(omega_src, 0.0, 0.0);
    src.is_integrated = false;
    f.add_volume_source(Ez, src, f.v, Jz_poly, scale);
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

void pi_steps_in_place_old(const double omega, const double Tend, const size_t Nt, const int N_iwh,
                           fields &f, realnum *dz_wh_vector, size_t N_D, realnum *bxy_wh_vector,
                           size_t N_B) {

  // realnum *dz_wh_old_vector_raw = new realnum[2 * N_D];
  // complex<realnum> *dz_wh_old_vector = reinterpret_cast<complex<realnum>
  // *>(dz_wh_old_vector_raw);

  // realnum *bxy_wh_old_vector_raw = new realnum[2 * N_B];
  // complex<realnum> *bxy_wh_old_vector = reinterpret_cast<complex<realnum>
  // *>(bxy_wh_old_vector_raw);
  realnum *dz_wh_old_vector = new realnum[N_D];
  realnum *bxy_wh_old_vector = new realnum[N_B];

  // realnum *dz_sol_raw = new realnum[2 * N_D];
  // complex<realnum> *dz_sol_vector = reinterpret_cast<complex<realnum> *>(dz_sol_raw);
  realnum *dz_sol_vector = new realnum[N_D];

  getsolution(f, dz_sol_vector);
  f.zero_fields();

  array_to_D(dz_wh_vector, f);
  array_to_B(bxy_wh_vector, f);

  // waveholtz
  for (int iwh = 0; iwh < N_iwh; iwh++) {
    double dfilt = 0.0;
    double t = 0.0;
    // implicit int it = 0;

    double rD2 = 0.0;
    double rB2 = 0.0;
    double eD = 0.0;
    D_to_array(f, dz_wh_vector);
    B_to_array(f, bxy_wh_vector);

    dfilt = std::cos(t * omega) - 0.25;
    for (size_t i = 0; i < N_D; i++) {
      dz_wh_vector[i] = dz_wh_vector[i] * dfilt * 0.5;
    }

    for (size_t i = 0; i < N_B; i++) {
      bxy_wh_vector[i] = 0.0;
    }

    // Time-loop
    for (size_t it = 1; it <= Nt; it++) {
      f.step();
      t = it * f.dt;
      dfilt = std::cos(t * omega) - 0.25;
      B_sum_to_array(f, bxy_wh_vector, dfilt); // Originally did the integration in-place.
      if (it == Nt) { dfilt = dfilt * 0.5; }
      D_sum_to_array(f, dz_wh_vector, dfilt);
    }

    rD2 = 0.0;
    for (size_t i = 0; i < N_D; i++) {
      dz_wh_vector[i] *= f.dt * 2 / Tend; // - 0.5*dz_vector[i]);
      rD2 += std::pow(dz_wh_old_vector[i] - dz_wh_vector[i], 2);
      dz_wh_old_vector[i] = dz_wh_vector[i];
      if (eD < abs(dz_sol_vector[i] - dz_wh_vector[i]))
        eD = abs(dz_sol_vector[i] - dz_wh_vector[i]);
      // eD += std::pow(real(dz_sol_vector[i] - dz_wh_vector[i]), 2) / N_D;
    }
    rB2 = 0.0;
    for (size_t i = 0; i < N_B; i++) {
      bxy_wh_vector[i] *= f.dt * 2 / Tend;
      rB2 += std::pow(bxy_wh_old_vector[i] - bxy_wh_vector[i], 2);
      bxy_wh_old_vector[i] = bxy_wh_vector[i];
    }

    // if (am_master()) outfile << std::sqrt(rD2) << std::endl;
    // if (am_master()) outfile << std::sqrt(rB2) << std::endl;
    master_printf(
        "Iter: %5d, error D: %2e,  D residaul norm: %2e B residaul norm: %2e, res total: %2e \n",
        iwh, (eD), std::sqrt(rD2), std::sqrt(rB2), std::sqrt(rD2 + rB2));

    // Write WH solution back to field
    array_to_D(dz_wh_vector, f);
    array_to_B(bxy_wh_vector, f);
    // Tol should be parameter for solver
    if (std::sqrt(rD2) < 1e-12) break;
    if (std::sqrt(rB2) < 1e-12) break;

  } // End of wh

  delete[] dz_wh_old_vector;
  delete[] bxy_wh_old_vector;
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

    // if (am_master()) outfile << std::sqrt(rD2) << std::endl;
    // if (am_master()) outfile << std::sqrt(rB2) << std::endl;
    master_printf("Iter: %5d, error (D,B): %2e, residaul norm (D,B): %2e \n", iwh, (eD),
                  std::sqrt(rD2));

    // Write WH solution back to field
    array_to_fields(wh_vector, f);
    // Tol should be parameter for solver
    if (std::sqrt(rD2) < 1e-12) break;

  } // End of wh

  delete[] wh_old_vector;
  delete[] sol_vector;
}



int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  const double wx = 2.0, wy = 2.0;
  const double res = (argc > 1) ? atof(argv[1]) : 32; // * PI;

  const double omega = 0.5; // std::sqrt(2.0) * PI;
  const double Tend = 2 * PI / omega;
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
  structure s(gv, eps_one, no_pml(), identity(), 0, courant);
  fields f(&s);

  f.use_real_fields();

  /* The continuous-wave forcing.  Must come before the N_D / N_B count below,
     because add_volume_source ends in require_component (src/sources.cpp:488)
     and so decides whether f[Dz] is allocated at counting time. */
  add_cw_source(f, "sin", omega, 1.);
  // g_sin_omega = omega;
  // custom_src_time src(sin_current, NULL, -meep::infinity,
  //                     meep::infinity); //,
  //                                      // omega_src); // / (2 * PI));
  // src.is_integrated = false;
  // f.add_volume_source(Ez, src, f.v, Jz_poly, 1.);

  f.require_component(Dz);
  f.require_component(Bx);
  f.require_component(By);
  // f.use_real_fields();
  // REAL SOLVER SHOULD BE ABLE TO DO THIS

  f.initialize_field(Bx, find_x);
  f.zero_fields();

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

  // THIS FUNCTION SETS FIELDS TO ZERO!
  // coordinates2file(f, N_D, N_B);

  realnum *d_wh_vector = new realnum[N_D];
  realnum *b_wh_vector = new realnum[N_B];

  //f.zero_fields();
  //g_eamp = std::cos(omega * t0);
  //f.initialize_field(Dz, dz_init);

  //g_bamp = (PI / omega) * std::sin(omega * (f.dt / 2 - t0));
  //f.initialize_field(Bx, bx_init);
  //f.initialize_field(By, by_init);

  f.zero_fields();
  //   ABOVE THIS SHOULD BE ALLOCATED OUTSIDE PI

  // Init GUESS
  // D_to_array(f, d_wh_vector);
  // B_to_array(f, b_wh_vector);

  // NOTE 4: Allocate x_wh = (d_wh, b_wh) to simplify the GMRES implementation?
  master_printf("pi(D,B) \n");
  // pi_steps_in_place_old(omega, Tend, Nt, 100, f, d_wh_vector, N_D, b_wh_vector, N_B);

  master_printf("pi(x) \n");
  f.zero_fields();

  const int N = N_D + N_B;
  realnum *wh_vector = new realnum[N];

  pi_steps_in_place(omega, Tend, Nt, 100, f, wh_vector, N);

  h5file *dz_file = f.open_h5file("dz", h5file::WRITE, 0, false);
  f.output_hdf5(Dz, f.v, dz_file);

  h5file *bx_file = f.open_h5file("bx", h5file::WRITE, 0, false);
  f.output_hdf5(Bx, f.v, bx_file);

  h5file *by_file = f.open_h5file("by", h5file::WRITE, 0, false);
  f.output_hdf5(By, f.v, by_file);
  // GOAL:
  // pi_steps(opts, f, x_wh, N_X)

  // f.zero_fields()
  // pi_steps(opts, f, x0, N_X);  x0 = PI0

  // GMRES(@(x) x - PI(x) + PI0, PI0)

  return 0;
}
