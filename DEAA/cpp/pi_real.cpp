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
      fr[idx] = x[ix++];                                                                             \
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
      fr[idx] = x[ix++];                                                                             \
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
// static complex<double> find_x(const vec &p) { return p.x()+0.5; }
// static complex<double> find_y(const vec &p) { return p.y()+0.5; }
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

// static complex<double> p_exact(const vec &p) {
//   return std::pow(std::pow(p.x(), 2) - 1, 2) + std::pow(std::pow(p.y(), 2) - 1, 2);
// }

// static complex<double> px_exact(const vec &p) {
//   return 4 * p.x() * std::pow(p.x() - 1, 2) * std::pow(std::pow(p.y(), 2) - 1, 2);
// }

// static complex<double> py_exact(const vec &p) {
//   return 4 * p.y() * std::pow(p.y() - 1, 2) * std::pow(std::pow(p.x(), 2) - 1, 2);
// }

// static complex<double> lap_exact(const vec &p) {
//   return 4 * ((3 * std::pow(p.x(), 2) * std::pow(std::pow(p.y(), 2) - 1, 2)) +
//               3 * std::pow(p.y(), 2) * std::pow(std::pow(p.x(), 2) - 1, 2));
// }

// px_exact(x, y) = 32 * (2x ^ 3 - 3x ^ 2 + x) * y ^ 2 * (y - 1) ^ 2 py_exact(x, y) =
//                      32 * (2y ^ 3 - 3y ^ 2 + y) * x ^ 2 * (x - 1) ^
//                      2 lap_exact(x, y) = 32 * ((6x ^ 2 - 6x + 1) * y ^ 2 * (y - 1) ^
//                                                2 + (6y ^ 2 - 6y + 1) * x ^ 2 * (x - 1) ^ 2)

// static complex<double> p_exact(const vec &p) {
//   return 16 * std::pow(std::pow(p.x(), 2), 2) * std::pow(std::pow(p.x() - 1, 2), 2) * \
//          std::pow(p.y(), 2) * std::pow(p.y() - 1, 2);
// }
//
// static complex<double> px_exact(const vec &p) {
//   return 32 * (2 * std::pow(p.x(), 3) - 3 * std::pow(p.x(), 2) + p.x()) * std::pow(p.y(), 2) *
//          std::pow(p.y() - 1, 2);
// }
// static complex<double> py_exact(const vec &p) {
//   return 32 * (2 * std::pow(p.y(), 3) - 3 * std::pow(p.y(), 2) + p.y()) * std::pow(p.x(), 2) *
//          std::pow(p.x() - 1, 2);
// }
// static complex<double> lap_exact(const vec &p) {
//   return 32 * (6 * std::pow(p.x(), 2) - 6 * p.x() + 1) * std::pow(p.y(), 2) *
//              std::pow(p.y() - 1, 2) +
//          (6 * std::pow(p.y(), 2) - 6 * p.y() + 1) * std::pow(p.x(), 2) * std::pow(p.x() - 1, 2);
// }

// static complex<double> Ez_exact(const vec &p) { return p_exact(p); }
// static double g_omega = 0.0;
// static complex<double> Bx_exact(const vec &p) { return -1 / g_omega * px_exact(p); }
// static complex<double> By_exact(const vec &p) { return 1 / g_omega * py_exact(p); }
// static complex<double> Jz_exact(const vec &p) {
//   return -g_omega * std::complex<double>(1., 0.) * p_exact(p) -
//          std::complex<double>(1., 0.) / g_omega * lap_exact(p);
// }

static complex<double> poly(const vec &p) {
  return (p.x() - 1) * (p.x() + 1) * (p.y() - 1) * (p.y() + 1);
}

static double poly_xx(const vec &p) { return (2) * (p.y() - 1) * (p.y() + 1); }

static double poly_yy(const vec &p) { return (2) * (p.x() - 1) * (p.x() + 1); }

static double g_omega = 0.0;
static complex<double> Jz_poly(const vec &p) {
  return (g_omega * poly(p) + poly_xx(p) / g_omega + poly_yy(p) / g_omega);
}

// Mutable arguments (zeros fields f)
// void coordinates2file(fields &f, size_t N_D, size_t N_B) {
//
//  realnum *d_vector_raw = new realnum[2 * N_D];
//  complex<realnum> *d_vector = reinterpret_cast<complex<realnum> *>(d_vector_raw);
//
//  realnum *b_vector_raw = new realnum[2 * N_B];
//  complex<realnum> *b_vector = reinterpret_cast<complex<realnum> *>(b_vector_raw);
//
//  f.zero_fields();
//  f.initialize_field(Dz, find_x);
//  D_to_array(f, d_vector);
//  print_to_file("D_xcoord.txt", d_vector, N_D);
//
//  f.zero_fields();
//  f.initialize_field(Dz, find_y);
//  D_to_array(f, d_vector);
//  print_to_file("D_ycoord.txt", d_vector, N_D);
//
//  f.zero_fields();
//  f.initialize_field(Bx, find_x);
//  f.initialize_field(By, find_x);
//  B_to_array(f, b_vector);
//  print_to_file("B_xcoord.txt", b_vector, N_B);
//
//  f.zero_fields();
//  f.initialize_field(Bx, find_y);
//  f.initialize_field(By, find_y);
//  B_to_array(f, b_vector);
//  print_to_file("B_ycoord.txt", b_vector, N_B);
//
//  delete[] d_vector;
//  delete[] b_vector;
//}

void getsolution(fields &f, realnum *dz_sol_vector) {

  f.zero_fields();
  f.initialize_field(Dz, poly);
  D_to_array(f, dz_sol_vector);
  f.zero_fields();
}

static double g_sin_omega = 0.0;
static double g_dt = 0.0;
static complex<double> sin_current(double t, void *) {
  return std::sin(g_sin_omega * (t - 0. * g_dt));
}

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

static void add_sin_point_source(fields &f, double omega_src, const vec &p) {
  g_sin_omega = omega_src;
  custom_src_time src(sin_current, NULL, -meep::infinity, meep::infinity,
                      omega_src); // / (2 * PI));
  // gaussian_src_time src(g_sin_omega, 0.000001);
  src.is_integrated = false; /* NOT the C++ default -- see add_cw_source */
  f.add_point_source(Dz, src, p, -1);
}

void pi_steps_in_place(const double omega, const double Tend, const size_t Nt, fields &f,
                       realnum *dz_wh_vector, size_t N_D, realnum *bxy_wh_vector, size_t N_B) {

  int N_iwh = 400;

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
    master_printf("Iter: %5d, error D: %2e,  D residaul norm: %2e B residaul norm: %2e \n", iwh,
                  (eD), std::sqrt(rD2), std::sqrt(rB2));

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
  // add_sin_point_source(f, omega, vec(0.25, 0.25));
  //  add_sin_point_source(f, omega, vec(-0.25, 0.25));
  //  add_sin_point_source(f, omega, vec(-0.25, -0.25));
  //  add_sin_point_source(f, omega, vec(0.0, 0.0));

  // ABOVE HERE IS DEFINED OUTSIDE CW-SOLVE()

  /* MEEP allocates field components lazily, so straight after the fields
     constructor every chunks[i]->f[c][0] is still NULL and the loops below
     would count zero owned points -- the buffers would then be new[0] and the
     first D_to_array would run off the end of them.  require_component is what
     initialize_field and add_volume_source call internally; doing it here (on
     every rank -- it ends in a collective sync_chunk_connections) makes the
     count independent of which of them ran first. */
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

  /* State goes into D, never E: MEEP time-steps D and recomputes E = D/eps, so
     a write to Ez is discarded on the first step.  initialize_field(Dz, ...)
     propagates to Ez for use via update_eh (src/initialize.cpp:140-143). */
  f.zero_fields();
  g_eamp = std::cos(omega * t0);
  f.initialize_field(Dz, dz_init);

  g_bamp = (PI / omega) * std::sin(omega * (f.dt / 2 - t0));
  f.initialize_field(Bx, bx_init);
  f.initialize_field(By, by_init);

  // f.zero_fields();
  //   ABOVE THIS SHOULD BE ALLOCATED OUTSIDE PI

  // NOTE 1: dz_vector and bxy_vectors are not used as initial guess only output atm.
  // to set initial values change f directly.
  // NOTE 2: f is already updated with the latest solution after a pi_step()
  // NOTE 3: setup struct {omega, Tend, Nt, Tol, N_iwh=1}
  // NOTE 4: Allocate x_wh = (d_wh, b_wh) to simplify the GMRES implementation?
  pi_steps_in_place(omega, Tend, Nt, f, d_wh_vector, N_D, b_wh_vector, N_B);
  //for (int i = 0; i < N_D; i++) {
  //  d_wh_vector[i] = 1.;
  //}

  //array_to_D(d_wh_vector, f);
  //D_sum_to_array(f, d_wh_vector, 1.5);
  //array_to_D(d_wh_vector, f);

  h5file *dz_file = f.open_h5file("dz", h5file::WRITE, 0, false);
  f.output_hdf5(Dz, f.v, dz_file);
  // GOAL:
  // pi_steps(opts, f, x_wh, N_X)

  // f.zero_fields()
  // pi_steps(opts, f, x0, N_X);  x0 = PI0

  // GMRES(@(x) x - PI(x) + PI0, PI0)

  return 0;
}
