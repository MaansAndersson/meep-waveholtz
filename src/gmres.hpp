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

#ifndef GMRES_H
#define GMRES_H

#include "meep.hpp"

namespace meep {

/* signature of a user callback function, passed to gmres, that computes y = Ax.  data is an
   opaque pointer to any user data that you need, passed through from the Adata pointer to gmres.
   This is the same type as bicgstab_op, so the same operator can be passed to either solver. */
typedef void (*gmres_op)(const realnum *x, realnum *y, void *data);

/* solve Ax=b by restarted GMRES(m).  m is the restart length (the dimension of the Krylov
   subspace built before each restart) — larger m gives more robust convergence at the expense of
   more memory; n is the dimension of the vectors and matrices (x and b are length n, A is n by n);
   x is the initial solution guess on input (often zero) and the approximate solution on output; b
   is the right hand side; A is a function that implements A-times-vector (see above) and Adata is
   an opaque user data pointer passed through to A; tol is the stopping tolerance |Ax-b| < tol*|b|
   (always checked on the true residual); on input *iters is the maximum allowed number of
   iterations (applications of A, excluding residual evaluations) and on output it is the actual
   number; quiet=1 to suppress status messages during the solver.

   The calling sequence is identical to bicgstabL (with m in place of L), so the two solvers are
   interchangeable.  You must also allocate a workspace, usually by calling the routine twice: on
   the first call, pass NULL for the work pointer, and the return value nwork is the required size
   of the workspace. Then allocate work = new realnum[nwork], and call it again.

   For non-NULL nwork, returns 0 on success, 1 if the maximum number of iterations was reached, and
   -1 if a breakdown in convergence was detected. */
ptrdiff_t gmres(const int m, const size_t n, realnum *x, gmres_op A, void *Adata,
                const realnum *b, const double tol,
                int *iters,    // input *iters = max iters, output = actual iters
                realnum *work, // if you pass work=NULL, gmres returns nwork
                const bool quiet);

} // namespace meep

#endif /* GMRES_H */
