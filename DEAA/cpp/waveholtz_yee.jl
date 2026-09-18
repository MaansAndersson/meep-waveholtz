# =====================================================================
#  waveholtz_yee.jl
#
#  WaveHoltz solver for the 2D time-harmonic Maxwell (TM) problem,
#  using the D and B fields on the MEEP Yee lattice and the MEEP
#  space-time staggering.
#
#  ---------------------------------------------------------------
#  Equations (MEEP units: c = 1, eps0 = mu0 = 1)
#  ---------------------------------------------------------------
#      dB/dt = -curl E ,        E = D / eps
#      dD/dt = +curl H - J ,    H = B / mu
#
#  TM polarization (d/dz = 0, fields Dz, Bx, By):
#
#      dBx/dt = -dEz/dy
#      dBy/dt = +dEz/dx
#      dDz/dt =  dHy/dx - dHx/dy - Jz
#
#  Time-harmonic ansatz  F(x,y,t) = f(x,y) exp(i*omega*t)  gives
#
#      i*omega*bx = -dez/dy,   i*omega*by = dez/dx,
#      i*omega*dz =  dhy/dx - dhx/dy - jz,
#
#  i.e. (for eps = mu = 1)  Laplace(ez) + omega^2 ez = i*omega*jz,
#  with the PEC (perfect electric conductor) conditions
#
#      n x E = 0   ->  Ez = 0 on all four walls,
#      n . B = 0   ->  Bx = 0 on x = x0, x1 ;  By = 0 on y = y0, y1 .
#
#  ---------------------------------------------------------------
#  MEEP Yee lattice  (this is the staggering used below)
#  ---------------------------------------------------------------
#  A component of E sits half a step along its OWN direction; a
#  component of H sits half a step along the two OTHER directions.
#  With d/dz = 0 that places, on a cell of size (dx,dy),
#
#      Dz, Ez, Jz, eps  at  ( x_i     , y_j     )   i=0..Nx, j=0..Ny
#      Bx, Hx           at  ( x_i     , y_{j+1/2} ) i=0..Nx, j=0..Ny-1
#      By, Hy           at  ( x_{i+1/2}, y_j     )  i=0..Nx-1, j=0..Ny
#
#      x_i = x0 + i*dx ,  x_{i+1/2} = x0 + (i+1/2)*dx  (same in y).
#
#         y_{j+1}  +--------By-------+
#                  |                 |
#        y_{j+1/2} Bx       .        Bx
#                  |                 |
#         y_j      +--------By-------+
#                 x_i    x_{i+1/2}   x_{i+1}
#
#  Note that Ez lives exactly ON the boundary nodes, so the PEC
#  condition Ez = 0 is imposed directly -- no ghost points are needed.
#
#  Time staggering (also MEEP): D and E live at integer time steps
#  t_n = n*dt, while B and H live at half-integer steps
#  t_{n+1/2} = (n+1/2)*dt.  The state of the simulation at "time n" is
#  therefore the pair
#
#      nu = ( D^n at t_n ,  B^{n-1/2} at t_n - dt/2 ) ,
#
#  and one MEEP step is (in this order, as in MEEP's step_db):
#
#      B^{n+1/2} = B^{n-1/2} - dt * curl E^n
#      D^{n+1}   = D^n       + dt * ( curl H^{n+1/2} - J^{n+1/2} )
#
#  Both updates are centered, so the scheme is second-order in time;
#  the source is sampled at the half-integer time t_{n+1/2}, again as
#  in MEEP.
#
#  ---------------------------------------------------------------
#  WaveHoltz
#  ---------------------------------------------------------------
#  With T = 2*pi/omega the filter is
#
#      Pi(nu) = (2/T) * int_0^T ( cos(omega*(t - t_ref)) - 1/4 ) F(t) dt
#
#  applied to each field F, where t_ref is the time level at which
#  that field is stored (t_ref = 0 for D, t_ref = -dt/2 for B).  The
#  shift by t_ref is what makes the filter return the field at its own
#  storage time: if F(t) = a*exp(i*omega*t) then the integral above is
#  exactly a*exp(i*omega*t_ref) = F(t_ref).
#
#  Pi is affine, Pi(nu) = S*nu + Pi_0 with Pi_0 = Pi(0), and the
#  time-harmonic solution is the unique fixed point,
#
#      nu = Pi(nu)   <=>   (I - S) nu = Pi_0 .
#
#  The quadratures used below (trapezoid on the integer grid for D,
#  the offset uniform rule on the half-integer grid for B) integrate
#  exp(i*k*omega*t), k = 1, 2, exactly over one period whenever
#  Nt >= 3.  Consequently the discrete filter reproduces the discrete
#  time-harmonic solution of the Yee scheme EXACTLY (not just to
#  O(dt^2)), so the fixed point carries no quadrature error at all and
#  the only error left is the O(h^2) Yee discretization error.
#
#  Two solvers are provided: the plain WaveHoltz fixed-point iteration
#  and matrix-free GMRES applied to (I - S) nu = Pi_0.  (CG is *not*
#  offered: S is symmetric in the energy inner product, not in the
#  Euclidean one used by a standard CG, and here the state is complex.)
# =====================================================================

using LinearAlgebra
using Printf

# ---------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------
struct Grid
    Nx::Int          # number of cells in x
    Ny::Int          # number of cells in y
    x0::Float64
    y0::Float64
    dx::Float64
    dy::Float64
end

function Grid(Nx::Int, Ny::Int; x0=-1.0, x1=1.0, y0=-1.0, y1=1.0)
    Grid(Nx, Ny, x0, y0, (x1 - x0) / Nx, (y1 - y0) / Ny)
end

# Coordinates of the Yee locations. The arrays are 1-based, so array
# entry [i,j] of a field carries the logical index (i-1, j-1):
#
#   Dz[i,j] -> ( x_{i-1}     , y_{j-1}     )
#   Bx[i,j] -> ( x_{i-1}     , y_{j-1/2}   )
#   By[i,j] -> ( x_{i-1/2}   , y_{j-1}     )
#
xnode(g::Grid, i::Int) = g.x0 + (i - 1) * g.dx        # x_{i-1}
ynode(g::Grid, j::Int) = g.y0 + (j - 1) * g.dy        # y_{j-1}
xhalf(g::Grid, i::Int) = g.x0 + (i - 0.5) * g.dx      # x_{i-1/2}
yhalf(g::Grid, j::Int) = g.y0 + (j - 0.5) * g.dy      # y_{j-1/2}

size_Dz(g::Grid) = (g.Nx + 1, g.Ny + 1)
size_Bx(g::Grid) = (g.Nx + 1, g.Ny)
size_By(g::Grid) = (g.Nx, g.Ny + 1)

ndof(g::Grid) = prod(size_Dz(g)) + prod(size_Bx(g)) + prod(size_By(g))

# ---------------------------------------------------------------------
# Manufactured solution (used to build the source and to measure error)
#
#   ez(x,y) = p(x,y) = 16 x^2 (x-1)^2 y^2 (y-1)^2          (vanishes to
#                                                    2nd order at walls)
#   by =  -i/omega * p_x ,   bx = +i/omega * p_y
#   jz =  -i*omega*p - (i/omega)*Laplace(p)
#
# (mu = eps = 1 is assumed for the manufactured solution to be valid.)
# ---------------------------------------------------------------------
# (0,1) x (0,1)
#p_exact(x, y)   = 16 * x^2 * (x - 1)^2 * y^2 * (y - 1)^2
#px_exact(x, y)  = 32 * (2x^3 - 3x^2 + x) * y^2 * (y - 1)^2
#py_exact(x, y)  = 32 * (2y^3 - 3y^2 + y) * x^2 * (x - 1)^2
#lap_exact(x, y) = 32 * ((6x^2 - 6x + 1) * y^2 * (y - 1)^2 +
#                        (6y^2 - 6y + 1) * x^2 * (x - 1)^2)
#
# (-1,1) x (-1,1)
p_exact(x, y)   = (x - 1) * (x + 1) * (y - 1) * (y + 1);
px_exact(x, y)  = (2*x - 1) * (y - 1)*(y + 1)
py_exact(x, y)  = (2*y - 1) * (x - 1)*(x + 1)
lap_exact(x, y) = (2) * (x + 1)*(x - 1) + (2) * (y + 1)*(y - 1)
                  

Ez_exact(x, y, ω) = complex(p_exact(x, y))
By_exact(x, y, ω) = -im / ω * px_exact(x, y)
Bx_exact(x, y, ω) = +im / ω * py_exact(x, y)
Jz_exact(x, y, ω) = -im * ω * p_exact(x, y) - (im / ω) * lap_exact(x, y)


#Ez_exact(x, y, ω) = complex(p_exact(x, y))
#By_exact(x, y, ω) = -1. / ω * px_exact(x, y)
#Bx_exact(x, y, ω) = +1. / ω * py_exact(x, y)
#Jz_exact(x, y, ω) = ω * p_exact(x, y) + (1. / ω) * lap_exact(x, y)

# ---------------------------------------------------------------------
# Problem: grid, frequency, materials, source, time stepping data
# ---------------------------------------------------------------------
struct Problem
    g::Grid
    ω::Float64
    T::Float64              # one period, 2*pi/omega
    dt::Float64             # T/Nt, at or below the Courant limit
    Nt::Int                 # steps per period (integer by construction)
    inv_eps::Matrix{Float64}     # 1/eps at the Dz points
    inv_mu_x::Matrix{Float64}    # 1/mu  at the Bx points
    inv_mu_y::Matrix{Float64}    # 1/mu  at the By points
    Jz::Matrix{ComplexF64}       # spatial profile of the current at Dz points
end

"""
    Problem(N, ω; courant=0.5, eps=..., mu=..., source=...)

Build the `N x N` problem on the unit square. `courant` is MEEP's
Courant factor: `dt = courant * min(dx,dy)` before the adjustment that
makes one period an integer number of steps. The 2D stability limit is
`courant <= 1/sqrt(2)`; MEEP's default 0.5 is used here as well.
"""
function Problem(N::Int, ω::Float64; courant::Float64 = 0.5,
                 eps = (x, y) -> 1.0,
                 mu  = (x, y) -> 1.0,
                 source = (x, y) -> Jz_exact(x, y, ω))

    g = Grid(N, N)
    T = 2π / ω

    # MEEP's time step, then shrink it so that Nt steps tile one period.
    dt_cfl = courant * min(g.dx, g.dy)
    @assert courant <= 1 / sqrt(2) + 1e-14 "courant factor exceeds the 2D Yee stability limit"
    Nt = ceil(Int, T / dt_cfl)
    Nt = max(Nt, 3)          # the filter quadrature needs Nt >= 3
    dt = T / Nt

    inv_eps  = Matrix{Float64}(undef, size_Dz(g)...)
    inv_mu_x = Matrix{Float64}(undef, size_Bx(g)...)
    inv_mu_y = Matrix{Float64}(undef, size_By(g)...)
    Jz       = Matrix{ComplexF64}(undef, size_Dz(g)...)

    for j in 1:g.Ny+1, i in 1:g.Nx+1          # Dz points (x_{i-1}, y_{j-1})
        x, y = xnode(g, i), ynode(g, j)
        inv_eps[i, j] = 1 / eps(x, y)
        Jz[i, j] = source(x, y)
    end
    for j in 1:g.Ny, i in 1:g.Nx+1            # Bx points (x_{i-1}, y_{j-1/2})
        inv_mu_x[i, j] = 1 / mu(xnode(g, i), yhalf(g, j))
    end
    for j in 1:g.Ny+1, i in 1:g.Nx            # By points (x_{i-1/2}, y_{j-1})
        inv_mu_y[i, j] = 1 / mu(xhalf(g, i), ynode(g, j))
    end

    Problem(g, ω, T, dt, Nt, inv_eps, inv_mu_x, inv_mu_y, Jz)
end

# ---------------------------------------------------------------------
# State vector <-> field arrays
#
# nu = [ vec(Dz) ; vec(Bx) ; vec(By) ], where Dz is understood at time
# t = 0 and Bx, By at time t = -dt/2 (the MEEP convention).
# ---------------------------------------------------------------------
struct Work
    Dz::Matrix{ComplexF64}      # marching fields
    Bx::Matrix{ComplexF64}
    By::Matrix{ComplexF64}
    aDz::Matrix{ComplexF64}     # filter accumulators
    aBx::Matrix{ComplexF64}
    aBy::Matrix{ComplexF64}
end

function Work(g::Grid)
    Work(zeros(ComplexF64, size_Dz(g)...), zeros(ComplexF64, size_Bx(g)...),
         zeros(ComplexF64, size_By(g)...), zeros(ComplexF64, size_Dz(g)...),
         zeros(ComplexF64, size_Bx(g)...), zeros(ComplexF64, size_By(g)...))
end

function offsets(g::Grid)
    nD = prod(size_Dz(g)); nBx = prod(size_Bx(g)); nBy = prod(size_By(g))
    (nD, nD + nBx, nD + nBx + nBy)
end

"Copy the flat state vector `ν` into the field arrays of `w`."
function unpack!(w::Work, ν::AbstractVector{ComplexF64}, g::Grid)
    o1, o2, o3 = offsets(g)
    copyto!(w.Dz, 1, ν, 1,      o1)
    copyto!(w.Bx, 1, ν, o1 + 1, o2 - o1)
    copyto!(w.By, 1, ν, o2 + 1, o3 - o2)
    return w
end

"Copy the filter accumulators of `w` into the flat state vector `ν`."
function pack!(ν::AbstractVector{ComplexF64}, w::Work, g::Grid)
    o1, o2, o3 = offsets(g)
    copyto!(ν, 1,      w.aDz, 1, o1)
    copyto!(ν, o1 + 1, w.aBx, 1, o2 - o1)
    copyto!(ν, o2 + 1, w.aBy, 1, o3 - o2)
    return ν
end

# ---------------------------------------------------------------------
# PEC boundary conditions:  Ez = 0 on all walls,  n.B = 0 on all walls.
#
# Once imposed, the Yee updates below preserve them automatically: the
# Bx update on the x-walls differences Ez values that are both zero,
# and likewise for By on the y-walls, while Dz is only ever updated at
# interior nodes. Imposing them on the *input* of the WaveHoltz
# operator makes these degrees of freedom identically zero throughout,
# so that (I-S) acts as the identity on them and the right-hand side
# Pi_0 has no component there.
# ---------------------------------------------------------------------
function impose_pec!(w::Work, g::Grid)
    @inbounds begin
        for j in 1:g.Ny+1                     # Ez = 0 on x = x0, x1
            w.Dz[1, j]        = 0
            w.Dz[g.Nx+1, j]   = 0
        end
        for i in 1:g.Nx+1                     # Ez = 0 on y = y0, y1
            w.Dz[i, 1]        = 0
            w.Dz[i, g.Ny+1]   = 0
        end
        for j in 1:g.Ny                       # n.B = Bx = 0 on x = x0, x1
            w.Bx[1, j]        = 0
            w.Bx[g.Nx+1, j]   = 0
        end
        for i in 1:g.Nx                       # n.B = By = 0 on y = y0, y1
            w.By[i, 1]        = 0
            w.By[i, g.Ny+1]   = 0
        end
    end
    return w
end

# =====================================================================
#  Yee updates.  The loops are written out index by index so that the
#  stencil and the staggering are visible; `@inbounds` is the only
#  concession to performance.
# =====================================================================

"""
    step_B!(w, p)

Advance the magnetic flux one full step, `B^{n-1/2} -> B^{n+1/2}`,
using the electric field `E^n = D^n / eps` at the integer time level:

    Bx[i,j] -= dt * ( Ez(x_{i-1}, y_j) - Ez(x_{i-1}, y_{j-1}) ) / dy
    By[i,j] += dt * ( Ez(x_i, y_{j-1}) - Ez(x_{i-1}, y_{j-1}) ) / dx
"""
function step_B!(w::Work, p::Problem)
    g = p.g; dt = p.dt
    Dz, Bx, By, ie = w.Dz, w.Bx, w.By, p.inv_eps
    @inbounds begin
        # dBx/dt = -dEz/dy       (Bx at (x_{i-1}, y_{j-1/2}))
        for j in 1:g.Ny
            for i in 1:g.Nx+1
                Ez_up = ie[i, j+1] * Dz[i, j+1]      # Ez at (x_{i-1}, y_j)
                Ez_dn = ie[i, j]   * Dz[i, j]        # Ez at (x_{i-1}, y_{j-1})
                Bx[i, j] -= dt * (Ez_up - Ez_dn) / g.dy
            end
        end
        # dBy/dt = +dEz/dx       (By at (x_{i-1/2}, y_{j-1}))
        for j in 1:g.Ny+1
            for i in 1:g.Nx
                Ez_rt = ie[i+1, j] * Dz[i+1, j]      # Ez at (x_i, y_{j-1})
                Ez_lf = ie[i, j]   * Dz[i, j]        # Ez at (x_{i-1}, y_{j-1})
                By[i, j] += dt * (Ez_rt - Ez_lf) / g.dx
            end
        end
    end
    return w
end

"""
    step_D!(w, p, tsrc, source_on)

Advance the electric displacement one full step, `D^n -> D^{n+1}`,
using `H^{n+1/2} = B^{n+1/2} / mu` and the current sampled at the
half-integer time `tsrc = (n+1/2)*dt`:

    Dz[i,j] += dt * ( (Hy[i,j] - Hy[i-1,j])/dx
                    - (Hx[i,j] - Hx[i,j-1])/dy
                    - Jz[i,j]*exp(i*omega*tsrc) )

Only interior nodes are touched; `Dz` on the four walls stays zero
(the PEC condition `Ez = 0`).
"""
function step_D!(w::Work, p::Problem, tsrc::Float64, source_on::Bool)
    g = p.g; dt = p.dt
    Dz, Bx, By = w.Dz, w.Bx, w.By
    imx, imy = p.inv_mu_x, p.inv_mu_y
    phase = source_on ? exp(im * p.ω * tsrc) : zero(ComplexF64)
    @inbounds begin
        for j in 2:g.Ny                       # interior nodes only
            for i in 2:g.Nx
                # Hy at (x_{i-1/2}, y_{j-1}) and (x_{i-3/2}, y_{j-1})
                Hy_rt = imy[i, j]   * By[i, j]
                Hy_lf = imy[i-1, j] * By[i-1, j]
                # Hx at (x_{i-1}, y_{j-1/2}) and (x_{i-1}, y_{j-3/2})
                Hx_up = imx[i, j]   * Bx[i, j]
                Hx_dn = imx[i, j-1] * Bx[i, j-1]
                Dz[i, j] += dt * ((Hy_rt - Hy_lf) / g.dx -
                                  (Hx_up - Hx_dn) / g.dy -
                                  phase * p.Jz[i, j])
            end
        end
    end
    return w
end

# ---------------------------------------------------------------------
# Filter accumulation:  a += c * F, written as explicit 2D loops.
# ---------------------------------------------------------------------
function accum_D!(w::Work, g::Grid, c::Float64)
    @inbounds for j in 1:g.Ny+1, i in 1:g.Nx+1
        w.aDz[i, j] += c * w.Dz[i, j]
    end
    return w
end

function accum_B!(w::Work, g::Grid, c::Float64)
    @inbounds for j in 1:g.Ny, i in 1:g.Nx+1
        w.aBx[i, j] += c * w.Bx[i, j]
    end
    @inbounds for j in 1:g.Ny+1, i in 1:g.Nx
        w.aBy[i, j] += c * w.By[i, j]
    end
    return w
end

# =====================================================================
#  The WaveHoltz operator
# =====================================================================
"""
    apply_Pi!(out, ν, p, w; source_on=true)

Evolve `(D^0, B^{-1/2}) = ν` over one period with the MEEP-staggered
Yee scheme and return the filtered fields in `out`.

With `source_on = true` this is the affine operator `Pi(ν) = S*ν + Pi_0`;
with `source_on = false` it is the linear part `S*ν` alone.

Quadrature (`Nt >= 3` is required, and both kernels happen to be
evaluated at the same set of times `(n+1)*dt`):

  * `D` at `t_n = n*dt`, `n = 0..Nt`, trapezoidal weights
    `dt/2, dt, ..., dt, dt/2`, kernel `cos(omega*t_n) - 1/4`;
  * `B` at `t_{n+1/2}`, `n = 0..Nt-1`, uniform weights `dt`, kernel
    `cos(omega*(t_{n+1/2} + dt/2)) - 1/4` -- the shift by `dt/2` is the
    `t_ref = -dt/2` reference of the magnetic field.
"""
function apply_Pi!(out::AbstractVector{ComplexF64}, ν::AbstractVector{ComplexF64},
                   p::Problem, w::Work; source_on::Bool = true)
    g, ω, dt, Nt, T = p.g, p.ω, p.dt, p.Nt, p.T

    unpack!(w, ν, g)
    impose_pec!(w, g)
    fill!(w.aDz, 0); fill!(w.aBx, 0); fill!(w.aBy, 0)

    # t = 0 endpoint of the trapezoidal rule for D
    accum_D!(w, g, (2 / T) * (dt / 2) * (cos(0.0) - 0.25))

    for n in 0:Nt-1
        # --- B^{n-1/2} -> B^{n+1/2}, then accumulate (kernel at (n+1)*dt)
        step_B!(w, p)
        ck = cos(ω * (n + 1) * dt) - 0.25
        accum_B!(w, g, (2 / T) * dt * ck)

        # --- D^n -> D^{n+1} with the current at t_{n+1/2}, then accumulate
        step_D!(w, p, (n + 0.5) * dt, source_on)
        wt = (n + 1 == Nt) ? dt / 2 : dt        # trapezoid endpoint at t = T
        accum_D!(w, g, (2 / T) * wt * ck)
    end

    pack!(out, w, g)
    return out
end

# =====================================================================
#  Solvers
# =====================================================================

"""
    waveholtz_fixedpoint(p; tol, maxiter)

Plain WaveHoltz iteration `ν <- Pi(ν)` started from `ν = 0` (so that
the first iterate is `Pi_0`). Returns `(ν, history)` where `history`
holds the relative increments `||ν_{k+1} - ν_k|| / ||ν_{k+1}||`.
"""
function waveholtz_fixedpoint(p::Problem; tol::Float64 = 1e-10,
                              maxiter::Int = 500, verbose::Bool = false)
    n = ndof(p.g)
    w = Work(p.g)
    ν  = zeros(ComplexF64, n)
    νn = zeros(ComplexF64, n)
    hist = Float64[]
    for k in 1:maxiter
        apply_Pi!(νn, ν, p, w)
        r = norm(νn - ν) #/ max(norm(νn), eps())
        push!(hist, r)
        ν, νn = νn, ν                 # swap: ν holds the new iterate
        verbose && k % 10 == 0 && @printf("  WHI  iter %4d   rel. increment %.3e\n", k, r)
        r <= tol && break
    end
		#println(hist)
    return ν, hist
end

"""
    gmres_mf(Aop!, b; tol, maxiter)

Matrix-free GMRES without restart for complex systems. `Aop!(y, x)`
must write `A*x` into `y`. Returns `(x, history)` with the history of
relative residual norms.
"""
function gmres_mf(Aop!, b::Vector{ComplexF64}; tol::Float64 = 1e-10,
               maxiter::Int = 200, verbose::Bool = false)
    n = length(b)
    β = norm(b)
    hist = Float64[]
    β == 0 && return zeros(ComplexF64, n), hist

    m = maxiter
    V  = Vector{Vector{ComplexF64}}(undef, m + 1)
    V[1] = b ./ β
    H  = zeros(ComplexF64, m + 1, m)
    cs = zeros(ComplexF64, m)          # Givens cosines (complex)
    sn = zeros(Float64, m)             # Givens sines (real, since H[j+1,j] >= 0)
    gv = zeros(ComplexF64, m + 1); gv[1] = β
    z  = Vector{ComplexF64}(undef, n)

    k = 0
    for j in 1:m
        Aop!(z, V[j])
        for i in 1:j                                  # modified Gram-Schmidt
            H[i, j] = dot(V[i], z)
            z .-= H[i, j] .* V[i]
        end
        hj = norm(z)
        H[j+1, j] = hj
        for i in 1:j-1                                # old rotations
            t        =  cs[i] * H[i, j] + sn[i] * H[i+1, j]
            H[i+1,j] = -sn[i] * H[i, j] + conj(cs[i]) * H[i+1, j]
            H[i, j]  = t
        end
        a = H[j, j]; r = hypot(abs(a), hj)            # new rotation
        if r == 0
            cs[j], sn[j] = one(ComplexF64), 0.0
        else
            cs[j], sn[j] = conj(a) / r, hj / r
        end
        H[j, j]   = cs[j] * a + sn[j] * hj            # == r
        H[j+1, j] = 0
        gv[j+1] = -sn[j] * gv[j]
        gv[j]   =  cs[j] * gv[j]
        k = j
        push!(hist, abs(gv[j+1]) / β)
        verbose && @printf("  GMRES iter %4d   rel. residual %.3e\n", j, hist[end])
        (hist[end] <= tol || hj <= eps(Float64) * β) && break
        V[j+1] = z ./ hj
    end

    y = zeros(ComplexF64, k)                          # back substitution
    for i in k:-1:1
        s = gv[i]
        for l in i+1:k
            s -= H[i, l] * y[l]
        end
        y[i] = s / H[i, i]
    end
    x = zeros(ComplexF64, n)
    for i in 1:k
        x .+= y[i] .* V[i]
    end
    return x, hist
end

"""
    waveholtz_gmres(p; tol, maxiter)

Krylov-accelerated WaveHoltz: solve `(I - S) ν = Pi_0` with GMRES.
One matrix-vector product is one source-free wave solve over a period,
`S*ν = Pi(ν)|_{J=0}`, so `Pi_0` (one forced solve) is formed once and
never re-subtracted -- this is the same system as the `ν - Pi(ν) + Pi_0`
formulation but without the cancellation.
"""
function waveholtz_gmres(p::Problem; tol::Float64 = 1e-10,
                         maxiter::Int = 200, verbose::Bool = false)
    n = ndof(p.g)
    w = Work(p.g)
    Π₀ = zeros(ComplexF64, n)
    apply_Pi!(Π₀, zeros(ComplexF64, n), p, w; source_on = true)

    nmv = Ref(0)
    function ImS!(out, x)
        apply_Pi!(out, x, p, w; source_on = false)    # out = S*x
        @inbounds for i in eachindex(out)
            out[i] = x[i] - out[i]
        end
        nmv[] += 1
        return out
    end

    ν, hist = gmres_mf(ImS!, Π₀; tol = tol, maxiter = maxiter, verbose = verbose)
    return ν, hist, nmv[] + 1                         # + the solve for Pi_0
end

# =====================================================================
#  Post-processing
# =====================================================================

"""
    harmonic_fields(ν, p) -> (Dz, Bx, By)

Unpack a converged state into time-harmonic amplitudes referenced to
`t = 0`. The magnetic field is stored at `t = -dt/2` (MEEP staggering),
so it is rotated forward by half a step, `b <- b * exp(i*omega*dt/2)`.
This phase differs from one by `O(omega*dt)`, i.e. `O(h)`, so skipping
it would destroy the second-order accuracy of `B`.
"""
function harmonic_fields(ν::AbstractVector{ComplexF64}, p::Problem)
    g = p.g
    o1, o2, o3 = offsets(g)
    Dz = reshape(ν[1:o1],       size_Dz(g))
    Bx = reshape(ν[o1+1:o2],    size_Bx(g))
    By = reshape(ν[o2+1:o3],    size_By(g))
    ph = exp(im * p.ω * p.dt / 2)
    return Dz, ph .* Bx, ph .* By
end

"""
    max_errors(ν, p) -> (eD, eBx, eBy)

Max-norm errors against the manufactured solution, each field sampled
at its own Yee location.
"""
function max_errors(ν::AbstractVector{ComplexF64}, p::Problem)
    g, ω = p.g, p.ω
    Dz, Bx, By = harmonic_fields(ν, p)
    eD = eBx = eBy = 0.0
    for j in 1:g.Ny+1, i in 1:g.Nx+1
        eD = max(eD, abs(Dz[i, j] - Ez_exact(xnode(g, i), ynode(g, j), ω)))
    end
    for j in 1:g.Ny, i in 1:g.Nx+1
        eBx = max(eBx, abs(Bx[i, j] - Bx_exact(xnode(g, i), yhalf(g, j), ω)))
    end
    for j in 1:g.Ny+1, i in 1:g.Nx
        eBy = max(eBy, abs(By[i, j] - By_exact(xhalf(g, i), ynode(g, j), ω)))
    end
    return eD, eBx, eBy
end

"Relative residual of the fixed-point equation, ||Pi(ν) - ν|| / ||ν||."
function fixedpoint_residual(ν::AbstractVector{ComplexF64}, p::Problem)
    w = Work(p.g)
    r = similar(ν)
    apply_Pi!(r, ν, p, w)
    return norm(r - ν) / norm(ν)
end

# =====================================================================
#  Drivers
# =====================================================================

"""
    run_case(N, ω; courant, tol, maxiter) -> NamedTuple

Solve one case with both the fixed-point iteration and GMRES and report
errors, iteration counts and timings.
"""
function run_case(N::Int, ω::Float64; courant::Float64 = 0.5,
                  tol::Float64 = 1e-10, maxiter::Int = 400,
                  verbose::Bool = true)
    p = Problem(N, ω; courant = courant)
    if verbose
        @printf("\nN = %d,  omega = %.4f (= %.3f pi),  dx = %.4e,  dt = %.4e,  Nt = %d,  dof = %d\n",
                N, ω, ω / π, p.g.dx, p.dt, p.Nt, ndof(p.g))
    end

    t1 = @elapsed νw, hw = waveholtz_fixedpoint(p; tol = tol, maxiter = maxiter)
    t2 = @elapsed νg, hg, nmv = waveholtz_gmres(p; tol = tol, maxiter = maxiter)

    eDw, eBxw, eByw = max_errors(νw, p)
    eDg, eBxg, eByg = max_errors(νg, p)

    if verbose
        @printf("  WHI   : %4d wave solves, %7.2f s, final %.2e | err (Dz,Bx,By) = %.3e %.3e %.3e\n",
                length(hw), t1, hw[end], eDw, eBxw, eByw)
        @printf("  GMRES : %4d wave solves, %7.2f s, final %.2e | err (Dz,Bx,By) = %.3e %.3e %.3e\n",
                nmv, t2, hg[end], eDg, eBxg, eByg)
        @printf("  fixed-point residual of the GMRES solution: %.3e\n",
                fixedpoint_residual(νg, p))
        @printf("  ||nu_WHI - nu_GMRES|| / ||nu_GMRES||      : %.3e\n",
                norm(νw - νg) / norm(νg))
    end

    return (; p, νw, νg, hw, hg, nmv,
            err_whi = (eDw, eBxw, eByw), err_gmres = (eDg, eBxg, eByg),
            time_whi = t1, time_gmres = t2)
end

"""
    convergence_study(ω; Ns, courant, tol)

Refinement study: max-norm errors and observed rates. Second order is
expected for all three fields.
"""
function convergence_study(ω::Float64 = 2π; Ns = [20, 40, 80, 160],
                           courant::Float64 = 0.5, tol::Float64 = 1e-12)
    @printf("\nConvergence study, omega = %.4f (= %.3f pi), courant = %.3f\n", ω, ω / π, courant)
    @printf("%6s %10s %8s %12s %8s %12s %8s %12s\n",
            "N", "err Dz", "rate", "err Bx", "rate", "err By", "rate", "solves")
    prev = nothing
    for N in Ns
        p = Problem(N, ω; courant = courant)
        ν, _, nmv = waveholtz_gmres(p; tol = tol, maxiter = 400)
        e = max_errors(ν, p)
        if prev === nothing
            @printf("%6d %12.4e %8s %12.4e %8s %12.4e %8s %8d\n",
                    N, e[1], "--", e[2], "--", e[3], "--", nmv)
        else
            r = ntuple(k -> log2(prev[k] / e[k]), 3)
            @printf("%6d %12.4e %8.2f %12.4e %8.2f %12.4e %8.2f %8d\n",
                    N, e[1], r[1], e[2], r[2], e[3], r[3], nmv)
        end
        prev = e
    end
end

# =====================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    run_case(40, 2π)
    run_case(40, 6π)
    convergence_study(2π; Ns = [20, 40, 80])
end
