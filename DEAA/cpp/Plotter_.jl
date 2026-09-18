using DelimitedFiles
using Plots

function read_D(file)
tmp = readdlm(file)
N = Int(sqrt(length(tmp)))
X_D = reshape(tmp,N,N)
X_D[:,N] .= 2*X_D[:,N-1] .- X_D[:,N-2]
X_D[N,:] .= 2*X_D[N-1,:] .- X_D[N-2,:]
X_D[N,N] = 2*X_D[N-1,N-1] - X_D[N-2,N-2]
return X_D
end 


function read_B(file)
tmp = readdlm(file)
N = Int(sqrt(length(tmp)/2))
BX = reshape(tmp[1:N*N],N,N)
BY = reshape(tmp[N*N+1:2*N*N],N,N)

BX[:,N] .= 2*BX[:,N-1] .- BX[:,N-2]
BX[N,:] .= 2*BX[N-1,:] .- BX[N-2,:]
BX[N,N] = 2*BX[N-1,N-1] - BX[N-2,N-2]


BY[:,N] .= 2*BY[:,N-1] .- BY[:,N-2]
BY[N,:] .= 2*BY[N-1,:] .- BY[N-2,:]
BY[N,N] = 2*BY[N-1,N-1] - BY[N-2,N-2]

return BX, BY
end

XD = read_D("D_xcoord.txt")
YD = read_D("D_ycoord.txt")
XBX, XBY = read_B("B_xcoord.txt")
YBX, YBY = read_B("B_ycoord.txt")

##pld = plot(XD, YD, lc=:black, lw=2, label="")
##plot!(pld, XD', YD', lc=:black, lw=2, label="", m=:circle, ms=6, mc=:black)
##
##
##plot!(pld, XBX, YBX, lc=:red, lw=2, label="")
##plot!(pld, XBX', YBX', lc=:red, lw=2, label="", m=:square, ms=6, mc=:red)
##
##
##plot!(pld, XBY, YBY, lc=:blue, lw=2, label="")
##plot!(pld, XBY', YBY', lc=:blue, lw=2, label="", m=:square, ms=6, mc=:blue)
##
##plot!(pld, ratio=1, xlims=(-1,1), ylims=(-1,1))
##savefig(pld, "yee_grid.png")


#
#D_0 = read_D("D_sol_0.txt")
#D_1 = read_D("D_sol_1.txt")
#D_2 = read_D("D_sol_2.txt")
#D_99 = read_D("D_sol_99.txt")
#
#
#BX, BY = read_B("B_sol_0.txt")
#BX_1, BY_1 = read_B("B_sol_1.txt")
#BX_2, BY_2 = read_B("B_sol_2.txt")
#BX_99, BY_99= read_B("B_sol_99.txt")



# ---------------------------------------------------------------- animation --
# test_cavity.cpp dumps every timestep of the run (D_sol_<it>.txt / B_sol_<it>.txt,
# it = 0..Nt).  Count them rather than hardcoding Nt, so this still works when
# test_cavity is rerun at another resolution (Nt = Tend*res*2).
nframes = count(f -> occursin(r"^D_sol_\d+\.txt$", f), readdir("."))

Ds = [read_D("D_sol_$(it).txt") for it in 0:nframes-1]
Bs = [read_B("B_sol_$(it).txt") for it in 0:nframes-1]   # (BX, BY) tuples

# The coordinate arrays are tensor-product, so surface() can take vectors.
# A[i,j] has i indexing y and j indexing x -- which is already the orientation
# surface(x, y, Z) wants (Z[row = y, col = x]), so no transpose.
xD,  yD  = XD[1,:],  YD[:,1]
xBX, yBX = XBX[1,:], YBX[:,1]
xBY, yBY = XBY[1,:], YBY[:,1]

# max|Dz| passes through ~0 twice per period, so per-frame autoscaling would blow
# those frames up into pure noise.  Fix the z and color limits globally.
zD  = maximum(maximum(abs, d)    for d in Ds)
zBX = maximum(maximum(abs, b[1]) for b in Bs)
zBY = maximum(maximum(abs, b[2]) for b in Bs)

dV = 1. #2*sqrt((xD[2] - xD[1])*(yD[2] - yD[1]))
T  = sqrt(2)                 # = 2pi/omega, one full period
dt = T / (nframes - 1)
Dexact = (XD.+1.0).*(XD.-1.0).*(YD.+1.0).*(YD.-1.0)

# clims as well as zlims: with only zlims fixed, GR still recolors each frame and
# the surface strobes through the colormap as the amplitude crosses zero.
panel(x, y, Z, ttl, zm) = surface(x, y, Z;
        title = ttl, zlims = (-zm, zm), clims = (-zm, zm),
        c = :balance, camera = (35, 30),
        xlabel = "x", ylabel = "y", colorbar = false,
        xlims = extrema(x), ylims = extrema(y))

anim = @animate for it in 1:nframes
    BX, BY = Bs[it]
    plot(panel(xD,  yD,  Ds[it], "Dz", zD),
				 panel(xD,  yD,  Dexact, "Dz exact", 1.),
         panel(xBX, yBX, BX./dV,     "Bx", zBX),
         panel(xBY, yBY, BY./dV,     "By", zBY);
         layout = (1, 5), size = (1800, 700),
         plot_title = "t = $(round((it-1)*dt, digits=3))  (T = $(round(T, digits=3)))")
end

gif(anim, "fields.gif", fps = 10)
