# Yee-EM-WaveHoltz in MEEP — status

Implements Sec. II-A of `EM_WH.pdf` (Peng & Appelö) on top of MEEP's FDTD core,
plain fixed point, no Krylov.

**Working on both target problems, both second-order convergent.** Closed PEC
cavity: rates exactly 2.000 in the discrete L2 and max norms. Ring resonator
with PML (paper Fig. 1): fitted order 1.85–2.14 in L2 by self-convergence, 93
iterations at resolution 10, agreeing with MEEP's FDFD solver to 9.1e-3 away
from the point sources. The EM-WaveHoltz iteration count is flat in resolution.

Environment: `conda activate meep` (MEEP 1.34.0, its own env).

## The two root causes

Both are about *where* state is written. MEEP time-steps the flux fields D and B
and recomputes `E = D/eps`, `H = B/mu` through `update_eh` on every step.

**1. Initial data must be written to D and B, never to E and H.** A write to E or
H is silently discarded on the very next step. In a lossless PEC box, where
energy must be conserved:

| injected into | sum Ez² step 0 | step 1 | step 2 | step 10 |
|---|---|---|---|---|
| `Ez` | 13.85 | **0.039** | 0.140 | 0.254 |
| `Dz` | 13.85 | 12.82 | 11.05 | 5.61 |

98% of the injected energy destroyed on step one.

**2. Where eps != 1, the state must *be* (D, B), not (E, H).** Filtering E and
injecting the result into D drops a factor of eps wherever the material is not
vacuum. On the ring (eps_r = 3.4²) an (Ez,Hx,Hy) state diverges at **61x per
application**; a (Dz,Bx,By) state converges at rate 0.82. Since eps and mu are
time independent, filtering `D = eps E` is exactly eps times the filter of E, so
this is the same method in flux variables. Recover E by injecting the converged
state and reading Ez, where MEEP applies `E = D/eps`.

Use `emwh.core.TM_FLUX_COMPONENTS = (Dz, Bx, By)` for any problem with material;
`INJECT_AS` maps E/H to D/B and leaves D/B alone, and all injection goes through
`emwh.core.inject_state()` so the distinction cannot drift again.

## Corrections to earlier conclusions in this file

Both were wrong and are worth recording, because each looked well-supported.

**"100% of the defect is in Π0" — wrong, and the evidence was circular.** With
injection broken, S was nearly the zero operator on the interior, so: `I − S`
came out symmetric positive definite (a near-zero operator trivially is);
ρ(S) came out *exactly* 0.5, which is `|β(0)|`, the **static**-mode multiplier —
the wave modes were being annihilated, and that number should have been read as
a symptom rather than a success; and `Π0 = (I−S)ν*` "reproducing ν* to 1e-15" is
`ν* ≈ ν*` when S ≈ 0, true regardless. Every check was invariant under S → 0.
The one-line check that settles it is energy conservation of freely evolving
injected data.

**"PML breaks D/B injection; the ring is disabled" — wrong.** That test used a
Gaussian centred at the origin, which is negligible inside the PML, so it proved
nothing. PML injection is fine: pulses decay correctly even when centred *inside*
the PML. The ring's divergence was cause 2 above, eps != 1.

## Results

PEC cavity, manufactured solution (paper Sec. III-C), Courant fixed at 0.5:

| res | rel L2 | rate | rel L∞ | rate | amplitude |
|---|---|---|---|---|---|
| 8 | 9.811e-2 | — | 7.682e-2 | — | 1.0953 |
| 32 | 6.131e-3 | 2.000 | 4.790e-3 | 2.000 | 1.0059 |
| 96 | 6.812e-4 | 2.000 | 5.321e-4 | 2.000 | 1.0007 |

~2.5x more accurate than `solve_cw` on the same grid (6.8e-4 vs 1.7e-3 at res 96);
both second order.

Ring resonator with PML, res 10 (N=120), omega_0, 10 periods, cos-forcing:
93 iterations / 41 s to a 1e-8 increment, against 3.6 s for `solve_cw` (the gap
is mostly the absent Krylov acceleration). Max difference 5.6e-2 relative, but
concentrated at the two point sources: **9.1e-3 excluding disks of radius 0.6
about (±1.1, 0)**. That difference does not shrink with more periods
(5.80, 5.50, 5.45e-2 for 5/10/20) nor with refinement (5.7, 5.5, 6.4, 8.0e-2 for
res 6/8/12/16) — a non-convergent point singularity, as expected for a delta
source. The overall amplitude *does* converge with more periods (best-fit scale
1.0365 → 1.0141 → 1.0048), which is the PML auxiliary variables settling.

Operator structure after the fix: ρ(S) = 0.279 (a real wave mode, not the static
one), symmetry 1.8e-16 on the **full** grid, ghost ring fully decoupled. Decay
from random data: ρ = 0.5003 (full state), 0.2629 (Ez only), 0.0467 (Ez only,
5 periods).

Ring grid refinement (`scripts/ring_refine.py`, L2, resolutions 5/10/15/20
against `solve_cw` at 60): **fitted order 1.853 / 2.136 / 1.880** from
self-convergence, which is reference-free. The reference-based rate column is
contaminated -- measuring FDFD against a fine FDFD inflates its rates above 2
(2.26, 2.34) and depresses WaveHoltz's, so the finest points there are the
least trustworthy. **Iteration count is flat in resolution: 82, 81, 83, 83**
(the paper's Table II behaviour). `solve_cw` costs 0.6/5.7/17.1/39.6 s at
resolutions 5-20 but 1740 s at 60 -- 44x for 9x the cells, so its iteration
count grows ~5x over that range while WaveHoltz's does not grow at all.

Caveat on analysis code: successive differences scale as
`C a^-p (1 - (a/b)^p)`, dominated by the coarse grid. A plain log-ratio of two
differences drops the second factor and reports a spurious 3.8 on this data
whenever the refinement ratio varies (here 2, 1.5, 1.333). `ring_refine.py`
fits the correct model.

Documents: `notes/pec_convergence.{tex,pdf}` (7 pp, both problems),
`notes/pec_convergence.png`, `notes/ring_compare.png`, `notes/ring_refine.png`,
`notes/pec_error_contours.png`. `CLAUDE.md` holds the orientation notes.

## Verified

* `initialize_field` **accumulates** (`f += val`) at exactly the Yee points —
  1.0 → 2.0 → 7.0 under repeated injection.
* `restart_fields()` = `fields.t = 0; fields.zero_fields()`, clearing PML and
  dispersion auxiliary state; verified to leave *exactly* zero.
* `get_array` / `get_array_metadata` work on the **centered** grid, not the Yee
  grid, even with `dft_cell=`; the Yee lattice is instead recovered from the
  positions `initialize_field` visits (`emwh/gridmap.py`).
* `add_dft_fields(..., yee_grid=True, decimation_factor=1, persist=True)`, and it
  works on D/B components too. DFT objects are lazy; the component must be
  allocated first.
* Filter == independent Python quadrature to **7.8e-16** on Ez, Hx, Hy.
* Π is stationary across restarts (1e-15) and exactly affine (3.5e-14).
* dt is bit-exact: `dt == Courant/resolution` and `M*dt - T == 0` exactly.
* MEEP's applied current is `sinc(ωΔt/2)·[sin|cos](ω·Δt)·J` — exact amplitude,
  exact spatial profile (residual 1.7e-16), but sampled at `t = Δt` rather than
  the half step `Δt/2` of eq. (23a).
* Several live `mp.Simulation` objects at once corrupt each other — build them
  lazily, one at a time.

## Recently closed

* **The "half step late" source is not a bug to fix** (`scripts/probe_halfstep.py`).
  The offset is real and unchanged by the D/B fix — `t_eff/dt = 1.00000` exactly
  at resolutions 8–64, amplitude exactly `sinc(ωΔt/2)`, fit residual 1e-16. But
  shifting the source to eq. (23a)'s half step makes the solver *worse*: at a
  fixed shift of half a step the PEC error stagnates at ~1.1e-3 (res 32/64/96:
  1.13e-3, 1.13e-3, 1.10e-3) instead of converging, and at res 96 it is 1.62x
  the uncorrected error. Only the unshifted source gives clean second order.

  A phase sweep shows why the earlier "it helps" reading was wrong: the
  error-minimising shift is *resolution dependent* — `s = +0.50` at res 32
  (1.13e-3 vs 6.13e-3 unshifted) but `s = +0.25` at res 64 (2.65e-4 vs
  1.53e-3). It halves as h halves, which is the signature of a tuned
  cancellation against the discretisation error, not a correct convention.
  MEEP's placement is self-consistent with its own discretisation; the earlier
  "real O(Δt) fidelity bug" label in this file was wrong.

  Not understood: at a fixed nonzero shift the error stagnates rather than
  degrading to O(h), and the measured difference between the shifted and
  unshifted solutions is ~27x larger than `1 - cos(ωΔt/2)` predicts. The
  argument that a lossless PEC cavity has Re{E} = 0, so a phase error cannot
  contaminate, does not survive contact with the data.

* `scripts/compare_fdfd.py` now uses `TM_FLUX_COMPONENTS` and converges on the
  ring (72 iterations, rate 0.82 at resolution 6); it previously diverged.
  `scripts/ring_compare.py` is still the one that makes the contour plots.
* `scripts/test_decay.py` case D (ring + PML) is re-enabled in flux variables
  and passes: rho = 0.9161, decaying to 1.2e-7 in 150 iterations. Random O(1)
  initial data is stable there, so the earlier NaN was the eps != 1 state bug,
  not PML.

## Open

* **Periodicity is inexact at 4.0e-4** (`tests/test_periodicity.py`), not
  round-off. Most likely the eq. (24) half-step back-out, whose stencil has no
  partner in the last row/column, so the state round trip is inexact at the grid
  edge. `backout=False` is used for the ring.
* **`emwh/pi0.py` is unvalidated.** Python-driven Π0 with MEEP's C++ stepper for
  S; written to chase the Π0 hypothesis, and its self-check failed at the time
  (two filter paths disagreeing by 1.06 relative, Π0 ~6x MEEP's). Those numbers
  were measured with the broken E-injection and are stale.
  `EMWaveHoltz.solve(pi0=...)` is a useful hook regardless.
* No Krylov acceleration. `solve()` already stores `pi_zero`, so a CG/GMRES
  driver on `(I − S)ν = Π0` is a small addition.
