# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An implementation of **EM-WaveHoltz** (Peng & Appelö, `EM_WH.pdf`, Sec. II-A "Yee-EM-WaveHoltz")
on top of MEEP's FDTD core — a frequency-domain Maxwell solver built as a filtered fixed-point
iteration over a time-domain solver. Plain fixed point; no Krylov acceleration yet.

`STATUS.md` is the working log: what is verified, with numbers, and what is open. **Read it first**
— it also records earlier conclusions that turned out to be wrong and why, which is the fastest way
to avoid repeating them. Not a git repository, so `STATUS.md` is the only history.

## Commands

MEEP lives in its own conda env; the base anaconda env has no `meep`.

```bash
conda activate meep                       # or use the interpreter directly:
/Users/appelo/anaconda3/envs/meep/bin/python scripts/convergence_pec.py
```

Run everything **from the repo root** — scripts write relative paths (`notes/…`).

```bash
# tests: plain scripts with asserts and __main__ blocks. pytest is NOT installed.
python tests/test_operator.py             # I-S symmetric SPD, filter multiplier, boundary ring
python tests/test_filter.py               # DFT filter vs independent Python quadrature
python tests/test_periodicity.py          # fixed point is the T-periodic solution
python scripts/test_decay.py              # zero forcing: ||Pi^k nu|| -> 0  (lives in scripts/)

# studies (regenerate figures/data under notes/)
python scripts/convergence_pec.py         # PEC manufactured solution, fixed Courant
python scripts/ring_compare.py            # ring vs solve_cw, contour plots
python scripts/ring_refine.py             # ring grid refinement, L2

# document
cd notes && pdflatex -interaction=nonstopmode pec_convergence.tex
```

`scripts/probe_*.py` are single-purpose MEEP-behaviour probes; each answers one question and prints
its evidence. Prefer extending one over re-deriving a MEEP fact from the docs — several documented
behaviours turned out to be wrong.

## Architecture

`emwh/` is the library; `scripts/` are experiments; `tests/` are assertions; `notes/` is the
LaTeX write-up plus generated figures.

One iteration (`EMWaveHoltz.apply_pi`, `emwh/core.py`) is:
`restart_fields` → optional half-step back-out → inject state → M × `fields.step()` → read filter.
The time stepping and the filter are pure C++; the only per-gridpoint Python is the injection
callback.

- **`core.py`** — the operator and fixed-point driver, `tune_courant`, `inject_state`, `INJECT_AS`.
- **`filters.py`** — `DFTFilter` (production; two native DFT accumulators) and `ProbeFilter`
  (pointwise Python reference, validation only).
- **`gridmap.py`** — Yee lattice discovery and position→index injection.
- **`problems.py`** / **`manufactured.py`** — the ring resonator and PEC cavity; the Sec. III-C
  manufactured solution with its exact `u`, forcing `J`, and norms.

Note `inject_state`, `INJECT_AS` and `TM_FLUX_COMPONENTS` are not re-exported by
`emwh/__init__.py`; import them from `emwh.core`.

## Invariants that are easy to break

**State goes into D and B, never E and H.** MEEP time-steps the flux fields and recomputes
`E = D/eps`, `H = B/mu` every step, so a write to E or H is discarded on the next step (injecting a
pulse into `Ez` in a lossless box lost 98% of its energy in one step). All injection must go through
`emwh.core.inject_state`, which applies the `INJECT_AS` mapping. Where `eps != 1` the state must
additionally *be* `(Dz, Bx, By)` — see `TM_FLUX_COMPONENTS` — because filtering E and injecting into
D drops a factor of eps in the material; with an E/H state the ring diverges at 61x per application.

**The filter window must be an exact integer number of timesteps.** `tune_courant` picks the Courant
number so `M = T/dt` is integral; the constructor rejects anything else. The filter's
"multiplier exactly 1 on the forced mode" property depends on summing over whole periods.

**`DFTFilter.read()` must be called after every window of stepping.** MEEP has no way to zero a DFT
accumulator and `restart_fields` deliberately leaves transforms running, so `read()` differences
running totals. Skipping it makes the *next* `apply_pi` difference against a stale baseline and
silently absorb the extra window. `reset()` is not a substitute.

**`add_dft_fields` needs `yee_grid=True, decimation_factor=1`.** The defaults interpolate to voxel
centres and auto-decimate from the Nyquist rate; either silently corrupts the operator.

## MEEP behaviours worth knowing

Each was measured here, and several contradict the natural reading of the docs:

- `get_array` and `get_array_metadata` work on the **centered** grid, not the Yee grid — even when
  given `dft_cell=`. The Yee lattice is instead recovered from the positions `initialize_field`
  visits (`gridmap.component_grid`). To read raw Yee values, use `fields.get_field` pointwise.
- `initialize_field` **accumulates** (`f += val`), so the field must be zeroed first.
- `restart_fields()` = `fields.t = 0; fields.zero_fields()`; it clears PML and dispersion auxiliary
  state and keeps sources (unlike `fields::reset()`).
- DFT objects are lazy (`swigobj_attr(...)` to instantiate) and the component must be allocated
  (`fields.require_component`) before the DFT is added.
- `ContinuousSource.dipole()` is a dipole moment, not a current. The driven current is
  `sinc(w*dt/2) * exp(-i*w*(t + dt/2))` — and MEEP applies it at `t = dt`, half a step later than
  paper eq. (23a) specifies. Known, uncorrected, O(dt).
- Several live `mp.Simulation` objects corrupt each other; build them lazily, one at a time.

## Validation strategy

Structural checks alone are not enough — a near-zero operator passes symmetry, positive-definiteness
and contractivity trivially, which is exactly how a fatal injection bug survived a full test suite
here. Any check that would still pass under `S -> 0` proves nothing. The cheap decisive tests are
**energy conservation of freely evolving injected data** (lossless box, no sources) and
**convergence against the manufactured solution**.
