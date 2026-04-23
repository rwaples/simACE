# PCGC Phase 3 — land the rare-K Hermite correction

**Drafted:** 2026-04-22.  Supersedes the short "Implementation scope
for M6 commit" list at the tail of `phase2_bias_derivation.md` §5.

**Superseded by Phase 4 for K<0.1 (2026-04-23):** the
`hermite_order=2` Phase 3 correction is the production path for
K≥0.1; at lower K the Phase 4 `moment="bivnor"` exact-moment Newton
is the new default under `moment="auto"` (cutoff frozen at 0.1 in
`notes/PCGC/phase4_m36_auto_cutoff.md`).  Phase 4 replaces this
plan's `hermite_order` API surface with `moment=`; the old parameter
is preserved as a deprecated pass-through through 2026.07.  See
`notes/PCGC/phase4_bivnor_landing_plan.md` for the landing and
`jackknife_calibration_notes.md` for the SE miscalibration that
remains unfixed (not a bivnor issue).  Content below preserved as
historical record.

## Context

Phase 2 (notes/PCGC/phase2_bias_derivation.md) derived and
prototype-validated a rare-K correction: the ρ²·τ²/2 term from the
Hermite expansion of the bivariate probit, absorbed via damped Newton
on the nonlinear moment equation

    E[r_ij | ρ_ij]  =  c_factor · ρ_ij · (1 + ρ_ij · τ² / 2) + O(ρ³) .

The prototype lives in `fitACE/fitace/pcgc/{prototype,validate}_rare_k_correction.py`
and runs pure-Python Newton on top of the existing kinship + pair
enumeration.  Validation across the 80-cell bias grid dropped max
|bias| by 85–91% per component; 13/16 cells pass |bias| ≤ jackknife SE.

Phase 3 turns the prototype into a production estimator: add a
`hermite_order` knob to `fit_pcgc`, implement the Newton iteration in
all three backends (reference / numba / cpp) with byte-parity, keep
the existing jackknife pipeline working, update tests + fixtures +
the bias note recommendations.

## Success criteria

1. **API**: `fit_pcgc(…, hermite_order=2)` returns a `PCGCResult` whose
   `.vc` has the corrected point estimates; `hermite_order=1` keeps
   the Golan path and reproduces current outputs byte-for-byte.
2. **Backend parity**: all three backends agree to `atol=1e-3` on σ²
   and `atol=5e-3` on SE at `hermite_order=2`, matching the existing
   Phase 1 Track-1 tolerance used for the Golan path.
3. **Bias**: re-running the Phase 2 bias grid under the default
   (`hermite_order=2`) reproduces the prototype's bias-reduction
   numbers (max |bias| ≤ 0.06 across A/C/E, 13+/16 cells pass
   |bias| ≤ SE).
4. **Tests**: new `tests/pcgc/test_rare_k_correction.py` covers
   convergence, Jacobian correctness, and clean-cell safety; existing
   parity tests pass at both orders.
5. **Bias-note update**: `notes/pcgc_phase1_bias.md` §Recommendations
   can lift the "K ≥ 0.15 only for V(C)/Vе partition" caveat.
6. **Runtime**: corrected fit walls ≤ 2× baseline at n=100k (Newton
   adds ~10-15 linear passes through the pair data on top of 1 OLS
   in the Golan path).

## Milestones

### P3.M1 — API + reference backend
Smallest self-contained slice.  Adds the parameter surface and the
numpy reference implementation.

1. Add `hermite_order: Literal[1, 2] = 2` to `fit_pcgc` signature
   in `fitACE/fitace/pcgc/fit.py` (default = 2 per Phase 2 §5).
2. Thread the parameter through backend dispatch and the
   `fit_pcgc_reference` signature in `reference.py`.
3. In `reference.py`, after the existing OLS solve, run the Newton
   iteration when `hermite_order == 2`:
   ```python
   beta = beta_scaled.copy()  # Golan starting point
   for _ in range(max_iter):
       Xb = X @ beta
       g = Xtr - X.T @ (Xb + τ²/(2·c_factor) · Xb²)
       Jac = -(XtX + τ²/c_factor · X.T @ (X * Xb[:, None]))
       beta -= damping · np.linalg.solve(Jac, g)
       if change < tol: break
   ```
4. Hyperparameters (defaults): `max_iter=50`, `tol=1e-5`,
   `damping=0.5`.  Expose them as keyword args with the same defaults
   the prototype used; keep `fit_pcgc`'s public surface clean and
   only expose them via an internal `_newton_kwargs` dict used by
   tests.
5. Meta fields: emit `hermite_order`, `newton_iters`,
   `newton_converged`, `newton_max_step` (max |Δβ / c_factor| over
   iters; diagnostic for numerical health).
6. Per-block Newton for the jackknife: re-use `loo_beta`'s per-block
   (X'X, X'r) contributions but do one Newton step per block starting
   from the full-sample β.  Single step is O(m) per block and gives a
   first-order approximation of β⁽⁻ᵇ⁾.  Flag this in `fit.vc.tsv.meta`
   as `jackknife_newton_steps=1`; plan to revisit in P3.M5.
7. Validation: on (A=0.5, C=0.2, K=0.05) rep1, the reference-backend
   corrected V(A) and V(C) must land within 1e-6 of the prototype's
   Newton output on the same fixture.  Add this as a golden-number
   regression test.

### P3.M2 — numba backend
Port the Newton iteration to the numba kernel so n ≤ 100k runs
don't fall back to reference (which caps at 20k).

1. Add `_newton_iter_numba` kernel in `numba_impl.py` that takes the
   pre-enumerated (a_ij, c_ij, r_ij) arrays and runs the same Newton
   loop.  The per-iteration compute is 3 linear passes: (a) `Xb`, (b)
   `X'(Xb² + 2·c_factor/τ² · Xb)` for the gradient, (c) `X'(Xb · X)`
   for the off-diagonal Jacobian term.  2×2 linear algebra stays in
   python (negligible).
2. Backend-parity test: 10k-pair synthetic fixture, numba vs
   reference β agree to atol=1e-6, SE to atol=1e-4.
3. Re-enable the numba path for n ≤ 100k under `hermite_order=2`
   (no cap increase needed — the Phase 2 bias grid fits at
   n_phen=102k used cpp).

### P3.M3 — cpp backend
Port to the OpenMP-parallel C++ kernel — the production hot path.

1. Add `newton_iter_cpp` function in
   `fitACE/fitace/ace_pcgc/src/pcgc.cpp` matching the numba kernel
   signature.  Same three linear passes per iteration, each
   `#pragma omp parallel for reduction` over the pair vector.
2. Expose via the existing cpp driver (`cpp_driver.py`) with the
   same `hermite_order` dispatch.
3. Backend-parity test: numba vs cpp on n=50k fixture — β atol=1e-3,
   SE atol=5e-3 (matching existing Track-1 Phase 1 tolerances).
4. Re-run the Phase 2 bias-grid under cpp `hermite_order=2` and
   verify byte-identical results to the prototype's validator
   output at `results/pcgc_bias_map/summary/bias_matrix_corrected.tsv`.

### P3.M4 — test + diagnostic suite
1. `tests/pcgc/test_rare_k_correction.py`:
   - tiny synthetic fixture where the Hermite expansion can be
     computed to fourth order by hand; check that the Newton solver
     lands within 1e-6 of the analytic `hermite_order=2` truth.
   - convergence under pathological `damping` (α ≥ 1 should blow up
     at K=0.01; α = 0.5 should converge).
   - clean-cell safety: corrected V(A), V(C) stay within 0.02 of
     baseline on the (A=0.25, C=0, K=0.15) fixture (i.e., the
     correction doesn't "over-correct" on clean data).
   - `hermite_order=1` reproduces current Golan outputs bit-for-bit
     on existing fixtures (regression guard).
2. Extend `tests/pcgc/test_ascertainment.py` with a test that
   numerically verifies the Hermite-expansion coefficient (τ²/2) by
   differencing `Φ₂(τ, τ; ρ)` with finite ρ against the exact series.
3. Add `fit.vc.tsv.meta` checks in any existing end-to-end test that
   uses PCGC — the new diagnostic fields (hermite_order,
   newton_iters, newton_converged) must appear.

### P3.M5 — jackknife SE honesty
The single-Newton-step per block (P3.M1 step 6) is a linearization.
It may under-estimate SE at K=0.01 where higher-order curvature
matters.

1. Add an opt-in `jackknife_newton_steps: int = 1` parameter.  At
   `jackknife_newton_steps=5` or `full`, each block runs its own
   Newton to convergence.  Cost: ~5× the baseline jackknife wall,
   but still O(m) per step so tractable at n=100k.
2. Compare SE honesty on the Phase 2 bias grid: the ratio
   SD-across-reps / jackknife-SE should be closer to 1 under
   multi-step per-block Newton than under single-step.  If it
   doesn't improve meaningfully, keep single-step as default and
   note why in the phase3 close-out.
3. Update `notes/PCGC/phase2_bias_derivation.md` §4 with the SE
   calibration check under the corrected estimator.

### P3.M6 — re-run Phase 1 fixtures + update bias note
1. Re-fit every scenario currently enabled for PCGC in
   `fitACE/config/iter_reml_bench.yaml` at `hermite_order=2`.
   Specifically: iter_reml_{10k,50k,100k}_noam × rep1;
   iter_reml_300k_noam × rep{1..5} (already on disk, just the fit
   changes).
2. Re-run the Phase 2 bias grid's 80 cells; snapshot the new
   summary_aggregate.tsv alongside the baseline for the bias note
   update.
3. Edit `notes/pcgc_phase1_bias.md` §Recommendations:
   - Replace "Trust V(C)/Vе partition only for K ≥ 0.15" with the
     corrected estimator's trust regime (expected: usable down to
     K = 0.05 at n ≥ 10k).
   - Add a pointer to phase2_bias_derivation and phase3 landing notes.
4. Edit `notes/PCGC/phase2_bias_derivation.md` §5 to mark the
   landing decision as executed; link to the Phase 3 commits.

### P3.M7 — production-scenario rerun + scale spot check
1. Re-run one iter_reml_bench scenario at K=0.05 and n=300k AM=0.3
   under the correction.  This isn't a bias validation (AM=0.3 has
   its own Vp>1 story) but a wall-time regression test at biobank
   scale.
2. Extend `benchmarks/pcgc/pcgc_scale.tsv` with a `hermite_order`
   column; confirm Newton overhead stays ≤ 2× at n ≥ 100k.

## Critical files

Listed by milestone, primary edits in **bold**:

- P3.M1: **`fitACE/fitace/pcgc/fit.py`** (signature + dispatch),
  **`fitACE/fitace/pcgc/reference.py`** (Newton impl + jackknife
  linearization), **`fitACE/fitace/pcgc/io.py`** (new meta fields).
- P3.M2: **`fitACE/fitace/pcgc/numba_impl.py`** (Newton kernel).
- P3.M3: **`fitACE/fitace/ace_pcgc/src/pcgc.cpp`** (Newton kernel),
  **`fitACE/fitace/ace_pcgc/src/pcgc.h`** (signature),
  `fitACE/fitace/pcgc/cpp_driver.py` (dispatch).
- P3.M4: **`fitACE/tests/pcgc/test_rare_k_correction.py`** (new),
  `fitACE/tests/pcgc/test_{ascertainment,cpp_parity,numba_parity}.py`.
- P3.M5: `fitACE/fitace/pcgc/jackknife.py` (multi-step per-block
  Newton option).
- P3.M6: `notes/pcgc_phase1_bias.md`, `notes/PCGC/phase2_bias_derivation.md`.

## Existing utilities to reuse

- **Prototype algorithms** in
  `fitACE/fitace/pcgc/{prototype,validate}_rare_k_correction.py`:
  the Newton loop is already validated; port it into the backends
  byte-for-byte.  The `validate_rare_k_correction.py` batch runner
  stays useful as a regression / validation harness even after
  backend integration — keep it, but point it at `hermite_order=2`
  via `fit_pcgc` rather than its current inline numpy kernel.
- `fitace.pcgc.ascertainment.pcgc_c_factor` — unchanged; the
  Hermite correction acts on top of c_factor without modifying it.
- `fitace.pcgc.jackknife.loo_beta` — reusable for the single-Newton-
  step-per-block linearization (just compute one Newton update
  against the reduced block sums, using the full-sample β as warm
  start).
- `fitace.pcgc.summarize_dev_grid.summarize_pcgc_dev_grid` — the
  output schema already carries the meta fields we need; just add
  `hermite_order`, `newton_iters`, `newton_converged` to the emitted
  row dict and the aggregator will pick them up.
- `numba`'s `@njit` compile path in `numba_impl.py` — the Newton
  kernel can reuse the same compilation idiom.
- `_scale_mem` / `_scale_runtime` / `_pcgc_mem` / `_pcgc_runtime` in
  `fitACE/workflow/common.py` — no resource-heuristic changes for
  this phase; Newton overhead is ≤ 2× and already inside the
  heuristic's 1.7× headroom in practice (measured in P3.M7).

## Risks & decisions

1. **Jackknife linearization vs full Newton per block** (P3.M1 step 6
   / P3.M5).  Default to single-Newton-step, measure SE honesty,
   promote to multi-step only if needed.  Trade-off: single-step is
   fast but possibly under-estimates SE at K=0.01; full is honest
   but expensive.
2. **Damping α**.  Prototype used 0.5 across the grid.  Could
   adaptive damping (Armijo line search on `||g(β)||²`) shave iters
   on easy cells?  Probably not worth it — convergence is already
   10–15 iters, each O(m).  Default α=0.5, expose as kwarg.
3. **Divergence handling**.  If Newton doesn't converge in max_iter,
   the current prototype falls through with the last iterate.  For
   production, add `newton_converged=False` in meta and a
   `allow_newton_divergence` gate that defaults to False → raise.
   User must opt in to return a non-converged result.
4. **hermite_order > 2**.  The derivation series continues with a
   ρ³·(τ²−1)²/6 term.  Phase 2 validation shows hermite_order=2 is
   sufficient at K ≥ 0.05; hermite_order=3 would address K=0.01
   residual.  Out of scope for Phase 3; flag as `NotImplementedError`
   so users see the extension point.

## Verification

End-to-end:

1. `pytest fitACE/tests/pcgc/ -v` → all pre-existing tests pass;
   new `test_rare_k_correction.py` passes; backend parity at both
   hermite orders.
2. `python -m fitace.pcgc.validate_rare_k_correction` (with the
   inner kernel swapped to `fit_pcgc(hermite_order=2)`) →
   `results/pcgc_bias_map/summary/bias_matrix_corrected.tsv`
   byte-identical to the Phase 2 prototype's output.
3. `cd fitACE && snakemake --cores 8 -j 1 -f
   $(list of 80 bias-grid + 20 iter_reml_bench fit.vc.tsv targets)`
   completes, with meta files including the new
   `hermite_order=2`, `newton_iters`, `newton_converged` fields.
4. `python -m fitace.pcgc.summarize_dev_grid results/pcgc_bias_map
   --scenarios-glob 'pcgc_bias_*' --out-dir results/pcgc_bias_map/
   summary_corrected` produces a summary where 13+/16 cells have
   |bias| ≤ SE on every component.
5. Updated `notes/pcgc_phase1_bias.md` renders cleanly in preview
   with the relaxed recommendations.
6. A representative n=300k re-fit (iter_reml_300k_noam rep1)
   completes within 2× of baseline wall and the corrected σ²
   table is committed in the Phase 3 close-out commit.

## Open items (for follow-on)

- **Weissbrod 2018 sample-prevalence correction**.  Currently raises
  NotImplementedError; still out of scope.  The Hermite correction
  derivation in Phase 2 works at P = K; the P ≠ K case needs its own
  derivation with the corrected-c_factor extended to the new
  threshold.  Natural Phase 4 scope.
- **AM > 0 handling**.  Phase 2 restricted scope to AM = 0; the Vе < 0
  AM-equilibrium sentinel is preserved under the correction, but
  correcting the AM-induced Vp > 1 distortion is an orthogonal
  methodological extension.  Another future phase.
- **low_e edge case**.  Not addressed by the rare-K correction;
  separate failure mode from `dev_laplace_low_e` (Vе ≈ 0 saturating
  Vp).  Still open.
