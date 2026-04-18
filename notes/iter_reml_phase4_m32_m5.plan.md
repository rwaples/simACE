# Phase 4 M3.2 – M5: MCEM outer loop, wiring, validation, release

**Status:** plan, 2026-04-19.  Follow-up to
`iter_reml_binary_phase4_mcem.plan.md` and the M3.1 landing
(ace_iter_reml `34366bd` — C++ TMVN sampler primitive + unit tests).

## Dependencies & context

Already landed (Phase 4 M1 + M3.1):
* `fitace/iter_reml/mcem_ref.py` — Python MCEM reference; validated
  MCEM dominates Laplace at rare K (6–12× better σ²_A recovery at
  n=300).
* `fitace/ace_iter_reml/src/tmvn.{h,cpp}` — C++ TMVN sampler
  (dense V⁻¹ Gibbs, Acklam inverse-CDF, tail-stable).  6/6 unit
  tests pass.

Not yet: the MCEM E-step is in C++ but there's no outer loop
calling it, no CLI to run it, no Snakemake rule using it.  M3.2
through M5 build the rest.

## M3.2 — MCEM outer-loop driver (3-5 days)

Goal: A callable C++ path from `run_ai_reml` → TMVN E-step →
MC M-step → updated σ², β.

### M3.2.1 — New layer type + options (~0.5 day)

`phenotype_layer.h`:

```cpp
enum class LayerType {
    Continuous,
    LiabilityMeanOnce,
    LiabilityLaplace,
    LiabilityMCEM,           // NEW
};

struct MCEMOptions {
    int n_samples   = 50;
    int burn_in     = 50;
    int thin        = 3;
    std::uint64_t seed = 42;
    // Converge when σ² relative range over em_tol_step_window iters
    // falls below em_tol_step (same convention as Phase 3 EM, but
    // the window is typically larger to average over MC noise).
    int em_tol_step_window = 5;
    double em_tol_step     = 1e-2;
    double em_initial_delta_trust = 0.2;
};

PhenotypeLayer make_liability_mcem_layer(
    const std::vector<double>& y_binary, double K);
```

`ai_reml.h` adds `MCEMOptions mcem_opts;` to `AiRemlOptions`.

**Decision: separate layer type, not reuse LiabilityLaplace.**
Clean CLI separation (`--liability-model mcem` vs `laplace`),
different convergence semantics (noisier gradient), different
logLik formula.

### M3.2.2 — `mcem_step` function (~1.5 days)

New function in `em_mstep.cpp` analogous to `em_step`:

```cpp
struct MCEmStepResult {
    std::array<double, 3> sigma2_next;
    std::array<double, 3> grad;              // MC-estimated
    std::array<double, 3> step;
    std::array<std::array<double, 3>, 3> AI; // Gaussian AI, used as preconditioner
    double mc_stderr_grad_norm;              // Monte Carlo stderr on |grad|
    int pcg_iters_total;
    int n_samples_used;
    double gibbs_time_s;                     // wall time spent in TMVN
    double grad_eval_time_s;
};

MCEmStepResult mcem_step(EmStepInputs& in);
```

Algorithm:

1. Build `xbeta_vec` (Xβ̂ + mean_l_prev, same convention as Phase 3).
2. Instantiate `TMVNSampler(y_raw, V, xbeta_vec, tau, seed)`.
3. Draw K samples (burn + thin per MCEMOptions).
4. For each sample l^(k):
   - Solve u^(k) = V⁻¹·(l^(k) − xbeta) via PCG.
   - Compute per-sample q_k^(k) = u^(k)'·M_A·u^(k), and likewise for C, E.
   - AI rows: e_A^(k) = M_A·u^(k); w_A^(k) = V⁻¹·e_A^(k) (one PCG per
     sample per component).  Too expensive at large K — restrict AI
     computation to a small subset of samples (e.g. K_ai=3), average.
5. Compute Hutchinson estimate of `tr(V⁻¹·M_k)` (reuse Phase 3
   probe-solve machinery).
6. MC gradient:
   ```
   q_bar_k = (1/K)·Σ_k q_k^(k)
   grad_k  = 0.5·(q_bar_k − tr_V_inv_M_k)
   ```
7. Estimate MC stderr of the gradient from sample-variance of q_k^(k).
8. Step rule: AI-preconditioned Newton + trust region, same as
   Phase 3.  Reject step if step size < MC stderr · safety factor
   (avoid chasing noise).
9. Return updated σ² + diagnostics.

**Cost analysis (n=1000, K=50 samples):**
* TMVN sampling: O(n²) per sweep · (burn_in + K·thin) sweeps
  = O(n² · 200) = 2·10⁸ ops ≈ 1 s at 200 MFLOP/s
* V⁻¹ build + inverse: O(n³) once = 1·10⁹ ≈ 5 s
* Per-sample u = V⁻¹·(l − Xβ): K PCG solves × ~20 iters × O(nnz)
  = 50·20·30000 = 3·10⁷ ≈ 0.3 s
* Per-sample AI computation: K_ai × 3 more PCG solves ≈ 0.03 s
* Hutchinson tr(V⁻¹·M): same m probes as Phase 3 ≈ 2 s
**Total: ~10 s per outer iter.**  At 30 iters: ~5 min per fit.
Roughly 3–5× Phase 3 Laplace at same n.  Acceptable.

### M3.2.3 — Outer-loop driver (~1 day)

Modify `ai_reml.cpp`:

1. Extend the `is_laplace_layer` pattern to also detect
   `is_mcem_layer` at the top of `run_ai_reml`.
2. In the outer loop, after E-step variants:
   * Laplace path: `find_laplace_mode` (unchanged).
   * **MCEM path: no mode-find**.  TMVN sampler is instantiated
     inside `mcem_step`.  The layer's `l_hat` is kept as a *running
     sample-mean* across iters for Xβ continuity.
3. Dispatch: if `is_laplace_layer`, call `em_step` (Phase 3 path).
   Else if `is_mcem_layer`, call `mcem_step`.
4. Convergence check: same σ²-range criterion but with the larger
   MCEM window (`opts.mcem_opts.em_tol_step_window`, default 5).
5. Skip `compute_diag_hinv` + `compute_laplace_correction` for
   MCEM (they're Laplace-specific).
6. Observed info: finite-difference MCEM gradient (slow — extra
   K·3 samples).  Defer to v2; use `invert3x3_sym(AI)` for v1 SE.

### M3.2.4 — Per-iter diagnostic emission (~0.5 day)

`AiRemlIterRow` gets optional MCEM-specific fields:

```cpp
struct AiRemlIterRow {
    ...  // existing Phase 3 fields
    double mc_stderr_grad_norm;    // NaN for non-MCEM
    int    n_samples_used;         // NaN for non-MCEM
    double gibbs_time_s;           // NaN for non-MCEM
};
```

`write_phase2_tsv` appends these as columns.  Extend
`--em-debug-jsonl` to include them.

Update `fitace/iter_reml/plot_trajectory.py` with a 5th panel when
the MCEM columns are present.

### M3.2 exit criteria

* `mcem_step` compiles fp32 + fp64.
* All 82 Phase 3 tests still pass (regression).
* C++ TMVN unit tests still pass.

## M3.3 — CLI + Python wrapper + Snakemake (~1-2 days)

### M3.3.1 — C++ CLI (~0.5 day)

`main.cpp`:

```
--liability-model {mean, laplace, mcem}
--mcem-samples N         (default 50)
--mcem-burnin B          (default 50)
--mcem-thin T            (default 3)
--mcem-seed S            (default = --seed)
```

Validation: `--liability-model=mcem` requires `--prevalence K`.
Mutually exclusive with `--skip_phase2`.

### M3.3.2 — Python wrapper (~0.5 day)

`fitace/iter_reml/fit.py`:

```python
def fit_iter_reml(
    ...,
    liability_model: Literal["mean", "laplace", "mcem"] = "mean",
    mcem_samples: int = 50,
    mcem_burnin: int = 50,
    mcem_thin: int = 3,
    mcem_seed: int | None = None,
    ...
):
```

Passes through to CLI.  `IterREMLResult.meta` records
`liability_model`, `mcem_samples`, etc.

### M3.3.3 — Snakemake (~0.5 day)

Extends `iter_reml.smk`:

* New config keys: `iter_reml_mcem_samples`, `iter_reml_mcem_burnin`,
  `iter_reml_mcem_thin`, `iter_reml_mcem_seed`.
* Defaults in `config/_default.yaml`.
* New dev-grid scenarios in `config/dev.yaml`:
  `dev_mcem_n1k`, `dev_mcem_K_rare`, `dev_mcem_n3k_K_rare`, etc.

## M3.4 — Integration tests (~1 day)

`tests/iter_reml/test_mcem_path.py`:

1. **Convergence smoke** — fits MCEM on tiny_ace_inputs at K=0.20,
   asserts `converged=True`, n_iter < max_iter, σ² all > 0.
2. **Non-degeneracy** — MCEM recovers σ²_A > 0.02 (truth 0.5) where
   Laplace pins at floor.  Documented scientific improvement.
3. **Cross-validation with Python reference** — same fixture, same
   seed, same K.  Expect C++ MCEM σ² within MC noise of Python
   reference (~3·σ_sd/√K).  Validates the C++ port is correct.
4. **Determinism** — same seed twice produces identical σ².
5. **Wall-time sanity** — MCEM at n=300 finishes in < 5 minutes.

## M4 — Dev-grid validation (~2-3 days)

Rerun `config/dev.yaml` + `dev_mcem_*` scenarios with MCEM.
Aggregate via `summarize_dev_grid.py`.  Key metrics per scenario:

| Metric | Threshold | Action if exceeded |
|---|---|---|
| `|bias(h²)|` | ≤ 0.05 typical, ≤ 0.10 rare-K | Investigate, tune K/burn |
| `|bias(c²)|` | ≤ 0.05 | Ditto |
| `|bias(Vp)|` | ≤ 0.1 | Ditto |
| σ²_E not floored | always true | Bug if violated |
| Convergence rate | ≥ 90% across reps | Increase em_max_iter |
| SE coverage | ≥ 85% at 95% nominal | Need better SE estimator |
| Wall-time | ≤ 10× Laplace | Profile + optimise |

Write `notes/iter_reml_phase4_m4_findings.md` documenting the
empirical MCEM bias / coverage / wall-time.  Decision gate: if
MCEM recovers σ² within the thresholds, ship; else iterate on
K/burn_in/step-rule.

## M5 — Release + docs (~1-2 days)

### M5.1 — User guide

Add `fitace/iter_reml/README.md` (or append to existing):

* When to use which `liability_model`:
  * `mean`: Phase 1 Mills-ratio imputation.  Fastest, continuous-
    like.  OK at n ≥ 10k with moderate kinship.
  * `laplace`: Phase 3 EM.  Recommended for n ≥ 10k K ≥ 0.15.
    Biased down by O(1/n) — negligible at production scale.
  * `mcem`: Phase 4.  Required for rare-K (K ≤ 0.10) or small-N
    (n ≤ 2000).  ~3-5× slower than Laplace.  Not scalable above
    n=2000 in v1 (dense V⁻¹ memory bound).
* CLI + Python + Snakemake usage snippets.
* Diagnostic tools (`plot_trajectory`, `verify_converged_gradient`,
  `em_diag`).

### M5.2 — Phase 3 ship note amendment

Update `iter_reml_phase3_ship_ready.md` to reference MCEM as the
rare-K escape hatch (not "deferred to Phase 4" — now shipped).

### M5.3 — CalVer tag

After M4 sign-off, tag `v2026.MM` across the three repos
(simACE / fitACE / ace_iter_reml).

### M5.4 — Deferred items for a follow-up phase

Not in scope for Phase 4 first-cut:

* **MCEM logLik** for LRT/AIC/BIC.  Needs bridge sampling or path
  sampling.  ~1 week research + implementation.
* **Louis-style observed info** for proper SE.  Computes
  `Cov(score)` across samples.  ~3-5 days.
* **Sparse-precision Gibbs** (Henderson A⁻¹ + household C⁻¹) for
  scalability beyond n=2000.  ~1-2 weeks.
* **Vectorised / parallel sample draws** via OMP.  Gibbs is
  inherently sequential per-chain, but K independent chains
  parallelise trivially.  ~1 day.

## Effort rollup

| Milestone | Estimate | Cumulative |
|---|---:|---:|
| M3.2 MCEM outer-loop | 3-5 days | 3-5 |
| M3.3 CLI + wrappers | 1-2 days | 4-7 |
| M3.4 Integration tests | 1 day | 5-8 |
| M4 Validation | 2-3 days | 7-11 |
| M5 Release + docs | 1-2 days | **8-13 days** |

Gated by M4: if validation shows MCEM doesn't meet ship criteria,
loop back to M3.2 for tuning.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| MC gradient noise dominates → EM oscillates | Larger K (100-200) + step rule that rejects sub-noise steps |
| Gibbs mixing bad on pedigree-scale correlation | Longer burn-in; fallback to sparse Cholesky + Botev in M3.2.v2 |
| Wall-time > 10× Laplace at n≥1k | Profile; parallelise K independent chains; cap n_samples |
| Dense V⁻¹ fails at n=2000 (~30 MB memory) | Fine at n=2000.  Document n ≤ 2000 as v1 limit |
| Observed-info SE using AI⁻¹ under-covers | Warn in release notes; Louis SE in follow-up phase |
| Convergence criterion too strict/lax for MC noise | `mcem_tol_step` separate knob; tune during M4 |

## Decision checkpoints

* **After M3.2**: does C++ MCEM match the Python reference on the
  tiny fixture within MC noise?  If not, halt and debug before
  wiring the plumbing.
* **After M4**: does MCEM ship-gate pass on typical scenarios?
  If not, examine failures case-by-case, iterate.
* **After M5**: CalVer tag.
