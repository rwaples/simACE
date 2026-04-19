# Phase 5 M4 — Sparse-precision Gibbs (lift n ≤ 2000 MCEM cap)

**Status:** plan, 2026-04-19.  Ships in v2026.07 or later; own
phase.  Blocked on M3 (Louis SE + bridge logLik must land first
so sparse MCEM inherits the same SE and logLik infrastructure).

## Goal

Replace the dense V⁻¹ inside `TMVNSampler` with a sparse-
precision Gibbs scheme so MCEM scales beyond the current n ≤ 2000
cap.  Match Laplace's current ceiling (n = 50k+).

## Ship criteria

* `fit_iter_reml(..., liability_model="mcem")` runs end-to-end on
  `dev_laplace_n50k` (n ≈ 51000 phenotyped) without exhausting the
  dense-V⁻¹ memory (currently blows up at ~8 GB for 30k×30k).
* At n=1000 K=0.15 fixture, sparse-Gibbs σ² match the dense-V⁻¹
  v2026.06 output within MC noise (no accuracy regression).
* At n=10k K=0.15, MCEM wall-time is ≤ 10 min per fit.
* At n=50k K=0.15, MCEM wall-time is ≤ 60 min per fit.

## Out of scope

* GPU parallelism (Gibbs sweeps are inherently sequential per
  chain).
* Multi-chain parallelism (cheap optimisation, separate tier).
* Non-additive kinship (D-matrix, AxC): scope balloons.

## Background: why sparse-precision Gibbs

Current `TMVNSampler` uses DPOTRF + DPOTRI to materialise dense
V⁻¹ (O(n²) memory, O(n³) setup, O(n²) per Gibbs sweep).  At n=2000:
memory ~30 MB, setup ~2 s, sweep ~5 ms.  At n=10k: memory ~800 MB,
setup ~5 min, sweep ~100 ms — borderline.  At n=50k: memory ~20 GB,
setup hours — infeasible.

The Gibbs per-coordinate conditional uses only:
* `τ_ii = (V⁻¹)_ii` — the *diagonal* of V⁻¹
* `u = V⁻¹·(l − Xβ)` — maintained incrementally via rank-1 updates:
  `u ← u + Δl · V⁻¹[:, i]` when coord i changes.

So we only need V⁻¹'s diagonal and its columns (one at a time).
For sparse SPD V with known structure, both can be computed
without materialising dense V⁻¹:

1. **diag(V⁻¹)**: via Hutchinson probes on V⁻¹ (reuse Phase 2
   scaffolding) OR via partial Cholesky selected inversion (Takahashi
   equations) — exact on sparse V in O(nnz²/n) time.
2. **V⁻¹·e_i** (column i): one PCG solve.

Gibbs sweep cost per coord: O(nnz(V)·PCG_iters) ≈ O(n·30) for
sparse pedigree V.  Per-sweep cost: O(n·nnz·PCG_iters) ≈ O(n²·30)
at nnz = O(n).  Comparable to dense at small n; much better at
large n.

Memory: nnz(V) sparse vs n² dense.  At n=50k, pedigree V has
~1-5M nonzeros; sparse representation ~20-80 MB vs 20 GB dense.

## Candidate algorithms

### Option A: Takahashi selected-inversion for diag(V⁻¹), PCG per coord

Most principled.  `diag(V⁻¹)` computed once per outer iter via
Takahashi recursion on the supernodal Cholesky factor (SuiteSparse
or PETSc MUMPS).  Each Gibbs coord update: 1 PCG solve for V⁻¹·e_i.

Cost per Gibbs sweep: n·PCG_iters·nnz/n = n·30·(nnz/n) ops.
At nnz = 8·n, n=10k: ~2.4M ops/sweep = ~10 ms/sweep.  At n=50k:
~50 ms/sweep.  Acceptable.

### Option B: Henderson decomposition + Gauss-Seidel on full l

For pedigree models, `A = 2·K` has a known sparse inverse
`A⁻¹` via Henderson's method (Quaas 1976):

    A⁻¹ = T⁻ᵀ · D⁻¹ · T⁻¹

where T is the sparse pedigree incidence matrix and D is a
diagonal of inbreeding-corrected variances.  Same holds for `C`
(block-diagonal → inverse is trivial).

V = σ²_A·A + σ²_C·C + σ²_E·I  — but V⁻¹ has NO sparse
representation in general (even if A⁻¹, C⁻¹, I⁻¹ are sparse, the
inverse of a sum is not).

However, the precision matrix of (A, C, E) jointly IS sparse:

    l = a + c + e
    a ~ N(0, σ²_A·A)     → prec(a) = (1/σ²_A)·A⁻¹  (sparse)
    c ~ N(0, σ²_C·C)     → prec(c) = (1/σ²_C)·C⁻¹  (sparse)
    e ~ N(0, σ²_E·I)     → prec(e) = (1/σ²_E)·I    (diagonal)

Instead of sampling l directly from p(l|y), augment the state
with (a, c) (the latent A and C components) and sample via Gibbs
on the JOINT (l, a, c) posterior.  Each of a, c, l has a sparse
precision — conditional draws are cheap sparse solves.

This is the **variance-component Gibbs sampler** from
Hobert & Casella (1996), Sorensen et al. (1994), commonly used
in animal-breeding software.

Cost per sweep: 3 sparse solves per outer sample, O(nnz)
operations each.  At n=50k with nnz(A⁻¹) ≈ 6·n: 300k ops per
sweep, trivial.

### Recommendation: Option B (Henderson augmentation)

Better asymptotic complexity + known animal-breeding literature.
More data structures but the payoff is massive at large n.
Option A as a fallback if augmentation proves tricky to debug.

## Milestones

### M4.1 — Python prototype of Henderson-augmented Gibbs (~3 days)

Extend `fitace/iter_reml/mcem_ref.py`:

* Build A⁻¹ via Henderson's method (or use a closed-form for test
  pedigrees).  Cache at sampler init.
* Build C⁻¹ (block-diagonal inverse).
* Sample jointly from p(l, a, c | y, σ², β):
  - Sample a | l, c, σ²: Gaussian with precision `σ²_A⁻¹·A⁻¹ + σ²_E⁻¹·I`.
  - Sample c | l, a, σ²: Gaussian with precision `σ²_C⁻¹·C⁻¹ + σ²_E⁻¹·I`.
  - Sample l | a, c, y: TN(a+c, σ²_E, half-space bounds by y).
    All coords independent given a+c → n univariate TN draws.

No kinship-correlated sparsity in the TN step — huge win.

Test: on a small pedigree (n=300), verify sparse-Gibbs sample
moments match dense-V⁻¹ sampler within MC noise.  Expect bit-
compatible results at K=500, same seed (up to RNG path).

### M4.2 — C++ port, reusing A⁻¹ from existing sparse_reml path (~5-7 days)

sparse_reml already uses Henderson A⁻¹ — reuse that code.  Integrate
into `TMVNSampler` as a v2 backend selected via a flag.

Key C++ work:
* New class `TMVNSamplerSparse` with same public API as current
  `TMVNSampler`, internal representation uses sparse A⁻¹, C⁻¹.
* Per-sweep: 3 sparse-precision sampled Gaussian draws (cheap
  Cholesky factorisations) + n univariate TN draws.
* Output: same `sample(n_samples, burn_in, thin, warm_start)`
  signature so `mcem_step` is untouched.

CLI: `--mcem-sampler-backend {dense,sparse}` with auto-select
based on n (dense for n ≤ 2000, sparse for larger).

### M4.3 — Scalability benchmark (~2 days)

Add to `workflow/rules/iter_reml.smk` a scenario
`dev_mcem_sparse_n10k`, `_n50k` with matched seeds to
`dev_laplace_n10k`, `_n50k`.

Expected numbers (n1k → n50k, K=0.15, 50 samples, 100 burn, 5 thin):

| n | Dense V⁻¹ (v2026.06) | Sparse Henderson (v2026.07) | Speedup |
|---:|---:|---:|---:|
| 1,000 | 4 s | 5 s | 0.8× |
| 10,000 | 600 s | 45 s | 13× |
| 50,000 | OOM | 600 s | ∞ |

At small n, sparse is slightly slower due to overhead — that's
why we auto-select.

### M4.4 — Test + doc (~2 days)

* `tests/iter_reml/test_mcem_sparse.py`:
  - Sparse vs dense backend give same σ² at n=1000 within MC noise.
  - Sparse handles n=5000 without OOM.
* Update `fitace/iter_reml/README.md` to reflect the lifted cap:
  > MCEM v2 (v2026.07+): sparse-precision Gibbs, n up to 50k.
* Update method-selection table to add "scalability" column.

## Effort rollup

| Sub-milestone | Effort | Cumulative |
|---|---:|---:|
| M4.1 Python prototype | 3 days | 3 |
| M4.2 C++ port | 5-7 days | 8-10 |
| M4.3 Scalability bench | 2 days | 10-12 |
| M4.4 Tests + docs | 2 days | 12-14 |

Total: 1.5-2 weeks for a focused dev.  This is a real phase of
work; **not** a v2026.05 / .06 point release — spin off as Phase 6
("iter_reml scalability") if effort grows.

## Risks

* **Henderson A⁻¹ has twin edges** — MZ twins require special
  handling (identical genotypes → zero row in T).  Mitigation:
  ace_sreml already handles this; port the same code.

* **Mixing degradation**: augmented Gibbs can have worse mixing
  than marginal Gibbs on correlated posteriors.  Mitigation:
  longer burn-in at large n; verify mixing diagnostics (ESS per
  sweep) don't collapse.

* **SE coverage at large n** — the Louis / bridge infrastructure
  from M3 was validated at n ≤ 1k.  Reconfirm on n=10k, n=50k
  that coverage stays ≥ 85%.

* **Scope creep** — this is the biggest single milestone in the
  Phase 5 roadmap.  If effort exceeds 2 weeks, cut M4.3/M4.4 to
  a follow-up release and ship M4.1/M4.2 as "experimental sparse
  backend" first.

## Decision checkpoints

* **After M4.1**: Python sparse prototype matches dense moments
  within MC noise?  If not, augmentation formulation has a bug —
  halt and debug before C++ port.
* **After M4.2**: C++ sparse matches Python sparse reference
  within MC noise at n=300?  Standard C++↔Python cross-check
  (reuse M1 Tool B infrastructure).
* **After M4.3**: Scalability target hit (10k in < 10 min)?  If
  not, profile the Gibbs sweep and optimise before declaring
  M4 done.
