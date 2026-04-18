# iter_reml Phase 3 — ship-ready status

**Status:** algorithm complete (2026-04-19).  Ships with documented
caveats for small-N rare-K applications.

## What Phase 3 delivered

Phase 3 replaces Phase 2's AI-REML retrofit on Laplace with a real
EM outer loop for the binary-trait liability model.  Key
deliverables:

### Algorithm

* **Analytic Laplace-correction gradient** (D5'/Option 2) —
  `½·(q_k^μ − tr(W^½·M_k·W^½·S⁻¹)) − ½·Σ_i W'_i·(H⁻¹)_ii·(H⁻¹·w_k)_i`.
  Implemented as `compute_w_prime` + `compute_diag_hinv` +
  `compute_laplace_correction` in `fitace/ace_iter_reml/src/laplace.{h,cpp}`.
  Verified by the dense-NumPy reference to match ∂logL_Laplace/∂σ²
  to ≤1 log unit at self-consistent Xβ.

* **EM M-step** (`em_mstep.{h,cpp}`) — AI-preconditioned damped
  gradient ascent with trust-region clamping and a 6-halving σ²-
  floor safeguard.  Supports `--em-preconditioner {ai, diag, identity}`.

* **Live-EM Xβ consistency fix** (ai_reml.cpp) — mode-find and
  em_step now share the SAME Xβ per iter (lagged one iter), so
  the gradient formula is consistent with how the mode was found.
  Caught by the finite-diff harness; prior inconsistency had wrong-
  sign gradients at nonzero Xβ.

* **Monotone logLik guard** — after each em_step, re-evaluates
  Laplace logLik at the proposed σ² and rejects the step if logLik
  drops > tol (default 1.0 log unit, Hutchinson noise budget).
  On reject: restore prior state and declare convergence.  Prevents
  drift past the Laplace MLE under noisy Q-gradient estimates.
  Effect on n=900 K=0.20 fixture: n_iter 100→27, σ²_E 1e-5→0.037
  (off floor), logLik −436→−430.

* **Post-convergence logLik + observed-info SE** (D4) — Laplace
  marginal logLik = `log p(y|l̂) − ½·(l̂−Xβ)'V⁻¹(l̂−Xβ) − ½·log|S|`
  via SLQ on a MATSHELL `S = I + W^½·V·W^½`.  Observed info by
  finite-difference of the EM gradient (3 trial E-steps, 3 gradient
  evals).  Populates `result.logLik`, `result.cov`, `result.se_sigma2`.

### Diagnostic tooling

All tooling under `fitace/iter_reml/` unless noted.

| Tool | Purpose |
|---|---|
| `em_diag.em_diag` + `finite_diff_gradient` | One-shot evaluation at a probe σ²; finite-diff cross-check of EM gradient against logLik |
| `plot_trajectory` | 4-panel PDF of logLik / σ² / grad / δ_trust across outer iters |
| `verify.verify_converged_gradient` | Post-fit check that `|grad·σ²|_∞` is small at claimed convergence |
| `ref_dense.LaplaceReference` | Pure-NumPy reference: exact mode / logLik / FD gradient / MLE optimiser at small n (≤2000) |
| `summarize_dev_grid` | Walk `results/dev/*/rep*/iter_reml_*/` → bias table vs truth |
| `--em-debug-jsonl PATH` (CLI) | Per-iter JSONL dump of every intermediate used for the step decision |
| `--diag-sigma2 A,C,E` (CLI) | Standalone evaluation at specified σ² without running the outer loop |
| `tests/test_laplace_primitives.cpp` | C++ unit tests for `compute_w_prime` and `compute_diag_hinv` |

### Existing infrastructure reused

* Per-stage `ACE_BENCH_RSS` → `fit.bench.tsv` already tracks peak
  RSS at every stage boundary.
* ε-annealing (Phase 2.9) still active — starts at `epsilon_start=1.0`,
  decays geometrically toward `epsilon_end=0.3` to ease the inner
  Newton basin then tighten the threshold-MLE approximation.

### Test status

All **82 iter_reml tests pass** on both fp32 and fp64 builds
(up from 75/77 at start of Phase 3, with 5 new EM-invariant tests).

## Known limitations

### Laplace approximation bias — the big one

The Laplace MLE is NOT at truth on binary/threshold data.  The
dense reference (no Hutchinson / SLQ / PCG noise) confirms this
at n=1k K=0.15:

| σ² point | dense logLik |
|---|---:|
| truth (0.5, 0.2, 0.3) | −494 |
| mid (0.25, 0.1, 0.15) | −465 |
| **Laplace MLE ≈ (0.19, 0.04, 0.025)** | **−442** |

Live EM finds the Laplace MLE correctly; the bias is in the
approximation itself.  Pattern: Laplace compresses σ² downward,
worse at small N and rare K.

**Prevalence effect** (gap between interior MLE and all_E degenerate
at n=1k):
* K=0.05: gap 4 (marginal — MLE nearly degenerate)
* K=0.15: gap 79
* K=0.30: gap 96
* K=0.50: gap 101

Higher K = more balanced binary = more strongly-identified MLE.

### Recommended usage

* **Works well**: n ≥ 10k, K ≥ 0.15.  Laplace bias shrinks with N;
  h² estimates are approximately unbiased even when V(A), V(C),
  V_E individually underestimate.
* **Warn**: n < 5k OR K < 0.10.  σ² biased downward ~30-50%.  h²
  more reliable than individual σ² components.
* **Unsuitable**: K < 0.05.  Dense reference shows the Laplace
  surface is nearly flat between interior MLE and degenerate "all-
  in-E" solution.  Recommend MCEM (Phase 4) or external tool
  (Stan) for these applications.

### Wall-time

Compared to Phase 2 at same fp32, n=10k:
* Phase 2 Laplace (old AI-REML): ~60s / ~10 iters (but σ²_E pinned
  at floor — biased fit).
* Phase 3 EM (this work): ~200-500s / 27-50 iters (monotone-ascending
  logLik, σ²_E off floor, genuinely finds MLE).

Cost per iter:
* Phase 2: 1 mode-find + AI-REML gradient.
* Phase 3: 1 mode-find + EM gradient (with Hutchinson trace) +
  Laplace correction (with Hutchinson diag(H⁻¹)) + 1 mode-find +
  SLQ on S (for logLik monotone guard).

~3-5× per iter, × 2-5× iters = **~10-20× slower** than Phase 2.
Above the original plan's 3× budget but justified by correctness.

## Commit trail

In rough order.  See `git log fitace/ace_iter_reml` for details.

* `bdee60b` — Option 2 analytic Laplace correction + Xβ consistency
* `5e4439f` — per-iter logLik + --em-preconditioner
* `e6482d9` — C++ unit tests for `compute_w_prime` / `compute_diag_hinv`
* `42b22f8` — --em-debug-jsonl
* `23636fb` — monotone logLik guard
* fitACE `5bdfed5` — plot_trajectory, verify_converged_gradient
* fitACE `190d1f1` — dense reference + Stan model file
* fitACE `6ac1fcf` — dev-grid summariser
* simACE `bfe1c91` / `fda87fa` — D6 findings note

## Open items (deferred)

* **D7**: Continuous/MeanOnce regression — existing test suite
  covers this; formal regression harness not written.
* **D8**: CLI flags / docs tidy-up — --em-preconditioner and
  --em-debug-jsonl are on the binary but not on the Python
  wrapper.  Minor plumbing.
* **Stan reference via cmdstanpy** — Stan model file exists
  (`fitace/stan/fit_pedigree_ace_binary.stan`) but not wired to a
  Python driver.  Cmdstanpy install pending.

These are small polish items.  Algorithm and tooling are
production-ready.
