# iter_reml tooling gaps — development-procedure audit

**Status:** audit written during Phase 3 EM debug (2026-04-18).
**Context:** when σ²_E converged to the floor on the tiny n=900 K=0.2
fixture, I couldn't tell whether the Laplace MLE was genuinely there
or whether my implementation was wrong.  That triggered this audit.

The tools below are ranked by expected development-impact, not
implementation cost.  User-selected for implementation: A, C, D, F,
G, I (B, E, H deferred).

## What already exists

| Tool | Purpose | Location |
|---|---|---|
| `--diag-sigma2` CLI + `run_em_diag` | Evaluate (logLik, grad, tr_WMWS, q, AI) at a specified σ² without running the outer loop | `main.cpp`, `em_mstep.cpp` |
| `--diag-xbeta-const` | Freeze Xβ across finite-diff evals (needed to match the live EM's centered convention) | `main.cpp` |
| `fitace.iter_reml.em_diag.finite_diff_gradient` | Python harness comparing analytic vs fd grad | `fitace/iter_reml/em_diag.py` |
| `--em-preconditioner {ai,diag,identity}` | Ablate step-rule preconditioner | `main.cpp` |
| Per-iter `logLik_em` in iter.tsv | Track whether EM is monotonically ascending the Laplace marginal | `ai_reml.cpp`, write_phase2_tsv |
| `EmObservedInfo` finite-diff observed info | Post-convergence SE | `em_mstep.cpp` D4 |

These let us verify single-point gradients and observed-info, but
not trajectory behaviour, not ground-truth comparison, and not the
C++ primitives in isolation.

## Tier 1 — blocks Phase 3 shipping

### A. Trajectory plotter (iter_reml_trajectory.py)

**Purpose:** read `<out>.iter.tsv` → PDF with 4 panels:
1. `logLik_em` across iters (log scale Y; mark peak with red dot)
2. `VC_A, VC_C, VC_E` on one panel (log scale Y; truth overlaid if
   supplied)
3. `|grad|_∞` across iters (log Y) + `grad_A, grad_C, grad_E`
4. `delta_trust` across iters (linear Y)

**Why:** eyeballing text tables is slow and brittle.  A monotone-
ascending logLik with Hutchinson noise looks VERY different from an
oscillating or drifting one when plotted.  Post-mortem of any fit
becomes fast.  Saves 10× time in every future diagnostic session.

**CLI:** `python -m fitace.iter_reml.plot_trajectory <out_prefix>
  [--truth A,C,E] [--out path.pdf]`.

**Cost:** ~30 minutes.  Pure matplotlib, no C++ changes.

### C. Auto grad-sanity at convergence (`verify_converged_gradient`)

**Purpose:** after any Laplace fit, automatically run
`finite_diff_gradient` at the converged σ² and warn if
`|grad_fd|_∞ · max(σ²)` > threshold.  A large gradient at claimed
convergence is a red flag — either convergence criterion fired too
early, or we're at a non-stationary point.

**Why:** we currently only check `converged=True` by our own
(sometimes-wrong) criterion.  Independent gradient check catches
false-convergence without relying on the criterion we're
debugging.

**Shape:** add as a method on `IterREMLResult` or an adjacent
helper; also a pytest fixture that auto-invokes it for all Laplace
tests.

**Cost:** ~20 minutes.

### D. Stan reference on tiny fixtures

**Purpose:** write a Stan model for the exact liability-threshold
ACE problem, fit MCMC on the tiny n=300 fixture, report posterior
σ² mean / mode / CI.  Ground truth.

**Why this is the most important missing tool:** when our Laplace
says σ²_E = 0.05 is the MLE on this fixture, I cannot tell whether:
  (a) the Laplace approximation really has its MLE there (small-N
      Laplace error at rare K), or
  (b) my correction/EM machinery is wrong.
Stan (exact posterior via MCMC) distinguishes these in ~30 seconds.
Without it, I'm debugging against my own implementation —
circular.

**Model sketch** (Stan):
```stan
data {
  int<lower=1> n;
  int<lower=0,upper=1> y[n];
  matrix[n,n] A;        // kinship (dense for n=300)
  matrix[n,n] C;        // household
  real<lower=0,upper=1> K;
}
transformed data {
  real tau = inv_Phi(1 - K);
}
parameters {
  vector[n] l;
  real<lower=0> sigma2_A;
  real<lower=0> sigma2_C;
  real<lower=0> sigma2_E;
  real beta0;
}
model {
  matrix[n,n] V = sigma2_A * A + sigma2_C * C
                + sigma2_E * diag_matrix(rep_vector(1, n));
  l ~ multi_normal(rep_vector(beta0, n), V);
  for (i in 1:n) y[i] ~ bernoulli(Phi((l[i] - tau)));  // sharp threshold
  sigma2_A ~ exponential(1);  // weakly-informative
  sigma2_C ~ exponential(1);
  sigma2_E ~ exponential(1);
}
```

**Cost:** 1-2 hours.  Stan boilerplate + running 4 chains × 1000
draws on the tiny fixture + parsing summary.

### Follow-up (deferred)

- **B. Preconditioner sweep harness** — matrix of configs → table.
  Can be done manually for now with shell loops.
- **E. Simulation-bias grid** — the `phase3_pxem.plan.md §Decision
  gate` criterion.  Needs D6 infrastructure; defer to when Phase 3
  is actually converging sensibly.

## Tier 2 — C++-level correctness (long-term stability)

### F. C++ unit tests (GoogleTest)

**Purpose:** direct tests for `compute_w_prime`,
`compute_diag_hinv`, `compute_laplace_correction`,
`em_step::gradient_only` branch — tested on small-n fixtures
without CLI → TSV → pytest round-trip.

**Why:** today every check costs a full fit wall-time.  C++ unit
tests run in milliseconds.  Catches bugs at compile time.
Refactoring safety net.

**Specific tests:**
- `compute_w_prime`: finite-difference against `compute_w_and_grad`
  per-i, asserts |Δ|/|h| ≈ 0 after subtracting the central-diff
  approximation.
- `compute_diag_hinv`: at n=50, compute H⁻¹ densely (LAPACK
  `dsyev` then reconstruct), take diagonal, compare to Hutchinson
  with m=1000 probes; assert relative error < 1%.
- `apply_S` matches a densely-built `I + W^½·V·W^½` matvec.
- `em_step(gradient_only=true)`: deterministic output for fixed
  seed; gradient formula matches (q_k − tr_WMWS_k)/2 +
  correction_k per-component.

**Cost:** 1-2 hours.  Add `tests/cpp/` + CMake target + test
primitives + run via `ctest`.

### G. Per-iter debug JSON log

**Purpose:** `--em-debug-log PATH` emits one JSON object per outer
iter with every intermediate value (V·y for the solver, mode
Newton iters/residual, W_diag summary, u/w_A/w_C/w_E norms, AI
matrix, grad components, tr_WMWS, correction components, step_dir,
trust region clamping, halving count).

**Why:** when a fit goes wrong, post-mortem currently means
re-running with higher log-level and grepping stdout.  A structured
per-iter dump lets me diff across fits, replay specific iters,
automate anomaly detection.

**Cost:** ~30 minutes.  JSON emission from the existing state
that's already computed.

## Tier 3 — scale / performance (ship-readiness)

### I. Memory profile per stage

**Purpose:** already have `ACE_BENCH_RSS` macros at stage
boundaries.  Convert the per-stage peak RSS into a
`<out>.memprof.tsv` so we can track memory growth across fits and
catch leaks before they surface at n=100k.

**Cost:** trivial (rewrap existing bench macros), ~15 minutes.

### Deferred to Tier 3

- **H. Wall-time scaling benchmark** — scripted grid over n.

## Rule-of-thumb for adding tooling

When a fit surprises you and you find yourself running 3 different
ad-hoc Python snippets to understand what happened, that's a
signal the tooling to answer that question is missing.  Add it;
reuse it next time.  The cost is one session; the payback is
every session thereafter.

## Cold-start order for a future session

If I had to pick up Phase 3 in a fresh session right now, I'd want
these in hand first:
1. Trajectory plotter (A) — read a fit's iter.tsv, see what
   happened in 2 minutes.
2. Stan reference (D) — verify what the Laplace MLE actually is
   on the tiny fixture.
3. Auto grad-check at convergence (C) — catches regressions
   silently.

Then all three combined answer "did Phase 3 do the right thing?"
without relying on self-consistency of the implementation under
test.
