# Phase 5 M1 — Multi-rep dev-grid + Python↔C++ cross-check

**Status:** plan, 2026-04-19.  Ships in v2026.05.

## Goal

1. Verify that v2026.04 MCEM's σ² estimates are **stable across
   random seeds** — the current dev-grid has 1 rep per scenario,
   which means the published σ²_A = 0.57 on dev_mcem_n1k (truth
   0.5) could be a lucky draw.  Need ±SD over 3-5 replicates.
2. Wire up **Tool B** (the Python↔C++ cross-check from the Phase 4
   plan §M3.4) — verifies that C++ and Python reference
   implementations of MCEM agree within MC noise on a shared
   fixture.  The skeleton in `tests/iter_reml/test_mcem_path.py::
   TestMCEMPythonCrosscheck` is ready to turn on.

## Ship criteria

* `summary_aggregate.tsv` for all `dev_mcem_*` scenarios at ≥ 3
  replicates; `bias_h2_mean`, `bias_c2_mean`, `bias_Vp_mean` all
  within ±2·SD of zero on K ≥ 0.15 scenarios.
* `TestMCEMPythonCrosscheck::test_mcem_cpp_matches_python_ref` is
  un-skipped and passes: σ²_MLE_cpp within 3·MC_stderr of
  σ²_MLE_python on a shared fixture + fixed K / burn / thin.
* `notes/iter_reml_phase5_m1_findings.md` documents the replicate
  variability and cross-check tolerance used.

## Out of scope

* Adding new dev scenarios (a scenario-sweep is a separate
  exercise).
* Re-running Laplace / Mean scenarios (unchanged from v2026.04).
* Bias correction if bias is larger than expected — that's an
  algorithmic change (Phase 5 M3 or later).

## Milestones

### M1.1 — Bump replicates on MCEM dev scenarios (~1 hr)

`config/dev.yaml` in both simACE and fitACE: change `replicates: 1`
→ `replicates: 5` on `dev_mcem_*` scenarios.  Matching-seed
convention across simACE and fitACE means rep₁…rep₅ use
`seed=base+0…4`.

Artefacts:
* `results/dev/dev_mcem_*/rep{1..5}/` from simACE simulate.
* `results/dev/dev_mcem_*/rep{1..5}/iter_reml_fp32/fit.*` from
  fitACE.

Wall-time: ≤ 5 min per (scenario × rep) at n=1020 K=100 burn=50.
4 scenarios × 5 reps = 20 fits ≈ 100 min.

### M1.2 — Aggregate & publish (~30 min)

```
python -m fitace.iter_reml.summarize_dev_grid \
    results/dev --scenarios-glob 'dev_mcem_*' \
    --out-dir results/dev/summary_mcem_v2026.05
```

Emits:
* `summary.tsv` — 20 rows (scenario × rep)
* `summary_aggregate.tsv` — 4 rows (scenario) with mean ± SD of
  bias_{A,C,E,Vp}, converged_frac, diverged_frac, wall_s.

Commit the aggregated TSV to `notes/phase5_m1_mcem_dev_grid.tsv`
for reference (not normally committed — exception for ship-gate
evidence).

### M1.3 — Tool B cross-check (~2 hr)

Replace the skip in
`fitACE/tests/iter_reml/test_mcem_path.py::TestMCEMPythonCrosscheck::
test_mcem_cpp_matches_python_ref` with a real test:

```python
def test_mcem_cpp_matches_python_ref(self, tiny_ace_inputs):
    K, K_samples, burn_in, thin = 0.15, 100, 100, 5
    seed = 42
    # 1. Run C++ MCEM.
    r_cpp = fit_iter_reml(..., liability_model="mcem",
                          mcem_samples=K_samples, mcem_burnin=burn_in,
                          mcem_thin=thin, mcem_seed=seed, max_iter=30)

    # 2. Run Python MCEMReference on the same fixture.
    ref = dense_reference_from_sparse(y_binary, kinship, hh,
                                      prevalence=K)
    mcem = MCEMReference(ref, n_samples=K_samples, burn_in=burn_in,
                         thin=thin, seed=seed)
    r_py = mcem.optimise(sigma2_init=sigma2_init_match_phase1,
                         max_iter=30, tol=1e-3, trust_region=0.15)

    # 3. Compare.  Both use different RNG streams in practice (C++
    #    seed mixing differs) so require σ² within 3·SE where SE
    #    is estimated from the M1 replicate study.
    se_A, se_C, se_E = 0.10, 0.05, 0.08  # from M1.2 aggregate.
    assert abs(r_cpp.σ²_A - r_py.σ²_mle[0]) < 3 * se_A
    ...
```

Test marked with `needs_bin` so it's skipped when no build is
present.  Tolerances are empirical — keyed off M1.2's measured
SEs rather than theoretical MC stderr.

### M1.4 — Findings note + tag (~30 min)

Write `notes/iter_reml_phase5_m1_findings.md`:
* Replicate variability table (bias mean, bias SD per scenario).
* Cross-check test passes at which tolerance, and why that
  tolerance is statistically defensible.
* Any scenario where MCEM fails > 10% of replicates — flag as
  documented limitation, not a bug.

Tag `v2026.05` if M1 + M2 land together, else spin M1 out as
`v2026.04.1`.

## Testing

* Existing ctests (3/3) unchanged.
* `pytest tests/iter_reml/test_mcem_path.py` — 5 tests:
  current 3 pass + 1 skip + 1 new cross-check = 5 pass.
* No simACE test changes.

## Risks

* **RNG-mismatch makes cross-check noisy** — C++ and Python
  samplers use different splitmix64 seed mixings; they will NOT
  produce identical samples.  Compare only σ² at convergence,
  allowing 3·SE tolerance from the replicate study.
* **Replicate variability larger than expected** — if SD of σ²_A
  across 5 reps is > 0.15 at n=1020, MCEM is hitting local
  extrema, not a unique MLE.  Escalate to M3 step-rule work
  rather than shipping.

## Effort rollup

| Sub-milestone | Effort | Cumulative |
|---|---:|---:|
| M1.1 bump replicates + run | 1 hr | 1 hr |
| M1.2 aggregate | 0.5 hr | 1.5 hr |
| M1.3 Tool B cross-check | 2 hr | 3.5 hr |
| M1.4 findings + tag | 0.5 hr | 4 hr |

Total: half a day to one day wall-clock (compute limited).
