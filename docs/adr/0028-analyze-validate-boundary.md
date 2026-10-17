# ADR 0028: Quality checks in Analyze, calibrated tests in `simace validate`

## Status

Accepted 2026-10-17 for issue #65, decided in a design interview; plan in
`plans/issue-65-analyze-validate-boundary.md` (session-local draft). Amends
[ADR 0027](0027-onset-tau-check.md) §1 and §8: onset τ keeps its cells, target
and SE, but its per-rep pass/fail gives way to a verdict across reps. Amends
[ADR 0008](0008-curated-analyze-report.md): the name `validation.yaml`, retired
there, returns for a different file. Gives [ADR 0026](0026-mate-correlation-checks-by-generation.md)
its deferred multiplicity rule, which arrives with the per-generation
mate-correlation calibrated test.

## Context

ADR 0027 introduced `simace validate`, which runs on-demand checks on
complete reps, and placed onset τ there on cost: about 2.4 s per rep at
`bench500K` and 4.6 s at `bench5M` when it ran in the cohort stage. Its
registry was described as the home for any check "that costs too much for
every rep or needs outcomes a rep does not store". Issue #65 asked which other
checks should follow it.

Cost turned out to be the wrong axis.

- **Measured cost.** Analyze phase 1 on `bench5M` (5,000,000 individuals;
  medians of 3 processes, 4.19 GHz) takes 8.38 s of the rep's 41.1 s. Its
  largest piece, `sibling_moments` (2.39 s), is a shared precompute that also
  feeds `truth`. The families whose output feeds only `quality_checks` cost
  0.11 s (`c_transmission`), 0.21 s (`structural`) and 0.00 s
  (`am_equilibrium`). `recorded_pedigree_truth` returns early without C
  transmission; forced on for the same pedigree it took 6.20 s and raised
  peak RSS from 1.93 to about 2.75 GB, but it computes truth, not checks.
  Nothing in Analyze is worth moving on cost.
- **Analyze's families produce more than checks.** `extract_truth` and
  `extract_estimators` (`simace/analysis/report.py`) read realized variances,
  per-generation moments, family structure, mate correlations and the
  liability-scale heritability estimators from the same family results. A
  family cannot leave Analyze; only its pass/fail can.
- **Two kinds of pass/fail.** Most Analyze checks are invariants or
  gross-error floors such as `max(0.05, 4·SE)`, so a failure means the rep
  is wrong. Onset τ is different by design: its Bonferroni z fails about 5%
  of correct reps (ADR 0027 §8), and its acceptance was a script,
  `scripts/onset_tau_calibration.py`, testing mean z, SD z and the count of
  failing reps across 500 reps. Its per-rep `passed: false` reads like a
  broken rep and is not one.
- **Some Analyze checks are mis-sized.** An audit of the current checks
  found several whose SE ignores clustering within sibships and households:
  `half_sib_shared_C{t}` (RMS z 3.8 over the current-code reps in
  `results/test`; `coverage_scenario` rep1 fails at −0.125 against a
  tolerance of 0.080), `half_sib_A{t}`, `dz_sibling_A{t}`, and at about 3
  effective SE near N = 1000 `total_variance_trait{t}`, `cross_trait_rC` and
  the per-generation `mate_corr_*_gen{g}` checks. At `bench5M` their 0.05 or
  0.1 floors bind and they are gross; at small N they fail correct reps.

## Decision

1. **The rule.** A check's placement follows what its failure means, not its
   cost.
   - A **quality check** fails only when the rep is wrong: an invariant, or a
     gross-error floor that a correct simulation essentially never crosses.
     Analyze runs quality checks on every rep. Analyze also writes all of
     `truth`, `observed` and `estimators`.
   - A **calibrated test** has a designed false-positive rate, so one rep's
     result is one draw. It belongs to `simace validate`, whatever it costs.
   - An **on-demand check** is anything `simace validate` runs: every
     calibrated test, plus a quality check that needs outcomes a rep does not
     store, or that costs too much for every rep. Cost decides only this last
     case.
2. **Verdicts are across reps.** A calibrated test stores per-cell results
   in each rep's `validation.yaml` (observed, expected, SE, z, scored or
   informational with a reason) and no per-rep pass. Each test declares a
   rule over its cells across reps and a minimum number of reps; below it the
   verdict is `insufficient`, never pass. Onset τ's rule is the former
   script's: mean z within `Φ⁻¹(1 − α/2)/√n`, `SD²·(n − 1)` within the
   χ²ₙ₋₁ quantiles, the count of reps whose largest |z| exceeds the per-rep
   Bonferroni bound at most the binomial(reps, 0.05) quantile at 1 − α, and
   every cell scored in at least 95% of reps; minimum 20 reps. An on-demand
   quality check keeps a per-rep pass.
3. **Two verdict levels and their families.** `simace validate` writes
   `results/{folder}/{scenario}/validation_summary.yaml`, one familywise-5%
   Bonferroni family over that scenario's tests, the same whatever the
   target. For a folder target it also writes
   `results/{folder}/validation_summary.yaml`: one familywise-5% family over
   every scenario test plus tests pooled per cell across scenarios. That is
   ADR 0027's acceptance, and the folder verdict is the one to cite. Pooling
   is refused when two scenarios' rep seed ranges overlap.
4. **Which reps.** A verdict covers every complete rep of the scenario, all
   validated by one simace version. `simace validate` skips a rep whose
   `validation.yaml` is schema-valid, from the current version and holds the
   requested checks, and (re)validates any other. A recompute of a rep
   deletes its `validation.yaml` and both summaries.
5. **Files.** `validation.yaml` and `validation_summary.yaml` each carry a
   required `schema` block (`kind`, `version`), and every reader refuses a
   file without it. `results/` holds 315 `validation.yaml` files from before
   ADR 0008 whose top-level `summary.passed` would otherwise read as a
   verdict.
6. **Surfacing.** `simace validate` prints each verdict, then rebuilds each
   validated scenario's plots and atlas (`--no-plots` skips it); the
   scenario atlas shows the verdict beside the per-rep z. `simace gather`
   copies per-rep z columns and the verdicts and never decides; its
   validation atlas gets a verdict page. The `validation_passed` column is
   removed. `simace validate` exits 3 when a target's own verdict is
   `fail`: the folder verdict for a folder target, the scenario verdict for
   a scenario target. A scenario `fail` inside a folder target is printed
   but sets no exit code, since each scenario family is at a familywise 5%
   and five of them would fail a correct folder about 23% of the time.
   `insufficient` exits 0 and is printed as such. A rep error, a failed plot
   rebuild or a busy lock exits 1 and takes precedence over 3, and usage
   errors keep 2.
   Reps are seeded, so a `fail` repeats until the code or the seeds
   change.
7. **The registry contract.** A calibrated test is an optional per-rep step,
   which reads the rep's inputs (pedigree, params, seed, phenotype arguments,
   `report.yaml`) as it needs them and returns cells, plus the rule across
   reps. A test with no per-rep step reads its cells from `report.yaml`.
8. **Package.** Analyze's check families move from `simace/analysis/validate/`
   to `simace/analysis/quality/`. `simace/analysis/validate/` keeps only
   on-demand checks, calibrated tests and verdicts. `simace validate` without
   `--checks`, a debug run of Analyze's checks, is removed; `simace analyze`
   covers it.
9. **What moves, and when.** The mis-sized checks listed in Context become
   calibrated tests, each with a cluster-robust SE and its own multi-rep
   acceptance, one issue per group. Analyze keeps their values in `truth` and
   `estimators` and drops their pass/fail in the same change that adds each
   calibrated test, so no quantity is ever left untested. The tetrachoric SE
   calibration (`scripts/se_calibration.py`, #55) becomes a calibrated test
   with no per-rep step as part of #60, which changes the SE it checks.
   Checks on regenerated pre-ascertainment outcomes (censoring and prevalence
   on the full phenotyped population) wait until one is proposed.

## Rejected alternatives

- **Placement by cost.** ADR 0027's rule. It would leave calibrated tests in
  Analyze whenever they are cheap, and nothing in Analyze is costly enough
  to move.
- **A per-rep pass for calibrated tests**, kept or renamed beside the
  verdict. A correct 100-rep folder would show about five failed reps.
- **Each verdict file its own 5% family.** A 5-scenario folder's six
  verdicts would raise at least one false alarm in about 26% of correct runs.
- **One folder-wide family everywhere.** A scenario's verdict would then
  depend on whether it was validated alone or with its folder.
- **Verdicts mixing simace versions.** An acceptance could silently combine
  reps validated before and after a fix.
- **Verdicts in `simace gather`.** It is the descriptive aggregator, and
  would decide on every gather.
- **A cheap fixed floor in Analyze for each moved check.** A floor wide
  enough for N ≈ 1000 catches only gross wiring bugs, and keeps two checks
  on one quantity.
- **Widening the mis-sized checks to fixed floors in place.** No power at
  small N, and the reported SE stays wrong.
- **New file names** (`on_demand.yaml`, `calibration.yaml`). The command is
  `simace validate`; the schema block removes the collision with old files.

## Consequences

- `simace run` costs nothing more, and a plain run reports only quality
  checks, whose failures mean something is wrong.
- `scripts/onset_tau_calibration.py` is deleted once the folder verdict
  reproduces its 161 tests on the same reps.
- A scenario with one rep, such as `bench5M`, gets per-rep z but only an
  `insufficient` verdict.
- `external/pg-phenotype/tools/consumer_simace.py` and
  `make_simace_po_regression_golden.py` import from
  `simace.analysis.validate`; the package move needs a matching
  pg-phenotype change.
- A recompute or a new simace version means re-validating before a verdict
  exists. At N = 20,000 that is minutes for 500 reps with `--jobs 8`.
