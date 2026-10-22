# Changelog

All notable changes to simACE are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/).
simACE uses [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`)
from Git tags via `setuptools-scm`. Before 1.0, the minor version goes up for a
change that breaks the command line, configuration keys, or result files
([ADR 0023](adr/0023-lockstep-semver.md)). Releases through 2026.09.2 used
CalVer (`YYYY.MM[.patch]`). The 0.1.0 section also lists the changes those
releases shipped after 2026.05.3.

## Unreleased

### Single-trait assortment hits its target, and each mating iteration's pairing is recorded ([#73](https://github.com/rwaples/simACE/issues/73), [#74](https://github.com/rwaples/simACE/issues/74))

- **Single-trait assortative mating now pins its mate correlation**
  ([ADR 0032](adr/0032-single-trait-pinning-and-pairing-records.md)). After
  the copula and dedup, Metropolis swaps bring the active trait's
  correlation within `METROPOLIS_TOL` of `assort`, without creating a
  duplicate couple. The copula alone fell short of the target.
- **Both-trait Metropolis proposes every couple.** With an odd number of
  slots it never proposed the last one; both kernels now share one
  proposal draw over all slots. Assortative reps change for a given seed;
  random-mating and Wright-Fisher pedigrees do not.
- **A new rep file, `mating.yaml`**, written by the simulate stage: one
  pairing record per mating iteration (mode, final errors against the
  targets, duplicates left, converged). Existing complete reps lack it,
  so `simace run` recomputes them. `simace simulate --output-mating` writes
  it, and `simace validate --mating` reads it.
- **`mate_correlation`** applies the finite-population factor on every
  assortative iteration, not only where both traits assort. It makes an
  iteration's scorable cells information when its record says the pairing missed
  its target after the swaps and dedup.

### Mate correlations per generation as a calibrated test ([#69](https://github.com/rwaples/simACE/issues/69))

- **A new calibrated test in `simace validate`**, `mate_correlation`
  ([ADR 0031](adr/0031-mate-correlation-calibrated-test.md)). Per parents'
  generation it scores the mate correlations of liability, and the two
  cross-trait ones where both traits assort, against the iteration's target.
  The SE follows the pairing mode: iid where one trait assorts, times a
  finite-population factor where Metropolis pins both. Induced cells, strata
  under 250 Mating Pairs, and scenarios with no assortment are not
  applicable; its verdict needs 20 replicates. It has its own atlas page.
- **Analyze drops its mate-correlation checks.** `quality_checks` no longer
  lists `assortative_mating.mate_corr_*` or any `mate_corr_*_gen{g}`; every
  mate correlation is reported as information. `report.yaml`'s
  `truth.recorded_pedigree.assortative_mating` is unchanged.
- **`METROPOLIS_TOL`** names the Metropolis refinement's stopping tolerance
  in `simace/simulation/assortment.py`; the value is unchanged.
- **`report_summary.tsv`** gains the test's columns, through parents'
  generation 11.
- **`config/inheritance_calibration.yaml`** gains `inh_am_mixed` and
  `inh_am_matrix`.

### Total variance and cross-trait rC as calibrated tests ([#68](https://github.com/rwaples/simACE/issues/68))

- **Two new calibrated tests in `simace validate`**
  ([ADR 0030](adr/0030-generation0-total-variance-and-rc-tests.md)), on
  recorded generation 0 with household-cluster SEs. `total_variance` scores
  Var(A + C + E) per trait against A + C + E at burn-in, on the relative
  scale. `cross_trait_rC` scores r(C1, C2) against rC. Both leave the verdict
  under any C transmission, and `total_variance` also under assortative
  mating up to burn-in; their verdicts need 20 replicates.
- **Analyze drops two pass/fail checks.** `quality_checks` no longer lists
  `statistical.total_variance_trait{t}` or `statistical.cross_trait_rC`;
  their values stay in `truth`, unchanged.
- **`statistical.c{t}_inheritance` covers recorded generation 0**: without
  paternal C transmission it groups every row by `household_id`, so a broken
  household share of C in generation 0 fails it.
- **`report_summary.tsv`** gains the two tests' columns.
- **`config/inheritance_calibration.yaml`** gains `inh_rc` (rC 0.5).

### Sibling inheritance calibrated tests ([#67](https://github.com/rwaples/simACE/issues/67))

- **Two new calibrated tests in `simace validate`**
  ([ADR 0029](adr/0029-inheritance-tests-score-residuals.md)).
  `mendelian_residual` scores, per trait with A > 0, the correlation of
  A − midparent between full sibs, maternal and paternal half-sibs (expected
  0) and its variance (expected A/2). `household_draw` scores, per trait with
  C > 0, the paternal half-sib correlation of C − b_m·C_mother − b_f·C_father
  (expected 0). Both take pg-phenotype's cluster SE; their verdicts need 20
  replicates.
- **Analyze drops three pass/fail checks.** `quality_checks` no longer lists
  `heritability.dz_sibling_A{t}_correlation`,
  `half_sibs.half_sib_A{t}_correlation` or `half_sibs.half_sib_shared_C{t}`:
  their tolerances ignored clustering. The correlations stay in `truth` and
  `estimators`, unchanged.
- **A calibrated-test cell can be `not_applicable`**, when its quantity is
  degenerate in the replicate; it is left out of the verdict's coverage
  requirement.
- **The per-replicate `simace validate` command requires `--report`**, the
  replicate's `report.yaml`, for the scenario parameters.
- **`report_summary.tsv`** gains `{test}_n_scored`, `{test}_max_abs_z`,
  `{test}_rep_bound` and `{test}_z_{cell}` columns for the two new tests;
  the onset τ columns are unchanged.
- **The scenario atlas** gains a calibrated-tests page: each replicate's z
  per cell of every test other than onset τ, under the scenario verdict.
- **pg-phenotype ≥ 0.5** is required, for the relative-pair correlation SE.

## 0.2.0 — 2026-10-19

### Quality checks and calibrated tests ([#65](https://github.com/rwaples/simACE/issues/65))

- **Analyze's check families moved to `simace.analysis.quality`**
  ([ADR 0028](adr/0028-analyze-validate-boundary.md)). Analyze runs these
  quality checks on every replicate; `simace.analysis.validate` now holds
  only `simace validate`'s on-demand checks. `build_validation_report` is
  now `build_quality_report`, and the `validate_*` functions import from
  `simace.analysis.quality`. `run_validation` is removed.
- **`validation.yaml` records cells, not a pass.** It carries a required
  `schema` block (`kind: on_demand_checks`, `version: 1`); every reader
  ignores a file without it, with a warning that names the file. Onset τ's
  `checks.onset_tau.cells` hold each cell's `z`, and its `summary` holds
  `n_scored`, `max_abs_z` and the per-replicate Bonferroni bound `z_bound`;
  the per-replicate `passed` and `n_failed` are removed, since a correct
  simulation exceeds the bound in about 5% of replicates. `simace validate`
  prints each replicate's largest |z| and bound. Every `validation.yaml`
  written before this change is ignored, including those from the onset-τ
  check below; rerun `simace validate` on those replicates.
- **`simace validate` decides verdicts across replicates.** After the
  per-replicate step it writes `results/{folder}/{scenario}/validation_summary.yaml`
  (`schema: {kind: calibration_verdict, version: 1}`): one Bonferroni family
  at a familywise 5% over the scenario's calibrated tests, with verdict
  `pass`, `fail` or `insufficient`. A folder target also writes
  `results/{folder}/validation_summary.yaml`, one family over every
  scenario's tests plus the tests pooled per cell across scenarios; the
  pooled tests are `insufficient` when two scenarios' rep seed ranges
  overlap. Onset τ's rule is the
  former `scripts/onset_tau_calibration.py` (mean z, SD z, the count of
  replicates over their per-replicate bound, and 95% coverage per cell),
  with at least 20 replicates; below that, or with no cell reported, the
  verdict is `insufficient`. A
  verdict covers every registered calibrated test and needs every complete
  replicate validated with them by one build (`validation.yaml` now records
  `source`, the checkout's `git describe`, beside `simace_version`, and
  builds compare without its `-dirty` mark); otherwise, or when a replicate
  fails validation, none is written and the command says why. A replicate
  whose `validation.yaml` is from the running build and holds the requested
  checks is skipped; `--force` validates it again, as is needed after
  uncommitted code changes. Validating a scenario without its folder removes
  the folder's verdict unless it still records that scenario's replicates
  and build, and a recompute of a
  replicate removes both verdicts. `scripts/onset_tau_calibration.py` is
  removed.
- **The verdicts reach the plots, the atlases and `simace gather`.**
  `simace validate` prints each scenario's verdict and the folder's, then
  rebuilds the plots and atlas of each scenario whose replicates are all
  complete and whose plots no longer show the current verdict, under the
  scenario's lock; `--no-plots` skips this and `--format pdf` works as in
  `simace run`. The onset-τ page states the scenario verdict above the
  per-replicate z, or why there is none, and no longer marks a replicate
  failed. `simace gather` copies the folder's and each scenario's verdict
  into `results/{folder}/verdicts.tsv` and decides none; the validation
  atlas opens with a verdict page. When a verdict changes after a gather,
  `simace validate` says to rerun `simace gather`.
- **Breaking: `simace validate` exits 3 on a failed verdict.** It used to
  exit 0 whatever the checks found. It now exits 3 when a target's own
  verdict is `fail`: the folder verdict for a folder target, the scenario
  verdict for a scenario target. A scenario's `fail` inside a folder target
  is printed but sets no exit code, since five scenario families at a
  familywise 5% each would fail a correct folder about 23% of the time.
  `insufficient` and no verdict exit 0. A replicate error, a failed plot
  rebuild or a busy lock exits 1 and takes precedence over 3; a bad target
  exits 2. Scripts that treat any nonzero exit as a crash need updating.
- **`simace validate --pedigree` requires `--checks`.** The recorded-pedigree
  debug run (`--pedigree … --params … --output …` without `--checks`) is
  removed; `simace analyze` runs those checks.

### Quality-check fixes ([#66](https://github.com/rwaples/simACE/issues/66))

- **Checks with nothing to test are informational.** A quality check that
  found too few pairs, no twins, no non-founders or no mating pairs, or an
  assortative-mating regime the equilibrium recursion does not model, used
  to report a pass, so `report.yaml`'s `quality_checks.summary` counted it
  in `checks_passed`. It is now informational, and the summary skips it.
  `twin_rate` with zero twins keeps its verdict, since it tests the
  no-twins invariant.
- **`generation_count` checks the pedigree's generation labels.** It now
  compares the distinct values of the generation column with
  `0..G_ped-1`, so a missing, extra, shifted or stray label fails; before,
  it compared a count that matched by construction. `population.n_generations`
  is unchanged for a correct pedigree.
- **`cross_trait_rA` is informational under assortative mating**, when
  either trait assorts, as `variance_A{t}` already was: assortment shifts
  the founders' A1–A2 correlation away from the configured `rA` (0.591
  against 0.5 at N = 1M in `epimight_h2_bias/am`). Its observed value, and
  `report.yaml`'s `cross_trait.rA`, are unchanged.

### Onset τ check for the frailty model ([#29](https://github.com/rwaples/simACE/issues/29))

- **`simace validate <scenario|folder>` runs on-demand checks** on complete
  replicates and writes each replicate's `validation.yaml`. These are checks
  that cost too much for every replicate or need outcomes a replicate does
  not store, so `simace run` never runs them and its cost is unchanged. It
  takes `--rep`, `--jobs`, `--checks` and `--dry-run`, and a later recompute
  of a replicate removes its `validation.yaml`. The per-replicate form takes
  explicit files (`--pedigree … --seed … --checks …` plus the phenotype
  flags).
- **The frailty model's onset times are validated** by its first on-demand
  check. For each frailty trait it compares Kendall's τ between relatives'
  raw onsets, per relationship type (MZ, FS, MHS, PHS, MO, FO), and
  within-person across the two traits, with its expectation given everyone's
  realized linear predictor. That expectation does not depend on the
  baseline hazard. The raw onsets are regenerated from the replicate's seed,
  over the whole phenotyped population before ascertainment. The cells share
  one Bonferroni family at a familywise 5% and have no tolerance floor. A
  cell over 100,000 entries is scored on a seeded subset of whole families
  ([ADR 0027](adr/0027-onset-tau-check.md)).
- `simace gather` adds `onset_tau_n_scored`, `onset_tau_max_abs_z`,
  `onset_tau_rep_bound` and one `onset_tau_z_{cell}` column per cell, empty
  for replicates not validated. The atlas gains an onset-τ page after the
  liability violins, a placeholder until the scenario's replicates are
  validated. `config/onset_tau.yaml` and `scripts/onset_tau_calibration.py`
  calibrate the check.

### Mate-correlation checks by generation ([#64](https://github.com/rwaples/simACE/issues/64))

- **Per-generation assortment and C/E are now validated.** Each check
  compares against the simulator's target for every mating whose parents
  are both recorded. A check whose pooled correlation over all Mating Pairs
  has no single target becomes one check per parents' generation,
  `mate_corr_liability{t}_gen{g}` or `mate_corr_cross_{12,21}_gen{g}`. Its
  pooled name stays as information with no pass/fail, so `report.yaml` and
  `gather` keep the realised value
  ([ADR 0026](adr/0026-mate-correlation-checks-by-generation.md)).
  The same-trait checks split when their target varies by generation. The
  cross-trait checks split when their target or C/E varies, because a
  pooled cross-trait Pearson mixes the generations' trait scales: in a
  probe with the target fixed at 0.2, the generations measured 0.20 and the
  pooled value 0.135. Per-generation `assort1`/`assort2` and per-generation
  C/E used to pass these checks unexamined. Scenarios whose targets are the
  same in every generation keep their check names.
- **Single-trait assortment no longer fails the other trait's check.** The
  same-trait target is now the expected mate correlation, including the
  part that assortment on the other trait induces (`rho_w² · assort`), not
  the configured `assort_t`. With `assort1: 0`, `assort2: 0.8` and a
  within-person correlation of 0.5, `mate_corr_liability1` failed a correct
  simulation: observed 0.259, expected 0.0, target 0.2. The targets also
  follow the `assort_matrix` diagonal, which the simulator mates on;
  `params.yaml` echoes the raw `assort1`/`assort2`.
- **The `mate_correlation` plot's expected panel** shows the same targets. A
  cell whose target varies by generation shows its range.

### Vertical transmission of C ([ADR 0025](adr/0025-vertical-c-transmission.md), [#58](https://github.com/rwaples/simACE/pull/58))

- **A child's C can take a share of each parent's C:**
  `C_child = b_m C_mother + b_f C_father + u_household`, per trait. Set it
  with `pedigree.trait{t}.C_transmission.{maternal,paternal}` in a scenario
  (`Ctm_maternal{t}` / `Ctm_paternal{t}` in `params.yaml` and the `simace
  simulate --Ctm-*` flags). Both default to 0, and with every coefficient 0
  the pedigree is bit-identical to before. The household draw tops up to
  the configured C, so C stays the total in expectation.
- **Config loading rejects coefficients outside [0, 1], `b_m² + b_f² > 1`,
  and a configured C that random-mating transmission cannot fit**, and warns
  when assortative mating is likely to overfill C. At run time a mates' C
  cross-covariance past sampling error stops the replicate; a smaller excess
  is clipped, logged, and recorded in `report.yaml`
  (`truth.recorded_pedigree.c_transmission_clips`).
- **With transmission on, `report.yaml` records the recorded pedigree's
  truth**: per-relationship realized Cov(A), Cov(C) and Cov(A, C) and the
  midparent slope, plus a `c{t}_transmission` check that regresses a child's
  C on both parents'. `params.yaml` and `run.yaml` gain the four `Ctm_*`
  keys; a `run.yaml` without them compares as 0, so finished replicates stay
  complete.
- **Reference lines are divided by V_P**, and liability-scale h² uses
  Var(A + C + E) directly, so both move for scenarios whose components do
  not sum to 1. New scenarios are in `config/c_transmission.yaml`.

### Two-trait assortative mating ([#34](https://github.com/rwaples/simACE/issues/34))

- **Mates are now drawn from a 4-variate Gaussian copula when both traits
  assort.** The old pairing hit the configured Pearson mate correlations but
  put too many pairs in the joint tails, so thresholded mate concordance
  matched a latent correlation 0.05 to 0.10 above `R_mf` at prevalence 0.1.
  Each female's ideal mate is now drawn from the Gaussian conditional given
  her two liabilities, and real males are matched one to one to those draws.
  The draw uses the realised within-person trait correlation, not the
  configured one, which cross-trait AM moves. On a regenerated
  `epimight_h2_bias/am` pedigree, the tetrachoric and Pearson mate
  correlations agree within sampling error. See
  [Methods](concepts/methods.md#assortative-mating).
- **Results change** for every scenario with both `assort1` and `assort2`
  nonzero or with `assort_matrix` set: `epimight_h2_bias/{am, am_asym,
  am_plus_c}`, `epimight_adhd_composite/{adhd_composite_v1, v2}`,
  `epimight_h2_temporal/growing_AM`, `epimight_basics/eb_realistic`,
  `family/assortative_mating_both`, `bench_scale/bench1M_am`, and
  `test/coverage_scenario`. A code change does not make a rep stale, so
  existing reps keep the old pairing until rerun with `--force`; `simace ls`
  flags reps built by another version.
- **Fewer than two mating slots no longer crash or hang** on the two-trait
  path: zero slots raised `ZeroDivisionError` and one slot looped forever.
- **Infeasible or degenerate two-trait assortment now stops the replicate**
  ([ADR 0024](adr/0024-realised-assortment-feasibility.md)). Each mating
  iteration checks the target `R_mf` against the realised within-female and
  within-male trait correlations. A conditional covariance eigenvalue below
  -1e-8 used to be clipped to 0 with a warning; it now raises an error that
  names `R_mf`, both realised correlations, the eigenvalue, and the mating
  iteration. A constant trait or collinear traits (|r| > 1 - 1e-6) within
  either sex used to fall back to the configured `rho_w`; they now raise and
  point to single-trait assortment. So does a sex whose mating slots come
  from at most two distinct parents, such as any two-slot population, where
  the realised correlation is undefined or +/-1. A configuration
  that used to finish with a warning now fails its replicate and writes no
  `run.yaml`.

### Mate correlation through pg-phenotype ([#47](https://github.com/rwaples/simACE/issues/47))

- **simACE now depends on pg-phenotype** (`>=0.4,<0.6`, locked at 0.5.0
  for this release; the first pin was `>=0.3,<0.4`). The report's
  `mate_correlation` matrix, the `assortative_mating` checks, and the genetic
  mate correlation `mu_A` behind the AM-aware expected relative correlations
  come from `pg_phenotype.assortative.mate_correlation` with `inference=False`
  (point estimates only). simACE's own pair finder is gone. The pairs are the
  same: a distinct mother and father, both pedigree rows, of at least one
  child. The estimates match the old ones to within 1e-12.
- Validation computes the mate correlations once per run. Before, every
  AM-aware check recomputed them.
- `r_ho` (the liability mate correlation behind the half-sib expectation) now
  reads the `liability{t}` column instead of summing the float32 `A`, `C` and
  `E` columns. The two differ by about 1e-9.
- An infinite liability or A value is now missing, like NaN: its pairs leave
  the cells that read it. Before, the whole cell was NaN.

### pedigree-graph 0.13

- **Moved to pedigree-graph 0.13** (`pedigree-graph>=0.12,<0.14`, locked
  at 0.13.0). Its pairwise kinship and relationship engine are faster; its
  Python API is unchanged.

### Tetrachoric SE fixed ([#48](https://github.com/rwaples/simACE/issues/48))

- **Tetrachoric SEs were too small, by 4x at r = 0 and more elsewhere.** The
  numba estimator used `n φ2² / (p00 p01 p10 p11)` as the Fisher information
  where it should have used `n φ2² Σ 1/p`. `tetrachoric_from_table` and
  `tetrachoric_corr_se` now call `pg_phenotype.correlation.tetrachoric`:
  two-step maximum likelihood with its sandwich SE, which also covers the
  estimated thresholds. On pg-phenotype's 2457-table gate the new SEs are 4 to
  2832 times the old (median 9.3), and r moves by at most 1.1e-7 on interior
  fits.
- **The SE is now `None` at a boundary fit** (an empty cell, or |r| within
  1e-3 of 1), where r is not identified. r is still reported. Before, many of
  these tables had a finite SE.
- **r moves on boundary fits.** There r is wherever the fit stops on a flat
  likelihood, and the new search runs to ±0.9999 instead of ±0.999: for
  example, (n11, n10, n01, n00) = (0, 7, 3, 100) goes from -0.938 to -0.901,
  and (10, 0, 0, 10) from 0.999 to 0.9999. Categories with an empty cell, and
  fitACE's Falconer h² from them, change accordingly.
- The SE still treats every pair as independent. Pairs that share a person,
  such as full sibs in a sibship of three or more, make it too small
  ([#55](https://github.com/rwaples/simACE/issues/55)). Treat it as a lower
  bound. Over 100 replicates of each `config/se_calibration.yaml` scenario,
  the SD of r across replicates was 0.96 to 1.39 times the mean reported SE
  (median 1.12 without shared C, 1.16 with C = 0.2), so a nominal 95%
  interval covers as little as 84%. `scripts/se_calibration.py` reruns the
  comparison.
- `se` in the report's `tetrachoric`, `tetrachoric_by_generation`,
  `tetrachoric_by_sex` and `cross_trait_tetrachoric` sections changes, and so
  do fitACE's Falconer `se_r` and `se_h2`. Existing reps keep the old SEs until
  rerun with `--force`.
- The numba tetrachoric core and `simace.core.numerics.tetrachoric_core` are
  gone. The table is still put in one orientation before fitting, so swapping
  the arrays or relabeling both cannot move a boundary r (#14).

### Stage defaults and censoring

- **`simace censor` and `simace cohort` flag defaults now match
  `config/_default.yaml`.** `--censor-age` goes from 100 to 80,
  `--death-scale` from 79.433 to 164, and `--death-rho` from 10 to 2.73. A
  hand-run stage with default flags now uses the pipeline's death model.
  `simace run` always passed these values, so its results do not change.
  `--gen-censoring` still defaults to no per-generation windows. This changes
  the command line's behaviour, so it takes a minor version (ADR 0023).
- **Removed `simace.censoring.censor.death_censor`.** Only tests called it.
  `run_censor` draws one death age per individual and applies it to both
  traits; `death_censor` drew a new age on every call.
- **Phenotype models reject `params` keys they do not use.** A misspelled
  key such as `standardise_hazard`, or another model's key such as `cip_k` on
  `frailty`, used to be ignored, so the run went ahead with that setting at
  its default. Config loading and each model's `from_config` (a hand-run
  `simace cohort`) now raise an error naming the key and listing the valid
  ones. This subsumes the old `prevalence` check for `frailty` and
  `first_passage`. Every scenario in `config/` passes. A scenario file
  outside the repo that carries a stray key now fails to load, so this takes
  a minor version (ADR 0023). The accepted keys live in
  `simace.core.phenotype_keys`, which config loading imports without numba.
- **`simace run` no longer fails off Linux.** It sized pedigree-graph's
  thread budget with `os.sched_getaffinity`, which only Linux has, so on
  macOS it raised `AttributeError` before any stage started. It now uses
  `os.process_cpu_count()`, which honours the affinity mask where there is
  one, so Linux behaviour is unchanged.
- **`simace run --max-memory` refuses off Linux.** The cap is enforced by
  a cgroup or by polling `/proc`, and macOS has neither, so the cap was
  accepted and never fired. It now exits 2 while checking arguments, before
  it resolves scenarios or takes a lock, and `--dry-run` refuses it too.
  Without a delegated cgroup the warning names `/proc` polling only when a
  cap was given.

- **`adult` with `method: cox` now clips the cumulative incidence before
  inverting it**, as `method: ltm` and the blended diagnosis already did. One
  shared helper, `logistic_cif_onset`, does the inversion. Output changes only
  when a group's latest-onset case has a rank fraction within 1e-10 of the
  prevalence (about 0.1% of prevalence values at 10 million individuals per
  group, none in tests). That one onset then comes out slightly earlier.

### Analyze stage

- **Relationship pair counts come from the moments pass.** The analysis
  sample's pairs used to be walked twice: once for the correlation moments,
  once more by `relationship_counts` for the per-category counts. One
  `relationship_moments` call over the correlation codes and every code up to
  `max_degree` now gives both. Every value in `report.yaml` and
  `plot_payload.yaml` is unchanged.
- **The gain depends on the scenario.** When the sample is the whole analysis
  pedigree (no ascertainment and `G_ped` equal to `G_pheno`), a 5M-row
  analyze stage takes about 23% less wall time with the same peak RSS.
  Otherwise, as under the default `G_ped: 6` / `G_pheno: 3`, the full
  pedigree is still counted in a second pass. There, at 1M rows, the
  relationship step is 25% faster at `max_degree: 3` and 2% faster at
  `max_degree: 5`, and its peak RSS is about 3% higher.
- **Breaking: the parent-offspring regressions in `report.yaml` drop
  `pvalue`.** Every `parent_offspring_corr`, `parent_offspring_corr_by_sex`
  and `parent_offspring_affected_corr` entry is now fitted by
  pg-phenotype's point-estimate mode, which gives no p-value; `stderr` is
  its `se_ols`. The other values and `n_pairs` are unchanged.
- **`max_degree` errors come from simACE.** A value outside 0 to 5 raises
  `ValueError`, and a bool or non-integer raises `TypeError`. These used to
  come from pedigree-graph (`max_degree_out_of_range`).
- **Aalen-Johansen curves stop at a risk-set gap**
  ([#51](https://github.com/rwaples/simACE/issues/51)). With disjoint
  `gen_censoring` windows, such as `0: [0, 30]` and `1: [40, 80]`, nobody is
  at risk between 30 and 40. The estimator used to treat the hazard there as
  zero, and if the last person at risk before the gap had an event, survival
  fell to 0 and the CIF ignored every later entrant. The disease, death, and
  survival curves are now NaN past the last exit before the gap, and the
  analyze log names the gap. No scenario in `config/` has such a gap, so their
  results are unchanged.

### Plots

- **Generation panels sort numerically.** With 11 or more phenotyped
  generations, `gen10` used to sort before `gen2` in the per-generation
  incidence, censoring, tetrachoric, and liability plots. Keys without a
  number, such as `all`, come after the numbered ones.
- **Two `simple_ltm` traits with different onset kinds** now get a two-model
  title on the atlas title page and the pipeline diagram, matching the
  equations.
- **The Aalen-Johansen incidence plot requires `true_values`** in every
  empirical cumulative-incidence payload it is given, as the empirical plot
  already did. A replicate missing them used to drop silently out of the
  true-curve average; it now fails the plot. AJ data without any empirical
  payload still renders, without the Empirical and True CIF lines.

### PLINK export ([#53](https://github.com/rwaples/simACE/issues/53))

- **`simace export-plink` writes a rep's pedigree for PLINK, GCTA and other
  pedigree tools.** It takes `--pedigree` (the rep's `pedigree.parquet`), an
  optional `--cohort`, and `-o <stem>`. It writes `<stem>.fam`,
  `<stem>.pheno` holding both traits, and `<stem>.twins` (MZ pairs, which
  PLINK has no field for). With the cohort, the files hold the analysis
  pedigree and the `.fam` PHENO is trait 1; without it, they hold the
  recorded pedigree and every phenotype is `-9`.
- An IID is `id_<id>` with the simACE id unchanged, so a missing parent is
  PLINK's `0`. Everyone shares FID `1`, because PLINK links a child only to
  parents with the same FID. Sex becomes `1` male, `2` female; case status
  `2` case, `1` control, `-9` for cohort members outside the analysis sample
  (ancestors, phenotyped or not). On a 107,105-person rep, with a one-variant
  dummy `.bed`/`.bim`, PLINK 1.9 and 2.0 loaded the `.fam` and `.pheno`
  without warnings; no test runs PLINK.

### Removed

Code that only tests reached:

- `simace.plotting.plot_heritability.plot_broad_heritability_by_generation`.
  The combined A/C plot replaced it in the atlas.
- The `greenwood=` argument of `compute_cumulative_incidence_aj` and its two
  by-sex variants ([#42](https://github.com/rwaples/simACE/issues/42)). No
  stage passed it.
- The pipeline diagram's handling of a phenotype `baseline` key, which no
  model accepts, and of a `_prev12` step that was never in the diagram.
- `plot_cumulative_incidence`'s fallback for the pre-`observed_values`
  cumulative-incidence format.
- `simace.analysis.validate.validate_effective_size`
  ([#40](https://github.com/rwaples/simACE/issues/40)). The validation runner
  never called it. Ne stays a reported statistic in `effective_size.yaml` and
  its plots, with no pass/fail check.
- The `@stage` decorator's `reads`, `writes` and `stage_name` attributes and
  its `name=` parameter ([#41](https://github.com/rwaples/simACE/issues/41)).
  No stage passed `name=`, and nothing read the attributes. Schema errors
  still name the stage after its function, without the `run_` prefix.

## 0.1.0 — 2026-10-02

### Versioning

- **The family version is SemVer from 0.1.0.** simACE, fitACE, and
  fitACE_epimight still release together at one version. fitACE pins
  `simace>=0.1.0,<0.2`, so it accepts any 0.1.x simACE and refuses 0.2.0 and
  every CalVer-era install. The CalVer tags stay in git.

### pedigree-graph 0.12

- **Moved to pedigree-graph 0.12** (`pedigree-graph>=0.12,<0.13`). Its
  kinship walk skips provable zeros and its moments arithmetic runs in the
  Rust core; neither changes a value. Across the relock from 0.11.0,
  `small_test`'s report, effective sizes, pair list, sparse GRM, inbreeding
  export and generation kinship summary are byte-identical
  (`external/pedigree-graph/tools/byte_parity.sh`).
- The test oracle `tests/analysis/moments_oracle.py` builds its table with
  `RelationshipMoments.from_exact`, since the constructor no longer takes
  the exact integers.

### Exact memory meters

- **`timing.tsv` gains `tree_peak_mb`.** Each stage runs in its own child
  cgroup of a delegated cgroup v2 root, and `tree_peak_mb` is its
  `memory.peak`: the exact peak of the stage and every process it starts,
  shared pages once, page cache included. It is empty without a delegated
  cgroup, and `simace run` says why once. `simace show` reports it per
  stage. See [Stage timing](user-guide/output-structure.md#stage-timing).
- **`max_rss_mb` is `wait4`'s `ru_maxrss` alone,** the largest single
  process's peak. It no longer takes the larger of that and a 0.1 s sample
  of the tree's summed resident memory.
- **`--max-memory` is the kernel limit `memory.max`** on each stage's
  cgroup, with swap off, so no spike slips between polls. Without a
  delegated cgroup it still polls `/proc`. See
  [ADR 0020](adr/0020-standalone-simace-cli.md).
- **Old four-column `timing.tsv` files still read.** `simace show`,
  `simace gather`, and `tools.benchmark` read columns by header name.
- **`tools.benchmark` records `cgroup_peak_kb`,** the whole run's
  `memory.peak`. Each summary names its `memory_meter` (`cgroup` or
  `sampled`), and `compare` does not compare memory across meters.
- **`tools/bench_pytest_workers.sh` adds `tree_peak_mib`,** the peak of all
  xdist workers together. It refuses a `results.tsv` with other columns.

### `simace run --until`

- **`simace run --until cohort` skips `analyze`.** Each replicate stops
  after the named stage, and its `run.yaml` records the stages run. Such a
  replicate is `partial`: a later run without `--until` resumes it at the
  next stage and keeps the earlier outputs. Plots and `simace gather` wait
  for replicates complete through `analyze`; `simace ls` lists partial
  replicates. See [ADR 0020](adr/0020-standalone-simace-cli.md).

### Python 3.14 ([#21](https://github.com/rwaples/simACE/issues/21))

- **The family runs on Python 3.14.** simACE, fitACE, fitACE_epimight and
  pedsum lock Python 3.14 in pixi and require `>=3.14`; ruff targets `py314`
  and ty checks against 3.14. pedigree-graph locks 3.14 for development but
  keeps its published `>=3.13` floor and its one `cp313-abi3` wheel, which
  CI now installs on both 3.13 and 3.14. Every product of simACE's smoke
  scenario and the fitACE exports is byte-identical to the 3.13 build.
- **ruff's py314 style is applied.** The formatter drops the parentheses
  from multi-exception `except` clauses (PEP 758), and imports used only in
  annotations moved under `TYPE_CHECKING` (annotations are lazy under
  PEP 649).
- **`TY_PYTHON_VERSION` is gone from `tools/family_repos.py`.**
  `tests/test_python_floor_consistency.py` now checks that each check
  unit's ty `python-version` and ruff `target-version` match its own
  `requires-python` floor, so units can hold different floors.

### Two Parquet files per replicate ([ADR 0021](adr/0021-two-canonical-replicate-parquets.md))

- **A replicate holds `pedigree.parquet` and `cohort.parquet`.**
  `pedigree.parquet` is now the recorded pedigree, formerly
  `pedigree.full.parquet`. `cohort.parquet` has one row per member of the
  analysis pedigree, with censored outcomes for the analysis sample.
  `simace.core.cohort.selected_views` rebuilds the analysis pedigree and
  analysis sample. `pedigree.full.parquet`, `trait.raw.parquet`,
  `trait.full.parquet`, and `trait.parquet` are no longer written.
- **`simace run` has three per-replicate stages: simulate, cohort,
  analyze.** The new `simace cohort` runs phenotype, censoring, and
  ascertainment in one process and also writes
  `phenotyped_population.yaml`, which `simace analyze` reads for the
  phenotyped-population scope. `simace analyze` takes `--pedigree`,
  `--params`, `--cohort`, and `--phenotyped-population`. The standalone
  `simace phenotype`, `censor`, and `ascertain` commands are unchanged.
- **Results layout marker.** `run.yaml` records `layout: 2`, and both
  Parquet files carry the metadata `simace_layout=2`. Existing replicates
  read as stale; `simace run --force` recomputes them and deletes the old
  files.
- **`scopes.phenotyped_population.n_generations`** in `report.yaml` now
  counts phenotyped generations; it was 1 before.
- **Hard cut for fitACE.** fitACE and fitACE_epimight move to the new pair
  in lockstep; no command writes the old one.

### `simace` CLI replaces Snakemake ([ADR 0020](adr/0020-standalone-simace-cli.md))

- **`simace run <scenario>`** runs every replicate through simulate,
  phenotype, censor, ascertain, and analyze, then the scenario plots and
  atlas. **`simace gather <folder>`** writes `report_summary.tsv` and the
  validation atlas. `simace show` and `simace ls` inspect config and
  replicate state. `Snakefile`, `workflow/`, the Snakemake adapter, and the
  Snakemake, SLURM-plugin, and snakefmt dependencies are removed.
- **Replicate-level resume.** Each replicate's `run.yaml` records the config
  it was computed from. Reruns skip matching replicates, recompute
  interrupted ones, and refuse changed ones unless `--force`.
- **One `simace` console script** with a subcommand per stage replaces the
  ten `simace-*` scripts. Stage subcommands take explicit paths and never
  read config. `simace simulate` no longer writes `params.yaml`;
  `simace run` does.
- **`params.yaml` gains `G_pheno`.** No key is removed or renamed.
- **`timing.tsv`** per replicate records each stage's wall time and peak RSS;
  `report_summary.tsv`'s `simulate_seconds` / `simulate_max_rss_mb` and
  `tools.benchmark` read it. `trait.raw.parquet` and
  `plotting_sample.parquet` are no longer deleted.
- **Scripts moved.** Example comparison scripts are in `scripts/examples/`
  and gene-drop scripts in `scripts/gene_drop/`, each with its own argparse
  CLI. `simace run` refuses `use_gene_drop` / `drop_from` scenarios.

### Relationship moments ([ADR 0022](adr/0022-relationship-moments.md))

- **Analyze computes every pair statistic from pedigree-graph 0.11
  relationship moments**, one engine pass per graph, with no pair list.
  `simace.analysis.stats.relationship_moments_for(df, source)` is the entry
  point; the six pair-correlation functions in `simace.analysis.stats`
  (`compute_liability_correlations`, `compute_affected_correlations`,
  `compute_tetrachoric`, `compute_tetrachoric_by_generation`,
  `compute_cross_trait_tetrachoric`, `compute_tetrachoric_by_sex`) take
  `moments=` instead of `pairs=` and lose their unused `seed=`.
  `tetrachoric_from_table(n11, n10, n01, n00)` joins the public
  primitives. Descriptive statistics keep parity: counts and 2×2 tables
  are identical, correlations agree within 1e-10.
- **Validate is exact over all sibling pairs.** The 5,000-pair cap and its
  seeded subsample are gone; `n_pairs` is the true count and the
  tolerances are evaluated with it. Sibling correlations on large
  scenarios move to their all-pairs values (up to 0.03 on `bench1M`).
  `validate_half_sibs` and `validate_heritability` take the sibling
  moments (`simace.analysis.validate._common.sibling_moments`) instead of
  relationship pairs. The offspring-with-sibling and
  offspring-with-maternal-half-sib counts come from household sizes.
- The stats runner's tetrachoric `ThreadPoolExecutor` is removed; the
  Analyze log times each graph build and the moments pass.

### pedigree-graph 0.11

- **Moved to pedigree-graph 0.11** (`pedigree-graph>=0.11,<0.12`). MZ
  co-twins now take part in sibling groups (pedigree-graph#29): a co-twin
  has FS, MHS and PHS pairs with its non-twin siblings, and the collateral
  codes built from sibling lists (Av, HAv, GAv and further) follow.
  `relationship_pair_counts` grows on every scenario with twins, in both
  `analysis_sample` and `analysis_pedigree`; on `small_test` rep1 the
  sample's FS goes 2348 → 2378, MHS 1426 → 1456, PHS 1010 → 1034, Av
  6197 → 6261, and on `bench1M` FS 619322 → 625802, MHS 307858 → 315498,
  PHS 306709 → 314515. Statistics computed over those pairs move with them.

### Performance benchmarks

- Replaced the mutable shell benchmark scripts with
  `pixi run python -m tools.benchmark`. Each invocation now writes an immutable,
  schema-versioned run directory with git, environment, input, cache, ordering,
  and thread provenance. The runner isolates memory sampling to its process
  group and distinguishes GNU time, largest-process, and summed concurrent RSS.
- Added compatible-run comparison with configurable wall-time and memory gates,
  explicit warm and cold Numba-cache policies, seeded rotating scenario order,
  a small and a release profile, and a manually dispatched smoke workflow.

### pedigree-graph 0.8

- **Migrated to pedigree-graph 0.8** (`pedigree-graph>=0.8,<0.9`). Relationship
  pairs come from `PedigreeGraph.relationship_pairs`, ascertained samples are
  `PedigreeGraph.view(ids=...)` rows, nominal kinship reads
  `RELATIONSHIPS[code].nominal_kinship`, and effective size runs through
  `pedigree_graph.effective_size.estimate_effective_sizes`. Behaviour that
  moves with it, measured by `tools/pg08_migration_diff.py` in the pedigree-graph repo against the 0.7.1
  baseline in its `docs/pedigree-graph-0.8-migration/` evidence tree:
  - `relationship_pair_counts` reports each pair under its one closest
    category, so a pair that 0.7.1 listed under two codes counts once; a code
    outside the requested depth is `null` ("not computed"), never `0`.
  - `effective_size.yaml` always carries all eight estimator keys. An
    estimator that was not requested (`ne_coancestry` under the default
    `skip_ne_coancestry`) or that the library refused holds
    `{reason, code, fields}` with no `ne`; the validator passes
    `not_requested` and fails `missing_metadata`, naming the code.
  - Effective-size arrays are indexed by observed generation label
    (`generations`, `parent_generations`, `transition_from`/`transition_to`)
    rather than a dense `0..max(label)` range, and the by-generation figure
    plots those labels.
  - Pair blocks are canonically sorted, so validation correlations that
    subsample pairs draw a different subset within their tolerance bands.
  - Ne_C's mean-kinship prerequisite is the pinned float32 recurrence of
    pedigree-graph ADR 0009, so its last digits may differ.
- The 0.7.1 old-API scanner (`tools/pedigree_graph_old_api.py`), its snapshot,
  and the guard test are deleted; the 0.8 root namespace makes any removed name
  an `ImportError`.

### Effective population size

- **`analysis.skip_ne_coancestry` now defaults to `true`.** The coancestry DP
  dominates the `effective_size` rule's memory — roughly 31x the per-individual
  slope of the other seven estimators — so the default put 152 of 331 scenarios
  over a 24 GB budget, up to 150 GB for a single job. The remaining seven Ne
  estimators are unchanged and `ne_coancestry` is reported as null, the same
  shape the slot already took when skipped. Scenarios needing Ne_C opt back in
  with `analysis.skip_ne_coancestry: false`; the `ne_coancestry_stream` folder
  already does. The Python API default is unchanged (`compute_effective_size`
  and `main` still take `skip_ne_coancestry=False`), staying in sync with
  upstream `pedigree_graph.compute_all_ne`.
- **CLI flag renamed to the positive form.** `--skip-full-kinship-matrix` is
  replaced by `--ne-coancestry`, an opt-in that defaults to off — matching both
  the new pipeline default and pedsum's flag of the same name. The old
  double-negative flag is gone; `--ne-coancestry` now turns Ne_C *on* rather
  than a `--skip-…` flag turning it off.

### Phenotype models

- **`simple_ltm` is now a phenotype model**, not a parallel output. It sets case
  status by a probit liability threshold at prevalence `K`, then assigns an
  age-of-onset via an `onset` sub-model (`{kind: fixed, age}` or
  `{kind: normal, mean, sd}`), and flows through the standard
  `phenotype → censor → ascertainment` pipeline like every other model. Select
  it with `phenotype.trait{N}.model: simple_ltm`.
- **Parallel `trait.simple_ltm.*` outputs removed** (ADR 0011 amendment). The
  `phenotype_simple_ltm` rule, `simace.phenotype.threshold`, the `simple_ltm`
  trait kind, and the third ascertainment branch are gone; `run_ascertainment`
  now returns `(pedigree, trait)`.
- **fitACE LTM stats repointed.** Falconer h², tetrachoric/prevalence stats, and
  the former `*.simple_ltm` plots now read `affected1`/`affected2` from the
  censored `trait.parquet` for every scenario and are emitted as
  **observed-binary** outputs (`observed_binary_stats.yaml`,
  `*.observed_binary` plots).

## 2026.05.3 — 2026-05-30

Headline: **Analyze replaces separate Validate and Stats stages.** It writes
`report.yaml` v2 plus `plot_payload.yaml`, adds applied scenario suites, and
cleans up plotting/relationship semantics.

### Analyze (ADR 0008)

- **Merged stage.** `simace-analyze`, `simace.analysis.analyze`, and the
  `analyze` rule run validation, full-population summaries, and sample stats in
  one pass.
- **Report v2.** `report.yaml` now uses scoped groups (`schema`, `replicate`,
  `inputs`, `scopes`, `quality_checks`, `truth`, `observed`, `estimators`);
  dense plot arrays moved to `plot_payload.yaml`.
- **Old outputs removed.** `validation.yaml`, `stats_report.yaml`, and
  `validation_summary.tsv` were replaced by `report.yaml`, `plot_payload.yaml`,
  and `report_summary.tsv`.
- **Ascertainment comparison.** `trait.full.parquet` is retained so reports
  compare pre- and post-ascertainment populations.

### Added

- **Applied scenario suites.** Added `neurodev`, `neurodev_ltm`,
  `paper_simulation`, and `paper_simulation_ltm` for ADHD/ASD/ID and
  validation-study runs.
- **Bias grid.** Added 64-cell `config/epimight_onset_censoring.yaml` sweep over
  lifetime prevalence, onset midpoint, and right-censoring.
- **Frailty calibration.** Added `scripts/calibrate_frailty_scale.py` to tune
  baseline scales to target gen-0 prevalence.

### Changed

- **Expected correlations centralized.** Correlation plots now use
  `core.relationships.expected_liability_corr` instead of hard-coded formulas;
  unknown relationship types now error.
- **Plot atlases registry-driven.** Phenotype, validation, and effective-size
  atlases now dispatch through renderer registries covered by manifest tests.

### Fixed

- `death_censor()` no longer mutates inputs.
- `gather` handles non-pattern paths.
- Mortality plot labels fit at very low death rates.

### Testing, workflow & environment

- Added CLI, edge-case, report, registry, and shared-parquet coverage.
- Added no-op stage performance checks and a 30 GB workflow `mem_mb` cap.
- Pinned pandas `<3` and adopted CIF terminology.
- Dropped unused `jupyter` / `notebook` from the conda env files.

## 2026.05.2 — 2026-05-20

Headline: **Unified ascertainment, Wright-Fisher mating, K-free Ne_C, and grouped
stats reports.**

### Added

- **Unified ascertainment (ADR 0001).** Dropout and case-weighted `N_sample`
  selection now run in one post-censor stage.
- **Wright-Fisher mating (ADR 0002).** Added `mating_model: wright_fisher` with
  two retained sexes, independent parent draws, no persistent pairs, and no MZ
  twins.
- **Ascertainment-bias example** plus pedigree-filter and effective-size
  analysis modules.
- **Ne_C benchmark scenarios.** Added `stream100K` and `stream500K` in
  `config/ne_coancestry_stream.yaml`; `stream100K` runs in ~73 s per replicate.
- `ne_coancestry(..., theta_per_gen=...)` can use pre-streamed per-generation
  mean kinship and skip K construction/DP.

### Changed

- **Grouped stats reports.** Replaced `phenotype_stats.yaml` and
  `phenotype_samples.parquet` with `stats_report.yaml` and
  `plotting_sample.parquet`; no compatibility reader. Superseded by the
  Unreleased Analyze merge.
- **K-free Ne_C.** `compute_all_ne` streams per-generation mean kinship via
  `PedigreeGraph.per_gen_mean_kinship()` instead of building sparse `K`; at
  N=100K peak RSS drops ~30% (12→9 GB), and the path avoids CSC index overflow
  above ~3M.
- **Config flag rename.** `skip_full_kinship_matrix` is now
  `skip_ne_coancestry`; no alias.
- **pedigree-graph pinned to `v0.5.1`.** Adds int64 DP `row_start` support and
  the streaming-θ kernel.
- **Python pinned to `3.13`** with Snakemake/SLURM plugin updates.

## 2026.05.1 — 2026-05-07

### Highlights

1. **Phenotype model architecture.** Replaces much of the monolithic
   phenotype dispatch with a `PhenotypeModel` ABC and separate model
   modules for `frailty`, `cure_frailty`, `adult`, and `first_passage`.
2. **Prevalence config moved into model params.** Threshold-style
   prevalence is now model-owned under
   `phenotype.traitN.params.prevalence`, with migration tooling and
   validation for old top-level prevalence keys.
3. **Hazard and standardization overhaul.** Adds a hazard registry,
   three-way liability standardization (`none`, `global`,
   `per_generation`), and per-trait `standardize_hazard` overrides for
   hazard-bearing models.
4. **Hierarchical config support.** Config loading now supports
   sectioned YAML (`pedigree`, `phenotype`, `censoring`, `sampling`,
   `analysis`, `tstrait`) while still flattening internally for
   workflow use.
5. **Snakemake wrapper simplification.** Adds shared Snakemake adapter
   utilities, reduces repeated wrapper boilerplate, splits simulation
   into pedigree/params stages, and adds an explicit `emit_params`
   path.
6. **Documentation + repo cleanup.** Moves docs fully into MkDocs,
   adds concept/user-guide/example pages, refreshes API docs, removes
   stale standalone docs and the public `notes/`, adds rule graph
   generation, and updates README/setup docs for Python `>=3.13`.
7. **Plotting and atlas refactor.** Introduces an atlas manifest,
   collapses plot dispatch, updates captions, redesigns
   tetrachoric/reference panels, and improves plot styling and example
   figures.
8. **Pipeline schema contracts.** Adds explicit DataFrame schema
   contracts for pedigree → phenotype → censor/sample handoffs, plus a
   `@stage` decorator to enforce and expose stage input/output
   metadata.
9. **PedigreeGraph externalized.** Removes internal pedigree graph and
   kinship-kernel code in favor of the standalone `pedigree-graph`
   package pinned at `v0.2.0`.
10. **Stats package split.** The old `simace/analysis/stats.py` was
    split into focused modules: correlations, tetrachoric, incidence,
    censoring, pedigree, sampling, effective size, and runner
    orchestration.
11. **Effective population size estimators.** Adds per-replicate Ne
    summaries, theoretical expectations for the ZTP mating model, and
    validation/reference tests for those estimators.
12. **Gene-drop + tstrait pipeline.** Adds a full tskit/tstrait branch
    for realistic genotype inheritance: preprocessing SimHumanity
    trees, fixed-pedigree drops, causal-effect assignment,
    genetic-value calculation, and `A1` augmentation.

Secondary theme: test coverage expanded across config loading,
phenotyping models, tskit/tstrait, stats, core schema/stage helpers,
workflow scripts, and plotting.

### Added

- MkDocs documentation site with Material theme
- Three-way liability standardization: `standardize` now accepts `none`,
  `global`, or `per_generation` (replacing the previous boolean flag,
  which is still accepted via a back-compat shim with `true → "global"`
  and `false → "none"`). Default is `"global"`.
- Per-trait `standardize_hazard` override inside
  `phenotype.trait{N}.params` for the four hazard-bearing model families
  (`frailty`, `cure_frailty`, `first_passage`, and `adult` with
  `method: cox`). Defaults to `None` → inherit from the global
  `standardize` flag. `cure_frailty` is the only family that honors both
  knobs independently (threshold step + hazard step). See [ACE Model §
  Standardisation](concepts/ace-model.md#standardisation).
- New `e_*_pergen` example scenarios (`e_flat_pergen`,
  `e_rise_mild_pergen`, `e_rise_steep_pergen`, `e_fall_steep_pergen`)
  matching the existing `_std`/`_nostd` E-trajectory pairs but with
  `standardize: per_generation`. The
  `docs/images/examples/increasing_e/prevalence_drift.png` figure now
  shows three lines per panel (`global` / `none` / `per_generation`)
  instead of two.

### Changed

- `phenotype.threshold.apply_threshold` now standardizes liability **once**
  outside the per-generation loop using the chosen mode. Under
  `standardize=true` (now `"global"`), this changes behaviour for any
  scenario whose `apply_threshold` call previously relied on per-gen
  standardization to preserve K (including the `phenotype_simple_ltm`
  benchmark output produced for every scenario). Set
  `standardize: per_generation` in the scenario YAML to restore exact
  per-gen prevalence preservation.
- `_apply_threshold_sex_aware` now z-scores liability once across both
  sexes (per the chosen mode) before applying per-(sex, gen) prevalence
  thresholds. Sex-shifted liability means now translate into
  sex-specific realised prevalences within each generation.
- `--standardize` CLI flag changed from `action="store_true"` (a
  no-op flag that could not be turned off) to
  `choices=["none", "global", "per_generation"]` with default
  `"global"`.
- `simace.plotting.compare_scenarios.compare_prevalence_drift` now
  accepts an optional third series via
  `pergen_paths_per_trajectory` / `pergen_label`; the existing two-series
  call signature is unchanged.
- `simace.phenotype._prototypes.bimodal_phenotype` (`phenotype_mixture_cip`,
  `phenotype_mixture_cure_frailty`, `phenotype_two_threshold`) ported to
  the mode-aware standardize API (`StandardizeMode | bool`, with
  `per_generation` rejected since these prototypes don't take a
  `generation` array). The same `cure_frailty` raw-vs-standardized-L
  fix applies here too.

### Fixed

- The phenotype-stage `--standardize` CLI flag could previously not be
  set to `false` from the command line.
- `cure_frailty` was passing the threshold-standardized liability `L_z`
  to its hazard kernel where the kernel expects raw liability. Combined
  with the `mean` and `scaled_beta` derived from the raw liability, this
  produced a hazard of
  `exp((beta/std)·(L_z − mean))` instead of the intended
  `exp((beta/std)·(L_raw − mean)) = exp(beta·z_score(L_raw))`.
  Equivalent to a `1/std²` instead of `1/std` scaling and an extra
  constant offset of `−beta·mean/std`. Silent under the default
  `A + C + E = 1` ACE configurations (where liability has mean ≈ 0 and
  std ≈ 1) but real for any scenario with non-zero-mean or
  non-unit-variance liability — including all per-generation `C`/`E`
  configurations. After the fix, the case-onset distribution under
  `standardize="global"` is invariant to additive shifts and
  multiplicative scales of liability, as it should be.
