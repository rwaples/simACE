# Changelog

All notable changes to simACE are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/).
simACE uses [CalVer](https://calver.org/) versioning (`YYYY.MM[.patch]`) from
Git tags via `setuptools-scm`.

## Unreleased

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

### Analyze (ADR 0006–0008)

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

- **Grouped stats reports (ADR 0003).** Replaced `phenotype_stats.yaml` and
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
