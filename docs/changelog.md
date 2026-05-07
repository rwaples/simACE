# Changelog

All notable changes to simACE are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/).
simACE uses [CalVer](https://calver.org/) versioning (`YYYY.MM`) derived from git tags.

## Unreleased

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
- `simace.phenotyping._prototypes.bimodal_phenotype` (`phenotype_mixture_cip`,
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
