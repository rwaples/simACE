# Changelog

All notable changes to simACE are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/).
simACE uses [CalVer](https://calver.org/) versioning (`YYYY.MM`) derived from git tags.

## Unreleased

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
