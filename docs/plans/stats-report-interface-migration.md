# Stats Report Interface Migration And Runner Refactor

## Summary

Migrate the Stats stage to grouped report artifacts: `stats_report.yaml` and `plotting_sample.parquet`. This is a breaking Interface change, shipped as a simACE PR followed immediately by a fitACE PR that updates its hard-coded simACE target path.

## Contract

- `build_stats_report(...) -> dict` returns the grouped report shape directly. No flat in-memory report.
- `compute_observed_h2_estimators(...)` accepts affected-correlation inputs directly, not a flat stats dict.
- Top-level groups are exactly: `metadata`, `incidence`, `censoring`, `pedigree`, `correlations`, `heritability`.
- Key migration:
  - `n_individuals`, `n_generations`, `case_ascertainment_ratio` -> `metadata.*`
  - `prevalence`, `mortality`, `regression`, all `cumulative_incidence*` -> `incidence.*`
  - `censoring`, `censoring_confusion`, `censoring_cascade`, `person_years` -> `censoring.windows`, `censoring.confusion`, `censoring.cascade`, `censoring.person_years`
  - `family_size`, `pair_counts`, `parent_status` -> `pedigree.family_size`, `pedigree.relationship_pair_counts`, `pedigree.parent_status`
  - `pair_counts_ped`, `n_individuals_ped`, `n_generations_ped` -> `pedigree.full.relationship_pair_counts`, `pedigree.full.n_individuals`, `pedigree.full.n_generations`
  - `liability_correlations`, `affected_correlations`, `parent_offspring_corr`, `parent_offspring_corr_by_sex`, `parent_offspring_affected_corr`, `mate_correlation`, all `tetrachoric*`, `cross_trait_tetrachoric`, `joint_affection` -> `correlations.*`, preserving old leaf names such as `mate_correlation`
  - `observed_h2_estimators` -> `heritability.observed_h2_estimators`
- No-pedigree behavior: always emit `pedigree.parent_status.phenotyped`; omit `pedigree.parent_status.in_pedigree`, `pedigree.full`, and `correlations.mate_correlation` when no full pedigree path is provided.

## Implementation Changes

- Rename workflow surface:
  - `rule stats_phenotype` -> `rule build_stats_report`
  - `workflow/scripts/simace/compute_phenotype_stats.py` -> `workflow/scripts/simace/build_stats_report.py`
  - `phenotype_stats.yaml` -> `stats_report.yaml`
  - `phenotype_samples.parquet` -> `plotting_sample.parquet`
  - `phenotype_stats.log/.tsv` -> `stats_report.log/.tsv`
- Update Plot file readers to require grouped reports. Plot internals may use a local rendering view, but old flat report files are not accepted.
- Validate stage and `validation.yaml` stay unchanged.
- Read full `pedigree.parquet` with projected columns: `id`, `mother`, `father`, `twin`, `sex`, `generation`, `liability1`, `liability2`.
- Add timing logs around input load, incidence, censoring/person-years, relationship context, pedigree summaries, fast correlations, tetrachoric block, YAML write, and plotting sample write.

## Rollout

- Add ADR `docs/adr/0003-stats-report-interface.md` explaining the breaking artifact rename, grouped schema, no compatibility reader, and fitACE impact.
- ADR rationale includes: `joint_affection` moves under `correlations` because it is a bivariate Trait 1 x Trait 2 affection summary, even though its current function lives in `incidence.py`.
- ADR non-goals: no Validate-stage change, no fitACE behavior change beyond target-path migration, no one-cycle dual emit.
- fitACE scope is currently one path: `fitACE/workflow/common.py` changes `phenotype_stats.yaml` to `stats_report.yaml`.
- Merge order: simACE first, then fitACE immediately after. fitACE CI/build failures in the gap are expected and signal the required coordinated migration.
- Operator regeneration guidance:
  - remove stale old artifacts: `find results -name 'phenotype_stats.yaml' -delete` and `find results -name 'phenotype_samples.parquet' -delete`
  - rerun the renamed rule or scenario targets so Snakemake creates `stats_report.yaml`, `plotting_sample.parquet`, plots, and atlases.

## Test Plan

- Stats runner tests assert the six top-level groups exist and old flat keys are absent.
- Add builder tests for grouped in-memory shape, no-pedigree shape, gen-censoring shape, and `compute_observed_h2_estimators(...)` input change.
- Update Plot smoke/helper coverage to exercise grouped report loading at file boundaries.
- Add target-generation checks for new filenames in simACE and fitACE workflow common helpers.
- Run:
  - `pytest tests/analysis/test_stats_runner.py`
  - `pytest tests/plotting/test_plot_smoke.py tests/plotting/test_plot_helpers.py`
  - `pytest tests/analysis/test_stats_computations.py`
  - fitACE workflow-common target test after the matching fitACE path change
  - `ruff check simace/analysis/stats simace/plotting workflow/scripts/simace`

## Assumptions

- Existing old result directories must be regenerated; they will not remain plot-compatible.
- Performance success is timing visibility plus reduced full-pedigree read width; no hard wall-clock target until logs from representative scenarios identify the dominant cost.
- Glossary updates for "per-replicate stats report" and "plotting sample" are part of the docs change.
