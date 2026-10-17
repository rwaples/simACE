# ADR 0026: Split mate-correlation checks by generation when pooling is invalid

## Status

Accepted and implemented 2026-10-16 for issue #64, in
`simace/analysis/validate/assortative_mating.py` (`recorded_matings`,
`validate_assortative_mating`) and
`simace/analysis/mate_correlation.py` (`mate_correlation_by_generation`).

**Status note (2026-10-17).** [ADR 0028](0028-analyze-validate-boundary.md)
classes the per-generation `mate_corr_*_gen{g}` checks as calibrated tests: at
about 3 SE near N = 1000 they fail correct reps. They move to `simace
validate` with a valid SE, and the multiplicity rule this ADR deferred
arrives with that move. Until then they stay in Analyze unchanged.

## Context

`validate_assortative_mating` compares the Pearson mate correlation over every
Mating Pair in the recorded pedigree against one expected value per cell. The
simulator instead targets each **mating iteration** separately
(`AssortmentPlan.for_generation`). Per-generation `assort1`/`assort2` and
per-generation C/E therefore give pairs from different generations different
targets. Until now the validator answered both cases with a passing skip.

A constant target is not enough for pooling. The pooled Pearson weights each
generation's covariance by its scale. Same-trait cells survive a change of
scale: with a common correlation and equal means, the pooled value equals that
correlation. Cross-trait cells mix the two traits' scales and do not. In a
probe (N=20000, cross target 0.2 throughout, E1 and E2 swapping between 0.1
and 5.0), the per-generation cross correlations were 0.204, 0.197 and 0.206,
while the pooled value was 0.135.

The target was also incomplete. Single-trait assortment induces correlations
in the other cells (`expected_mate_corr_matrix`), and the simulator mates on
the `assort_matrix` diagonal, not on the raw `assort1`/`assort2` that
`params.yaml` echoes. Comparing the diagonal against the configured `assort_t`
made `mate_corr_liability1` fail on a correct simulation: observed 0.259
against expected 0.0, where the induced target is 0.2.

## Decision

1. **Target.** The expected value of each recorded mating iteration $i$ is
   the full matrix `expected_mate_corr_matrix`, with assortment resolved at
   $i$ and C/E of the parents' generation, `max(0, i - 1)`. As in the
   simulator, a set `assort_matrix` supplies the assortment through its
   diagonal and is mated on only when both diagonal entries are nonzero, and
   the Wright-Fisher mating model has zero targets. Diagonal cells are always
   scored. Cross cells are scored when both traits assort at $i$.
2. **Pool only when valid.**
   - `mate_corr_liability{t}` stays pooled when its expected value is
     constant over recorded matings.
   - `mate_corr_cross_12` and `mate_corr_cross_21` stay pooled only when,
     over recorded matings, the cell is scored in every iteration, its target
     is constant, and C1, C2, E1 and E2 are constant.
   - Each check family decides on its own.
3. **Split otherwise.** A check that cannot pool becomes one check per
   parents' generation, `{name}_gen{g}`, where $g$ is the parents' 0-based
   `generation` value. A cross check is emitted only for generations where
   it is scored. The pooled correlation stays under the check's own name as
   information, with no pass/fail: it is still the realised value of the
   recorded pedigree, but it has no single target.
4. **Tolerance.** Each check, pooled or split, keeps `max(0.1, 3·se)`, with
   se from its own pair count. A split generation with no Mating Pairs
   fails, because the simulator recorded matings there.

Rejected alternatives:

- **Always split.** This costs $G_{\text{ped}} - 1$ extra mate-correlation
  passes on every assortment scenario and renames every existing check.
- **Stable names with nested per-generation results.** The atlas would show
  one row per check, and a failing generation would surface only in the
  details text.
- **pg-phenotype's `stratum=`.** It returns one pooled stratified estimate
  per cell, not a value per stratum, and it builds Mate Networks.

## Consequences

The set of scored mate-correlation checks in `report.yaml` depends on the
scenario. Consumers that read a pooled name (`report.py`, `report_schema.py`,
`plot_validation.py`) still get its `observed` value.

The `mate_correlation` plot keeps a pooled observed panel. Where a cell is
not comparable with its target that way, its title points to the
per-generation checks.

Split scenarios pay one pg-phenotype call per recorded parents' generation.
At N=200k with G_ped=6 that was 0.56 s, against 0.14 s for one pooled call.
Scenarios that pool keep one call.

The tolerance floor still catches only gross errors; tightening it needs a
multiplicity rule and a realised-versus-target bias study, which this ADR
leaves for later. Induced cross cells under single-trait assortment are still
unscored.

The cross-trait pooling rule reads only the configured C/E. Assortment also
grows $\mathrm{Var}(A)$ by generation, at different rates for two traits that
assort with different strength, so the trait-scale ratio can drift under
constant C/E and pull a pooled cross correlation slightly below its target.
The pooled cross checks had the same exposure before this decision.

Splitting triggers on any change in a target, however small next to the
tolerance. A threshold would pool targets that differ, which have no
single expected value.
